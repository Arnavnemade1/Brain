"""Continuous reconstruction of a real-life image stream.

The gameplay reconstruction answers "what was happening" over a session. This
answers the harder question — "what was he *looking at*" — over a continuous
stream of real-world photographs, moment by moment, in the order they were
actually seen.

Two reconstructions are produced for every moment, and showing both is the
point:

* **one viewing** — what the EEG from that single 200 ms presentation alone
  supports. This is the honest real-time number, and it is mostly wrong.
* **consolidated** — the same moment decoded from every held-out viewing of
  that object in the session, averaged. This is what repeated exposure buys,
  and it is the condition under which the reconstruction becomes worth
  looking at.

Neither track ever sees the training data of the other, and no trial is ever
scored by a model that was fitted on it: predictions come from k-fold
cross-validation over repetition-balanced folds, so every moment in the output
was held out when it was decoded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .decoding import build_features, repetition_folds
from .images import Stimulus, load_catalogue
from .reconstruct import RetrievalReconstructor
from .semantics import SemanticSpace

log = logging.getLogger("mindscape.stream")

#: How many held-out viewings the consolidated track averages. Each fold holds
#: roughly this many repetitions of every stimulus, so it is set by the fold
#: count rather than chosen freely.
DEFAULT_FOLDS = 5

#: Candidates kept per moment, for the ranked strip in the render.
TOP_K = 5


@dataclass
class Moment:
    """One presentation, with what was shown and what was recovered."""

    index: int
    onset: float

    truth_number: int
    truth_filename: str
    truth_label: str
    truth_level_a: str
    truth_level_b: str

    #: Single-viewing reconstruction.
    live_number: int
    live_filename: str
    live_label: str
    live_rank: int
    #: Standard deviations by which the winner stood above the field.
    live_confidence: float
    live_similarity: float

    #: Consolidated over held-out repetitions of the same object.
    solid_number: int
    solid_filename: str
    solid_label: str
    solid_rank: int
    solid_confidence: float
    solid_similarity: float
    solid_viewings: int

    #: Top candidates from the consolidated track, for the ranked strip.
    solid_top: List[Tuple[int, str, float]]


@dataclass
class StreamReconstruction:
    subject: str
    moments: List[Moment]
    metrics: Dict[str, float]

    @property
    def duration(self) -> float:
        return self.moments[-1].onset - self.moments[0].onset if self.moments else 0.0


def _auc(score: Sequence[float], hit: Sequence[bool]) -> float:
    """Area under the ROC of evidence predicting correctness.

    0.5 means the system's own certainty says nothing about whether it is
    right, which is worth knowing before any of it is used to dim a panel.
    """
    values = np.asarray(score, dtype=float)
    labels = np.asarray(hit, dtype=bool)
    positive, negative = values[labels], values[~labels]
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    ranks = values.argsort().argsort() + 1
    rank_sum = ranks[labels].sum()
    return float(
        (rank_sum - positive.size * (positive.size + 1) / 2)
        / (positive.size * negative.size)
    )


def _summarise(
    moments: Sequence[Moment], space: SemanticSpace, n_candidates: int
) -> Dict[str, float]:
    """Per-frame scores for both tracks, against measured chance."""
    if not moments:
        return {}

    total = len(moments)
    out: Dict[str, float] = {
        "moments": float(total),
        "candidates": float(n_candidates),
        "chance_exact": 1.0 / n_candidates,
        "chance_top5": 5.0 / n_candidates,
        "chance_similarity": space.chance_similarity(),
        # How many viewings the consolidated track actually averaged. This is
        # set by how much held-out data a split leaves, not chosen, and it is
        # the reason two regimes can differ without either one leaking.
        "solid_viewings": float(np.mean([m.solid_viewings for m in moments])),
    }

    for track, rank_of, sim_of, guess_of in (
        (
            "live",
            [m.live_rank for m in moments],
            [m.live_similarity for m in moments],
            [m.live_number for m in moments],
        ),
        (
            "solid",
            [m.solid_rank for m in moments],
            [m.solid_similarity for m in moments],
            [m.solid_number for m in moments],
        ),
    ):
        ranks = np.array(rank_of)
        truth = [m.truth_number for m in moments]

        out[f"{track}_exact"] = float(np.mean(ranks == 0))
        out[f"{track}_top5"] = float(np.mean(ranks < 5))
        out[f"{track}_similarity"] = float(np.mean(sim_of))
        out[f"{track}_median_rank"] = float(np.median(ranks))

        tiers = space.tier_counts(truth, guess_of)
        for name, count in tiers.items():
            out[f"{track}_tier_{name}"] = count / total

        # Animacy and category agreement of the top candidate.
        animacy = category = 0
        for m, guess in zip(moments, guess_of):
            gi = space.position(int(guess))
            if gi is None:
                continue
            animacy += int(space.level_a[gi] == m.truth_level_a)
            category += int(space.level_b[gi] == m.truth_level_b)
        out[f"{track}_animacy"] = animacy / total
        out[f"{track}_category"] = category / total

        confidence = [
            m.live_confidence if track == "live" else m.solid_confidence
            for m in moments
        ]
        out[f"{track}_confidence_auc"] = _auc(confidence, list(ranks == 0))

    # How often the right *kind of thing* shows up in the ranked strip, even
    # when the exact photograph does not. Only the consolidated track keeps a
    # full candidate list, so this is reported for it alone.
    concept_hits = 0
    for m in moments:
        ti = space.position(m.truth_number)
        if ti is None:
            continue
        truth_concept = space.level_c[ti]
        for number, _label, _p in m.solid_top[:5]:
            gi = space.position(int(number))
            if gi is not None and space.level_c[gi] == truth_concept:
                concept_hits += 1
                break
    out["solid_top5_concept"] = concept_hits / total

    return out


def reconstruct_stream(
    root: Path,
    subject: str,
    n_folds: int = DEFAULT_FOLDS,
    split: str = "folds",
    max_moments: Optional[int] = None,
    shuffle_labels: bool = False,
    seed: int = 0,
) -> StreamReconstruction:
    """Reconstruct one subject's whole viewing stream, in temporal order.

    ``split`` is ``"folds"`` for repetition-balanced cross-validation, or
    ``"time"`` for the strict control: fit on the earlier part of the session,
    reconstruct the later part, with a gap between them. RSVP epochs are 700 ms
    long at a 200 ms stimulus interval, so neighbouring trials share samples
    and a random split can leak; the time split is what rules that out.

    ``shuffle_labels`` runs the identical pipeline with stimulus identities
    permuted. Everything else — folds, templates, retrieval, scoring — is
    untouched, so whatever it scores is what this pipeline produces from a
    signal that carries nothing.
    """
    from .epochs import load_or_extract

    epochs = load_or_extract(root, subject)
    features = build_features(epochs)

    catalogue = load_catalogue(root / "stimuli")
    space = SemanticSpace.from_catalogue(catalogue)

    stimulus = features.stimulus.copy()
    if shuffle_labels:
        # Permute the mapping from trial to identity. Note this must permute
        # the *labels*, not the features: permuting features would also
        # destroy the repetition structure the folds rely on.
        rng = np.random.default_rng(seed)
        stimulus = rng.permutation(stimulus)

    onsets = epochs.onset_s
    order = np.argsort(onsets, kind="stable")

    splits = _splits(split, stimulus, order, n_folds)
    decoded = np.zeros(stimulus.size, dtype=bool)

    # Predictions land here indexed by trial, so the stream can be walked in
    # time order afterwards regardless of which fold each trial fell in.
    live_rank = np.full(stimulus.size, -1, dtype=np.int32)
    live_guess = np.zeros(stimulus.size, dtype=np.int32)
    live_conf = np.zeros(stimulus.size, dtype=np.float32)
    solid_rank = np.full(stimulus.size, -1, dtype=np.int32)
    solid_guess = np.zeros(stimulus.size, dtype=np.int32)
    solid_conf = np.zeros(stimulus.size, dtype=np.float32)
    solid_views = np.zeros(stimulus.size, dtype=np.int32)
    solid_top: Dict[int, List[Tuple[int, str, float]]] = {}

    embeddings = _Embeddings(catalogue)

    for fold_index, (train_idx, test_idx) in enumerate(splits):
        if test_idx.size == 0 or train_idx.size == 0:
            continue
        decoded[test_idx] = True

        model = RetrievalReconstructor(embeddings, root / "stimuli")
        model.fit(features.x[train_idx], stimulus[train_idx])

        # --- single viewing -------------------------------------------------
        for i in test_idx:
            result = model.reconstruct(
                features.x[i], truth_number=int(stimulus[i]), top_k=1
            )
            live_guess[i] = result.top.stimulus_number
            live_conf[i] = result.top_z
            live_rank[i] = result.truth_rank if result.truth_rank is not None else 199

        # --- consolidated over held-out repetitions -------------------------
        # Averaging only within the test fold keeps every trial that
        # contributes to a consolidated estimate held out from the model that
        # scores it.
        for value in np.unique(stimulus[test_idx]):
            rows = test_idx[stimulus[test_idx] == value]
            result = model.reconstruct(
                features.x[rows], truth_number=int(value), top_k=TOP_K
            )
            top = [
                (c.stimulus_number, c.label, c.probability) for c in result.candidates
            ]
            for i in rows:
                solid_guess[i] = result.top.stimulus_number
                solid_conf[i] = result.top_z
                solid_rank[i] = (
                    result.truth_rank if result.truth_rank is not None else 199
                )
                solid_views[i] = rows.size
                solid_top[int(i)] = top

        log.info("%s: split %d/%d decoded", subject, fold_index + 1, len(splits))

    moments: List[Moment] = []
    count = 0
    for i in order:
        if max_moments is not None and count >= max_moments:
            break
        # Under a time split only the held-out tail has predictions; those
        # trials are not part of the reconstruction and must not be scored.
        if not decoded[i]:
            continue

        truth_number = int(stimulus[i])
        truth = catalogue.get(truth_number)
        if truth is None:
            continue

        live_stim = catalogue.get(int(live_guess[i]))
        solid_stim = catalogue.get(int(solid_guess[i]))

        moments.append(
            Moment(
                index=count,
                onset=float(onsets[i]),
                truth_number=truth_number,
                truth_filename=truth.filename,
                truth_label=truth.label,
                truth_level_a=truth.level_a,
                truth_level_b=truth.level_b,
                live_number=int(live_guess[i]),
                live_filename=live_stim.filename if live_stim else "",
                live_label=live_stim.label if live_stim else "",
                live_rank=int(live_rank[i]),
                live_confidence=float(live_conf[i]),
                live_similarity=space.score(truth_number, int(live_guess[i])),
                solid_number=int(solid_guess[i]),
                solid_filename=solid_stim.filename if solid_stim else "",
                solid_label=solid_stim.label if solid_stim else "",
                solid_rank=int(solid_rank[i]),
                solid_confidence=float(solid_conf[i]),
                solid_similarity=space.score(truth_number, int(solid_guess[i])),
                solid_viewings=int(solid_views[i]),
                solid_top=solid_top.get(int(i), []),
            )
        )
        count += 1

    metrics = _summarise(moments, space, n_candidates=space.size)
    return StreamReconstruction(subject=subject, moments=moments, metrics=metrics)


class _Embeddings:
    """Minimal stand-in exposing the catalogue the reconstructor needs.

    The full :class:`StimulusEmbeddings` also carries visual features, which
    retrieval here does not use — templates are learned from EEG directly.
    """

    def __init__(self, catalogue: Dict[int, Stimulus]) -> None:
        self.catalogue = catalogue


#: Trials skipped between train and test in the strict time split. One epoch
#: spans 700 ms at a 200 ms stimulus interval, so a trial overlaps at most the
#: three either side of it; 25 is comfortably clear of that.
TIME_SPLIT_GAP = 25


def _splits(
    kind: str, stimulus: np.ndarray, order: np.ndarray, n_folds: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Train/test index pairs for the requested validation scheme."""
    if kind == "time":
        cut = int(order.size * 0.6)
        train = order[:cut]
        test = order[min(cut + TIME_SPLIT_GAP, order.size) :]
        return [(train, test)]

    if kind != "folds":
        raise ValueError(f"unknown split {kind!r}; expected 'folds' or 'time'")

    folds = repetition_folds(stimulus, n_folds=n_folds)
    return [
        (np.concatenate([f for k, f in enumerate(folds) if k != i]), folds[i])
        for i in range(len(folds))
    ]
