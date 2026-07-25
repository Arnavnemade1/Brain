"""Real-subject reconstruction sessions.

Holds the fitted retrieval reconstructor per subject and replays held-out
viewings through it, so the console can show what a person actually saw beside
what the system recovered from their EEG.

Subjects are loaded lazily and cached: epoch extraction reads a 500 MB
recording, which is far too slow to do per request.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .decoding import build_features, repetition_folds
from .epochs import available_subjects, load_or_extract
from .images import StimulusEmbeddings, load_or_build
from .reconstruct import Reconstruction, RetrievalReconstructor, encode_stimulus

log = logging.getLogger("mindscape.realdata.service")

DATA_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "data" / "ds004018"


@dataclass
class SubjectModel:
    """Everything needed to reconstruct viewings for one subject."""

    subject: str
    reconstructor: RetrievalReconstructor
    #: Held-out trials, never used to build the templates.
    test_x: np.ndarray
    test_stimulus: np.ndarray
    n_train: int
    n_channels: int
    n_times: int

    def viewings(self) -> List[int]:
        return sorted({int(v) for v in self.test_stimulus})


class RealDataService:
    """Lazily-loaded real EEG subjects and their reconstructors."""

    def __init__(self, root: Path = DATA_ROOT) -> None:
        self.root = root
        self._models: Dict[str, SubjectModel] = {}
        self._embeddings: Optional[StimulusEmbeddings] = None
        self._lock = threading.RLock()

    @property
    def available(self) -> bool:
        return self.root.exists() and bool(available_subjects(self.root))

    def subjects(self) -> List[str]:
        if not self.root.exists():
            return []
        return available_subjects(self.root)

    @property
    def stimuli_dir(self) -> Path:
        return self.root / "stimuli"

    def embeddings(self) -> StimulusEmbeddings:
        with self._lock:
            if self._embeddings is None:
                self._embeddings = load_or_build(self.root)
            return self._embeddings

    def model(self, subject: str) -> SubjectModel:
        """Fitted reconstructor for one subject, cached after first load."""
        with self._lock:
            if subject in self._models:
                return self._models[subject]

            log.info("loading real subject %s", subject)
            epochs = load_or_extract(self.root, subject)
            features = build_features(epochs)

            folds = repetition_folds(features.stimulus, n_folds=5)
            test_index = folds[0]
            train_index = np.concatenate(folds[1:])

            reconstructor = RetrievalReconstructor(
                self.embeddings(), self.stimuli_dir
            ).fit(features.x[train_index], features.stimulus[train_index])

            model = SubjectModel(
                subject=subject,
                reconstructor=reconstructor,
                test_x=features.x[test_index],
                test_stimulus=features.stimulus[test_index],
                n_train=int(train_index.size),
                n_channels=features.n_channels,
                n_times=features.n_times,
            )
            self._models[subject] = model
            log.info(
                "%s ready — %d training trials, %d held out",
                subject,
                model.n_train,
                test_index.size,
            )
            return model

    def reconstruct_viewing(
        self,
        subject: str,
        stimulus_number: Optional[int] = None,
        trials: int = 8,
        seed: int = 0,
    ) -> Dict:
        """Reconstruct one held-out viewing and return it ready for transport."""
        model = self.model(subject)
        rng = np.random.default_rng(seed)

        available = model.viewings()
        if not available:
            raise RuntimeError(f"{subject} has no held-out viewings")

        if stimulus_number is None:
            stimulus_number = int(rng.choice(available))

        rows = np.nonzero(model.test_stimulus == stimulus_number)[0]
        if rows.size == 0:
            raise KeyError(f"stimulus {stimulus_number} not held out for {subject}")

        rng.shuffle(rows)
        chunk = rows[: max(1, min(trials, rows.size))]

        reconstruction = model.reconstructor.reconstruct(
            model.test_x[chunk], truth_number=stimulus_number, top_k=8
        )

        return self._serialise(reconstruction, subject)

    def _serialise(self, reconstruction: Reconstruction, subject: str) -> Dict:
        stimuli = self.stimuli_dir

        candidates = []
        for candidate in reconstruction.candidates:
            candidates.append(
                {
                    "rank": candidate.rank,
                    "stimulusNumber": candidate.stimulus_number,
                    "label": candidate.label,
                    "levelA": candidate.level_a,
                    "levelB": candidate.level_b,
                    "score": round(candidate.score, 5),
                    "probability": round(candidate.probability, 5),
                    "correct": candidate.stimulus_number == reconstruction.truth_number,
                    "image": encode_stimulus(stimuli, candidate.filename),
                }
            )

        truth = self.embeddings().catalogue.get(reconstruction.truth_number or -1)

        return {
            "subject": subject,
            "trialsUsed": reconstruction.trials_used,
            "truth": {
                "stimulusNumber": reconstruction.truth_number,
                "label": truth.label if truth else "",
                "levelA": truth.level_a if truth else "",
                "levelB": truth.level_b if truth else "",
                "image": encode_stimulus(stimuli, reconstruction.truth_filename or ""),
            },
            "truthRank": reconstruction.truth_rank,
            "exact": reconstruction.is_exact,
            "inTop5": reconstruction.in_top5,
            "correctAnimacy": reconstruction.correct_animacy,
            "correctCategory": reconstruction.correct_category,
            "candidates": candidates,
        }


_service: Optional[RealDataService] = None


def get_real_data() -> RealDataService:
    global _service
    if _service is None:
        _service = RealDataService()
    return _service
