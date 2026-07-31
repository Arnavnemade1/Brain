"""Index every open EEG dataset worth pulling, without downloading any of it.

Two decisions shape this, and both are worth stating because the obvious
approach is different.

**It queries indices, not the open web.** OpenNeuro exposes a GraphQL API and
PhysioNet a directory listing; both carry licences, subject counts and BIDS
metadata. Scraping search results instead would return files with no
provenance and no licence, which is worse than useless for a project whose
value rests on every number tracing back to a control.

**It catalogues before it fetches.** Metadata for a thousand datasets costs a
few megabytes; the datasets themselves are terabytes. Deciding what is worth
downloading is a filtering problem, and filtering wants an index.

The other thing this encodes is what "more data" actually means. Pooling
heterogeneous EEG does not straightforwardly buy statistical power: a
motor-imagery recording contributes nothing to a question about emotion,
because the two measure different constructs under different task demands.
What helps is *more subjects on the same paradigm*, and *independent datasets
asking the same question*, so an effect can be tested for replication rather
than merely re-fitted. So every entry is tagged by paradigm, and the shortlist
is grouped by the open question it could answer.

Run with::

    backend/.venv/bin/python -m tools.harvest_catalogue
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

OPENNEURO = "https://openneuro.org/crn/graphql"
CATALOGUE = Path(__file__).resolve().parent.parent.parent / "data" / "catalogue"

#: Paradigm keywords, scored against name, tasks and README. Order breaks
#: ties only — the winner is whichever scores highest, not whichever appears
#: first. First-match-wins was tried and was badly wrong: with `affect` listed
#: first, any dataset whose README mentioned emotion anywhere in four thousand
#: characters was tagged affective, which filed a polysomnography corpus under
#: emotion.
PARADIGMS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "affect",
        ("emotion", "affect", "valence", "arousal", "mood", "empathy", "feeling"),
    ),
    (
        "naturalistic",
        ("naturalistic", "movie", "film", "narrative", "story", "free view",
         "free-view", "documentary", "cinema"),
    ),
    (
        "vision",
        ("visual", "object recognition", "image", "face", "scene", "rsvp",
         "perception", "visual oddball", "ssvep", "checkerboard", "n170",
         "n2pc", "flanker"),
    ),
    (
        "motor",
        ("motor", "movement", "imagery", "grasp", "reach", "finger", "hand",
         "gait", "walking", "erd"),
    ),
    (
        "memory",
        ("memory", "encoding", "retrieval", "recall", "recognition", "learning"),
    ),
    ("language", ("language", "reading", "semantic", "word", "sentence", "speech")),
    ("audio", ("auditory", "sound", "music", "listening", "tone", "mmn")),
    ("sleep", ("sleep", "dream", "nap", "drowsi")),
    ("rest", ("resting", "eyes open", "eyes closed", "baseline", "rest state")),
)

#: How much each paradigm could move the open questions in this project.
#: Stated here rather than inferred, so the ranking is auditable.
PRIORITY: Dict[str, Tuple[int, str]] = {
    "affect": (3, "emotion decoding failed; needs independent replication"),
    "naturalistic": (3, "the missing piece — real scenes with ground truth"),
    "vision": (2, "appearance and spatial-limit work rests on one dataset"),
    "motor": (2, "strongest domain; more subjects sharpen the ceiling"),
    "memory": (1, "adjacent, untested here"),
    "language": (0, "out of scope"),
    "audio": (0, "out of scope"),
    "sleep": (0, "out of scope"),
    "rest": (1, "useful only as a calibration baseline"),
    "other": (0, "unclassified"),
}


#: Labels added by hand, keyed by a substring of the dataset name.
#:
#: Keyword matching cannot recognise proper nouns, and some of the most useful
#: corpora name their tasks after them. The Healthy Brain Network releases list
#: `DespicableMe`, `ThePresent` and `DiaryOfAWimpyKid` — film titles, so the
#: word "movie" never appears anywhere in the metadata and no keyword list
#: reaches them. They are eleven releases of children watching actual film
#: clips, which is precisely the naturalistic stimulus this project lacks, so
#: they are tagged here after reading their task lists rather than left out.
OVERRIDES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("Healthy Brain Network", ("naturalistic",)),
)


@dataclass
class Entry:
    source: str
    accession: str
    name: str
    subjects: int
    sessions: int
    tasks: List[str]
    modalities: List[str]
    size_bytes: int
    paradigm: str
    paradigms: List[str]
    url: str
    readme_excerpt: str = ""

    @property
    def gigabytes(self) -> float:
        return self.size_bytes / 1e9


def _post(query: str, variables: Optional[Dict] = None, retries: int = 3) -> Dict:
    payload = json.dumps({"query": query, "variables": variables or {}}).encode()
    request = urllib.request.Request(
        OPENNEURO,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "mindscape-catalogue"},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt == retries - 1:
                raise
            # The index is a courtesy; back off rather than hammering it.
            time.sleep(2 * (attempt + 1))
    return {}


#: Field weights. A BIDS task label is the most deliberate description a
#: dataset gives of itself; a README is long, discursive, and mentions
#: everything the authors were thinking about.
WEIGHT_TASK = 6
WEIGHT_NAME = 4
WEIGHT_README = 1
#: README hits are capped so a chatty paragraph cannot outvote a task label.
README_CAP = 3


def classify(name: str, tasks: Sequence[str], readme: str) -> List[str]:
    """Every paradigm a dataset genuinely contains, strongest first.

    Multi-label, because single-label was hiding the most useful corpus in the
    index. The Healthy Brain Network releases carry `DespicableMe`, `ThePresent`
    and `DiaryOfAWimpyKid` — real movie clips, which is exactly the naturalistic
    stimulus this project lacks — alongside resting state and visual tasks. One
    label forced them to "memory", so a shortlist built for the naturalistic
    question never showed them.

    A paradigm qualifies only on a task-label or dataset-name hit. A mention in
    the README is not enough: authors discuss everything they were thinking
    about, and that is how a polysomnography corpus first got tagged affective.
    """
    task_text = " ".join(tasks).lower()
    name_text = name.lower()
    readme_text = readme[:4000].lower()

    scored: List[Tuple[int, int, str]] = []
    for rank, (paradigm, keywords) in enumerate(PARADIGMS):
        strong = WEIGHT_TASK * sum(k in task_text for k in keywords)
        strong += WEIGHT_NAME * sum(k in name_text for k in keywords)
        weak = WEIGHT_README * min(sum(k in readme_text for k in keywords), README_CAP)
        if strong > 0:
            # Negative rank keeps PARADIGMS order as the tie-break, toward the
            # more specific paradigm listed earlier.
            scored.append((strong + weak, -rank, paradigm))

    if not scored:
        # Nothing named a task or titled itself usefully; fall back to the
        # README so the entry is not simply lost.
        fallback = max(
            (
                (WEIGHT_README * min(sum(k in readme_text for k in kw), README_CAP), -r, p)
                for r, (p, kw) in enumerate(PARADIGMS)
            ),
            default=(0, 0, "other"),
        )
        return [fallback[2]] if fallback[0] > 0 else ["other"]

    scored.sort(reverse=True)
    return [paradigm for _, _, paradigm in scored]


def apply_overrides(name: str, labels: List[str]) -> List[str]:
    """Merge hand-checked labels for datasets keywords cannot reach."""
    for needle, extra in OVERRIDES:
        if needle.lower() in name.lower():
            for label in extra:
                if label not in labels:
                    labels = [label] + labels
    return labels


QUERY = """
query Page($after: String) {
  datasets(first: 25, after: $after) {
    pageInfo { hasNextPage endCursor }
    edges {
      node {
        id
        latestSnapshot {
          tag
          size
          readme
          description { Name }
          summary { modalities subjects sessions tasks totalFiles }
        }
      }
    }
  }
}
"""


def harvest_openneuro(limit: Optional[int] = None) -> List[Entry]:
    """Walk the OpenNeuro index, keeping anything with EEG in it."""
    entries: List[Entry] = []
    cursor: Optional[str] = None
    scanned = 0

    while True:
        data = _post(QUERY, {"after": cursor})
        block = data.get("data", {}).get("datasets")
        if not block:
            break

        for edge in block["edges"]:
            # The index returns null edges for datasets mid-deletion or with a
            # broken snapshot. One of them aborted a 1,150-dataset crawl, so
            # anything malformed is skipped rather than trusted.
            node = (edge or {}).get("node") if edge else None
            if not node or not node.get("id"):
                continue
            snapshot = node.get("latestSnapshot") or {}
            summary = snapshot.get("summary") or {}
            scanned += 1

            modalities = [m.upper() for m in (summary.get("modalities") or [])]
            # iEEG and MEG are different measurements with different ceilings;
            # only scalp EEG is comparable to what this project has.
            if "EEG" not in modalities:
                continue

            description = snapshot.get("description") or {}
            name = str(description.get("Name") or node["id"])
            readme = str(snapshot.get("readme") or "")
            tasks = [str(t) for t in (summary.get("tasks") or [])]
            subjects = summary.get("subjects") or []

            entries.append(
                Entry(
                    source="openneuro",
                    accession=node["id"],
                    name=name.strip(),
                    subjects=len(subjects) if isinstance(subjects, list) else int(subjects or 0),
                    sessions=len(summary.get("sessions") or []),
                    tasks=tasks,
                    modalities=modalities,
                    size_bytes=int(snapshot.get("size") or 0),
                    paradigm=(
                        labels := apply_overrides(name, classify(name, tasks, readme))
                    )[0],
                    paradigms=labels,
                    url=f"https://openneuro.org/datasets/{node['id']}",
                    readme_excerpt=re.sub(r"\s+", " ", readme)[:400],
                )
            )

        info = block["pageInfo"]
        print(f"  scanned {scanned}, kept {len(entries)} EEG", flush=True)
        if not info.get("hasNextPage"):
            break
        cursor = info.get("endCursor")
        if limit and scanned >= limit:
            break

    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="stop after N scanned")
    parser.add_argument("--out", default=str(CATALOGUE / "eeg_catalogue.json"))
    args = parser.parse_args()

    print("harvesting OpenNeuro…")
    entries = harvest_openneuro(limit=args.limit or None)

    entries.sort(
        key=lambda e: (-PRIORITY.get(e.paradigm, (0, ""))[0], -e.subjects)
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "harvested": time.strftime("%Y-%m-%d"),
                "sources": ["openneuro"],
                "count": len(entries),
                "totalTerabytes": sum(e.size_bytes for e in entries) / 1e12,
                "priority": {k: {"rank": v[0], "why": v[1]} for k, v in PRIORITY.items()},
                "datasets": [e.__dict__ for e in entries],
            },
            indent=2,
        )
    )

    # Grouped by *every* label, so a mixed dataset appears under each question
    # it could answer. Counts therefore sum to more than the dataset total.
    by_paradigm: Dict[str, List[Entry]] = {}
    for entry in entries:
        for label in entry.paradigms:
            by_paradigm.setdefault(label, []).append(entry)

    print(f"\n{len(entries)} EEG datasets, "
          f"{sum(e.size_bytes for e in entries) / 1e12:.1f} TB total\n")
    print(f"{'paradigm':<14} {'sets':>5} {'subjects':>9} {'size':>9}  relevance")
    for paradigm, _ in sorted(
        PRIORITY.items(), key=lambda kv: -kv[1][0]
    ):
        group = by_paradigm.get(paradigm, [])
        if not group:
            continue
        print(
            f"{paradigm:<14} {len(group):>5} {sum(g.subjects for g in group):>9} "
            f"{sum(g.size_bytes for g in group) / 1e12:>8.2f}T  "
            f"{PRIORITY[paradigm][1]}"
        )

    print("\nTop candidates by open question:\n")
    for paradigm in ("affect", "naturalistic", "vision", "motor"):
        group = sorted(by_paradigm.get(paradigm, []), key=lambda e: -e.subjects)[:6]
        if not group:
            continue
        print(f"  {paradigm.upper()} — {PRIORITY[paradigm][1]}")
        for entry in group:
            print(
                f"    {entry.accession:<12} {entry.subjects:>4} subj "
                f"{entry.gigabytes:>7.1f} GB  {entry.name[:58]}"
            )
        print()

    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
