"""Download the real EEG dataset MindScape decodes from.

ds004018 — Grootswagers, Robinson & Carlson (2019), *The representational
dynamics of visual objects in rapid serial visual processing streams*,
NeuroImage 188:668-679. Sixteen subjects, 63-channel EEG at 1000 Hz, 200 object
images presented in 5 Hz RSVP streams, 40 repetitions each.

This dataset matters for a specific methodological reason. The widely-cited
EEG-to-image results (Spampinato et al. 2017 and the work built on that data)
used a *block design* — every image of a class shown consecutively — and Li et
al. 2020 demonstrated those models were largely decoding block position in
time rather than visual content. Their accuracy collapses under a randomised
design. This dataset randomises stimulus order within RSVP streams, so
decoding here reflects stimulus-driven response and nothing else.

Run with::

    backend/.venv/bin/python -m tools.fetch_dataset --subjects 4
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path
from typing import List

DATASET = "ds004018"
TAG = "2.0.0"
BASE = f"https://openneuro.org/crn/datasets/{DATASET}/snapshots/{TAG}/files"

ROOT = Path(__file__).resolve().parent.parent.parent / "data" / DATASET

#: Small files that describe the experiment.
METADATA = [
    "README",
    "participants.tsv",
    "dataset_description.json",
    "task-rsvp_eeg.json",
    "task-rsvp_events.json",
]


def fetch(remote: str, destination: Path, retries: int = 3) -> bool:
    """Download one file, skipping if already present and non-empty."""
    if destination.exists() and destination.stat().st_size > 0:
        return True

    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/{remote}"

    for attempt in range(retries):
        try:
            temporary = destination.with_suffix(destination.suffix + ".part")
            with urllib.request.urlopen(url, timeout=180) as response:
                with open(temporary, "wb") as handle:
                    while True:
                        chunk = response.read(1 << 20)
                        if not chunk:
                            break
                        handle.write(chunk)
            temporary.replace(destination)
            return True
        except Exception as exc:  # noqa: BLE001
            if attempt == retries - 1:
                print(f"  ! failed {remote}: {exc}")
                return False
            time.sleep(2 * (attempt + 1))
    return False


def stimulus_names() -> List[str]:
    """The 200 object images plus target images, by convention of the archive."""
    # The archive lists stimuli as stim001..stim216 with descriptive suffixes.
    # Names are recovered from the events file rather than guessed.
    events = ROOT / "sub-01" / "eeg" / "sub-01_task-rsvp_run-01_events.tsv"
    if not events.exists():
        return []

    import csv

    names = set()
    with open(events) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            name = row.get("stimulusname", "").strip()
            if name and name.endswith(".png"):
                names.add(name)
    return sorted(names)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subjects", type=int, default=4, help="How many subjects to fetch (1-16)"
    )
    parser.add_argument(
        "--runs", default="01", help="Comma-separated run ids, e.g. 01 or 01,02"
    )
    args = parser.parse_args()

    subjects = [f"sub-{i:02d}" for i in range(1, min(args.subjects, 16) + 1)]
    runs = [r.strip() for r in args.runs.split(",") if r.strip()]

    ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Target: {ROOT}")
    print(f"Subjects: {', '.join(subjects)} · runs {', '.join(runs)}\n")

    print("Metadata…")
    for name in METADATA:
        fetch(name, ROOT / name)

    # Events first — they name the stimuli we then download.
    print("Event files…")
    for subject in subjects:
        for run in runs:
            remote = f"{subject}:eeg:{subject}_task-rsvp_run-{run}_events.tsv"
            fetch(remote, ROOT / subject / "eeg" / remote.split(":")[-1])

    print("Stimulus images…")
    names = stimulus_names()
    if not names:
        print("  ! no events file yet; cannot resolve stimulus names")
    else:
        missing = 0
        for i, name in enumerate(names):
            target = ROOT / "stimuli" / name
            if not fetch(f"stimuli:{name}", target):
                missing += 1
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(names)}")
        print(f"  {len(names) - missing}/{len(names)} images")

    print("\nEEG recordings (large — several hundred MB each)…")
    for subject in subjects:
        for run in runs:
            for extension in ("vhdr", "vmrk", "eeg"):
                filename = f"{subject}_task-rsvp_run-{run}_eeg.{extension}"
                remote = f"{subject}:eeg:{filename}"
                destination = ROOT / subject / "eeg" / filename
                if destination.exists() and destination.stat().st_size > 0:
                    continue
                started = time.perf_counter()
                ok = fetch(remote, destination)
                if ok and extension == "eeg":
                    size = destination.stat().st_size / 1e6
                    elapsed = time.perf_counter() - started
                    print(
                        f"  {filename}  {size:.0f} MB in {elapsed:.0f}s "
                        f"({size / max(elapsed, 1e-6):.1f} MB/s)"
                    )

    total = sum(f.stat().st_size for f in ROOT.rglob("*") if f.is_file())
    print(f"\nDone. {total / 1e9:.2f} GB in {ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
