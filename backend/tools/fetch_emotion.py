"""Download DENS — the continuous naturalistic emotion EEG dataset.

ds003751 — Mishra, Asif, Tiwary & Srinivasan (2023), *Dataset on Emotion with
Naturalistic Stimuli (DENS)*. Forty subjects watched emotionally evocative
multimedia video clips while EEG was recorded continuously.

This is the dataset the video reconstruction runs on, and it is the right one
for a specific reason: subjects clicked a mouse **at the moment they felt the
emotion**, so there is a per-moment timestamp to validate a per-frame decoder
against — not only a single rating for the whole clip.

Run with::

    backend/.venv/bin/python -m tools.fetch_emotion --subjects 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import List

DATASET = "ds003751"
TAG = "1.0.6"
BASE = f"https://openneuro.org/crn/datasets/{DATASET}/snapshots/{TAG}/files"
GRAPHQL = "https://openneuro.org/crn/graphql"

ROOT = Path(__file__).resolve().parent.parent.parent / "data" / DATASET

METADATA = ["README", "participants.tsv", "dataset_description.json"]


def fetch(remote: str, destination: Path, retries: int = 3) -> bool:
    if destination.exists() and destination.stat().st_size > 0:
        return True
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/{remote}"

    for attempt in range(retries):
        try:
            temporary = destination.with_suffix(destination.suffix + ".part")
            with urllib.request.urlopen(url, timeout=300) as response:
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
                print(f"  ! {remote}: {exc}")
                return False
            time.sleep(2 * (attempt + 1))
    return False


def list_subjects() -> List[str]:
    """Subject ids, read from the dataset's own participants file."""
    participants = ROOT / "participants.tsv"
    if not participants.exists():
        fetch("participants.tsv", participants)
    if not participants.exists():
        return []

    import csv

    with open(participants) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return sorted(r["participant_id"] for r in rows if r.get("participant_id"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", type=int, default=10)
    args = parser.parse_args()

    ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Target: {ROOT}\n")

    for name in METADATA:
        fetch(name, ROOT / name)

    subjects = list_subjects()[: args.subjects]
    if not subjects:
        print("Could not resolve subject list.")
        return 1

    print(f"Fetching {len(subjects)} subject(s): {', '.join(subjects)}\n")

    for subject in subjects:
        # Behaviour and events are small; get them first so a partial download
        # is still analysable for everything except the raw signal.
        for remote, local in [
            (
                f"{subject}:beh:{subject}_task-Emotion_beh.tsv",
                ROOT / subject / "beh" / f"{subject}_task-Emotion_beh.tsv",
            ),
            (
                f"{subject}:eeg:{subject}_task-emotion_events.tsv",
                ROOT / subject / "eeg" / f"{subject}_task-emotion_events.tsv",
            ),
            (
                f"{subject}:eeg:{subject}_task-Emotion_eeg.set",
                ROOT / subject / "eeg" / f"{subject}_task-Emotion_eeg.set",
            ),
        ]:
            fetch(remote, local)

        binary = ROOT / subject / "eeg" / f"{subject}_task-Emotion_eeg.fdt"
        if binary.exists() and binary.stat().st_size > 0:
            continue

        started = time.perf_counter()
        if fetch(f"{subject}:eeg:{subject}_task-Emotion_eeg.fdt", binary):
            size = binary.stat().st_size / 1e6
            elapsed = time.perf_counter() - started
            print(
                f"  {subject}  {size:.0f} MB in {elapsed:.0f}s "
                f"({size / max(elapsed, 1e-6):.1f} MB/s)"
            )

    total = sum(f.stat().st_size for f in ROOT.rglob("*") if f.is_file())
    print(f"\nDone. {total / 1e9:.2f} GB in {ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
