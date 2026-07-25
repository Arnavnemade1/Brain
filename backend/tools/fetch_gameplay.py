"""Download the gameplay EEG dataset.

ds003517 — Cavanagh et al., *Continuous video game play*, NeuroImage 2016
(10.1016/j.neuroimage.2016.02.075). Seventeen people played an 8-bit
side-scrolling game ("Escape from Asteroid Axon") while EEG was recorded, with
every button press and game event timestamped into the EEG stream.

This is the closest public data to "someone doing an activity with a headset
on". It is not passive viewing: the player is acting, and the log records what
they did and what happened to them, moment by moment. That log is the ground
truth the reconstruction is scored against.

Run with::

    backend/.venv/bin/python -m tools.fetch_gameplay --subjects 6
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

DATASET = "ds003517"
TAG = "1.1.0"
BASE = f"https://openneuro.org/crn/datasets/{DATASET}/snapshots/{TAG}/files"
ROOT = Path(__file__).resolve().parent.parent.parent / "data" / DATASET

#: The gameplay run. run-01 is a separate oddball task.
RUN = "02"
TASK = "ContinuousVideoGamePlay"

METADATA = ["README", "participants.tsv", "dataset_description.json"]


def fetch(remote: str, destination: Path, retries: int = 3) -> bool:
    if destination.exists() and destination.stat().st_size > 0:
        return True
    destination.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(retries):
        try:
            temporary = destination.with_suffix(destination.suffix + ".part")
            with urllib.request.urlopen(f"{BASE}/{remote}", timeout=300) as response:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", type=int, default=6)
    args = parser.parse_args()

    ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Target: {ROOT}\n")

    for name in METADATA:
        fetch(name, ROOT / name)

    subjects = [f"sub-{i:03d}" for i in range(1, args.subjects + 1)]
    print(f"Fetching {len(subjects)} subject(s)\n")

    for subject in subjects:
        stem = f"{subject}_task-{TASK}_run-{RUN}"
        eeg_dir = ROOT / subject / "eeg"

        # Small files first: events are the ground truth and are analysable
        # even if a large signal download is interrupted.
        for extension in ("events.tsv", "channels.tsv", "electrodes.tsv", "eeg.json", "eeg.set"):
            fetch(f"{subject}:eeg:{stem}_{extension}", eeg_dir / f"{stem}_{extension}")

        binary = eeg_dir / f"{stem}_eeg.fdt"
        if binary.exists() and binary.stat().st_size > 0:
            continue

        started = time.perf_counter()
        if fetch(f"{subject}:eeg:{stem}_eeg.fdt", binary):
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
