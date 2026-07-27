"""Download the PhysioNet motor movement recordings.

EEG Motor Movement/Imagery Database (eegmmidb), Schalk et al. 2004, via
PhysioNet. 109 subjects, 64-channel BCI2000 at 160 Hz, recorded while people
physically opened and closed their fists and feet.

Only the **executed movement** runs are fetched. The database also contains
matched motor *imagery* runs, which are the ones most BCI papers use; the
question here is what someone's body was actually doing, so imagined movement
is the wrong condition and is left on the server.

    run 03, 07, 11   left fist  vs  right fist
    run 05, 09, 13   both fists vs  both feet

Roughly 15 MB per subject.

Run with::

    backend/.venv/bin/python -m tools.fetch_motor --subjects 20
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import List

BASE = "https://physionet.org/files/eegmmidb/1.0.0"
DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "eegmmidb"

#: Executed-movement runs. See the module docstring for what each pair means.
RUNS = (3, 5, 7, 9, 11, 13)

#: Recordings the database documents as defective — wrong sampling rate or
#: truncated annotation streams. Excluding them here rather than letting them
#: fail downstream, so a subject count means what it says.
EXCLUDED = frozenset({"S088", "S089", "S092", "S100"})


def subject_ids(count: int) -> List[str]:
    out: List[str] = []
    index = 1
    while len(out) < count and index <= 109:
        name = f"S{index:03d}"
        if name not in EXCLUDED:
            out.append(name)
        index += 1
    return out


def fetch(subject: str, run: int, force: bool = False) -> bool:
    """Download one run. Returns True if bytes were transferred."""
    name = f"{subject}R{run:02d}.edf"
    target = DATA_ROOT / subject / name
    if target.exists() and target.stat().st_size > 0 and not force:
        return False

    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/{subject}/{name}"

    # Download to a temporary name and move into place, so an interrupted
    # transfer never leaves a half file that later looks cached.
    staging = target.with_suffix(".edf.part")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            staging.write_bytes(response.read())
    except (urllib.error.URLError, TimeoutError) as error:
        staging.unlink(missing_ok=True)
        raise SystemExit(f"failed to fetch {url}: {error}")

    staging.replace(target)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    subjects = subject_ids(args.subjects)
    total = 0
    downloaded = 0

    for subject in subjects:
        for run in RUNS:
            if fetch(subject, run, force=args.force):
                downloaded += 1
            total += 1
        size = sum(p.stat().st_size for p in (DATA_ROOT / subject).glob("*.edf"))
        print(f"  {subject}  {size / 1e6:5.1f} MB")
        sys.stdout.flush()

    on_disk = sum(p.stat().st_size for p in DATA_ROOT.rglob("*.edf"))
    print(
        f"\n{len(subjects)} subjects, {total} runs "
        f"({downloaded} newly downloaded), {on_disk / 1e6:.0f} MB in {DATA_ROOT}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
