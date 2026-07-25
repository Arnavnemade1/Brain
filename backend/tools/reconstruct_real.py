"""Reconstruct what a person was looking at, and show the result.

Runs retrieval reconstruction on held-out real EEG and writes a contact sheet:
each row is one viewing, with the true image beside the top candidates the
system recovered from brain activity alone.

This is the deliverable behind "recognizable". Where the top candidate matches,
the reconstruction is literally the photograph the person saw.

Run with::

    backend/.venv/bin/python -m tools.reconstruct_real --trials 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO, format="  %(message)s")

from app.realdata.decoding import build_features, repetition_folds  # noqa: E402
from app.realdata.epochs import available_subjects, load_or_extract  # noqa: E402
from app.realdata.images import load_or_build, thumbnail  # noqa: E402
from app.realdata.reconstruct import RetrievalReconstructor  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "ds004018"


def contact_sheet(
    rows: List[Dict], stimuli_dir: Path, path: Path, top_k: int = 5, cell: int = 72
) -> None:
    """Write a PNG: true image on the left, ranked candidates to the right."""
    from PIL import Image, ImageDraw

    pad = 6
    gap = 18
    width = pad + cell + gap + top_k * (cell + pad) + pad
    height = pad + len(rows) * (cell + pad) + 28

    sheet = Image.new("RGB", (width, height), (8, 11, 20))
    draw = ImageDraw.Draw(sheet)

    draw.text((pad, 8), "shown", fill=(150, 200, 230))
    draw.text((pad + cell + gap, 8), "reconstructed from EEG  (ranked)", fill=(230, 190, 120))

    for r, row in enumerate(rows):
        y = 28 + r * (cell + pad)

        truth_path = stimuli_dir / row["truth_filename"]
        if truth_path.exists():
            image = Image.open(truth_path).convert("RGB").resize((cell, cell))
            sheet.paste(image, (pad, y))
        draw.rectangle(
            [pad, y, pad + cell, y + cell], outline=(90, 140, 180), width=1
        )

        for c, candidate in enumerate(row["candidates"][:top_k]):
            x = pad + cell + gap + c * (cell + pad)
            candidate_path = stimuli_dir / candidate["filename"]
            if candidate_path.exists():
                image = Image.open(candidate_path).convert("RGB").resize((cell, cell))
                sheet.paste(image, (x, y))
            # Green outline marks the correct object wherever it landed.
            hit = candidate["stimulus_number"] == row["truth_number"]
            colour = (90, 210, 150) if hit else (60, 66, 82)
            draw.rectangle([x, y, x + cell, y + cell], outline=colour, width=2 if hit else 1)

    sheet.save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", default="", help="Comma-separated, or all")
    parser.add_argument(
        "--trials", type=int, default=10, help="Trials averaged per reconstruction"
    )
    parser.add_argument("--examples", type=int, default=14, help="Rows in the contact sheet")
    args = parser.parse_args()

    subjects = (
        [s.strip() for s in args.subjects.split(",") if s.strip()]
        or available_subjects(ROOT)
    )
    if not subjects:
        print("No subjects available. Run: python -m tools.fetch_dataset")
        return 1

    embeddings = load_or_build(ROOT)
    stimuli_dir = ROOT / "stimuli"

    print("=" * 78)
    print(f"Reconstruction by retrieval — {args.trials} trial(s) averaged per viewing")
    print("=" * 78)

    overall: List[Dict] = []
    sheet_rows: List[Dict] = []

    for subject in subjects:
        epochs = load_or_extract(ROOT, subject)
        features = build_features(epochs)

        folds = repetition_folds(features.stimulus, n_folds=5)
        test_index = folds[0]
        train_index = np.concatenate(folds[1:])

        reconstructor = RetrievalReconstructor(embeddings, stimuli_dir).fit(
            features.x[train_index], features.stimulus[train_index]
        )

        rng = np.random.default_rng(11)
        results: List[Dict] = []

        for number in np.unique(features.stimulus):
            rows = test_index[features.stimulus[test_index] == number]
            if rows.size == 0:
                continue
            rng.shuffle(rows)
            chunk = rows[: min(args.trials, rows.size)]

            reconstruction = reconstructor.reconstruct(
                features.x[chunk], truth_number=int(number), top_k=8
            )
            results.append(
                {
                    "truth_number": reconstruction.truth_number,
                    "truth_filename": reconstruction.truth_filename,
                    "truth_rank": reconstruction.truth_rank,
                    "exact": reconstruction.is_exact,
                    "top5": reconstruction.in_top5,
                    "correct_animacy": reconstruction.correct_animacy,
                    "correct_category": reconstruction.correct_category,
                    "candidates": [
                        {
                            "rank": c.rank,
                            "stimulus_number": c.stimulus_number,
                            "filename": c.filename,
                            "label": c.label,
                            "probability": c.probability,
                        }
                        for c in reconstruction.candidates
                    ],
                }
            )

        exact = float(np.mean([r["exact"] for r in results]))
        top5 = float(np.mean([r["top5"] for r in results]))
        animacy = float(np.mean([bool(r["correct_animacy"]) for r in results]))
        category = float(np.mean([bool(r["correct_category"]) for r in results]))

        print(
            f"  {subject}  exact {exact * 100:5.1f}%   top-5 {top5 * 100:5.1f}%   "
            f"top-1 animacy {animacy * 100:5.1f}%   top-1 category {category * 100:5.1f}%"
            f"   (n={len(results)})"
        )

        overall.append(
            {
                "subject": subject,
                "exact": exact,
                "top5": top5,
                "animacy": animacy,
                "category": category,
            }
        )

        if subject == subjects[0]:
            # Prefer a readable mix: some exact hits, some near misses.
            hits = [r for r in results if r["exact"]]
            near = [r for r in results if not r["exact"] and r["top5"]]
            misses = [r for r in results if not r["top5"]]
            rng.shuffle(hits)
            rng.shuffle(near)
            rng.shuffle(misses)
            take = args.examples
            sheet_rows = (
                hits[: take // 2] + near[: take // 4] + misses[: take - take // 2 - take // 4]
            )

    print()
    print(f"  Across {len(overall)} subject(s), {args.trials} trials averaged:")
    print(f"    exact match (200-way)     {np.mean([o['exact'] for o in overall]) * 100:5.1f}%   chance 0.5%")
    print(f"    correct in top 5          {np.mean([o['top5'] for o in overall]) * 100:5.1f}%   chance 2.5%")
    print(f"    top-1 right animacy       {np.mean([o['animacy'] for o in overall]) * 100:5.1f}%   chance 50%")
    print(f"    top-1 right category      {np.mean([o['category'] for o in overall]) * 100:5.1f}%   chance 10%")

    if sheet_rows:
        output = ROOT / "derivatives" / "reconstruction_sheet.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        contact_sheet(sheet_rows, stimuli_dir, output)
        print(f"\n  contact sheet → {output}")

    summary = ROOT / "derivatives" / "reconstruction_results.json"
    summary.write_text(json.dumps(overall, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
