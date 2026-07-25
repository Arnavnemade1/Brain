"""Real-subject reconstruction endpoints.

Serves reconstructions from actual human EEG (ds004018) rather than the
forward-model simulation. Everything here is held-out data: the templates a
reconstruction is matched against never saw the trials being reconstructed.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from ..realdata.service import get_real_data

log = logging.getLogger("mindscape.api.realdata")

router = APIRouter(prefix="/api/real", tags=["real data"])


@router.get("/status")
async def status() -> dict:
    """Whether real-subject data is present, and which subjects are ready."""
    service = get_real_data()
    subjects = service.subjects()
    return {
        "available": bool(subjects),
        "subjects": subjects,
        "dataset": "ds004018",
        "citation": (
            "Grootswagers, Robinson & Carlson (2019), NeuroImage 188:668-679"
        ),
        "stimuli": 200,
        "hint": (
            None
            if subjects
            else "Run `python -m tools.fetch_dataset` in backend/ to download."
        ),
    }


@router.get("/subjects/{subject}/viewings")
async def viewings(subject: str) -> dict:
    """Stimulus numbers held out for this subject."""
    service = get_real_data()
    if subject not in service.subjects():
        raise HTTPException(status_code=404, detail=f"No data for {subject}")

    try:
        model = service.model(subject)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "subject": subject,
        "viewings": model.viewings(),
        "trainingTrials": model.n_train,
        "channels": model.n_channels,
        "timepoints": model.n_times,
    }


@router.get("/subjects/{subject}/reconstruct")
async def reconstruct(
    subject: str,
    stimulus: Optional[int] = Query(default=None, ge=1, le=200),
    trials: int = Query(default=8, ge=1, le=16),
    seed: int = Query(default=0),
) -> dict:
    """Reconstruct one held-out viewing from this subject's EEG."""
    service = get_real_data()
    if subject not in service.subjects():
        raise HTTPException(status_code=404, detail=f"No data for {subject}")

    try:
        return service.reconstruct_viewing(subject, stimulus, trials, seed)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("real reconstruction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/results")
async def results() -> dict:
    """Measured decoding and reconstruction accuracy, if the harnesses ran."""
    import json

    service = get_real_data()
    root = service.root / "derivatives"

    payload: dict = {"available": False}

    decoding = root / "decoding_results.json"
    reconstruction = root / "reconstruction_results.json"

    if decoding.exists():
        try:
            records = json.loads(decoding.read_text())
            payload["available"] = True
            payload["subjects"] = len(records)
            payload["animacy"] = sum(r["animacy"] for r in records) / len(records)
            payload["animacyShuffled"] = sum(
                r["animacy_shuffled"] for r in records
            ) / len(records)
            payload["category"] = sum(r["category"] for r in records) / len(records)
            payload["preStimulus"] = sum(r["pre_stimulus"] for r in records) / len(records)
            payload["peakLatencyMs"] = sum(r["peak_latency_ms"] for r in records) / len(
                records
            )
            identification = {}
            for key in ("1", "4", "10"):
                values = [
                    r["identification"].get(key) or r["identification"].get(int(key))
                    for r in records
                ]
                values = [v for v in values if v]
                if values:
                    identification[key] = {
                        "top1": sum(v["top1"] for v in values) / len(values),
                        "top5": sum(v["top5"] for v in values) / len(values),
                        "pairwise": sum(v["pairwise"] for v in values) / len(values),
                    }
            payload["identification"] = identification
        except Exception as exc:  # noqa: BLE001
            log.warning("could not read decoding results: %s", exc)

    if reconstruction.exists():
        try:
            records = json.loads(reconstruction.read_text())
            payload["reconstruction"] = {
                "exact": sum(r["exact"] for r in records) / len(records),
                "top5": sum(r["top5"] for r in records) / len(records),
                "animacy": sum(r["animacy"] for r in records) / len(records),
                "category": sum(r["category"] for r in records) / len(records),
            }
        except Exception as exc:  # noqa: BLE001
            log.warning("could not read reconstruction results: %s", exc)

    return payload
