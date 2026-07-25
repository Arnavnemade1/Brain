"""Session persistence.

JSON files on disk, one per session, with an in-memory index. Deliberately
simple: the interesting engineering in MindScape is upstream of storage, and a
file-backed store keeps the whole stack runnable with no external services.

The interface is narrow enough that swapping in a real database later means
reimplementing this one class.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional

from ..config import get_settings
from ..models.memory import SessionRecord, SessionSummary, SessionVersion

log = logging.getLogger("mindscape.store")


class SessionStore:
    """Thread-safe file-backed store for completed sessions."""

    def __init__(self, directory: Optional[Path] = None) -> None:
        settings = get_settings()
        self.directory = directory or (settings.data_dir / "sessions")
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._index: Dict[str, SessionSummary] = {}
        self._load_index()

    # ------------------------------------------------------------------ index

    def _path_for(self, session_id: str) -> Path:
        # Session ids are generated from a fixed alphabet, but this store is
        # also reachable with ids from request paths — reject anything that
        # could escape the data directory.
        safe = "".join(c for c in session_id if c.isalnum() or c in "-_")
        if not safe or safe != session_id:
            raise ValueError(f"invalid session id: {session_id!r}")
        return self.directory / f"{safe}.json"

    def _load_index(self) -> None:
        for path in sorted(self.directory.glob("*.json")):
            try:
                raw = json.loads(path.read_text())
                record = SessionRecord.model_validate(raw)
                self._index[record.id] = SessionSummary.model_validate(
                    record.model_dump(by_alias=True)
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("Skipping unreadable session %s: %s", path.name, exc)

        if self._index:
            log.info("Loaded %d stored session(s)", len(self._index))

    # ------------------------------------------------------------------ public

    def save(self, record: SessionRecord) -> SessionRecord:
        """Persist a session, appending to its version history if it exists."""
        with self._lock:
            existing = self.get(record.id)
            if existing is not None:
                record.version = existing.version + 1
                record.history = [
                    SessionVersion(
                        version=existing.version,
                        created_at=existing.created_at,
                        confidence=existing.confidence,
                        coherence=existing.coherence,
                        fragment_count=len(existing.fragments),
                        delta=_describe_delta(existing, record),
                    ),
                    *existing.history,
                ][:12]
                record.bookmarked = existing.bookmarked

            path = self._path_for(record.id)
            # Write to a temporary file and replace, so an interrupted write
            # cannot leave a half-written session behind.
            temp = path.with_suffix(".json.tmp")
            temp.write_text(json.dumps(record.model_dump(by_alias=True), indent=2))
            temp.replace(path)

            self._index[record.id] = SessionSummary.model_validate(
                record.model_dump(by_alias=True)
            )
            return record

    def get(self, session_id: str) -> Optional[SessionRecord]:
        try:
            path = self._path_for(session_id)
        except ValueError:
            return None
        if not path.exists():
            return None
        try:
            return SessionRecord.model_validate(json.loads(path.read_text()))
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to read session %s: %s", session_id, exc)
            return None

    def list(self, limit: int = 100, query: Optional[str] = None) -> List[SessionSummary]:
        """Summaries, newest first, optionally filtered by a text query."""
        with self._lock:
            items = list(self._index.values())

        items.sort(key=lambda s: s.created_at, reverse=True)

        if query:
            needle = query.strip().lower()
            if needle:
                items = [s for s in items if _matches(s, needle)]

        return items[:limit]

    def delete(self, session_id: str) -> bool:
        with self._lock:
            try:
                path = self._path_for(session_id)
            except ValueError:
                return False
            self._index.pop(session_id, None)
            if path.exists():
                path.unlink()
                return True
            return False

    def set_bookmark(self, session_id: str, bookmarked: bool) -> Optional[SessionRecord]:
        with self._lock:
            record = self.get(session_id)
            if record is None:
                return None
            record.bookmarked = bookmarked

            path = self._path_for(session_id)
            temp = path.with_suffix(".json.tmp")
            temp.write_text(json.dumps(record.model_dump(by_alias=True), indent=2))
            temp.replace(path)

            self._index[session_id] = SessionSummary.model_validate(
                record.model_dump(by_alias=True)
            )
            return record

    @property
    def count(self) -> int:
        return len(self._index)


def _matches(summary: SessionSummary, needle: str) -> bool:
    haystack = " ".join(
        [
            summary.title,
            summary.biome.replace("_", " "),
            summary.primary_emotion,
            summary.dominant_cognitive_state,
            summary.source,
        ]
    ).lower()
    return needle in haystack


def _describe_delta(previous: SessionRecord, current: SessionRecord) -> str:
    """One line summarising how a re-reconstruction changed."""
    parts: List[str] = []

    confidence_change = current.confidence - previous.confidence
    if abs(confidence_change) >= 0.02:
        direction = "rose" if confidence_change > 0 else "fell"
        parts.append(f"confidence {direction} {abs(confidence_change) * 100:.0f}%")

    fragment_change = len(current.fragments) - len(previous.fragments)
    if fragment_change:
        direction = "recovered" if fragment_change > 0 else "lost"
        parts.append(f"{abs(fragment_change)} fragment(s) {direction}")

    if current.biome != previous.biome:
        parts.append(
            f"setting reinterpreted as {current.biome.replace('_', ' ')}"
        )

    if current.primary_emotion != previous.primary_emotion:
        parts.append(f"affect shifted to {current.primary_emotion}")

    resolved = sum(
        1
        for region in current.scene.regions
        if region.confidence >= current.scene.fragment_threshold
    )
    previously_resolved = sum(
        1
        for region in previous.scene.regions
        if region.confidence >= previous.scene.fragment_threshold
    )
    if resolved != previously_resolved:
        parts.append(f"{resolved - previously_resolved:+d} regions resolved")

    return "; ".join(parts) if parts else "no material change"


_store: Optional[SessionStore] = None


def get_store() -> SessionStore:
    """Process-wide store singleton."""
    global _store
    if _store is None:
        _store = SessionStore()
    return _store
