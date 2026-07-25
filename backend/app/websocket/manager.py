"""Active session registry.

Holds the runners for sessions currently streaming, enforces the concurrency
cap, and persists each session when it finishes.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional

from ..config import get_settings
from ..models.memory import SessionSource
from ..pipeline.runner import SessionRunner
from ..services.ingest import Recording
from ..services.store import get_store
from ..simulation.generator import SimulatedRecording

log = logging.getLogger("mindscape.sessions")


class SessionBusy(RuntimeError):
    """Raised when the concurrency cap is reached."""


class SessionManager:
    """Tracks live sessions for the process."""

    def __init__(self) -> None:
        self._runners: Dict[str, SessionRunner] = {}
        self._lock = asyncio.Lock()

    async def create(
        self,
        session_id: str,
        recording,
        source: SessionSource,
        title: Optional[str] = None,
        speed: float = 1.0,
    ) -> SessionRunner:
        settings = get_settings()

        async with self._lock:
            if len(self._runners) >= settings.max_concurrent_sessions:
                raise SessionBusy(
                    f"{settings.max_concurrent_sessions} sessions already running"
                )

            runner = SessionRunner(
                session_id=session_id,
                recording=recording,
                source=source,
                title=title,
                speed=speed,
            )
            self._runners[session_id] = runner
            log.info(
                "Session %s created (%s, %.1fs, %d ch)",
                session_id,
                source,
                runner.duration,
                len(recording.channels),
            )
            return runner

    def get(self, session_id: str) -> Optional[SessionRunner]:
        return self._runners.get(session_id)

    async def finish(self, session_id: str, persist: bool = True) -> None:
        """Remove a session, optionally writing it to the store."""
        async with self._lock:
            runner = self._runners.pop(session_id, None)

        if runner is None:
            return

        if persist:
            try:
                record = runner.to_record()
                if record is not None:
                    # File IO off the event loop.
                    await asyncio.to_thread(get_store().save, record)
                    log.info(
                        "Session %s stored (%s, confidence %.2f, %d fragments)",
                        session_id,
                        record.biome,
                        record.confidence,
                        len(record.fragments),
                    )
            except Exception:  # noqa: BLE001
                log.exception("Failed to persist session %s", session_id)

    @property
    def active_ids(self) -> List[str]:
        return list(self._runners)

    @property
    def count(self) -> int:
        return len(self._runners)


_manager: Optional[SessionManager] = None


def get_manager() -> SessionManager:
    global _manager
    if _manager is None:
        _manager = SessionManager()
    return _manager
