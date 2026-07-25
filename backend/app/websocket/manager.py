"""Active session registry."""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional

from ..config import get_settings
from ..reconstruction.runner import ReconstructionRunner
from ..services.store import get_store

log = logging.getLogger("mindscape.sessions")


class SessionBusy(RuntimeError):
    """Raised when the concurrency cap is reached."""


class SessionManager:
    """Tracks live reconstruction sessions for the process."""

    def __init__(self) -> None:
        self._runners: Dict[str, ReconstructionRunner] = {}
        self._lock = asyncio.Lock()

    async def register(self, session_id: str, runner: ReconstructionRunner) -> None:
        settings = get_settings()
        async with self._lock:
            if len(self._runners) >= settings.max_concurrent_sessions:
                raise SessionBusy(
                    f"{settings.max_concurrent_sessions} sessions already running"
                )
            self._runners[session_id] = runner
            log.info(
                "Session %s registered (%s, %.1fs, %d ch)",
                session_id,
                runner.clip.id,
                runner.clip.duration_seconds,
                len(runner.recording.channels),
            )

    def get(self, session_id: str) -> Optional[ReconstructionRunner]:
        return self._runners.get(session_id)

    async def finish(self, session_id: str, persist: bool = True) -> None:
        async with self._lock:
            runner = self._runners.pop(session_id, None)
        if runner is None:
            return

        if persist:
            try:
                record = runner.to_record()
                if record is not None:
                    await asyncio.to_thread(get_store().save, record)
                    log.info(
                        "Session %s stored (%s, fidelity %.3f)",
                        session_id,
                        record.stimulus_id,
                        record.fidelity,
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
