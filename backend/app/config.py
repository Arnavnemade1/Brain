"""Runtime configuration, resolved from the environment with safe defaults.

Everything here has a working default so the stack runs with no `.env` at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import List


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, "")) if os.getenv(key) else default
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, "")) if os.getenv(key) else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    app_name: str = "MindScape"
    version: str = "0.1.0"
    debug: bool = field(default_factory=lambda: _env_bool("MINDSCAPE_DEBUG", True))

    # --- Signal acquisition -------------------------------------------------
    sample_rate: int = field(default_factory=lambda: _env_int("MINDSCAPE_SAMPLE_RATE", 256))
    #: Analysis window length in seconds.
    window_seconds: float = field(
        default_factory=lambda: _env_float("MINDSCAPE_WINDOW_SECONDS", 2.0)
    )
    #: Fraction of each window shared with the previous one.
    window_overlap: float = field(
        default_factory=lambda: _env_float("MINDSCAPE_WINDOW_OVERLAP", 0.5)
    )
    #: Mains frequency to notch out (50 in most of the world, 60 in the Americas).
    line_frequency: float = field(
        default_factory=lambda: _env_float("MINDSCAPE_LINE_FREQUENCY", 60.0)
    )

    # --- Session behaviour --------------------------------------------------
    #: Default simulated session length in seconds.
    default_duration: float = field(
        default_factory=lambda: _env_float("MINDSCAPE_DEFAULT_DURATION", 120.0)
    )
    max_upload_bytes: int = field(
        default_factory=lambda: _env_int("MINDSCAPE_MAX_UPLOAD_BYTES", 64 * 1024 * 1024)
    )
    max_concurrent_sessions: int = field(
        default_factory=lambda: _env_int("MINDSCAPE_MAX_SESSIONS", 24)
    )

    # --- Persistence --------------------------------------------------------
    data_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("MINDSCAPE_DATA_DIR", Path(__file__).resolve().parent.parent / ".data")
        )
    )

    # --- CORS ---------------------------------------------------------------
    cors_origins: List[str] = field(
        default_factory=lambda: [
            o.strip()
            for o in os.getenv(
                "MINDSCAPE_CORS_ORIGINS",
                "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173",
            ).split(",")
            if o.strip()
        ]
    )

    @property
    def window_samples(self) -> int:
        return int(self.sample_rate * self.window_seconds)

    @property
    def hop_samples(self) -> int:
        return max(1, int(self.window_samples * (1.0 - self.window_overlap)))

    @property
    def hop_seconds(self) -> float:
        return self.hop_samples / float(self.sample_rate)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings
