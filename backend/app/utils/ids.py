"""Identifier helpers.

Session ids are short, readable and sortable-ish so they read well in the UI
and in exported reports.
"""

from __future__ import annotations

import secrets
import time
from typing import Optional

_ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"


def short_id(length: int = 8) -> str:
    """URL-safe, unambiguous id (no look-alike characters)."""
    return "".join(secrets.choice(_ALPHABET) for _ in range(length))


def session_id() -> str:
    """Time-prefixed session id, e.g. ``s-lz4k-9qh2``."""
    stamp = int(time.time()) % (36**4)
    prefix = _base36(stamp).rjust(4, "0")
    return "s-{0}-{1}".format(prefix, short_id(4))


def fragment_id(session: str, index: int) -> str:
    return "{0}-f{1:03d}".format(session, index)


def region_id(session: str, index: int) -> str:
    return "{0}-r{1:02d}".format(session, index)


def _base36(value: int) -> str:
    if value == 0:
        return "0"
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    out = ""
    while value:
        value, rem = divmod(value, 36)
        out = digits[rem] + out
    return out


def seed_from(*parts: Optional[str]) -> int:
    """Deterministic 32-bit seed from arbitrary string parts."""
    h = 2166136261
    for part in parts:
        if part is None:
            continue
        for ch in part:
            h ^= ord(ch)
            h = (h * 16777619) & 0xFFFFFFFF
    return h & 0x7FFFFFFF
