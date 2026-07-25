"""Optional OpenAI-compatible completion client.

The product is fully functional without this. When no endpoint is configured,
``complete`` returns ``None`` immediately and callers fall back to their local
generator. Failures are logged once and never propagate — a narrative is a
finishing touch, not something worth failing a session over.
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx

from ..config import get_settings

log = logging.getLogger("mindscape.llm")

_warned = False


async def complete(prompt: str, max_tokens: int = 320) -> Optional[str]:
    """Single-turn completion. ``None`` on any failure or missing config."""
    global _warned

    settings = get_settings()
    if not settings.llm_enabled:
        return None

    url = settings.llm_base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": settings.llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.8,
    }
    headers = {
        "Authorization": f"Bearer {settings.llm_api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=settings.llm_timeout) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        choices = data.get("choices") or []
        if not choices:
            return None

        content = choices[0].get("message", {}).get("content")
        return content.strip() if isinstance(content, str) and content.strip() else None

    except Exception as exc:  # noqa: BLE001 — never fail a session over this
        if not _warned:
            log.warning(
                "LLM narrative generation unavailable (%s); using local generator", exc
            )
            _warned = True
        return None
