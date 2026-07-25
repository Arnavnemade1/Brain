"""MindScape API entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api import library, meta, sessions
from .config import get_settings
from .services.store import get_store
from .websocket import session_stream

logging.basicConfig(
    level=logging.DEBUG if get_settings().debug else logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)-22s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mindscape")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    store = get_store()
    log.info(
        "MindScape %s — %d Hz · %.1fs windows · %d stored session(s) · narratives: %s",
        settings.version,
        settings.sample_rate,
        settings.window_seconds,
        store.count,
        "LLM" if settings.llm_enabled else "local",
    )
    yield
    log.info("MindScape stopped")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="MindScape API",
        version=settings.version,
        description=(
            "Neural signal interpretation and memory reconstruction services for "
            "the MindScape platform."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(meta.router)
    app.include_router(sessions.router)
    app.include_router(library.router)
    app.include_router(session_stream.router)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Errors reach the client as structured JSON so the UI can present
        # something better than a blank failure.
        log.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal error while processing the request.",
                "path": request.url.path,
            },
        )

    return app


app = create_app()
