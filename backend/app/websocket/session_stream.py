"""WebSocket endpoint for a live session.

Two concurrent tasks per connection: one pumps events out of the runner, the
other reads control commands. Whichever finishes first tears down the other.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from ..models.stream import StreamCommand
from .manager import get_manager

log = logging.getLogger("mindscape.ws")

router = APIRouter()


@router.websocket("/ws/session/{session_id}")
async def session_stream(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()

    runner = get_manager().get(session_id)
    if runner is None:
        await websocket.send_json(
            {
                "type": "error",
                "stage": None,
                "message": f"No active session {session_id}. Create one first.",
                "recoverable": False,
            }
        )
        await websocket.close(code=4404)
        return

    async def pump_events() -> None:
        async for event in runner.run():
            if websocket.client_state != WebSocketState.CONNECTED:
                break
            await websocket.send_json(event.wire())

    async def read_commands() -> None:
        while True:
            raw = await websocket.receive_json()
            try:
                command = StreamCommand.model_validate(raw)
            except Exception as exc:  # noqa: BLE001
                log.debug("Ignoring malformed command on %s: %s", session_id, exc)
                continue
            _apply(runner, command)

    producer = asyncio.create_task(pump_events(), name=f"events:{session_id}")
    consumer = asyncio.create_task(read_commands(), name=f"commands:{session_id}")

    try:
        done, pending = await asyncio.wait(
            {producer, consumer}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        for task in done:
            exc = task.exception()
            if exc is not None and not isinstance(exc, WebSocketDisconnect):
                raise exc

    except WebSocketDisconnect:
        log.info("Client disconnected from %s", session_id)
    except Exception:  # noqa: BLE001
        log.exception("Session stream %s failed", session_id)
    finally:
        producer.cancel()
        consumer.cancel()
        runner.stop()
        # Persist whatever was reconstructed, even on an early disconnect.
        await get_manager().finish(session_id, persist=True)

        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()


def _apply(runner, command: StreamCommand) -> None:
    if command.type == "pause":
        runner.pause()
    elif command.type in ("resume", "start"):
        runner.resume()
        if command.speed:
            runner.set_speed(command.speed)
    elif command.type == "stop":
        runner.stop()
    elif command.type == "speed" and command.speed:
        runner.set_speed(command.speed)
    elif command.type == "seek" and command.t is not None:
        runner.seek(command.t)
    elif command.type == "focus" and command.region_id:
        runner.focus_region(command.region_id)
