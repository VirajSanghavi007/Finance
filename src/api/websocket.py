from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from src.config.logging_config import get_logger

logger = get_logger(__name__)


class ConnectionManager:
    """Manages active WebSocket connections."""

    def __init__(self) -> None:
        self._active: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._active.append(ws)
        logger.info("ws_connected", n_clients=len(self._active))

    def disconnect(self, ws: WebSocket) -> None:
        self._active.remove(ws)
        logger.info("ws_disconnected", n_clients=len(self._active))

    async def broadcast(self, data: dict[str, Any]) -> None:
        payload = json.dumps(data)
        disconnected = []
        for ws in self._active:
            try:
                await ws.send_text(payload)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self._active.remove(ws)


manager = ConnectionManager()


async def signal_feed(ws: WebSocket) -> None:
    """
    WebSocket endpoint: pushes signal updates to subscribed clients.
    In production, this reads from a Redis pub/sub channel.
    For now, sends a heartbeat every 5 seconds.
    """
    await manager.connect(ws)
    try:
        while True:
            msg = {
                "type": "heartbeat",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            await ws.send_text(json.dumps(msg))
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        manager.disconnect(ws)
