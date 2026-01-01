# SPDX-License-Identifier: Apache-2.0

"""
Description: WebSocket Emitter for sending messages over WebSocket connections.
"""


import asyncio
import json
from typing import Dict, Any
from fastapi import WebSocket

from shared_types.emitters.base_emitter import BaseEmitter


class WebSocketEmitter(BaseEmitter):
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self._lock = asyncio.Lock()

    async def send(self, msg: Dict[str, Any]):
        # 단일 전송 락 – 동시 send 충돌 방지
        async with self._lock:
            await self.ws.send_text(json.dumps(msg, ensure_ascii=False))

    async def event(self, node: str, stream_id: str, event: str, extra: Dict[str, Any] | None = None):
        msg = {"type": "event", "node": node, "stream_id": stream_id, "event": event}
        if extra:
            msg.update(extra)
        await self.send(msg)

    async def token(self, node: str, stream_id: str, text: str):
        await self.send({"type": "token", "node": node, "stream_id": stream_id, "text": text})

    async def json(self, node: str, stream_id: str, payload: Dict[str, Any]):
        await self.send({"type": "json", "node": node, "stream_id": stream_id, "payload": payload})

    async def done(self, node: str, stream_id: str, usage: Dict[str, Any] | None = None):
        msg = {"type": "done", "node": node, "stream_id": stream_id}
        if usage:
            msg["usage"] = usage
        await self.send(msg)
