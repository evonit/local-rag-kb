# SPDX-License-Identifier: Apache-2.0

"""
Description: In ordered queue emitter implementation for streaming outputs Asyncronously.
"""


import asyncio, json
import traceback
from typing import Any, Dict
from fastapi import WebSocket

from shared_types.emitters.base_emitter import BaseEmitter


class QueueingEmitter(BaseEmitter):
    """메시지를 큐에 넣기만 하고, 실제 전송은 단일 writer task가 담당합니다."""
    def __init__(self, queue: asyncio.Queue[Dict[str, Any]]):
        self._q = queue

    async def send(self, msg: Dict[str, Any]):
        await self._q.put(msg)

    async def event(self, node: str, stream_id: str, event: str, extra: Dict[str, Any] | None = None):
        m: Dict[str, Any] = {"type": "event", "node": node, "stream_id": stream_id, "event": event}
        if extra:
            m.update(extra)
        await self.send(m)

    async def token(self, node: str, stream_id: str, text: str):
        await self.send({"type": "token", "node": node, "stream_id": stream_id, "text": text})

    async def json(self, node: str, stream_id: str, payload: Dict[str, Any]):
        await self.send({"type": "json", "node": node, "stream_id": stream_id, "payload": payload})

    async def done(self, node: str, stream_id: str, usage: Dict[str, Any] | None = None):
        m: Dict[str, Any] = {"type": "done", "node": node, "stream_id": stream_id}
        if usage:
            m["usage"] = usage
        await self.send(m)


async def websocket_writer(ws: WebSocket, queue: asyncio.Queue[Dict[str, Any]]):
    """큐에서 메시지를 하나씩 읽어 WebSocket으로 전송.
    예외가 나면 종료되며, 호출측에서 task를 취소/정리해야 합니다.
    """
    try:
        while True:
            msg = await queue.get()
            await ws.send_text(json.dumps(msg, ensure_ascii=False))
    except Exception as e:
        # WebSocketDisconnect 포함 – 상위에서 정리
        traceback.print_exc()
        raise


def make_queueing_emitter_factory(ws: WebSocket, *, maxsize: int = 1000):
    queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=maxsize)
    writer_task = asyncio.create_task(websocket_writer(ws, queue))
    def factory(stream_id: str) -> BaseEmitter:
        return QueueingEmitter(queue)
    return factory, writer_task
