# SPDX-License-Identifier: Apache-2.0

"""
Description: Console emitter implementation for streaming outputs to the console.
"""



import json
from typing import Dict, Any

from shared_types.emitters.base_emitter import BaseEmitter
from shared_types.emitters.emitter_factory import EmitterFactory


class ConsoleEmitter(BaseEmitter):
    async def send(self, msg: Dict[str, Any]):
        print(json.dumps(msg, ensure_ascii=False))

    async def event(self, node: str, stream_id: str, event: str, extra: Dict[str, Any] | None = None):
        line = {"type": "event", "node": node, "stream_id": stream_id, "event": event}
        if extra:
            line.update(extra)
        await self.send(line)

    async def token(self, node: str, stream_id: str, text: str):
        # 토큰은 실시간 이어붙여 보기 좋게 처리
        print(text, end="", flush=True)

    async def json(self, node: str, stream_id: str, payload: Dict[str, Any]):
        line = {"type": "json", "node": node, "stream_id": stream_id, "payload": payload}
        await self.send(line)

    async def done(self, node: str, stream_id: str, usage: Dict[str, Any] | None = None):
        line = {"type": "done", "node": node, "stream_id": stream_id}
        if usage:
            line["usage"] = usage
        await self.send(line)
        print()  # 줄바꿈



def make_console_emitter_factory() -> EmitterFactory:
    def factory(stream_id: str) -> BaseEmitter:
        return ConsoleEmitter()
    return factory