# SPDX-License-Identifier: Apache-2.0

"""
Description: Implements a protocol for capturing emitted events, tokens, and JSON payloads.
"""


from typing import Any, Dict, List

from shared_types.emitters.base_emitter import BaseEmitter
from shared_types.emitters.emitter_factory import EmitterFactory


class CaptureEmitter(BaseEmitter):
    def __init__(self, store: List[Dict[str, Any]]):
        self.store = store
    async def send(self, msg: Dict[str, Any]):
        self.store.append(msg)
    async def event(self, node, stream_id, event, extra=None):
        m = {"type":"event","node":node,"stream_id":stream_id,"event":event}
        if extra: m.update(extra)
        await self.send(m)
    async def token(self, node, stream_id, text):
        await self.send({"type":"token","node":node,"stream_id":stream_id,"text":text})
    async def json(self, node, stream_id, payload):
        await self.send({"type":"json","node":node,"stream_id":stream_id,"payload":payload})
    async def done(self, node, stream_id, usage=None):
        m = {"type":"done","node":node,"stream_id":stream_id}
        if usage: m["usage"] = usage
        await self.send(m)

def make_capture_emitter_factory(store: List[Dict[str, Any]]) -> EmitterFactory:
    def factory(stream_id: str) -> BaseEmitter:
        return CaptureEmitter(store)
    return factory