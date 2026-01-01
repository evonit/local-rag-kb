# SPDX-License-Identifier: Apache-2.0

"""
Description: Base Emitter Protocol Definition
            make LLM workflow and its response be independent with emitter implementation.
"""



from typing import Protocol, Dict, Any


class BaseEmitter(Protocol):
    async def send(self, msg: Dict[str, Any]) -> None: ...
    async def event(self, node: str, stream_id: str, event: str, extra: Dict[str, Any] | None = None) -> None: ...
    async def token(self, node: str, stream_id: str, text: str) -> None: ...
    async def json(self, node: str, stream_id: str, payload: Dict[str, Any]) -> None: ...
    async def done(self, node: str, stream_id: str, usage: Dict[str, Any] | None = None) -> None: ...