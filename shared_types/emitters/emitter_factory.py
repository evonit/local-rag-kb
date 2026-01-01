# SPDX-License-Identifier: Apache-2.0

"""
Description: Factory for creating emitter instances.
"""



from typing import Protocol, Tuple, Any

from shared_types.emitters.base_emitter import BaseEmitter


class EmitterFactory(Protocol):
    def __call__(self, stream_id: str) -> BaseEmitter | Tuple[BaseEmitter, Any]: ...