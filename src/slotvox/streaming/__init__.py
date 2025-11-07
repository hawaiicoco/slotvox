"""Chunked streaming understanding with bounded buffers.

Streaming sessions consume audio in chunks through a hard-bounded buffer,
commit frame-level slot tags as lookahead permits, and maintain partial
intent hypotheses with stability flags. For greedy decoding the committed
decisions equal the offline ones — see docs/streaming.md. The buffer and
its contract are torch-free; sessions require the torch extra and are
exported lazily. NOT real speech: streams are synthetic factory clips.
"""

from __future__ import annotations

import importlib
from typing import Any

from slotvox.streaming.buffer import AudioBuffer

_TORCH_NAMES = {
    "IntentHypothesis": "slotvox.streaming.session",
    "StreamResult": "slotvox.streaming.session",
    "StreamSession": "slotvox.streaming.session",
}

__all__ = [
    "AudioBuffer",
    "IntentHypothesis",
    "StreamResult",
    "StreamSession",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve torch-backed exports with a helpful error when absent."""
    module_name = _TORCH_NAMES.get(name)
    if module_name is None:
        raise AttributeError(f"module 'slotvox.streaming' has no attribute {name!r}")
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            f"slotvox.streaming.{name} requires the torch extra: "
            "pip install 'slotvox[torch]' "
            "(CPU wheels: --index-url https://download.pytorch.org/whl/cpu)"
        ) from exc
    return getattr(importlib.import_module(module_name), name)
