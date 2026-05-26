"""``slotvox predict`` — offline inference for one WAV file.

Loads a trained checkpoint, reads a (strictly validated) mono PCM16 WAV,
and serves the decision through the SAME adapter protocol the HTTP path
uses, so CLI and service answers cannot drift apart. Requires the torch
extra.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from slotvox.cli.common import EXIT_OK, emit
from slotvox.errors import ValidationError

_ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")


def register_predict_command(subparsers: Any) -> None:
    """Attach the ``predict`` subcommand to ``subparsers``."""
    parser = subparsers.add_parser("predict", help="run offline inference on one WAV file")
    parser.add_argument("--model", required=True, help="checkpoint file (slotvox.checkpoint)")
    parser.add_argument("--audio", required=True, help="mono PCM16 WAV file")
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(handler=cmd_predict)


def cmd_predict(args: Any) -> int:
    """Understand one WAV file with a checkpointed model."""
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ValidationError(
            "predict requires the torch extra: pip install 'slotvox[torch]'"
        ) from exc

    import numpy as np

    from slotvox.adapters.model import JointAdapter
    from slotvox.adapters.protocol import InferRequest
    from slotvox.audio import read_wav
    from slotvox.models.checkpoint import load_checkpoint
    from slotvox.tagging.bio import tags_to_spans

    loaded = load_checkpoint(args.model)
    audio = read_wav(args.audio)
    expected = loaded.spec.feature_config.sample_rate
    if audio.sample_rate != expected:
        raise ValidationError(
            f"audio sample rate {audio.sample_rate} does not match the model's {expected}"
        )
    stem = Path(args.audio).stem.lower()
    request_id = stem if _ID_PATTERN.fullmatch(stem) else "cli-predict"
    adapter = JointAdapter(loaded.model)
    request = InferRequest(
        request_id, tuple(float(value) for value in np.asarray(audio.samples)), audio.sample_rate
    )
    response = adapter.infer(request)
    spans = tags_to_spans(response.frame_tags)
    payload = {
        "intent": response.intent,
        "posterior": response.posterior,
        "frame_count": len(response.frame_tags),
        "spans": [
            {"label": span.label, "start_frame": span.start, "end_frame": span.end}
            for span in spans
        ],
        "model_hash": response.model_hash,
    }
    lines = [
        f"intent:    {response.intent} (posterior {response.posterior:.4f})",
        f"frames:    {len(response.frame_tags)}",
        f"slot spans: {len(spans)}",
    ]
    emit(args, payload, lines)
    return EXIT_OK
