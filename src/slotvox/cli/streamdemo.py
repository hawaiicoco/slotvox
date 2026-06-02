"""``slotvox stream-demo`` — chunked streaming over one synthetic clip.

Honest scope: this demonstrates streaming MECHANICS — bounded buffers,
commit lookahead, first-decision latency, partial hypotheses and
revisions — on a seeded UNTRAINED model. The decisions themselves are
arbitrary; no quality claim is made or implied. Fully offline; requires
the torch extra.
"""

from __future__ import annotations

from typing import Any

from slotvox.cli.common import EXIT_OK, emit
from slotvox.errors import ValidationError


def register_stream_demo_command(subparsers: Any) -> None:
    """Attach the ``stream-demo`` subcommand to ``subparsers``."""
    parser = subparsers.add_parser(
        "stream-demo", help="stream one synthetic clip through a session (mechanics demo)"
    )
    parser.add_argument("--domain", default="weather")
    parser.add_argument("--language", default="zh", choices=("zh", "en"))
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--chunk-ms", type=int, default=160)
    parser.add_argument("--encoder", default="tcn", choices=("tcn", "gru"))
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--n-mels", type=int, default=24)
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(handler=cmd_stream_demo)


def cmd_stream_demo(args: Any) -> int:
    """Stream a generated utterance chunk by chunk; report the mechanics."""
    try:
        import torch
    except ImportError as exc:
        raise ValidationError(
            "stream-demo requires the torch extra: pip install 'slotvox[torch]'"
        ) from exc

    from slotvox.config import FeatureConfig, GenerationConfig, StreamConfig, TrainConfig
    from slotvox.data.dataset import generate_dataset
    from slotvox.models.joint import JointIntentSlotModel, ModelSpec
    from slotvox.schema.builtins import builtin_domain
    from slotvox.streaming.session import StreamSession
    from slotvox.tagging.bio import repair_sequence, tags_to_spans
    from slotvox.util.seed import derive_seed

    feature_config = FeatureConfig(n_mels=args.n_mels)
    spec = ModelSpec.from_domain(
        builtin_domain(args.domain),
        feature_config,
        TrainConfig(encoder=args.encoder, hidden_size=args.hidden, layers=1, seed=args.seed),
    )
    torch.manual_seed(derive_seed("stream-demo", args.seed))
    model = JointIntentSlotModel(spec)
    model.eval()
    dataset = generate_dataset(
        GenerationConfig(domain=args.domain, counts={"train": 1}, n_speakers=1, seed=args.seed),
        language=args.language,
    )
    samples = dataset.examples[0].utterance.samples
    stream_config = StreamConfig(chunk_ms=args.chunk_ms, sample_rate=feature_config.sample_rate)
    session = StreamSession(model, feature_config, stream_config)
    chunk = stream_config.chunk_samples
    for start in range(0, len(samples), chunk):
        session.push_audio(samples[start : start + chunk])
    result = session.finalize()
    # an untrained head can emit structurally invalid greedy tags; the
    # promote repair documents exactly what the span count is based on
    spans = tags_to_spans(repair_sequence(result.frame_tags, "promote"))
    payload = {
        "intent": result.intent,
        "posterior": result.posterior,
        "stable": result.stable,
        "revisions": result.revisions,
        "frames": result.n_frames,
        "audio_ms": result.audio_ms,
        "first_frame_latency_ms": result.first_frame_latency_ms,
        "first_stable_latency_ms": result.first_stable_latency_ms,
        "span_count": len(spans),
    }
    lines = [
        "stream-demo: MECHANICS only — untrained seeded model, decisions are arbitrary.",
        f"audio:          {payload['audio_ms']:.1f} ms in {args.chunk_ms} ms chunks",
        f"frames:         {payload['frames']} committed",
        f"first frame at: {payload['first_frame_latency_ms']} ms of audio",
        f"first stable:   {payload['first_stable_latency_ms']} ms of audio",
        f"final intent:   {payload['intent']} (posterior {payload['posterior']:.4f}, "
        f"stable={payload['stable']}, revisions={payload['revisions']})",
        f"slot spans:     {payload['span_count']}",
    ]
    emit(args, payload, lines)
    return EXIT_OK
