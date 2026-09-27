# Usage guide

All examples use generated signals and run offline. No command downloads a
speech corpus or pretrained model.

## Set up an environment

The core package needs Python 3.11 or newer and NumPy. Install the development
environment from the lockfile:

```bash
uv sync --extra dev
```

Training, prediction, and model-backed streaming also need the CPU torch
extra:

```bash
uv sync --extra dev --extra torch
```

Run `uv run slotvox --help` to list commands and
`uv run slotvox <command> --help` for the authoritative options.

## Inspect a built-in schema

```bash
uv run slotvox schema weather
```

Built-in domains define the allowed intents, slots, value types, and language.
They are the input contract for dataset generation.

## Generate and featurize a dataset

```bash
uv run slotvox generate \
  --domain weather \
  --counts train=64,dev=16,test=16 \
  --seed 7 \
  --out outputs/weather

uv run slotvox featurize \
  --dataset outputs/weather \
  --out outputs/weather-features
```

Generation writes a versioned manifest, examples, and deterministic metadata.
Featurization writes log-mel arrays plus provenance that ties them to the
source dataset. See [dataset-format.md](dataset-format.md) for the exact layout.

## Train and run inference

The following commands require the torch extra:

```bash
uv run slotvox train \
  --domain weather \
  --counts train=64,dev=16 \
  --seed 7 \
  --epochs 12 \
  --out outputs/weather-model

sample_wav=$(find outputs/weather/audio -name '*.wav' -print -quit)
uv run slotvox predict \
  --model outputs/weather-model/model.ckpt \
  --audio "$sample_wav"
```

The model is intentionally small and CPU-oriented. Results measure the
synthetic task only and do not represent real-speech accuracy.

## Exercise streaming behavior

```bash
uv run slotvox stream-demo --domain weather --chunk-ms 160
```

This command demonstrates chunking, bounded buffering, partial hypotheses,
and finalization. See [streaming.md](streaming.md) for state and equivalence
guarantees.

## Export instruction data

```bash
uv run slotvox export-instructions \
  --data outputs/weather \
  --out outputs/weather-instructions
```

The export references generated audio through manifest IDs and includes schema
and provenance metadata. It does not include model weights or external data.

## Evaluate and render reports

```bash
uv run slotvox eval \
  --dataset outputs/weather \
  --model outputs/weather-model/model.ckpt \
  --split dev \
  --out outputs/evaluation.json

uv run slotvox report \
  --run outputs/evaluation.json \
  --out outputs/report
```

Reports include limitations and reproducibility metadata. Metric definitions
and confidence-interval behavior are available through the API guide and
source docstrings.

## Try the adapter protocol

```bash
uv run slotvox serve-mock --port 8765
```

The server binds to loopback for offline protocol tests. Use fixture replay
when a test does not need HTTP. The request and response envelopes are
documented in [adapters.md](adapters.md).

Runnable scripts for dataset generation, joint training, and instruction
export are indexed in [the examples directory](../examples/README.md).
