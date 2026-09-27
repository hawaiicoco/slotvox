# API guide

slotvox exposes focused public APIs from its subpackages. Import from the
subpackages shown below; modules whose names start with an underscore are
implementation details.

## NumPy-only core

`slotvox.audio` reads, validates, frames, and writes bounded PCM WAV data.
The main entry points are `read_wav`, `write_wav`, `parse_wav`,
`frame_signal`, and the PCM conversion helpers.

`slotvox.features` contains the deterministic feature frontend:
`stft_power`, `mel_filterbank`, `log_mel`, and `log_mel_frames`.
Configuration dataclasses live in `slotvox.config`.

`slotvox.tagging` provides tokenization, BIO/BIOES validation and repair,
and conversions between tag sequences and `Span` values.

`slotvox.schema` defines intent, slot, domain, and annotated-utterance
schemas. Use `builtin_domain` or `builtin_domains` to load the bundled
domains, and the JSON helpers for bounded artifact I/O.

`slotvox.data` generates and persists synthetic datasets. A typical API
workflow is:

```python
from slotvox.config import GenerationConfig
from slotvox.data import generate_dataset, save_dataset

config = GenerationConfig(
    domain="weather",
    counts={"train": 64, "dev": 16, "test": 16},
    seed=0,
)
dataset = generate_dataset(config)
save_dataset(dataset, "outputs/weather")
```

See [dataset-format.md](dataset-format.md) for the files and schema versions
written by this layer.

`slotvox.eval` computes intent, slot, and joint-turn metrics and renders
Markdown or self-contained HTML reports. Metric functions accept explicit
records and do not download data.

`slotvox.instructions` builds versioned speech-instruction JSONL exports,
checks quality and provenance, and detects exact or near duplicates.

`slotvox.adapters` defines the inference request/response protocol. The
replay and loopback HTTP adapters are available without torch; `JointAdapter`
is resolved lazily and requires the optional model dependency.

## Optional torch APIs

Install the extra before importing model-backed objects:

```bash
uv sync --extra dev --extra torch
```

`slotvox.models` contains the joint intent/slot model, checkpoint helpers,
dataset collation, and deterministic training loop. `slotvox.streaming`
provides `StreamSession` and related model-backed streaming types while
keeping `AudioBuffer` available to core-only installs.

Code that supports core-only installs should import optional objects only in
the path that uses them and should surface the package's dependency error to
the caller.

## Errors and compatibility

Public validation failures derive from the exceptions in `slotvox.errors`.
Persisted artifacts include a schema identifier and version and are strictly
validated on load. Unknown schema versions are rejected rather than guessed.
See [architecture.md](architecture.md) for package boundaries and
[adapters.md](adapters.md) for the full inference contract.
