# On-disk formats

All slotvox artifacts are versioned, bounded, and revalidated on load.
Everything below is derived from synthetic data; no file format here
implies real speech corpora.

## Dataset directory — `slotvox.dataset` v1

```text
<dir>/dataset.json      envelope (written last = commit point)
<dir>/manifest.jsonl    one JSON object per example
<dir>/audio/<id>.wav    mono PCM16 audio
```

`dataset.json` keys: `schema`, `schema_version`, `example_count`,
`provenance` (regeneration inputs: domain, language, policy, seed, config),
`stats` (see below), `dataset_hash`.

`manifest.jsonl` row keys: `utterance_id`, `split`, `speaker_id`,
`snr_db`, `pattern_id`, `annotation` (tokens/tags/intent), `slot_values`
(surface forms per slot), `sample_rate`, `n_samples`, `segments`
(exact sample boundaries per token).

Loading revalidates keys strictly, rebuilds every object through its
constructor, and recomputes `dataset_hash` — a mismatch means corruption
or tampering and is a hard error. Audio passes through PCM16, so samples
match to quantization tolerance while labels match exactly.

## Feature directory — `slotvox.features` v1

```text
<dir>/features.json        envelope: feature config, dataset hash, count
<dir>/<utterance_id>.npz   mel: float32 [T, n_mels]
                           frame_token: int64 [T]  (-1 = gap frame)
```

`frame_token` maps each log-mel frame to the token whose segment contains
the frame center, or `-1` for gap frames; `frame_tags` lifts token-level
BIO tags to frames through this map. The envelope records the source
`dataset_hash` so consumers can refuse features that do not belong to the
labels they are paired with. Utterance ids are validated as safe filename
components before any path is built.

## Statistics envelope — `slotvox.dataset-stats` v1

Produced by `slotvox.data.stats.summarize` and embedded in `dataset.json`:
per-split counts, speakers, patterns, SNR levels, intent distributions,
slot mention counts, token-length and duration summaries, and the
`dataset_hash`. Every number is computed from the examples present.
