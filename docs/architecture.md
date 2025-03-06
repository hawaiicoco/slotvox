# Architecture

slotvox is organized as a pipeline of layers; each layer only depends on the
layers below it.

```text
audio → features → synth ─────────────┐
                                      ├→ data (factory / splits / stats)
schema (slots / intents / domains /   │        ↓
        annotations / serialize) ─────┴→ tagging (BIO / spans / transitions /
                                                    tokenizer)
        ↓                                            ↓
models (torch extra)  ⇄  streaming          instructions (JSONL pipeline)
        ↓                                            ↓
adapters (protocol / replay / http)  →  eval (metrics / bootstrap / report)
                                                     ↓
                                       cli + examples (offline entry points)
```

## Layers

| Layer | Modules | Responsibility |
| ----- | ------- | -------------- |
| Signal | `slotvox.audio`, `slotvox.features`, `slotvox.synth` | bounded WAV/PCM I/O with strict validation, framing/windows, log-mel features, deterministic synthetic "utterances" |
| Task definition | `slotvox.schema` | versioned intents, slot types with value constraints, dialogue domains, utterance↔annotation pairs, strict validation, JSON roundtrip |
| Labeling | `slotvox.tagging` | BIO/BIOES algebra with validity checks and repair policies, span extraction with boundary confidence, transition constraints, zh-char/en-word tokenization |
| Data | `slotvox.data` | scripted dialogue factory, leakage-free splits (per speaker/pattern), statistics and provenance hashes |
| Model | `slotvox.models` (torch extra) | compact causal encoder (TCN/GRU) over log-mel + intent head + frame-level slot head, weighted joint loss, deterministic training, checkpoints |
| Streaming | `slotvox.streaming` | bounded audio buffer, chunked inference, partial hypotheses with stability flags, emission/revision semantics, resumable state |
| Data products | `slotvox.instructions`, `slotvox.adapters` | instruction JSONL pipeline for speech LLMs; inference-service protocol with replay and loopback HTTP adapters |
| Evaluation | `slotvox.eval` | metrics with documented boundary policies, paired bootstrap CIs, slices, confusion matrices, run metadata, Markdown/HTML reports with limitations |
| Interface | `slotvox.cli`, `examples/` | offline subcommands and runnable examples |

## Design principles

1. **Synthetic by construction.** Labels exist because the signal generator
   placed them; there is no annotation step that could disagree with the
   audio. Every fixture is documented as NOT real speech.
2. **Determinism.** Explicit seeds everywhere; artifacts carry schema
   versions and provenance hashes; golden tests make drift deliberate.
3. **Bounded resources.** Buffers, file sizes, retry counts, and report
   sizes have hard limits enforced by tests.
4. **Protocol-first adapters.** Requests and responses are validated in both
   directions; adapters prove protocol behavior, never model quality.
5. **Honest reporting.** Numbers in docs and reports come from executed
   synthetic experiments; limitations sections are mandatory.
