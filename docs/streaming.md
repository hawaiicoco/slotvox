# Streaming understanding

`slotvox.streaming` runs the joint model over an audio stream in chunks,
with hard resource bounds and offline-comparable decisions. Everything
here operates on synthetic factory clips — not real speech.

## Semantics

- `push_audio(chunk)` appends to a bounded buffer and commits every frame
  whose audio **and** lookahead (`StreamConfig.lookahead_ms`) have
  arrived. `finalize()` commits the remaining complete frames without
  lookahead and returns the `StreamResult`.
- Committed slot tags are final: greedy per-frame argmax, emitted once,
  never revised. Intent is different: it is a *partial hypothesis*
  recomputed from accumulated frame states, flagged `stable` when the
  softmax posterior reaches `stability_threshold`. A later different
  stable intent is counted as a **revision** — reported, never hidden.
- Latency accounting is in audio time: `first_frame_latency_ms` and
  `first_stable_latency_ms` are milliseconds of pushed audio when those
  decisions first became available.

## Bounded resources

The audio buffer has a hard capacity (`buffer_capacity_ms`); appends that
would overflow are explicit errors, and consumed audio is dropped. The
encoder context is bounded too: the TCN replays a window of at most
`receptive_field - 1` cached mel frames per commit; the GRU carries only
its hidden state. A session streaming an hour of audio uses the same
memory as one streaming a second.

## Offline equivalence (the central invariant)

For greedy decoding, a chunked session and the offline path agree:

- committed frame tags equal the offline per-frame argmax tags **exactly**
  for every chunk size tested (the frame front end and the causal encoder
  window cover the same inputs),
- the final intent equals the offline intent (the streaming intent head
  consumes a running sum/count of the same frame states).

The tests pin this for both encoders (TCN and GRU) at several chunk sizes
with fixed seeds. Equivalence assumes the model is in eval mode and CPU
determinism; it is a statement about the pipeline, not about accuracy.

## Resumable state

`StreamSession.to_dict()` serializes the full session (schema
`slotvox.stream-state` v1): bounded buffer contents, frame cursor,
committed tags, mel context window, GRU hidden state, intent accumulator,
stability bookkeeping, and both configs. `from_dict` validates strictly —
including a `model_hash` guard that rejects states produced by a
different model — and a resumed session reproduces the uninterrupted
result exactly (pinned by tests).

## Limitations

- Decisions are greedy; beam or CRF-style decoding would need revisions
  of committed tags, which this design deliberately does not do.
- Lookahead trades latency for stability: with `lookahead_ms=0` tags
  commit immediately but never see right context (still offline-equal,
  since the encoder is causal).
- Latency numbers are audio-time accounting, not wall-clock measurements.
