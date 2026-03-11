# Adapters and the inference protocol

`slotvox.adapters` defines the contract between slotvox and any serving
stack, plus fully offline test doubles for it. **Adapters validate
protocol behavior, not model quality** — a green adapter test says the
wire contract holds, never that predictions are good.

## Protocol

- Request (`slotvox.infer-request` v1): `request_id`, bounded raw
  `samples` (max `MAX_REQUEST_SAMPLES` = 480,000 ≈ 30 s at 16 kHz), and
  a `sample_rate` from a supported enum. JSON-native — no binary.
- Response (`slotvox.infer-response` v1): `request_id`, `intent`,
  `posterior` in [0, 1], `frame_tags` (a structurally valid BIO
  sequence), and a 64-hex `model_hash` pinning the deciding model.
- Both envelopes are described by `request_schema()` / `response_schema()`
  (a mini JSON-schema subset) and cross-checked against
  `slotvox.adapters.validator.validate` in tests. Validation is strict
  in BOTH directions: servers must reject bad requests, clients must
  reject bad responses.

## Implementations

| Adapter | Transport | Requires | Use |
| --- | --- | --- | --- |
| `ReplayAdapter` | in-process fixtures | nothing | deterministic offline tests, CLI demos |
| `MockInferServer` | loopback HTTP server | nothing | serving-path tests, `slotvox serve-mock` |
| `LocalHttpAdapter` | loopback HTTP client | nothing | timeouts/retries against the mock |
| `JointAdapter` | in-process model | torch extra | real inference from a trained checkpoint |

`LocalHttpAdapter` accepts only `http://127.0.0.1` / `http://localhost`
URLs — the offline guarantee is enforced, not promised. It retries only
documented transient conditions (network/timeout errors and HTTP 429,
500, 502, 503, 504) with a bounded attempt budget; every other status
fails fast. `MockInferServer.fail_next` / `delay_next` script those
transient conditions so retry and timeout paths are tested, not hoped
for.

## Fixture stores

`save_fixtures` / `load_fixtures` persist request/response pairs under
the `slotvox.adapter-fixtures` v1 envelope (atomic write, strict read).
A fixture whose response id does not match its request cannot be
constructed — the pairing rule is enforced at the type level.

## Limitations

- The protocol carries raw samples as JSON numbers: fine for tests and
  short clips, wrong for production bandwidth (a real deployment would
  use binary audio and gRPC-style framing).
- The mini validator implements only the keywords the protocol schemas
  use; unknown keywords are errors, not extensions.
- Nothing here measures accuracy. Evaluation lives in `slotvox.eval`.
