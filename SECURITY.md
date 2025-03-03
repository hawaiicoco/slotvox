# Security policy

## Scope

slotvox is an offline research toolkit. The default installation and test
path performs no network I/O, executes no untrusted models, and stores no
credentials. Components with a security-relevant surface:

- `slotvox.adapters.http` — loopback HTTP client used only against local
  mock servers in tests; enforces timeouts and bounded retries.
- `slotvox serve-mock` — local mock inference server for protocol testing;
  not intended for production exposure.
- WAV/PCM and JSON readers — strict validation, explicit size bounds, and
  schema checks guard against malformed or hostile input.
- Report export — all user/model-supplied text is escaped in HTML output.

## Reporting a vulnerability

Please report security issues privately through the repository's GitHub
private security advisory feature (Security tab → Report a vulnerability).
Include reproduction steps and impact. We aim to acknowledge reports within
7 days.

## Supported versions

| Version | Supported |
| ------- | --------- |
| 0.1.x   | yes       |
| < 0.1   | no        |
