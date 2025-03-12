## Summary

What changes, and why?

## Related issues

## Type

- [ ] Feature
- [ ] Fix
- [ ] Docs / chore
- [ ] Refactor (no behavior change)

## Checklist

- [ ] `make build`, `make test-all`, `make format-check`, `make lint` pass
- [ ] Tests cover boundaries and both directions of touched operations
- [ ] Golden/roundtrip tests updated deliberately (not to mask regressions)
- [ ] Docs updated; no claims beyond experiments actually executed here
- [ ] No new required runtime dependencies; torch code stays behind the
      `model` marker
- [ ] Deterministic behavior: explicit seeds wherever randomness appears
