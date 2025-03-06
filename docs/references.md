# References

Public material consulted for ideas and format facts while designing
slotvox. No code or text was copied; the implementation is original.

## Joint intent classification and slot filling

- Q. Chen, S. Zhuo, W. Wang. *BERT for Joint Intent Classification and Slot
  Filling* (2019). <https://arxiv.org/abs/1902.10909> — background on
  jointly modeling intent and slots. slotvox adapts the joint-loss idea to a
  compact causal speech encoder trained only on synthetic signals.

## Large-scale speech data pipelines

- C. Wang, A. Wu, J. Pino, A. Baevski, M. Auli, A. Conneau. *Large-Scale
  Self- and Semi-Supervised Learning for Speech Translation* (2021).
  <https://arxiv.org/abs/2104.06678> — background on assembling large speech
  datasets from mixed sources; informed the mixing/provenance design of the
  instruction-data pipeline. No data or models from this work are used.

## Data formats and schemas

- HuggingFace *Datasets* documentation. <https://huggingface.co/docs/datasets>
  — facts about common dataset layouts (JSON/JSONL manifests, split naming)
  consulted when designing slotvox export schemas. slotvox does not depend on
  the library.

## BIO tagging

- The BIO/BIOES chunk-tagging conventions are standard in sequence labeling
  (as described in the chunking literature and tooling docs such as
  `seqeval`). slotvox implements its own tag algebra with explicit validity
  and repair policies rather than importing one.
