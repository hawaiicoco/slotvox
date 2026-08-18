# 指令数据导出与回放评估示例 / Instruction export + replay evaluation

无需 torch。串联完整数据链路：合成数据集 -> 语音-文本指令语料
（JSONL + manifest + 质量标记）-> gold 回放夹具 -> 适配器协议决策 ->
评估 run envelope -> Markdown/HTML 报告。

Torch-free. Chains the full data pipeline: synthetic dataset -> interleaved
speech-text instruction corpus (JSONL + manifest + quality flags) ->
gold-replay fixtures -> adapter-served decisions -> scored run envelope ->
Markdown/HTML reports.

```bash
python examples/instruction_export/export_and_evaluate.py --out /tmp/slotvox-corpus
```

**诚实说明 / Honest scope:** 回放夹具由 gold 标注构造，因此评估指标
“按构造”满分——本示例验证的是服务与评估管线的正确性，而不是任何模型
的质量。真实模型评估见 `slotvox train` + `slotvox eval`（需要 torch
extra，同样仅针对合成数据）。

The replay fixtures are built FROM the gold annotations, so the metrics are
perfect BY CONSTRUCTION — this validates the serving + evaluation plumbing,
not model quality. For real (still synthetic-data-only) model evaluation see
`slotvox train` + `slotvox eval` (torch extra).
