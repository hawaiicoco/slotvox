# 数据集生成示例 / Dataset generation

离线生成一个合成 SLU 数据集：领域 schema、意图与槽位标注、BIO 标签、
说话人与模板两类无泄漏切分，并演示持久化往返与统计摘要。所有音频均为
合成信号（tone/chirp/formant 拼接），不是真实语音。

Generates a synthetic SLU dataset fully offline: domain schema, intent and
slot annotations, BIO tags, leak-free speaker/pattern splits, plus the
persistence round-trip and statistics summary. All audio is synthetic
(tones, chirps, formant patterns) — NOT real speech.

```bash
python examples/dataset_generation/generate_and_validate.py --out /tmp/slotvox-demo
```

输出包括域定义、切分无泄漏校验、数据集哈希与首个样例的文本/标注。
The output covers the domain definition, the split-leakage checks, the
dataset hash, and the first example's transcript and annotation.
