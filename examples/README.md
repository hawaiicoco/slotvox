# 示例 / Examples

本目录下的示例全部离线运行，仅使用 slotvox 自带的合成数据，不依赖任何真实
语音语料、预训练模型或外部下载。每个示例子目录附有独立的简短 README。

All examples here run fully offline on slotvox's built-in synthetic data.
They require no real speech corpora, pretrained models, or downloads.
Each example subdirectory ships its own short README.

1. `dataset_generation/` — 合成数据集生成 + schema 校验 + 无泄漏切分演示
   (synthetic dataset generation + schema validation + leak-free splits)
2. `joint_training/` — 联合意图-槽位模型训练，指标改善真实可见
   （需要 torch extra；仅针对合成数据，非基准测试）
   (joint model training with visible metric movement; torch extra; synthetic
   data only, not a benchmark)
3. `instruction_export/` — 指令语料导出 + gold 回放夹具上的评估报告
   （满分按构造，验证管线而非模型质量；无需 torch）
   (instruction-corpus export + evaluation report over gold-replay fixtures;
   perfect by construction — validates plumbing, not quality; torch-free)

运行方式 / How to run（仓库根目录 / from the repository root）:

```bash
python examples/dataset_generation/generate_and_validate.py --out /tmp/slotvox-demo
python examples/joint_training/train_joint_model.py --out /tmp/slotvox-train
python examples/instruction_export/export_and_evaluate.py --out /tmp/slotvox-corpus
```
