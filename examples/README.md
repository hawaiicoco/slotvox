# 示例 / Examples

本目录下的示例全部离线运行，仅使用 slotvox 自带的合成数据，不依赖任何真实
语音语料、预训练模型或外部下载。每个示例子目录附有独立的简短 README。

All examples here run fully offline on slotvox's built-in synthetic data.
They require no real speech corpora, pretrained models, or downloads.
Each example subdirectory ships its own short README.

Planned examples:

1. `dataset_generation/` — synthetic dataset generation + schema validation
2. `joint_training/` — joint intent-slot training with visible metric
   improvement (requires the `torch` extra)
3. `instruction_export/` — instruction-data export + evaluation report on
   replay-adapter outputs
