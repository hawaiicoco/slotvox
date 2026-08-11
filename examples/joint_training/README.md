# 联合训练示例 / Joint training

使用 torch extra（CPU）在合成数据上从零训练联合意图-槽位模型，并打印
训练前后的 dev 指标与每轮历史。相同 seed 结果完全一致（确定性训练）。
脚本与测试中的“指标改善”仅针对本合成数据与固定 seed，不构成任何真实
语音基准结论。

Trains the joint intent-slot model from scratch on synthetic data with the
torch extra (CPU), printing dev metrics before/after and the per-epoch
history. Identical seeds give identical runs. The visible improvement is
real for THIS synthetic dataset and seed — it is not a benchmark and makes
no claim about real speech.

```bash
pip install 'slotvox[torch]'   # CPU wheels suffice
python examples/joint_training/train_joint_model.py --out /tmp/slotvox-train
```

输出中的 `RESULT before/after` 行可被脚本化断言（仓库测试即如此做，
并标记为 slow + model）。`model.ckpt` 可直接用于 `slotvox predict`。
The `RESULT before/after` lines are machine-readable (the repo's slow test
asserts improvement on the default seed). The saved `model.ckpt` feeds
`slotvox predict` and `slotvox eval` directly.
