# slotvox

slotvox 是一个端到端口语理解（intent 分类 + slot 填充）实验工具箱：从合成语音波形、
log-mel 特征、BIO 序列标注，到联合意图-槽位模型、流式理解、指令数据导出、协议适配器
与离线评估报告，全部在同一仓库内闭环完成。

> **重要**：本项目只使用内置的确定性合成数据。仓库不包含、不下载任何真实语音语料或
> 预训练模型，也不宣称任何真实数据集上的成绩。所有“语音”均为按构造生成的
> tone/chirp/formant 信号，其片段与意图、槽位标签按构造一一对应（by construction），
> 并非真实语音。

## 主要特性

- **音频与特征原语**：严格校验的有界 WAV/PCM 读取、分帧与窗函数、log-mel 特征
  （带 golden 契约测试）
- **任务 schema**：意图、槽位类型（枚举/数值/时间/自由文本约束）、对话域
  （music-control、weather、navigation、calendar 等）、utterance↔annotation 配对；
  全部带 schema 版本号、严格校验（拒绝重复/未知标签）与 JSON roundtrip 测试
- **合成对话工厂**：确定性种子生成带意图标签的波形与 token 级 BIO 标注（与信号片段
  对齐），可控噪声与说话人分组，按 speaker/pattern 防泄漏的数据切分，统计摘要与
  provenance 哈希
- **序列标注核心（NumPy）**：BIO/BIOES 标签代数（双向有效性校验与修复策略）、带边界
  置信度的 span 抽取、标签转移约束、中文按字/英文按词的 unicode 安全分词
- **联合意图-槽位模型（可选 `torch` extra，仅 CPU）**：log-mel 上的紧凑因果编码器
  （TCN/GRU）+ 意图分类头 + 帧级槽位标注头，可配置加权的联合损失，确定性训练循环、
  梯度裁剪、checkpoint 保存/加载/续训；在合成 train→dev 切分上展示真实的损失下降与
  intent accuracy / slot F1 提升
- **流式理解**：有界音频缓冲、分块推理、带稳定性标志的部分意图假设、发射/修订语义、
  首次决策延迟统计、greedy 路径下“流式 == 离线”等价性测试、可序列化的流状态
- **指令数据管线（面向 speech LLM）**：交错语音-文本 instruction JSONL schema
  （以 manifest ID 引用音频）、模板渲染与变量校验、确定性权重的数据集混合、近重复检测
  （归一化文本哈希 + 特征哈希相似度，阈值均写入文档）、质量标志（时长界限、SNR 估计、
  标注完整性）、导出统计与带 schema 版本的 golden 测试。管线只产出数据，不附带任何
  模型权重或外部下载
- **适配器协议**：文档化的推理服务契约（请求/响应 dataclass + JSON schema）、离线
  fixture/replay 适配器、带超时/重试/可重试状态码处理的 loopback HTTP 适配器（对本地
  mock server 测试）。适配器验证协议行为，而非模型质量
- **评估**：intent accuracy、slot BIO F1（strict 与 partial-credit 两种边界策略均有
  文档说明）、joint turn accuracy、按噪声/说话人组/领域切片、带确定性种子的配对
  bootstrap 置信区间、混淆矩阵、可复现的运行元数据、Markdown/自包含 HTML 报告导出
  （敌意文本安全转义、无网络依赖），每份报告都带诚实的 limitations 章节
- **CLI**：`slotvox schema` / `generate` / `featurize` / `train` / `predict` /
  `stream-demo` / `export-instructions` / `eval` / `report` / `serve-mock`，
  全部可离线运行
- **可复现性**：固定版本 lockfile、所有产物带版本化 JSON schema 与 roundtrip/golden
  测试、配置校验带清晰错误信息、随机性处处显式种子

## 安装

需要 Python >= 3.11。核心运行依赖只有 NumPy；可训练联合模型需要 CPU 版 PyTorch
（可选 extra `torch`，缺失时相关测试干净跳过）。

```bash
uv sync --extra dev --extra torch   # 开发环境，版本由 uv.lock 固定
# 或者：
python -m pip install -e ".[dev,torch]" \
    --extra-index-url https://download.pytorch.org/whl/cpu
```

## 快速开始

```bash
slotvox generate --domain weather --counts 64:16:16 --out outputs/demo
slotvox featurize outputs/demo --out outputs/features
slotvox train --data outputs/features --epochs 12 --out outputs/model   # 需要 torch
slotvox predict --model outputs/model --data outputs/features --split dev
slotvox stream-demo --domain weather --chunk-ms 160
slotvox export-instructions --data outputs/demo --out outputs/instructions
slotvox serve-mock --port 0        # 本地协议 mock 服务（Ctrl-C 退出）
slotvox eval --pred preds.jsonl --gold outputs/demo --report report.md
```

各子命令的准确参数以 `slotvox <子命令> --help` 为准；`docs/usage.md` 与 `examples/`
提供完整的可运行示例。

## 开发

| 命令 | 作用 |
| --- | --- |
| `make build` | 构建 wheel |
| `make test` | 快速测试（排除 `slow` 标记） |
| `make test-all` | 全量测试（含 `slow` 与 `model`，CI 运行此项） |
| `make format` / `make format-check` | ruff 格式化 / 校验 |
| `make lint` | ruff 检查 |
| `make typecheck` | mypy 静态检查 |
| `make release` | 构建 sdist+wheel 并执行安装冒烟测试 |

测试标记：`slow`（长时训练/集成测试，`make test` 排除）与 `model`（需要 `torch`
extra，未安装时干净跳过）。

## 文档

- [docs/architecture.md](docs/architecture.md) — 架构与数据流
- [docs/usage.md](docs/usage.md) — 使用指南
- [docs/api.md](docs/api.md) — API 参考
- [docs/development.md](docs/development.md) — 开发指南
- [docs/references.md](docs/references.md) — 参考文献
- [examples/](examples/) — 可离线运行的完整示例

## 局限（务必阅读）

- 所有数据均为合成信号，**不是真实语音**；任何指标只反映“模型能否学会按构造生成的
  合成模式”，不能外推到真实 ASR/SLU 场景
- 联合模型是紧凑的 CPU 模型，用于验证端到端管线的正确性，不追求真实任务精度
- 适配器与 mock server 测试验证协议行为，不验证模型质量
- 文档与报告中的每个数字都来自本仓库实际执行过的合成实验；未执行过的实验一律不写

## 许可

MIT，见 [LICENSE](LICENSE)。
