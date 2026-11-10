# Feedback-Coded Speculative Decoding

八字节反馈编码与投机推理修复实验的研究代码。仓库包含参考解码、云边反馈协议、B/C/F-H 修复头、数据审计与配对评测工具。模型权重、实验数据、训练缓存和运行结果不包含在仓库中。

## 当前状态

- 原始参考实现提供冻结 Draft/Target、8 字节反馈、离线训练和逐请求输出一致性检查。`scripts/run_local_pilot.py` 是本地正确性实验入口，逐前缀重算，不代表部署吞吐。
- 新增的 [B/C/F-H Transformer 修复头](docs/TRANSFORMER_BCFH_LOCAL_V1.md)把输出侧 MLP 残差换成因果 Transformer；Draft 主干和词表头保持冻结。默认配置为 2 层、宽度 128、4 个注意力头。
- Transformer 候选目前是**本地代码和 CPU 合约测试**，尚未完成正式训练与 200 条请求的闭环评测，因此不能声称超过原始投机推理。
- `scripts/transformer_bcfh_experiment.py` 和 `scripts/transformer_bcfh_closed_loop.py` 是需要已审计数据与冻结模型后端的库入口，不会自行下载模型或派发 GPU 作业。

## 目录

| 路径 | 用途 |
| --- | --- |
| `feedback_codec/` | 反馈编码、修复头、参考解码及审计逻辑 |
| `scripts/` | 数据准备、训练和评测入口 |
| `tests/` | CPU 单元与合约测试 |
| `docs/` | 实验边界、阶段门槛和设计记录 |
| `program.md` | 研究方案与实验顺序 |

## 本地准备

使用 Python 3.11+，按自己的 CUDA/PyTorch 环境安装依赖。示例：

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests/test_v10_transformer_bcfh.py
```

参考实验需要本地已有的 Target/Draft 权重，以及预先冻结的 JSONL token ID 数据。它不会自动下载权重：

```bash
python scripts/run_local_pilot.py \
  --target /models/Qwen3-8B \
  --draft /models/Qwen3-0.6B \
  --dataset prompts.jsonl \
  --output runs/pilot-001 \
  --device cuda:0 --dtype bfloat16
```

`prompts.jsonl` 每行至少包含 `id`、`split`（`train`/`test`）和 `input_ids`。不同集合不能复用 ID 或 token 序列。请使用同一固定 tokenizer/chat template 准备所有比较组；`scripts/tokenize_prompts.py` 可用于冻结输入并记录哈希。

## Transformer B/C/F-H 的正式比较

目标是让同一批请求的**每次 Target 验证轮平均接受 Draft token 数**超过原始 Vanilla speculative decoding。先完成修复后的 1000 条训练请求（7982 个合法窗口）与独立的 200 条验证请求（1598 个合法窗口）的审计，再分别训练 B、C、F-H，之后运行完整请求闭环评测。原始基线取自真实记录的 packed Vanilla SD 请求轨迹；评测须先精确重放其提案、接受数和独立 Target AR 输出。修复臂也必须逐 token 保持 Target AR 输出一致。预设通过条件为配对收益为正且 95% 区间下界大于 0。局部窗口诊断的正收益不能代替该结论。

完整约束见 [Transformer 本地协议](docs/TRANSFORMER_BCFH_LOCAL_V1.md)。F-H 使用合法边界的 Target hidden，是诊断上界，不属于 8 字节反馈臂。旧 984 条子集仅保留历史结论，不用于新候选的正式训练或验证。

## 解释边界

仓库不附模型权重、受控数据或实测吞吐结果。参考后端与 Transformer 闭环评测会顺序重算前缀，适合检查因果性与输出一致性；不能据此报告 packed 推理速度、云边时延或 WAN 收益。实验结果应连同冻结索引、模型快照、源码哈希和完整回执单独保存。
