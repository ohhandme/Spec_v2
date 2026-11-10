# P1-B F 高带宽 Target 特征上界预注册

本方案在任何 F GPU 训练或验证结果出现之前锁定。它只诊断原 D 未超过 C/E 的原因，**不改变 H1 成功门槛**。`program.md`、原 P1-A B/C/D/E 结果、固定 train/val/test 划分、Target greedy oracle、`feedback_codec/`、冻结 fixed-prefix 生成和统计函数均不改。正式 test 继续封存。

## 单一干预与因果时序

原 D 的云端模块将当前验证所得的有效 `h_T` 变为 8 组×256 个分数，经过 hard Gumbel / argmax 得到 8 个字节索引；边端把这 8 个索引用模 256 加到第二组本地码，再查同一码本和修复器。

F **只移除云端的 8 字节 hard 选择**：对原云端模块的同一 8×256 分数作温度 1 的 softmax，发送全部 2048 个 FP32 概率，采用小端序序列化，恰好 **8192 字节/非首轮消息**。边端对本地 hard 索引作 one-hot，与云端概率作同一模 256 组合，使用原有的码本加权查找与低秩修复器。训练时云端概率直接参与可微的同一组合，推理时先序列化再解码；GPU P0 必须证明 FP32 线上的逐值精确往返。云端与边端模型层数、参数名、形状、可训练参数量、token 输入、纠正后前缀、CE 损失及输出头保持与 D 相同。训练中本地码仍用原 Gumbel/ST；云端从 hard Gumbel 改为确定的连续 softmax，这正是本次干预，不能据此声称 8 字节编码可部署。

这一表示替换同时移除了云端 Gumbel 噪声，使边端取得码本嵌入的概率加权期望；因此 F 优于 D 不能单独归因于传输位数，F 失败也不能排除一切有效 Target 特征。F 是**连续云码分布的上界诊断**，并非直接传输完整 `h_T` 的上界。

F 只允许读取**当前轮** Target 验证时预测纠正/bonus token 的 `h_T`，其前缀须等于原 fixed-probe 记录的 `valid_prefix`。纠正后 `corrected_prefix` 和普通协议 token 与 C/D 相同。下一轮 Target 状态、目标续文标签、完整 Target logits 和 val/test 接受情况绝不进入 F 的云消息；目标续文只用于训练 CE 或验证接受长度。第一轮没有反馈的处理不变。本实验不实现实际网络发送或在线闭环，8192 字节只是纯反馈载荷，不是 wire bytes 或时延。

## 模型、数据与固定预算

- Target 官方 Qwen3-8B BF16 revision `b968826d9c46dd6066d109eabc6255188de91218`；Draft 官方 Qwen3-0.6B BF16，权重 SHA256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`。沿用 P1-A 训练协议的逐文件 SHA 表和相同 tokenizer/Qwen thinking 配置。
- 训练 1024 prompt、9366 固定窗口，语料 SHA256 `dcdcde719da0dcd2071236f5eae50263dbdba83a330c9ac17535789a45e074e9`；验证 64 prompt、567 窗口，其中 427 个完整非终止窗口是主集，语料 SHA256 `96e15f8b28cd873a71c7714ecd3769497271422b933cfbbcab25e6f4f36a09fd`。test 60 prompt 封存。
- seed `42/43/44`，每 seed 一轮无放回 `9366` 步、LR `1e-4`、AdamW weight decay `0.01`、`prefix_weight=0`、温度 `1.0`、每 `256` 步原子 checkpoint。每 seed 的边端初始参数 SHA 与样本/轮次曝光 SHA 必须逐字等于原 D；参考训练协议 SHA256 `5a0a4a699208094ac33e9b4731e817e005166e334a52b503f904dcdccfd3a5243`，正式预算锁 SHA256 `9f7a046de99c14b950dc1df128d40ff1a85f3f68f1aad75ae8c60368179b91b2`。
- 原 D/C/E 使用已冻结的 P1-A val 逐窗口报告，F 在同一语料上自回归生成 L=8 草稿；不重训 B/C/D，不改评测器。原始 P1-B no-`h_T` 控制也保留，作为机制解释的旁证。

## 真实 GPU P0 与评测门槛

正式 F 训练前运行真实 4090 BF16 P0：核对模型/语料/原 D checkpoint 与源 SHA；证明 F 与 D 的初始边端参数相同；两个冻结 train 窗口做有限 loss、云端投影/编码器、两本地 head、码本、修复输出非零有限梯度；8192 字节精确序列化往返且对不同有效 `h_T` 的概率分布有差异；两条独立 Target AR 的单轮完整提交 token 完全相同；Target、Draft、输出头不更新。任何一项失败则不训练，保留失败报告并实证修复后重跑 P0。该小 gate 不替代既有 EOS/截断 P0；正式验证仍逐窗口保留 EOS/截断敏感性分析。

训练启动检查必须明确要求 GPU P0 报告中的 `wire_roundtrip_exact=true` 与 `target_dependence=true`；验证启动时再核对训练协议和 GPU P0 报告所记录的 gate 脚本 SHA，并与实际脚本字节一致。三层来源链不一致即停止，不得沿用旧 gate。

验证只读固定 `val`，结果逐窗口记录草稿 token、同一目标续文、连续接受长度、首次拒绝位置、prompt ID 与完整/终止状态；逐行重算须零差异。每 seed 报告按 prompt 配对的 **F−C** 与 **F−D**、原 D−C 与 D−E；主集 427 窗口，所有 567 窗口作敏感性分析。沿用冻结 2000 次 prompt bootstrap、seed `20261031`。不能用 teacher-forcing CE 或少量 P0 回归代替真实自回归接受长度。

## 解释和停止规则

- 若 F 显著超过 C 且原 D 未超过 C/E，说明有效 Target 特征在较高带宽下可能有用，优先定位 8 字节离散化、云端码训练和有效位数；这仍不支持当前 H1 的 8 字节方案。
- 若 F 也不超过 C，优先检查边端修复表达能力、训练 CE 与连续接受指标的错配、Target/Draft 组合和样本覆盖。不能直接推断 `h_T` 完全无信息。
- 若 F 超过 D 但两者都不超过 C，只能说移除 hard 云码局部缓解了问题；不进入 P2/P3 或声称加速。
- 约 8–12 个彼此可解释的预注册 P1 候选仍无稳定 D>C 时按原 `program.md` 给阴性/停止报告，不自动扩成多专家或多分支研究。高带宽 F 永远不能充当 8 字节实际速度基线，不据此报告 WAN 或净加速。

## 隔离实现与留痕

新增 `scripts/high_bandwidth_control.py`、`scripts/check_high_bandwidth_gpu_p0.py`、`scripts/train_high_bandwidth_p1.py`、`scripts/evaluate_high_bandwidth_p1.py` 和 `tests/test_high_bandwidth_control.py`。每阶段记录实际远端源码、模型、数据、原结果、checkpoint SHA 与独立输出目录。已有 worker 不重复启动，GPU 准入仍为无他人计算进程、空闲≥22000MiB 且连续三次利用率<10%。OOM/NaN/输出分歧先保留原件和旧源码，修改后重跑受影响的真实 GPU gate；不删失败样本、不放宽正确性。大型 checkpoint 和结果不纳入 Git，也不推送 GitHub。

本机版本测试：完整本地测试 `91/91` 通过（需要本机 loopback socket 权限）；新增三个 CLI 的 `--help` 均能运行。下表仅记录本机字节哈希，部署后必须核对**远端实际字节**，不能把 Windows 哈希当作 Linux 运行哈希。

| 新文件 | 本机 SHA256 |
|---|---|
| `scripts/high_bandwidth_control.py` | `936985e29537b132bf69a835a86aa1c3da55b275da12c40d83a3860f48520359` |
| `scripts/check_high_bandwidth_gpu_p0.py` | `24cdfcaf7708140562cece9fd7b2a76572f6b941d31f043b8bec33236f023480` |
| `scripts/train_high_bandwidth_p1.py` | `25a360e0758a7d91615ac49902272bf7a7f530d238557538c08d4989a56d7eb8` |
| `scripts/evaluate_high_bandwidth_p1.py` | `171ae10803a73492be81f41a4fab9688b95c0a0937cebf0e7efb431055e12e9e` |
| `tests/test_high_bandwidth_control.py` | `55bdc62665935395484151b24cb91a0c5aa181dd13a0c812d509bd799cdcf4ab` |
