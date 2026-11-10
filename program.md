# Autoresearch Program — 64-bit 验证反馈：先修正 Draft 分布，再预测拒绝风险，最后联合异步起草

> **身份**：本文件是交给 Coding/Research Agent 的可执行研究协议（`program.md`），不是已证实的论文结论。  
> **研究日期**：2026-10-31。**核心项目**：`experiments/feedback-coded-specdec/`（以真实仓库位置为准）。  
> **术语纠正**：**8 bit = 1 Byte；8 Bytes = 64 bit**。之前研究方案规定 **每次非终止验证最多新增 8 Bytes / 64 bits 下行反馈**。以下以 64 bits 为主预算，8 bits 作为严格消融，不要把二者混淆。  
> **现有证据**：本地交接文档记载已有基础实现和小模型功能测试，但**没有**真实 Qwen3-8B/0.6B 数据上的有效增益、可复现网络时延或最终加速结果。应在当前环境重新确认代码状态。  
> **优先级**：**P0 正确性 → P1 完整 Target 信息增量性 → P2 单任务 Hidden 压缩/修复 → P3 同预算排序信息竞争 → P4 单任务拒绝风险 → P5 双任务联合编码 → P6 异步协议 → P7 真实云边成本**。不允许为得到正结果绕过某个失败的关卡。  
> **研究依据（项目材料）**：`10.28(1).md`、`代码设计思路`、先前的 `program.md`（若存在 `SOURCE_TRACE.md` 同读）。研究设计**不得**把材料里的“预期效果”当成观测结果。参考论文链接见第 16 节。

---

## 0. Agent 的目标、原则、停止规则

### 0.1 两个待检验的目标（不要一开始联合训练）

- **目标 B / Distribution Repair（先做）**：正常 Target 验证时可获得的辅助信息，在只增加固定 64-bit 回包且不改变 Target 校验规则的条件下，能否改善 **纠正后的下一窗口** Draft 的实际 Greedy 连续接受长度？
- **目标 A / Rejection-Risk Prediction（后做）**：同一条或另一条低比特反馈，能否在 Draft 逐步生成时提高对**当前策略下**首拒位置/剩余可接受长度的预测，进而降低边端白算和异步预生成的回滚成本？
- **组合目标 A+B（最后做）**：在总反馈仍为 64 bits、边端计算/内存预算匹配时，共享反馈是否产生正的系统净收益？

**回答架构选择**：先单任务 **B-only**；B 的收益可信后做 **A-only**；两者分别证实后再做 **A+B joint**。风险标签取决于最终 Draft 分布，因此不能先把双任务 loss 简单相加。

### 0.2 可证伪的核心假设

- **H1（教师增量价值）**：在边端已有正确前缀、接受位置、被拒候选和纠正/奖励 token 的情况下，正常验证得到的 Target 内部状态/输出分布能为下一窗口提供**可利用的额外计算知识**。以同前缀、同参数的 protocol-only 修复器为强对照。
- **H2（可压缩性）**：把 H1 中的有用信息编码为真实序列化的 **≤64 bits**，仍能带来可靠、超过协议本身的下一窗口增益。
- **H3（编码目标）**：未来预测收益驱动编码（task-oriented）在相同位数下，是否优于纯 Hidden 重建编码、显式 Top-K 排序/置信度摘要？**不能假定答案。**
- **H4（风险）**：相对 WISP 风格的边端置信度/熵预测器，64-bit 教师反馈是否显著改善**已修复 Draft**的拒绝风险预测或调度决策？
- **H5（联合）**：B 与 A 共享 64-bit 消息的联合方法，相比 B-only、A-only 和显式分位预算，是否改善端到端收益？
- **H6（系统）**：真实或明确标记为网络整形/trace 驱动的云边配置下，节省的后续验证/等待成本是否大于反馈编码、传输、边端修复与预生成成本？

### 0.3 强制证据分级

每个结果标识为其**实际证据级别**，不得越级陈述：

1. `toy_smoke`：基础逻辑/数值检查；
2. `fixed_prefix_probe`：从相同纠正后前缀真实自回归生成 L 个 token；
3. `on_policy_sync`：各策略完整闭环、同步 Greedy 投机解码；
4. `async_trace_sim`：已校准时延轨迹的离散事件模拟；
5. `async_measured`：实际实现并测量的异步执行；
6. `network_measured`：受控物理链路或清晰标注的网络整形实验。

不能用 teacher-forcing CE、Hidden MSE、风险分类 Accuracy、加速的解析估算代替第 3–6 类性能证据。

### 0.4 禁止

- 禁止在正常验证之外额外运行一次 Target Forward，却把其成本记为零；特别禁止编码 **处理了纠正/奖励 token `b_t` 之后** 才会得到的 `h_T(x_{t+1})`。
- 禁止把未来 Target 参考续文、未来 Teacher logits、未来真实拒绝位置直接输入线上编码器；未来信息只可作为离线标签。
- 禁止重复发送正常协议里已有的 `b_t` 并把它的贡献计算为新增教师反馈；允许用它作为编码器条件和本地解码上下文。
- 禁止先测试 SpecBench/SPEED-Bench 再反复调参；禁止把同一 Prompt 的不同窗口拆到 train/test 两侧。
- 禁止在尚无单任务收益时叠加 MoE、ConFu 的额外 contemplate forward、SpecBranch 的全树计算或者 E²-SCI 的容差验证。
- 禁止不公平的模型/训练预算、修改 Target Greedy 验证标准、只比较加速比而不比较正确性。
- 禁止在有未提交的用户源码时使用破坏性 `git reset --hard`、覆盖原始权重或删除缓存。
- **禁止声称 Autoresearch 会自动在本对话中后台运行**；本文件只有交给具备执行条件的 Agent 后才会被执行。

---

## 1. 实验资源与冻结协议

### 1.1 第一版固定配置

| 配置 | 固定值 / 规则 |
|---|---|
| Target | `Qwen/Qwen3-8B`，冻结，记录精确 revision/hash |
| Draft | `Qwen/Qwen3-0.6B`，冻结 backbone 和 `lm_head`，记录精确 revision/hash |
| 推理 | Greedy，`do_sample=false`，首轮统一使用非思考模式（验证 `enable_thinking=false`） |
| 提议窗口 | `L=8`；窗口边界和 bonus token 规则所有方法一致 |
| 主链路 | 边端 Draft → 云端 Target Verify → 返回正常 `accepted_count`/`b` 和可选 `m` |
| 新增回包硬约束 | 每个非终止验证响应最多 **8 个原始 bytes**，不是 8 个 int64 或 8 个字符串 |
| 源隐状态 | Target 用于预测当轮 `b_t` 的**有效验证前缀**最终归一化 hidden；Qwen3-8B 默认维数 4096，以实际模型 config 为准 |
| 本地修复器 | 先用输出侧 `rank=32` 条件低秩修正；后测 8/16/32；位置为 final norm 与冻结 lm_head 之间 |
| 基础编码 | `G=8, K=256`，每组 1 byte 编号，共 `8 × 256` 个离散选择位址；本地可训练码表 `8×256×16` |
| 参数更新 | 仅新加入 encoder/decoder/adapter/risk head；在线不改变 0.6B/8B 权重 |
| 最终质量 | 与同 prompt 的 Target Greedy AR 逐 token **100% 一致**（含 EOS 与长度上限） |

`Qwen3-0.6B`、`Qwen3-8B` 的 `config.json` 名义 `vocab_size` 为 **151,936**，因而合法 token ID 在 [0,151935]，18 bit 足够；**实现时必须读取本地锁定 checkpoint 的 config 和 tokenizer 并断言匹配**。不准仅根据这一行静态假设词表一致。量化、内核、聊天模板、精度与终止条件必须固定。

### 1.2 有效因果状态与回包时机

令当前确认前缀为 `x_t`，Draft 提议 `y_{t,1:L}`；Target 验证接受 `a_t` 个，然后产生纠正 token/bonus token `b_t`：

`x_{t+1} = x_t || y_{t,1:a_t} || b_t`。

正常验证可用：

```text
valid_prefix_t       = x_t || y_{t,1:a_t}
valid_target_hidden_t = h_T(valid_prefix_t)    # 此状态预测 b_t
valid_target_logits_t = z_T(valid_prefix_t)    # 同一次 target verify 的结果
accepted_count       = a_t
rejected_draft_token = y_{t,a_t+1} or NULL
corrected_or_bonus   = b_t
```

`h_T` **不是**已经输入 `b_t` 后的新隐状态。云端 encoder 若用 `e(b_t)`，那是 token embedding lookup，不是 `h_T(x_{t+1})`。对于 `a_t=0`，有效 Target 隐状态应来自已有正确前缀末端；若程序没有该缓存，须正确计入为获取它需要的计算，不能假称免费。全部接受时 `b_t` 为奖励 token；`b_t` 终止请求时不应发无用下一窗口反馈。

可选的显式 logits 摘要也**只能**取 `z_T(valid_prefix_t)`；不能取 `z_T(x_{t+1})` 而不增加下一次 Target Forward。

### 1.3 为什么正常协议已经携带信息

边端已有：`x_t`、自己提出的所有 `y`、`a_t`、`b_t`，可重建 `x_{t+1}`；接受/纠正信息本身不是额外教师信息。H1 的公平比较必须提供给 **C（protocol-only）** 这些现有输入，不可把 C 故意削弱为仅持有 Prompt。目标是让边端以有限计算获取其自己无法廉价推算的 Target **计算性知识**，不宣称已知前缀条件下增加了信息论熵。

---

## 2. 数据与实验口径：先冻结，再 Autoresearch

### 2.1 原始 Prompt → 验证轨迹 → 下一窗口标签

初始 Pilot 与正式训练为**建议预算，不是预设的显著结果**。若本地资源不足，在 P0 manifest 中一次性记录缩放方案；同组对照不能用不同规模的数据。

| 数据用途 | Pilot | Formal | 注意事项 |
|---|---:|---:|---|
| Train 原始 Prompt | 1,000 | 6,000 | UltraChat / Dolly或Alpaca / CodeAlpaca / GSM8K-train，建议比例 60/20/10/10 |
| Val 原始 Prompt | 200 | 600 | 同类但**与 train 按 prompt/近重复组隔离**，用于所有反复调参 |
| Final test | 不触碰 | SpecBench + SPEED-Bench（有条件扩展） | 仅封版后使用，检查任务/样本重复 |
| 每 Prompt 新增生成上限 | 128 tokens | 128–256 tokens | 所有组一致、清晰报告 EOS |
| 收集的固定探针 | 每 Prompt 上限 8 个窗口 | 每 Prompt 上限 8 个窗口 | 在看任何方法结果之前锁定窗口抽样 |

具体数据源版本与许可证、URL、HF revision、采样 seed、sha256、去重策略都写进 `datasets_manifest.json`。不要在训练时使用 SpecBench 或 SPEED-Bench。GSM8K 的 test 不进入训练；如用于早期评估，必须另设完全独立的数学测试集。

轨迹由**原始未修复的 Vanilla SD** 首先产生（每次最多 8 个候选），每条保存：

```json
{
  "prompt_id": "...",
  "split": "train|val|test",
  "round_id": 0,
  "original_prefix_ids": [],
  "draft_candidates_ids": [],
  "accepted_count": 0,
  "rejected_token_id": null,
  "corrected_or_bonus_token_id": 0,
  "corrected_prefix_ids": [],
  "target_valid_hidden_path": "...",
  "target_valid_logits_path": "...",
  "target_future_greedy_ids": [],
  "is_terminal": false,
  "model_hashes": {},
  "trajectory_policy_id": "vanilla_L8_v1"
}
```

`target_future_greedy_ids` 是从**相同** `corrected_prefix_ids` 用独立 Target Greedy 生成最多 8 个 token 的监督/评测 oracle；绝不能喂给在线 encoder。Target full logits/teacher-forcing soft labels 只可用于**离线损失计算**，绝不能注入推理时的 64-bit encoder。

### 2.2 严格区分三种评测

- **Fixed-prefix probe**：锁定同一批 `corrected_prefix_ids`，从每个前缀由各版 Draft **真实自回归**生成 8 个 token，评测目标续文上的最长相同前缀。测试是否存在**反馈因果增益**。
- **Teacher forcing**：在目标真实续文上计算 CE/KL，允许离线训练，不是实际接受长度。
- **On-policy**：各策略从原 Prompt 开始、通过自己实际形成的验证边界运行完整 SD，报告验证轮数与延迟。不能拿 fixed-prefix 的窗口结果代替它。

主指标定义：

`A_probe(x, method) = max{k in [0,L] : draft_method(x)[1:k] == target_greedy(x)[1:k]}`。

为避免 EOS 混淆，主要 probe 使用“目标还有 ≥8 个非终止输出 token”的预封版子集；另外必须报告**所有窗口（包含 EOS）**的结果与被排除比例。不能按方法的表现筛选窗口。

### 2.3 统计单位与效果判定

- 同一 Prompt 的多个窗口是相关样本。**先逐 Prompt 汇总，再按 Prompt 作成对 bootstrap 95% CI**。每个变体至少 3 个训练随机种子（Pilot 可先 1 个用于查 bug，但不支持论文结论）。
- 主比较 `mean_prompt(A_D - A_C)`；风险任务用 `NLL/Brier` 的成对比较；系统任务用每请求 wall time 的成对比较。
- 固定 `seed`, `prompt_order`, `max_probe_per_prompt`, 模型版本、训练步数、解码与终止规则。
- 同时报告差值绝对量（例如 `+0.XX token/round`）、95% CI、每 seed 分布和每 Prompt 变化，不只报告相对百分比。
- 多个编码候选的筛选**只能**在 Val 上；建议用预注册 Primary Contrast + Holm 修正或分层确认策略处理多重比较。test 只运行最后锁定的少数候选一次。
- **实用效应门槛**：P0 基于真实成本测量或保守敏感性区间预先计算保本所需 `ΔA_min`，不得事后为结果修改。`ΔA>0` 可支持信号，但不等于系统可用。

---

## 3. P0 — 查仓库、锁模型、建 Baseline、保证严格正确

1. 阅读真实仓库中的 `README.md`, `SOURCE_TRACE.md`（如有）, `反馈编码设计`, `feedback_codec/{hf_backend,decoding,codec,training,losses,pipeline,evaluation,cloud_edge,cost_model}.py`, `scripts/`, `tests/`。文件/CLI 不存在时如实记录，必要时实现最小接口，**不要凭本文猜测已有脚本**。
2. `git status`，保存用户未提交工作；新建尚不存在的 `autoresearch/oct31-feedback64` 或按本地实际选择不冲突 tag。建立 manifest（commit, torch/transformers, dtype, model revision, tokenizer hash, chat template, thinking mode, GPU, RAM, dataset sha）。
3. 基础通路测试：消息 `pack/unpack` 严格 8 bytes，异常长度报错；Target 和 Draft 冻结；教师 encoder 不读取未来；refusing/bonus/EOS/首轮无消息；不能把正常 `b` 当成新增载荷。
4. 实现正确的独立 Target Greedy AR oracle，并对所有变体的闭环最终输出（token IDs、EOS、截断）逐 token 对比。任一不匹配 -> **BLOCKED**，先修正确性。
5. 统一 B/C/D 的本地额外参数数量与每 token 修复成本（目标误差 ≤5%；若做不到至少报告偏差，使用 parameter-matched adapter 对照）。云端 encoder 参数/计算单独统计；不允许不匹配架构还宣称纯教师收益。
6. 保留原来功能代码，不在 P0 静默更换 8B/0.6B 模型、改用 tiny model 或改变采样模式。如果 GPU 不足，特征提取可以分两段离线缓存，且明确 `feature_cached` 与真实在线成本区别。
7. 确认真实 Qwen 推理 packed verify/KV cache 与 reference 的 token 行为一致。现存逐 token 重算完整前缀的实现**只可用于功能正确性/最早的离线 probe**，不可据此宣称系统加速。
8. 封版 Val probe 和 Final test；`datasets_manifest.json`, `eval_protocol_version`, `oracle_hash` 写入所有 run。

### P0 的最小公平对照

| ID | 可用信息/模型 | 作用 |
|---|---|---|
| **A RAW** | 冻结 Qwen3-0.6B，无新增修复 | 原始底线 |
| **B EDGE** | 同样 rank/边端参数的本地修复器，只读本地前缀/特征 | 排除新增参数收益 |
| **C PROTOCOL** | B + 上轮 `accepted_count`, rejected candidate, `b`（均为普通协议数据） | **最重要强对照** |
| **F FULL-H** | C + 当前正常验证的完整有效 Target hidden，未经压缩 | H1 诊断性上界；非实际网络 baseline |
| **D 64-H** | C + 64-bit Hidden 反馈，下一轮使用 | 研究方法（P2 起） |
| **E SHUFFLE** | 完全相同 D 权重，但把消息在请求间打乱、分层打乱 | 负对照；排除无内容的 Code |

B/C/D/F 使用**同一**修复函数与相当本地容量；F 可多一个高维输入投影（成本须报告）。B/C 没有云端隐状态但可以使用**边端已有的所有协议字段**生成本地 128 维控制向量；不能只去掉 C 的整个 decoder 造成不公平。

---

## 4. P1 — 先问“完整合法 Target 信息有没有增量收益？”（严禁先量化）

### P1.1 单任务 B：无压缩辅助信息探针

1. 把 Target valid hidden `h_T∈R^4096`（必要时 valid logits summary）用于训练 **F FULL-H** 的轻量投影；Target 固定、Draft 固定、在线信息边界同第 1 节。
2. 用相同纠正后前缀和相同训练预算训练 **B EDGE**、**C PROTOCOL**、**F FULL-H**；初始只使用 next-token CE，后续不能同时改变 loss/模型/样本。
3. 先验证 `F vs C` 的 fixed-prefix Greedy 接受长度（同时记录 `F vs B`, CE, next-token top1, first reject index）。
4. 另一诊断上界 `F LOGIT`：使用正常验证边界的**当前** Target logits 摘要，与 C 同前缀比；不允许 Target 未来 logits 作为在线输入。
5. 验证 `h_T` 是否真的被使用：保持 F 权重只把 hidden 置零/分层打乱；如果效果不变，检查信息被忽略、训练失败或前缀泄漏。

### P1.2 判定和分支

- `F > C` 稳定：说明完整教师内部状态有可利用价值，进入 P2。
- `F ≈ C` 但 `F LOGIT > C`：隐藏向量/适配器不合适，仍可继续 P3 的显式概率摘要；**不得**宣称所有 Target 反馈无价值。
- `F ≈ C`、`F LOGIT ≈ C`：检查强 edge-only、输出修复容量、数据/模型组合、训练与因果状态；针对性修复后仍无收益，**NO-GO**，暂停双任务与系统集成。
- `F > C` 但成本太高：只支持可利用性，不支持部署；P2 试 64 bit 是否足够。

**P1 固定窗口至少 3 seeds 正向验证后再做大量 P2 搜索**。P1 的最终判定仍要参考预登记的收益阈值和 CI，不能因 0.01 token 的随机优势宣称成功。

---

## 5. P2 — 单任务分布修复：64-bit Hidden State 编码从简单到复杂

### P2.0 定义一次公平的 64-bit 接口

64-bit 消息 `m∈{0,…,255}^8`，线上是真正的 `bytes(m)`；**不可把 8 个 `int64` 当 8 bytes**。本地持久化的码表、量化刻度、字典与修复器权重属于一次性部署开销，需要汇报模型体积；不在每轮通信重复发送。记录实际线上 `payload_bytes=8` 和完整报文 `wire_bytes`。

修复器（沿用原设计）：

```text
c = decoder_64bit(m)                      # e.g. 128D 本地控制向量
u_j = V @ h_D,j                           # r = 32
alpha_j = f_psi(c, u_j, j/L)
h_tilde_j = h_D,j + U @ (alpha_j * u_j)
logits_D,j = W_out @ h_tilde_j            # W_out frozen
```

无消息时使用原 Draft 或 C 的 protocol-only 控制。修复只作用在最后输出层前，不改变历史 transformer KV 的计算；必须检验下一 token autoregressive 栈和正确的缓存截断。

### P2.1 H-INT8 / 简单 Hidden 线性量化（先打通）

定义 `z = P h_T`, `P: R^4096 → R^8`，每维在 train 数据上校准静态范围，执行 `uniform_uint8_quant(z_i)`，拼成 8 字节；边端 `dequant + small decoder` 得到 `c`。`P` 可以从 train-only PCA、随机正交投影或训练好的线性层初始化：**先分别记录版本，不混称一个 baseline**。

- `H-INT8-RECON`：投影/反投影以重建 `h_T` 的 MSE/余弦为目标训练，之后冻结编码器，训练同样的本地修复器；
- 不把全量 `P` 或量化 min/max 发到线上；它们只在部署时同步一次。
- 记录 8 维的溢出/截断率、有效熵、码值分布和 MSE；MSE 仅作辅助。

**目的**：提供最容易解释、最容易实现的“直接压缩 Hidden”基线。如果它无效，不要立即得出“64 bit 不够”的结论，下一步可做更灵活的任务导向编码。

### P2.2 H-VQ64-RECON / 8×256 离散码本重建

复用已有 `G=8, K=256` 编码器。云端输出 `S∈R^(8×256)`，每组 hard argmax 一个 index，序列化为 8 raw bytes；边端查 `C_g∈R^(256×16)` 后拼接成 128 维。先以重建合法 Target hidden（MSE + cosine）训练 `E_phi`/辅助重建器，再训练同结构修复器。

- 重建网络只能用 train split；Val 不回传梯度；
- 训练硬编号可使用直通 Gumbel-Softmax，推理必须 hard deterministic argmax；
- 对每组报告码字使用率、entropy/perplexity、塌缩率；
- 若出现训练硬码与推理 argmax 不一致，固定一套 Gumbel 退火/推理规则并在所有组统一使用。

### P2.3 H-VQ64-TASK / 任务导向 Hidden 编码（本方案首个正式候选）

与 `H-VQ64-RECON` **完全同样的 64-bit 物理编码、128D 边端 decoder、rank-32 修复器**，但直接按后续 Teacher Greedy token 的 CE 更新 Encoder + Decoder + Repair：

`L_CE = -mean_j log q_D(y*_{j} | x_{t+1}, y*_{<j}, m_t)`。

先仅 CE，不加 prefix loss、risk loss、MoE、异步或网络门控。比较：

- A RAW、B EDGE、C PROTOCOL；
- `H-INT8-RECON`、`H-VQ64-RECON`、`H-VQ64-TASK`；
- E SHUFFLE：同一 Task model 但把码跨 Prompt/按 `a_t` 分层随机打乱；
- F FULL-H 诊断参考上界。

**训练公平性**：报告统一训练样本/梯度步数的 CE 修复阶段；如果 RECON 用了额外预训练而 TASK 没有，需报告总训练算力，补做预算匹配版，不能将额外算力当作编码收益。

**主比较**：`H-VQ64-TASK − C` 的 `A_probe` 差值，95% CI；其次 `H-VQ64-TASK − H-VQ64-RECON`。仅在 CE 下降而 Greedy 连续接受长度不变时，不支持 H2/H3。

### P2.4 次级探索（只有 P2.3 成立后）

1. 消息预算：`0 / 8 / 16 / 32 / 64 bit`（每种都用真实 bytes pack；`8bit=1byte`）；调整 `G` 但**保持本地解码能力和训练预算合理匹配**，不让“更长消息=更多模型参数”成为混淆。
2. 损失函数：`CE` → `CE + prefix surrogate` → `CE + greedy-margin surrogate`，一次改一个；所有方案最终看自回归 `A_probe`。
3. `rank ∈ {8,16,32}` 与 `decoder width` 逐一变化，报告边端每 token 时间和总额外参数。
4. `Target last hidden` vs `Target 多层投影`：新增多层 hidden 若属于**本次 verify 已缓存的中间输出**，可以测试，但须计入导出/存储/编码成本，不得免费要求 Target 多层重跑。
5. 训练轨迹更新：用当前修复策略在**训练集**重新采集新验证边界，验证集的固定探针仍不改变；另单独测 on-policy rollout。

若任务编码因码本塌缩失效，只允许受控测试：温度 schedule、码本均衡正则、小初始化/EMA 等。不得反复在 Val 上大规模无目标搜索。

---

## 6. P3 — 第二类信息：64-bit 显式 Top-K 排序与概率摘要

**动机**：联邦 Top-K/排序蒸馏论文提示“教师所有高维特征未必值得传”。但本场景只允许正常验证边界的 `z_T(valid_prefix_t)`，它针对的是**当前 `b_t` 的分布**，不是纠正后下一 token 的未来分布；必须独立检验其增量价值。

### P3.1 可逐 bit 审计的 RANK64 消息格式

对 `softmax(z_T(valid_prefix_t))` 排名，`b_t` 本来就是 Greedy Top-1 且已通过正常协议发送，所以**不要重复传它的 ID**。构造一个严格的示例 64-bit 格式：

| Field | bits | 解释 |
|---|---:|---|
| `target_top2_token_id` | 18 | 教师第二候选（不含 `b_t`） |
| `target_top3_token_id` | 18 | 教师第三候选 |
| `top2_prob_q6` | 6 | 静态/训练集校准的概率量化 |
| `top3_prob_q6` | 6 | 同上 |
| `top1_prob_q8` | 8 | `b_t` 的教师概率，ID 不重复传 |
| `entropy_or_margin_q8` | 8 | 教师当前分布的熵或第一/第二候选 Margin（只选一种作预注册版本） |
| **合计** | **64** | **8 raw bytes** |

- 按 `uint64 little-endian` 固定掩码与位移完成 pack/unpack；不能发送 JSON、十进制字符串或 18bit 对齐到 4Byte 后继续宣称 8Byte。
- 词表索引限制 18bit；无效 token 保留哨兵 `2^18−1`（如果真正使用，需写入协议 spec）；特殊词表与空候选边界确定化。
- `top1_prob_q8`/其他概率使用固定 train-only 标定，不允许推理时额外传 scale。量化截断率与排序失真必须记录。
- full softmax/topk/entropy 计算与内存访问若不是标准 verify 原有必算操作，**加入云端 encoder 时间**。声称“不额外 Target Forward”，不等于没有计算开销。

### P3.2 公平对照矩阵

按以下顺序只改消息格式，保持同一边端修复器容量与训练预算：

1. `RANK64-TOP2`：用上述显式 RANK64 给边端 decoder，再训练 rank-32 修复；
2. `RANK64-MARGIN`：把末尾 8bit entropy 换成边界 logit margin，保持总位数 64；
3. `RANK64-ONLYSTATS`：仅发送当前目标分布的 entropy、p1、p2、margin 等量化统计，剩余 bits 置零或作为重复校验；**实际信息位数与 wire bytes 分别报告**；
4. `H-VQ64-TASK`：P2 锁定的隐式任务编码，不因看到 RANK 成绩而修改其评测器；
5. `C PROTOCOL` / `E SHUFFLE` / `F LOGIT`：相同前缀对照/诊断。

`RANK64` 只读取**本轮**验证分布，不能把未来第 1–8 步的 Target logits 暗中放进去。若 `RANK64` 失败，不等于“排序蒸馏无用”，可能是**本轮排序对下轮太陈旧**；必须区分信息类型和时间有效性。

### P3.3 可能的更强下一步：learned logit-summary encoder

如果 RANK64 显式格式在 pilot 中表现有信号，再尝试 `SCORE64-TASK`：输入本轮 Target logit statistics / 小样本候选集、经 8 组离散码编码，用后续 CE 训练。与 `H-VQ64-TASK` 对照“原始输入属于 Hidden 还是 Logits”；不能把额外量化表或整词表分布偷偷下传。只有 Val 上看到明显互补时才研究 `H+SCORE` 混合编码。

---

## 7. P4 — 第二个**独立**目标：在固定 Draft 分布上预测首拒风险

这一步先做 **A-only**，不是 P5 的 joint。

### P4.1 先冻结修复策略，确保风险标签固定

在 train 上选定一版已封存的 Draft：

- `R0 RAW`：未修复 Draft（最简单风险基线）；
- `R1 REPAIRED`：若 P2/P3 修复有效，取 Val 选出的**单一修复器**，冻结其所有参数与消息规则，**重新收集 train 的 on-policy 验证轨迹**。

风险研究的主场景应为 **R1**，因为以后异步执行的是被修复后的 Draft。原始 RAW 上的首次拒绝标签不能当成 R1 的正确风险监督。

### P4.2 风险预测的两个时间粒度

**(i) 窗口开始时的预算预测器**：在下一轮起草开始前，使用上一轮收到的消息 `m_t` + 正确前缀最后一个边端状态 + 协议已知字段，输出未来窗口 `A∈{0,...,L}` 的分布 `P_hat(A=k)`，用于选择草稿/预取预算。不能假装知道尚未生成的未来候选 hidden。

**(ii) 逐 token hazard head**：生成到第 `j` 个 Draft candidate 并有对应 Draft hidden/logits 后，计算

`r_j = P_hat(首次拒绝出现在 j | 前 j-1 个都接受, m_t, h_D,j, q_D,j)`。

`r_j` 在生成阶段可用，但**目标验证结果只在回包时可知**。它只能用于控制继续起草/触发验证/限制异步 lookahead，不能代替 Target 判定。

### P4.3 离散生存损失（首拒后不乱贴标签）

令真实首拒位置为 `r ∈ {1,...,L}`，全部接受则记为 **right-censored at L**。只对首次拒绝之前的可观察生存序列监督：

- 如果在 `r` 处拒绝：`NLL = −sum_{j<r} log(1−risk_j) − log(risk_r)`；
- 如果全部接受：`NLL = −sum_{j≤L} log(1−risk_j)`。

窗口开始的 `P(A=k)` 则可用明确多分类 CE 或基于 hazard 的联合一致性约束训练；不要把“第 r 个之后的 token”当作已经验证的负样本。训练和 Val 按**Prompt**划分；风险标签与该版本 Draft 配对。

### P4.4 风险单任务 Baseline

| ID | 风险预测使用的信息 | 说明 |
|---|---|---|
| `R-EDGE` | 本地 Draft logits entropy、top1 probability、margin、position/last hidden | WISP 风格的轻量边端 predictor；重要强基线 |
| `R-PROTOCOL` | `R-EDGE` + 上轮拒绝位置、纠正 token、接受长度 | 不新增 Target hidden 的最强协议基线 |
| `R-FULL-H` | `R-PROTOCOL` + 合法完整 Target hidden | 辅助上界，不可直接部署 |
| `R-H64` | `R-PROTOCOL` + H-VQ64 或同预算新风险码 | 教师特征的增量值 |
| `R-RANK64` | `R-PROTOCOL` + P3 排序/概率消息 | 显式分布摘要的增量值 |
| `R-SHUFFLE` | `R-H64` 权重固定但消息乱序/分层乱序 | 负对照 |

**主风险指标**：held-out Prompt 上 `first-reject NLL`、离散生存 `Brier`、不同位置 reliability/ECE、`predicted accepted length` 的 MAE、**决策条件下**浪费 Draft token 与可复用候选占比。Accuracy/AUC 只辅助，不能作为云边性能证据。

### P4.5 闸门

- `R-H64` 必须在同样参数预算下与 `R-PROTOCOL`、`R-EDGE` 比较；若仅在 RAW 而非 REPAIRED 上有效，要如实报告漂移。
- NLL/Brier 优化但并没有改善决策的实际效用，不进入联合模型和异步系统“加速”主张；可保留作为预测贡献。
- 若不需要任何教师信息、仅本地 entropy 就足以预测风险，那是重要**负面发现**：应该把剩余反馈比特优先用于修复，不为了做双任务而强行分配。

---

## 8. P5 — 两个单任务都有效后才做 64-bit 双任务联合编码

### P5.1 先做最小 2×2 模块消融

在同样 Draft 基座、Target、网络协议下分别运行：

| 变体 | Repair Head B | Risk Head A | 为什么 |
|---|---|---|---|
| `00 vanilla` | × | × | 原始异步/同步基线 |
| `01 B-only` | ✓ | × | 只增加下一窗口接受长度 |
| `10 A-only` | × | ✓ | 只减少风险/浪费计算 |
| `11 A+B` | ✓ | ✓ | 检验联合互补性 |

`10` 与 `11` 的风险标签必须来自各自被固定的 Draft 策略；如比较同一风险器可迁移性，作为**独立测试**报告，不混入主比较。

### P5.2 三种编码资源分配：严格相同总 64 bits

| 编码方式 | 下行位数 | 训练设计 | 目标 |
|---|---:|---|---|
| `SHARED64` | 64 | 一个 encoder / 一个码，分别喂 repair head 与 risk head；先分阶段训练，再小幅 joint finetune | 检验共享信息 |
| `SPLIT32+32` | 32+32 | 两条独立子码，每条 ≤32 bit；各自 head | 检验显式分配 |
| `SPLIT16+48` | 16+48 | 16 bit risk、48 bit repair | 检验非均匀预算（先锁为第二个消融） |
| `SINGLE64-B` | 64+0 | 保留 P2/P3 最强单任务编码和边端本地风险器 | 检验教师比特是否应全给修复 |
| `SINGLE64-A` | 0+64 | 本地修复器 + 教师风险编码 | 检验教师比特是否应全给风险 |

还可测 `SPLIT48+16`，但只在 Val 先显示资源权衡值得探索时做，避免事后“暴力搜”配比。**所有方案总消息均 ≤8 raw bytes**；不足 8 bytes 的方案要同时报告有效 bits 和固定帧的 wire bytes。

### P5.3 训练顺序（解决风险/修复相互影响）

1. **B 热身**：用 P2/P3 选择的修复目标训练消息编码器和修复器；在相同正式 train 上检查真实自回归 `A_probe`。
2. **A 校准**：冻结 B，运行 B 修复后的 train 轨迹，训练风险 head；不再使用未修复 Draft 的首拒标签。
3. **共享微调**（仅 `SHARED64`）：以预登记的 `L = λ_B L_CE + λ_A L_survival (+ λ_P L_prefix)` 联合训练；λ 在 Val 上有限小网格，或者使用先归一化损失尺度的固定权重。禁止通过 Final test 选择 λ。
4. **再次轨迹更新**：微调 B 改变提议分布后，只从 train 重新采样并更新 A 校准；固定 Val/Test Prompt、oracle 与评测协议。
5. 比较 `SHARED64` 与 `SPLIT`、单目标、边端本地 A，不仅比较 Task Loss，还看预测/接受与 P6 的真实决策成本。

**必要消融**：去掉 H 输入、去掉 Target logits 输入、风险头不用共享码、修复头不用共享码、消息跨 Prompt 乱序、仅本地信息、冻结/不冻结 B、码预算各占比。

---

## 9. P6 — 异步起草不能把“知道风险位置”误当作“知道纠正 token”

### P6.1 严格的可复用性约束

预测 `j` 处拒绝 **不等于** Target 纠正的 `b_t` 已知。所有基于错误 token 的后续 Draft/KV **不能复用**。异步提前起草只能沿某条**假设路径**计算，并在验证后逐 token 核对其路径前缀与真实确认前缀相同，否则丢弃并恢复正常计算。

特别注意**全部接受**时还有 Target 生成的 bonus `b_t`：沿原草稿直接生成的下一 token 必须与 `b_t` 匹配，后续缓存才有可能复用；否则不能忽略 bonus 而宣称异步命中。

### P6.2 先做同步效果，再做异步

- `SYNC-B`：只有 B 修复，真实闭环验证轮数、平均接受长度。
- `ASYNC-BASE`：PicoSpec 风格/简单单路径异步预起草（先确认可合法复用与网络时序）。
- `ASYNC-LOCAL-RISK`：仅使用本地风险，在云端验证当前窗口时自适应决定**是否继续沿假设路径提前起草**以及最大预取 token 数。
- `ASYNC-TARGET-RISK`：加入教师 64-bit 风险码；注意第 `t` 轮 Target 正在验证期间，第 `t` 轮**新反馈 m_t 尚未返回**，边端只能用上一次 `m_{t−1}` 与当前已有局部信号，不许时间穿越。
- `ASYNC-JOINT64`：P5 最终共享码同时影响分布修复与预取策略。
- `ASYNC-BRANCH2`：若硬件允许，额外可做最多 2 分支并匹配 Draft 计算/峰值内存的 SpecBranch-inspired 对照；否则标记因资源不适用，不得假称论文级完整复现。

停止提前起草（early-stop）可能**降低白算但增加气泡等待**，因此不能单靠白算减少认定加速。必须同时报告：prefetch produced tokens、prefetch valid reused tokens、discarded tokens、rollback cost、Target idle、Draft idle、trigger 频次、额外 GPU 时间、边端峰值 KV。

### P6.3 时间和成本先模拟、后实测

1. 先标定 Target packed verify latency、Draft token latency、编码/修复开销、socket/serialization、网络 RTT 及独立上下行 bandwidth；如还在 sequential-target reference 阶段，只能输出 `simulated` 结果。
2. 写可复现离散事件模拟器（事件包括 Draft start/end、request upload、Target verify、response download、speculative prefetch、cache invalidation），拿当前硬件的真实阶段时延做 trace 回放。
3. simulator 通过定制单元测试（网络 RTT=0、无限带宽、首拒为 0、全接受、bonus 不匹配、部分接受、EOS）再进行比较。
4. 实现真实 async pipeline 后，在同步/异步相同 Greedy 输出正确前提下，记录 wall-clock 的 per-request latency、inter-token latency 与 P95；模拟和实测结果用不同标记。
5. **系统闸门**：`ASYNC-JOINT64` 对 `ASYNC-LOCAL-RISK + B-only` 的附加收益须为正，才能证明教师风险码超越边端本地预测，并非仅仅“用了风险器”。

---

## 10. P7 — 带宽、RTT、端算力与反馈收益保本边界

分别控制：`RTT`、上行 Mbps、下行 Mbps、jitter、丢包/重传、设备算力、总请求数/并发。不要用单一“网络差”指标混合多种原因。

建议在**受控网络整形**中预登记的矩阵：

| 场景 | RTT（示例） | 上行 | 下行 | 要证实的瓶颈 |
|---|---:|---|---|---|
| Near edge | 5ms | 50 Mbps | 50 Mbps | 低 RTT 时额外修复是否反而拖慢 |
| WAN high RTT | 80ms | 10 Mbps | 10 Mbps | 减少验证轮次是否抵消计算成本 |
| Uplink-limited | 40ms | 0.5 Mbps | 10 Mbps | 上传 Draft/概率信息的成本 |
| Downlink-limited | 40ms | 10 Mbps | 0.1 Mbps | 8B 额外载荷是否值得 |
| Both constrained | 120ms | 0.1 Mbps | 0.1 Mbps | 系统是否仍保持净收益 |

**这些是建议测试点，不是现实网络统计值**；按硬件/链路可测性修改时应提前写入 manifest。对每项采集实际 wire bytes（帧头、序列化、正常 `a,b`、新增 `m`）、RTT 与 queue；不要只报 8 bytes 有多小。

近似保本条件（仅适用于同步固定窗口、平均轮次成本简化）：

`(Abar + ΔA + 1)/(C + ΔC) > (Abar + 1)/C`，等价于 `ΔA/(Abar+1) > ΔC/C`。

异步模式需计算**关键路径**而非把 Draft/Verify/Network 全部简单相加；真实系统必须同时比 `cloud_only_AR`、`colocated_SD`、`vanilla_edge_cloud_SD`、`async_edge_cloud_SD` 和本方法，并说明云边分离未必总是最优。

终点指标：逐 token Greedy exact match=100%、每请求生成 token、验证轮数、有效 token/s、P50/P95 延迟、边端每 token 计算、峰值内存/能耗（可测时）、上/下行 wire bytes、云端 Target GPU 活跃时间/闲置、异步有效复用率。

---

## 11. 预注册的实验矩阵（按顺序执行，不能一口气跑所有组合）

| ID | 阶段 | 实验 | 仅改变的内容 | 主要指标 | 通关条件/动作 |
|---|---|---|---|---|---|
| `00` | P0 | RAW / EDGE / PROTOCOL | 强基线与同前缀评测 | `A_probe` + exact match | 必须通过正确性 |
| `01` | P1 | FULL-H vs PROTOCOL | 添加未压缩合法 Target hidden | `ΔA_probe`, CI | 证明有可利用信号 |
| `02` | P1 | FULL-LOGIT vs PROTOCOL | 同边界 logits 摘要（诊断） | `ΔA_probe`, CI | 决定 Hidden 还是 logits 路线 |
| `03` | P2 | H-INT8-RECON | 8D × uint8 Hidden | `ΔA_probe`, MSE, 耗时 | 初级可部署基线 |
| `04` | P2 | H-VQ64-RECON | 8×256 PQ/VQ 重建 | `ΔA_probe`, 使用率 | 压缩架构基线 |
| `05` | P2 | H-VQ64-TASK | 只改编码训练目标为后续 CE | **vs C**, vs RECON, vs SHUFFLE | H2 / H3 初检 |
| `06` | P2 | 0/8/16/32/64 bit | 每次只改 budget | `ΔA / bit`, 增量成本 | 找饱和区间 |
| `07` | P2 | CE → CE+prefix | 只改 loss | `A_probe`、CE | 不能只看 CE |
| `08` | P3 | RANK64-TOP2 | 用显式概率排序摘要代替 Hidden | `A_probe`, wire, cloud cost | 同预算竞争 |
| `09` | P3 | RANK64-MARGIN | entropy 字段换 margin | `A_probe` | 测语义特征取舍 |
| `10` | P3 | SCORE64-TASK（有证据才做） | 编码 logits summary | `A_probe` | 找更好的知识接口 |
| `11` | P4 | R-EDGE | 仅本地 risk head | NLL/Brier/成本 | 风险强基线 |
| `12` | P4 | R-PROTOCOL | 加正常拒绝/纠正信息 | NLL/Brier/成本 | 排除协议混淆 |
| `13` | P4 | R-H64 / R-RANK64 | 再加真实教师消息 | NLL/Brier/可复用效用 | H4 是否成立 |
| `14` | P5 | SINGLE64-A/B | 只用一个任务码 | `A_probe` 与风险 | 建立两个单任务边界 |
| `15` | P5 | SPLIT32+32 /16+48 | 固定 64bit，显式分位 | 修复 + 风险 + 训练成本 | 资源竞争 |
| `16` | P5 | SHARED64 | 64bit 共享、双 head | 联合效用、CI | H5 是否成立 |
| `17` | P6 | ASYNC-BASE/LOCAL/TARGET | 只变预取控制信号 | 复用率、浪费、延迟 | H6 的前置条件 |
| `18` | P6 | ASYNC-JOINT | 双 head 与异步管线合并 | P50/P95、真实 token/s | 正确且有净收益 |
| `19` | P7 | 网络/边端矩阵 | RTT/BW/算力 | 全量系统指标 | 报告收益边界而非单点 |

**单变量不是指不同阶段永远只能改变一个组件**；而是每个受控比较必须明示其唯一差异。实验 `05` 对 `04` 的损失目标区别如果涉及多项训练超参变化，应拆分并标注。

---

## 12. 统计、假阳性保护与工程门槛

### 12.1 最小“确认有效”的判据

1. **Repair 增量**：在固定 Val 前缀上，`H-VQ64-TASK − C PROTOCOL` 的按 Prompt 成对均值为正；跨 3 训练种子方向一致；优于打乱消息 E；确认性样本上的预登记 95% CI 下界 > 0 才写“显著正信号”。仅 Pilot 正数写“探索性”。
2. **编码目标**：同等 64bit 和训练/本地参数预算，TASK 与 RECON/RANK 的差异做成对比较；比较对象数量和校正事先锁定，不择优挑指标。
3. **风险增量**：在**当前冻结 Draft**的验证分布上，`R-H64` 的 NLL/Brier 至少有可重复改善；若加速决策的成本无改进，就不要写系统收益。
4. **联合增量**：SHARED64 与固定预算 SPLIT 以及 SINGLE64-B + 本地 A 比较，至少一个系统指标改善，同时无明显灾难性退化；不能只以 joint loss 更低为证据。
5. **正确性**：所有异步及同步方式最终 target-token exact match 100%；否则停止报告性能。
6. **系统收益**：至少在明确的一类预登记网络/边端预算下，以**实测**请求配对时延或 throughput 证明净收益；如只模拟就只报告估计。

### 12.2 不要把“部分改善”误当成成功

- Full-H 比 C 好而 64-bit 不好 -> 信息有价值但压缩器不够，`REVISE`。
- 64-bit 比 RAW 好但不比 C 好 -> 可能只是多了参数或正常纠正信息，`NO H2`。
- CE 降了而 `A_probe` 不变 -> 代理优化不等于 Greedy 接受改善，`REVISE LOSS/HEAD`。
- 风险准确率升了但异步有效复用/等待不改善 -> 没有 H6。
- 提升只在 baseline 原始弱 Draft 上存在、较强本地修复器 C 上消失 -> 缺乏附加教师收益。
- Joint 比单任务弱 -> 保留单任务，**不能为了论文结构强行二合一**。

### 12.3 Stop / Go 决策

- **GO**：真实模型、正确性、Prompt 级 CI 支持对应阶段假设且存在成本上可实现的正收益。
- **REVISE**：诊断说明某一环节可能可修（例如 FULL-H 有效、VQ 塌缩；或已修复 Draft 风险标签漂移），只允许围绕已定位机制做有限 1–3 项小改动。
- **NO-GO**：经过 3 seed、合理强对照与足够 probe 仍无教师消息增量价值，停止堆叠架构，记录负结果。
- **BLOCKED**：模型未下载、GPU/磁盘不足、不能确认 tokenizer 一致、oracle 错误或 packed KV 不一致；不得用 toy 结果顶替。

### 12.4 训练资源公平

- 对需要训练的编码器、decoder 与修复器报告**总参数量、优化步骤/训练 Token、教师 forwards、GPU wall time**；云端编码与边端 decode 的额外运行开销分开报告。
- 单次修改后应以固定 seed 的 Pilot 快筛，再在预选少数候选上用 3 个 seed 确认；不能同一个 val 上做大量隐式选择后把未校正 CI 当确认性检验。
- 随机代码打乱需跨 prompt、也要按 `accepted_count/任务/长度` 分层打乱，保留 D 权重不重训。生成窗口仅能从预先锁定的样本索引取。
- 真实网络指标需 warmup、GPU synchronize 边界、报告重复次数、机器规格/功耗、发包时间戳、token 输出长度，且版本 hash 一致。

---

## 13. Autoresearch 自动实验循环：给 Agent 的实际执行规范

### 13.1 与 Karpathy 原版 autoresearch 的区别

本项目不是修改单个 `train.py` 并最小化 `val_bpb` 的任务。本项目必须锁定数据/评测和目标模型，**只允许修改本研究模块的模型/训练/调度策略**；主指标为真实 Greedy 连续接受长度、风险校准和后续端到端性能。不得把原版 5 分钟训练时间、原始 `prepare.py` 或 `val_bpb` 直接照搬。

### 13.2 封版后允许修改的文件范围

- 允许：`feedback_codec/codec.py` 中 encoder/decoder、`feedback_codec/training.py`、`feedback_codec/losses.py`、新增隔离的 `risk.py`, `message_formats.py`, `policy.py`, `configs/*.yaml`（以真实代码组织为准）。
- 锁定：`datasets_manifest.json`、锁定 Prompt IDs、`oracle.py`、`evaluation.py` 的核心指标、Target 验证规则、split 划分与最终测试集。
- 若发现核心评测 bug：**暂停、版本升级、对所有受影响历史实验统一重测**，不能只在新方法上使用新评测器。

### 13.3 每一轮 Auto-Experiment 必须执行

1. 检查 Git 状态、当前最强 **公平基线**与 `results.tsv`；不能覆盖用户未提交更改。
2. 写 `hypothesis.md`：阶段、单一假设、受控比较、唯一变量、期望影响、可能失败机制、统计单位。
3. 只改对应模块，运行静态检查和功能测试（含真实 `pack/unpack`、无未来泄漏、冻结梯度检查、Greedy exact-match）。
4. 在**同样训练样本/步数、同一 Val** 下运行 Pilot；所有日志重定向到 `runs/<run_id>/train.log`，不在上下文中刷屏。
5. 输出 `A_probe`, `vs C`, `vs E`, 95% CI（可用时）、risk NLL/Brier（风险阶段）、参数量、编码/修复时间、VRAM、有效 msg bits、真实 bytes。
6. 按固定规则选择 keep/discard/inconclusive；结果若落在噪声范围，追加 seed 而不是夸大结论。
7. `results.tsv` 追加本轮；保存小型 ckpt 和 config。用户工作必须保留，不可进行破坏性 reset；仅回退 Agent 当前实验代码。
8. 若阶段 Gate 未通过，只允许第 12 节有限诊断，不擅自越级扩大模块；阶段完成后更新 `reports/phase_P*.md`。

建议标准输出（字段需由新/已有 runner 真实实现，**本文件不假设这些 CLI 已存在**）：

```text
---
phase: P2
variant: H-VQ64-TASK
probe_mean_A: <float>
probe_delta_vs_protocol: <float>
probe_delta_ci95_low: <float>
probe_delta_ci95_high: <float>
probe_delta_vs_shuffle: <float>
feedback_payload_bytes: 8
feedback_wire_bytes: <int or null>
encoder_ms_per_round: <float or null>
decoder_repair_ms_per_token: <float or null>
train_seconds: <float>
peak_vram_mb: <float>
correctness_token_match: <float or null>
status: keep|discard|inconclusive|blocked|crash
```

建议 `results.tsv`：

```text
run_id\tcommit\tphase\tvariant\tseed\thypothesis\tdata_hash\teval_version\ttrain_steps\ttrain_time_s\tmean_A_probe\tdelta_vs_C\tci95_low\tci95_high\tdelta_vs_shuffle\trisk_nll\trisk_brier\tfeedback_bits\twire_bytes\tlatency_p50_ms\tlatency_p95_ms\tpeak_vram_mb\tstatus\tnotes
```

### 13.4 Runner 最小接口（如果不存在，应新增而不是虚构命令成功）

```bash
# 以下是期望的接口协议，具体脚本名可以按项目实际情况实现。
python -m pytest -q
python scripts/prepare_trajectories.py --config configs/data_v1.yaml
python scripts/run_feedback_experiment.py --config configs/p2_h_vq64_task.yaml --seed 1
python scripts/evaluate_fixed_prefix.py --manifest datasets_manifest.json --checkpoint <checkpoint_path>
python scripts/evaluate_on_policy.py --manifest datasets_manifest.json --checkpoint <checkpoint_path>
python scripts/summarize_results.py --results results.tsv
```

runner 必须支持 `--dry-run`/`--help`、可中断后恢复且不覆盖历史实验目录、错误 traceback、GPU OOM 明示、unavailable model checkpoint 的 BLOCKED 退出，而不能悄悄换模型。若没有真实模型/训练集，只能运行 toy smoke 并清楚标注。

### 13.5 目录交付物

```text
program.md                              # 研究协议（本文件）
datasets_manifest.json                  # 数据与模型 hash/split/固定配置
configs/                                # 一阶段一变量的实验定义
runs/<run_id>/config.json
runs/<run_id>/train.log
runs/<run_id>/summary.json
runs/<run_id>/per_prompt.jsonl          # Prompt 级配对结果
runs/<run_id>/message_stats.json        # code usage/entropy/packed bytes
results.tsv                             # 所有实验，包括阴性/失败
reports/phase_P0.md
reports/phase_P1.md
reports/phase_P2.md
reports/phase_P3.md
reports/phase_P4.md
reports/phase_P5.md
reports/phase_P6.md
reports/phase_P7.md
reports/final_research_decision.md
```

**大型缓存、模型权重和训练轨迹不加入 Git**；记录存储路径和 sha。结论段必须标注实际执行的日期、GPU/网络、成功运行数量，并明确区分“未运行”和“负结果”。

---

## 14. 需要重点避免的三种科学与工程假象

### 14.1 “压缩 Hidden 能重建”≠“对下游 Draft 有效”

Hidden 重建可作为对照，但主目标是下轮有效的 Greedy 连续接受长度。可重建大量不相关维度并不保证额外预测收益。要比较 `H-VQ64-RECON` 与 `H-VQ64-TASK` 才能判断任务导向编码有无增益。

### 14.2 “Top-K 有信息”≠“下一轮仍有用”

当前 Target `z_T(valid_prefix_t)` 是**生成 `b_t` 时**的分布；它未必预测 `b_t` 后多个 token 的路径。若显式 Rank64 无效，不等于排序型蒸馏在别的协议上无效；只是该零额外 Target Forward 的边界信号不够用。

### 14.3 “预测拒绝点”≠“未来 token 已知”

风险 head 无论多准确，也不能推断 Target 的实际纠正 token。只有提前计算的路径**恰好匹配**最终 Target 确认的前缀，之后的 token 和 KV 才能复用。异步必须同时正确地处理部分接受、零接受、全接受 + bonus、EOS 与分支覆盖不足。

---

## 15. 最终报告必须回答的问题

1. `F FULL-H` 和 `F LOGIT` 各自是否有超出 `C PROTOCOL` 的增量收益？能否证明任务值得压缩？
2. `H-INT8`, `H-VQ64-RECON`, `H-VQ64-TASK`, `RANK64` 在**相同 64-bit** 与相当边端成本下的 Greedy `ΔA`、CI 分别多少？
3. Teacher 消息打乱、移除 Target hidden、仅有正常协议信息后结果如何？
4. 是否存在码本塌缩、错误码字、量化饱和、更多参数带来的虚假增益？
5. 修复后的 Draft 改变了拒绝分布多少？A-only 的 NLL/Brier 和实际调度效用比本地 WISP 风格预测器改善多少？
6. `SHARED64` 是否确实比 `SINGLE64-B + local risk`、`SPLIT32+32` 更好？如更差，是否应该保留单任务？
7. 所有闭环输出是否与 Target Greedy AR 完全一致？bonus/纠正后的缓存是否严格合法？
8. 端到端是实测、网络整形、模拟还是纯数学模型？真实上下行字节、额外计算、P95 延迟、边端峰值内存如何？
9. 在哪些设备/网络区间有净收益，在哪些区间没有？与 cloud-only / co-located SD 相比部署的合理性如何？
10. 最终状态是 `GO` / `REVISE` / `NO-GO` / `BLOCKED`，能支持的具体贡献是什么、哪些仍待验证？

---

## 16. 研究相关论文与使用边界

以下文献为**实验动机和方法启发**，不是这些作者已经验证过本文件的 64-bit 布局或本项目的实验结果。

1. **ConFu**，ICML 2026：Target 的 future guidance 改善 Draft；原方法主要针对 EAGLE-3 并含 contemplate 机制，并非已验证的低带宽 8-byte 云边协议。 https://proceedings.mlr.press/v306/qin26c.html
2. **SpecBranch**，ICLR 2026：回滚感知的并行分支与动态起草，用于 P4/P6 的风险与路径复用对照；不能把预测拒绝当作已知纠正 token。 https://proceedings.iclr.cc/paper_files/paper/2026/hash/67089958e98b243d5cc1881ad60418b8-Abstract-Conference.html
3. **WISP**，POMACS 2026：使用本地风险预测减少无效 Draft，并关注服务端验证干扰。用于 P4 的**本地**风险强基线。 https://arxiv.org/abs/2601.11652
4. **PicoSpec**，2026 预印本：异步边云 SD 与通信问题，用于 P6 的系统底座/思想参照；完整复现需要相应协议与模型适配。 https://arxiv.org/abs/2603.19133
5. **FedFD**，NeurIPS 2025：异构模型的特征蒸馏与投影对齐，支持“原始 Target hidden 不一定适合直接给小模型”这一设计动机；不是 64-bit 实现。 https://proceedings.nips.cc/paper_files/paper/2025/hash/96dd63ab8fd79163a5399e0b385f208d-Abstract-Conference.html
6. **FLM-TopK**，IEEE INFOCOM 2025：稀疏传输必须考虑索引开销；原论文研究梯度传输，不是当前 LLM logits。 https://ieeexplore.ieee.org/document/11044514
7. **PLD**，NeurIPS 2025：排序蒸馏的知识表达思想；原实验并非 LLM 在线验证反馈，P3 是从中借鉴的**新对照设计**。 https://proceedings.neurips.cc/paper_files/paper/2025/hash/c6b2921f24f82dc05cda53bbe50bab90-Abstract-Conference.html
8. **Communication-Aware KD for Federated LLM Fine-Tuning**，2025：基于带宽的 adaptive Top-K logits 与隐藏层投影，P3 的启发。 https://arxiv.org/abs/2509.01750
9. **Categorical Reparameterization with Gumbel-Softmax**，ICLR 2017：可学习硬离散消息的训练工具；使用 STE 不意味着梯度无偏。 https://arxiv.org/abs/1611.01144
10. **Karpathy Autoresearch Program**：不可更改的评测与记录每次实验的思想来源；本项目定制的目标与时间预算不同。 https://github.com/karpathy/autoresearch/blob/master/program.md

---

## 17. Agent 的启动顺序（无需读完后再问已经确定的事情）

**从现在开始，只执行实际具备资源的最低尚未完成阶段。**

1. 检查目录、文件、git、模型、GPU 和训练集；输出当前阻塞项，不要伪造实验。
2. 若缺 `A/B/C` 或 fixed-prefix probe，先实现并补严格测试；不要训练 Joint 模型。
3. 若 `A/B/C` 齐全，运行 **P1 的合法 Full-H & Full-Logit 诊断**；基于结果决定 P2/P3 的可行性。
4. 按第 11 节顺序逐一实验，记录成功和失败的所有配置；不要用 test 做模型选择。
5. 未达到阶段 gate 时停止升级复杂度，生成 `reports/phase_P*.md` 的明确判断和下阶段最小修复计划。
6. 所有完成的结论必须能由 `results.tsv`、同版本评测、原始日志和可复现 checkpoint 追溯。

**最终优先级不变：先证明 64-bit Target 反馈相对于协议本身具有独立的未来预测收益；再证明它帮助拒绝风险预测；最后才研究两个目标共享这 64 bits 能否真正减少异步云边成本。**
