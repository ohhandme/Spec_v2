# v2 P0 预锁：数据、对照、统计与成本

状态：**PRELOCK / 不可启动采集或训练。** 本文在新验证集结果出现前写成，落实根目录 `program.md`（SHA256 `798a15439ffcbca43d15809bb1d492ee0e5b66daa09ed722288abfae72c418d0`）的 P0。需先取得精确数据源 revision/许可证/文件 SHA、独立 split 与真实成本标定，再把 `datasets_manifest.prelock.json` 封成不可变的 `datasets_manifest.json` 和数值 `cost_lock.json`。此前不能以“预锁”冒充已通过 P0。

## 已知事实与隔离

- 本 worktree 起点是 `21a43380a7a41cabceea03ce7f2e9ee59dd9fd35`，新方案在此分支更新。旧 balanced3000 的单种子验证结果见外层 `autoresearch/feedback-coded-specdec-balanced3000-261031/reports/BALANCED3000_SEED42_VAL_20261101.md`：D−C 为 +0.01942 token/提示，但 D−同层打乱 E 为 +0.00061，95% CI 跨零。旧 120 条 val 已参与方案修订，**不能作为本版确认集**；旧 60 条 test 继续封存。
- 保持官方 Qwen3-8B BF16 Target revision `b968826d9c46dd6066d109eabc6255188de91218` 与 Qwen3-0.6B BF16 Draft revision `c1899de289a04d12100db370d81485cdf75e47ca`。现有 8B 资产清单为 `qwen3-8b-expected.json`，旧 0.6B 权重 SHA256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`；均须在新 P0 对实际远端文件、config、tokenizer、chat template 再核验。已有历史远端 torch 2.8.0+cu128、transformers 4.57.3 和四张 4090 均仅作线索，以新 run 的现场采样为准。
- 当前 Git 工作树、运行资产、独立 AR oracle 与新 packed/KV 门槛必须另行记录 SHA。不得复用旧顺序 full-prefix 后端的结果来宣称 packed 正确或系统加速。

## 新数据方案：先查可行性，再封版

优先采用新版原文的 Pilot 预算：**1000 个训练原始提示、200 个开发验证原始提示，Greedy L8，每提示最多生成 128 token，最多取 8 个原始 Vanilla SD 窗口**。同源候选的建议配额先定为下表；仓库版本与许可已按官方源核对，**原始文件 SHA 和可用池仍待本次逐文件核验**。不满足任一配额即停在 P0 并记录实测池大小，不静默借样本。若因实测 GPU/磁盘/源池不足需要缩放，应在查看任何新 val 接受长度前一次性写入独立版本的缩放原因与所有对照共同预算。

| 来源候选 | Train | Pilot Val | Formal Train（仅过 Pilot gate 后） | Formal 确认 Val |
| --- | ---: | ---: | ---: | ---: |
| UltraChat-200K `train_sft` | 600 | 120 | 3600 | 360 |
| Databricks Dolly-15k | 200 | 40 | 1200 | 120 |
| CodeAlpaca | 100 | 20 | 600 | 60 |
| GSM8K **train** | 100 | 20 | 600 | 60 |

官方固定来源：UltraChat-200K revision `8049631c405ae6576f93f445c6b8166f76f5505a`（MIT，`train_sft.prompt`）；Dolly-15k revision `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`（CC-BY-SA-3.0，`train.instruction` 加非空 `context`）；CodeAlpaca-20k revision `152bb5e9a29651266b018106053980070a0521a1`（`train.instruction` 加非空 `input`，原作者 [DATA_LICENSE](https://github.com/sahil280114/codealpaca/blob/master/DATA_LICENSE) 为 **CC-BY-NC-4.0**，优先按此较严格许可，HF 镜像 CC-BY-4.0 标签有冲突）；GSM8K revision `740312add88f781978c0658806c59bc2815b9866`（MIT，只用 `main/train.question`）。只取提示，不读 assistant/response/output/answer 作输入或标签。每个本次原始文件仍须逐 SHA 锁定后才采样；UltraChat 旧 SHA 也须重新核对。不使用 GSM8K test、SpecBench 或 SPEED-Bench 做训练/调参；最终测试集只在候选封版后一次使用。Formal 6000/600 是后续确认预算，不能把 Pilot200 上反复选模后的区间写成最终确认。

| 预算上界 | Pilot（train+val=1200） | Formal（train+确认val=6600） |
| --- | ---: | ---: |
| 真实生成 token（每提示≤128） | 153,600 | 844,800 |
| 固定 probe 窗口（每提示≤8） | 9,600 | 52,800 |
| 独立 Target future oracle token（每窗口≤8） | 76,800 | 422,400 |
| 每窗口单个 4096 维 BF16 FULL-H 的裸存储上界 | 75 MiB | 412.5 MiB |

这些是**容量上界**，不把 oracle token 或特征缓存时间算成已测 GPU 用时；额外 full logits 若用于离线损失，须单列磁盘预算且不以完整词表矩阵作为 F-LOGIT 在线输入。Pilot 先不缩小，Formal 的 128 token 是原文允许的 128–256 下端。远端四卡只在各自满足无其他计算进程、空闲≥22,000 MiB、连续三次利用率<10%时互斥分片；若少卡就排队/分段缓存，不改变对照的数据、样本序或标签。Pilot 真正采集前按实测吞吐、空闲盘与显存记录预计 wall time；不能把旧 32-token 采集时长直接线性当新 128-token 承诺。

对每个来源只抽原始用户问题/指令；原数据集的 assistant 答案不进入模型输入或在线消息。选定来源后先统一 Qwen3 非思考聊天模板（`enable_thinking=False`, `add_generation_prompt=True`）、tokenizer 和最长输入 256 token，再做 NFKC、大小写/空白归一化的 exact family hash 与字符 5-gram Jaccard≥0.85 近重复屏蔽。新 Pilot train、Pilot val、Formal train、Formal 确认 val、最终 test **两两按 family 隔离**；将旧 1024 与 balanced3000 的 train/val/test 家族纳入排除清单，旧封存 test 只使用已有锁文件中的身份/哈希做泄漏排除，不读其模型输出。先冻结全部验证身份，再选训练；具体采样 seed `20261101`、source IDs、排除计数、每 split token JSONL SHA 写入最终 `datasets_manifest.json`。

Vanilla Draft/Target Greedy L8 轨迹从原始 prompt 采集。窗口选择在看任何新方法表现前锁为**每提示按 `round_index` 升序取前 8 个存在下一窗口的验证响应纠正后前缀**；不按方法的接受长度筛窗口。选中的短续文/EOS 窗口仍留在所有窗口敏感性分析；主集只用预定义的目标至少还有 8 个非终止 token 的窗口，报告排除比例。离线独立 Target greedy future 只作标签/评测，绝不进入在线 encoder。保存有效 Target hidden 与同一验证边界 logits 的来源 forward ID；`a=0` 缺有效前缀缓存时如需额外 Target forward，必须逐轮计数和入账。

## P0/P1 对照与确认规则

- A RAW：冻结 Draft。B EDGE：本地信息；C PROTOCOL：B 加正常 `accepted_count`、被拒候选、纠正/奖励 token；F FULL-H：C 加**未经压缩的合法 Target 有效隐藏状态**，高带宽诊断上界；F LOGIT：C 加**同一次合法验证 forward** 的当前边界 logits 摘要，预定使用 Top32 token ID + FP16 logit、logsumexp 和 margin（非未来 logits）。D 64-H 与 E SHUFFLE 留到 P2。旧 F 的 8192 字节 `8×256` soft 概率不是本版 FULL-H。
- B/C/F/F-LOGIT 使用相同输出侧 rank32 修复器、同一步数/窗口/种子和同一边端优化器。F 与 F-LOGIT 的高维输入投影必须有 B/C 的**同参数量、同每 token 修复预算**本地可用特征投影对照；目标边端可训练参数偏差≤5%，否则将差额与参数匹配消融作为阻塞项。云端 encoder/取特征成本单列；F/F-LOGIT 高带宽字节单列，不能记成 8 字节可部署方案。
- P0 须先通过 8 原始字节 pack/unpack、非法长度拒绝、冻结梯度、无未来泄漏、零/部分/全接受+bonus/EOS/截断、独立 Target AR 完整 token 一致，以及真实 Qwen packed verify/KV 与 reference 的逐 token 一致；任一失败标 `BLOCKED`。首次无消息，`m_t` 仅在下一窗口使用。
- Pilot seed `42` 只用于管线和候选筛选；确认 seed 预锁 `42,43,44`。若 Pilot 中调过模型/损失，Formal 的 600 个**新提示**只在候选与步骤封版后评估一次。每个提示先聚合≤8窗口，再按提示配对；相同提示索引在三 seed 同步重采样，10,000 次 bootstrap。主对比固定为 FULL-H−C 与 F-LOGIT−C；若同时作确认性声称，对两者的按提示配对置换检验作 Holm step-down，FWER 0.05。三个 seed 的点估计必须同号，独立 Formal 提示的配对 95% CI 下界也须>0，且固定权重的 hidden/logit 置零和分层跨请求置换须显示真实内容效应；否则仅写探索/不确定。B 对照及 CE/top1/首拒为次指标，不替代主门槛。避免用同一 Val 大量隐式选择后仍称原区间为确认性。

## ΔA_min 的预锁计算法（数值 P0 实测后才能填）

当前 `cost_model.py` 是**串行情景计算器**，不是测得的加速。P0 要在真实 GPU 用同步边界、预热与重复测量锁定 Draft 每 token、packed Target verify 每轮、encoder 每条消息、edge decode 每消息和 rank32 repair 每 token 的 P50/P95；分别记录额外 Target forward 计数、真实序列化 bytes 与本地 peak VRAM。网络情景先按原文建议锁定：near edge `(RTT 5ms, up/down 50/50 Mbps)`、high RTT `(80ms,10/10 Mbps)`、uplink limited `(40ms,0.5/10 Mbps)`、downlink limited `(40ms,10/0.1 Mbps)`、both constrained `(120ms,0.1/0.1 Mbps)`。这些是**模拟/整形情景，不是实测网络分布**；若有真实电脑↔4090 回路测量，另存原始 RTT/上传/下载序列与时间戳，不能把该电脑 CPU 当已测边缘 GPU。

对非终止一轮令 `T_C = T_draft8 + T_verify_packed + T_up + T_down_ordinary + RTT`，`T_extra64 = T_encode + T_decode + 8*T_repair + 8/B_down`（`B_down` 用 byte/s；另计帧头/序列化增量）；反馈与正常回包同包时**不额外加一次 RTT**。以旧验证集 C=3.1228 只作历史规划值，并对 `A_C∈{2,3.1228,4.5}` 全部列值；保守保本阈值为 `ΔA_min=(1+A_C)*T_extra64/T_C`，在每个预登记网络情景中用 `T_extra64` 的 P95 与 `T_C` 的 P10 计算。若获得合法在线实测，须再以请求级 wall time 验证；这个简式不代替端到端结论。P0 输出不可变 `cost_lock.json`，包含实际计时、场景、数值阈值与 SHA，并在读取新 val 结果之前签定。Full-H 与 F-LOGIT 仅判断信息可利用性，不以 64-bit 的成本阈值冒充其可部署性。

## P0 交付门槛和仍待查事实

1. 完成数据源精确仓库/revision/许可及原件 SHA、池大小/近重审计，冻结 split 与最大窗口索引、`datasets_manifest.json`、独立 test 封存 SHA；绝不先看新 val 效果。
2. 对远端模型实际文件重新逐 SHA、tokenizer/chat template/config、BF16/torch/transformers、GPU UUID/RAM/磁盘检查并记录运行协议；旧历史值不能代替新证据。
3. 验证新 packed/KV、独立 AR、全部边界/消息时间规则；训练后每臂闭环 SD→AR 仍须单独做，fixed-prefix 结果不能代替。
4. 实测并封版成本阈值与共同训练曝光/预算；建立 `oracle_hash`, `eval_protocol_version` 和候选比较家族。此四项未全绿，状态为 `P0_BLOCKED/PENDING`，不得启动 P1 质量结论或宣称系统收益。
