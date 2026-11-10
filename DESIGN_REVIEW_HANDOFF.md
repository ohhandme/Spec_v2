# 反馈编码投机推理：代码设计与审查交接

**状态（2026-10-30）：** 已完成本地参考实现和小模型功能测试；尚未运行真实 Qwen3-8B／Qwen3-0.6B 配对实验，也没有真实云边时延或加速结论。本文供另一位 AI 对照原方案与源码审查，不把方案稿中的设想当作已证实结果。

**来源：** 用户给出的《面向后续预测收益的云边投机推理反馈编码》`10.28(1).md`，原始本机路径为 `C:/Users/asus/xwechat_files/wxid_zwms7t0hqv5622_4135/temp/RWTemp/2026-10/cac5679a2468eb6f77a09ffbec7b7a5c/10.28(1).md`，SHA256 为 `1569e29487822bedb351bc2bf39d6e0986e2c7e31259a822e17b15e1351c5c21`。原稿注明“实验部分为设计，尚未形成实测结论”。工程根目录即本文件所在的 `experiments/feedback-coded-specdec/`。详细逐节索引另见 `SOURCE_TRACE.md`。

## 1. 要实现的机制

云端冻结目标模型 (T)（方案首版为 Qwen3-8B），边端冻结草稿模型 (D)（Qwen3-0.6B），greedy 解码，草稿窗口 (L=8)。正常投机验证给出接受数 (a) 和纠错／奖励 token (b)。方案新增一个**固定 8 字节的离散反馈消息** (m)，用本轮正常验证已得到的信息指导**下一轮**草稿。消息不参与目标模型的接受规则，也不是输出文本。

数据流可以写成：

```text
边端 D：共享正确前缀 x + 上轮消息 m  →  起草 y1…yL
云端 T：在正确前缀上验证 y          →  接受 a 个、产生 b
云端编码器：有效前缀的 hT、首拒 token 或空标记、b、a/L
                                        → 8 个 0..255 编号 → 8 原始字节
边端：正确前缀更新为 x || y1:a || b，查本地码表
                                        → 下一轮每个位置的输出侧修复
```

例如 (a=3) 时，第 4 个草稿 token 是首拒项。编码器取的是 `x || y1:y3` 上**用于预测 b** 的目标隐藏状态，另取第 4 个草稿 token 与 (b) 的输入嵌入和 (3/8)。第 4 个之后的错误草稿隐藏状态、以及 (b) 之后尚未计算的目标隐藏状态，均不能进入编码器。若 8 个草稿全接受，则首拒项换成可训练的空标记，(b) 是奖励 token。首轮没有反馈；结束请求时也不再生成消息。

## 2. 原方案与源码逐项对应

| 原稿要求 | 代码入口 | 实际行为 |
|---|---|---|
| 冻结模型、词表一致、取最终归一化隐藏状态 | `feedback_codec/hf_backend.py:25,88,106,136,169,193` | 本地 HF 后端仅加载已有权重；冻结两模型；检查词表、特殊 token 和模板一致；用 final-norm hook 检查 `hidden_states[-1]`。|
| 有效验证边界与纠错／奖励 (b) | `feedback_codec/decoding.py:111` | 逐个验证候选；只把已接受 token 加进目标前缀。首拒时记录该有效前缀的目标 step；全接受时再取 bonus step；消息仅给下一轮。|
| 云端输入 (v=[P_h h^T,P_e e(y_{a+1}),P_e e(b),a/L]) | `feedback_codec/codec.py:94,125` | 两个低维投影、共享 token 嵌入投影、拒绝空标记与接受比例；没有领域标签，也不读取未来教师续文。|
| 8 组×256 离散编号，新增载荷 8 字节 | `feedback_codec/codec.py:20,50,61,69,189`；`feedback_codec/training.py:50` | 推理 `argmax` 得 `[8]` 编号，打包／解包为恰好 8 字节；错误长度及越界编号报错。8 字节仅指新增反馈负载，不含正常 `a,b` 和帧头。|
| 边端 (8\times256\times16) 码表、rank-32 修复 | `feedback_codec/codec.py:94,223,241`；`feedback_codec/hf_backend.py:211` | 码表拼成 128 维控制；`u=Vh`、`gate=f(c,u,j/L)`、`h̃=h+U(gate⊙u)`，再经冻结输出头。无消息时使用草稿模型原 logits。|
| CE、前缀代理损失、直通离散训练 | `feedback_codec/losses.py:11`；`feedback_codec/codec.py:204`；`feedback_codec/training.py:146` | 教师强制续文上计算 CE 和 (1-\frac1L\sum_{k=1}^{L}\prod_{j=1}^{k}p_j)。`gumbel_softmax(hard=True)` 以前向硬选择、反向软梯度联合更新编码器／码表／修复器。|
| 普通轨迹→CE 热身→当前策略重采→CE+前缀训练 | `feedback_codec/training.py:78`；`feedback_codec/pipeline.py:88` | 每条轨迹记录有效前缀、首拒 token、(b)、纠正后前缀；离线目标 AR 从**纠正后前缀**生成未来标签。热身后重新起草、重新收集验证边界。|
| 云边分离与回包时序 | `feedback_codec/cloud_edge.py:79,209` | 本机 loopback socket 分离云验证器和边草稿器；非终止轮回包附 8 原始字节，边端在下一轮使用。当前控制帧是 JSON，且为一致性检查重发完整前缀。|
| 完整输出正确性与接受长度比较 | `feedback_codec/evaluation.py:37,96`；`feedback_codec/decoding.py:217` | 同请求对照无反馈、学习反馈、跨请求打乱码；每条输出须与独立目标 AR 的完整 token／EOS／截断一致，之后才统计指标和配对 bootstrap 95% 区间。|
| 云边成本 | `feedback_codec/cost_model.py:15,43` | 只接收外部实测的 RTT、带宽及 GPU 耗时，做明确标注的串行假设计算；不是本代码测得的加速。|

## 3. 训练时的信息边界

每条训练例包含：本轮的**有效验证状态**、纠正后的前缀，以及目标模型从该前缀继续生成的参考 token。`collect_trajectories` 会先用独立目标 AR 检查整个投机输出是否正确，再生成离线续文。训练器把前一部分交给编码器，把续文交给草稿的 teacher-forcing 损失；它不会将未来 token 塞进 `FeedbackCodec.scores`。

`HFDraftBackend.teacher_hidden`（`hf_backend.py:237`）对 `corrected_prefix + continuation[:-1]` 前向，取每一位置预测相应未来 token 的最终归一化隐藏状态。目标／草稿基座和输出头冻结；优化器只含反馈编码器、码表、低秩矩阵和控制网络。第一阶段前缀损失权重为 0，第二阶段使用配置的权重。`pipeline.py` 保存原始与刷新轨迹、训练历史、阶段权重和协议文件。

## 4. 推理与评测口径

- `SpeculativeDecoder` 的第一轮 `codes=None`。边端先拟稿，云端确认接受前缀并给 (b)，然后才产生下一轮的码。被拒后的草稿尾部丢弃；已接受草稿 EOS、目标纠错 EOS、奖励 EOS、输出上限都作为终止条件。
- `accepted_draft_tokens_per_round` 是每次验证平均兑现的草稿 token 数；`verification_rounds` 是完成请求的验证轮数。`draft_token_acceptance_fraction = accepted / proposed` 更准确地说是**提案 token 利用比例**：首拒之后的候选虽然进入分母，并没有逐个被目标验证，不应称作独立逐 token 接受概率。
- `feedback_bytes` 只数实际发出的 8 字节消息。完整 socket 帧另有长度、JSON 控制字段和本地参考实现重发的前缀；不能把 `feedback_bytes` 当作总链路流量。
- 配对评测对每个 prompt 运行无反馈、学习反馈、打乱反馈；对学习反馈减无反馈的逐请求差值做 bootstrap。只有测试数据与训练数据严格分离、所有请求完整正确时，接受长度差值才可解释。

## 5. 运行与现有证据

`scripts/tokenize_prompts.py` 可将固定 chat template 下的原始提示转成锁定的 `id/split/input_ids` JSONL；`feedback_codec/pipeline.py:40` 拒绝重复 ID 和跨划分重复 token 序列。`scripts/run_local_pilot.py` 需要已下载的两个本地模型目录、预先固定的 JSONL 与新输出目录；它记录输入、源码及模型文件 SHA256。运行命令见 `README.md`。

本机已运行 **43 项测试通过**，包括编码字节数、梯度到编码器／码表／修复器、冻结输出头、tiny Qwen3 端到端训练、EOS／截断逐 token 正确性和 loopback socket 回包。复查命令（从工程根目录运行）：

```powershell
& '../specbranch-reproduction/.venv/Scripts/python.exe' -m pytest -q --basetemp=.pytest_tmp_review
```

在当前 Windows 沙箱里，真实 `socketpair` 测试可能需要允许**仅本机 loopback**；无此权限时该套件可能阻塞。此前 43 项通过是在允许本机 loopback 后获得的，未连接远端服务器。

该测试数量是小模型／模拟逻辑的功能证据，**不是** Qwen3-8B／0.6B 的接受率结果。

## 6. 已知边界与建议审查重点

1. **性能实现尚未完成。** HF 后端每个 token 重算完整前缀，目标模型顺序验证，不是并行 packed 验证；虽然后置修复在结构上不依赖历史 KV，当前代码没有实现真正的 KV 缓存复用。不能由本地运行时间推断云边加速。
2. **线协议是本机参考。** 只有反馈负载严格为 8 字节。JSON 帧、完整前缀重发、无跨机认证／传输保障，尚不能作为生产或真实 WAN 协议。
3. **真实模型和强基线尚未运行。** 未在 Qwen3-8B／0.6B 上验证接受长度，也未完成原稿提出的等参数 edge-only、8 字节 hidden 重建、ConFu/SPARK 等强基线；目前只实现无反馈与打乱反馈对照。
4. **实验设计尚未封口。** CLI 支持 train/test，但没有内置领域数据构造、独立 validation/选模、预注册样本量或多随机种子；默认超参数只是首版起点，输出不构成论文统计结论。
5. **训练与推理存在可审查的近似。** 训练以 teacher-forced CE／前缀乘积代理优化，推理使用 greedy 硬 argmax；直通 Gumbel 的梯度是近似。请检查码本是否真正学习到了超出正常 `a,b` 与文本上下文的增量信息，而非仅记忆训练分布。
6. **数值一致性仍需实卡门槛。** 本地后端使用最终归一化隐藏状态，真实 BF16、packed 并行验证和不同内核可能产生边界 argmax 差异。任何真实加速实验都应先逐请求核对完整 token/EOS/截断，再计时。
7. **8 字节容量不等于有效信息量。** 8 组各 256 个选项至多提供 64 bit 的码空间，但训练可能只使用少数码字；当前没有码字频率、熵或塌缩监控。真实实验应记录各组使用直方图，并做预算／打乱反馈消融，再判断目标信息是否真的带来增益。

建议审查者优先回答：因果输入是否严守本轮有效前缀？全接受与 EOS 分支是否遗漏反馈或错误追加 bonus？硬码前向和训练梯度是否确实贯通？冻结模型与输出头是否意外更新？8 字节是否只算新增负载？接受长度统计是否把未验证候选误称为逐 token 概率？在加入 packed/KV 和真实 WAN 测量前，哪些性能主张必须保持未验证？
