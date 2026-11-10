# Packed BF16 数值复核候选 v1（未过 GPU P0）

本候选只在隔离 v2 工作树的 `feedback_codec/packed.py` 中增加**默认关闭**的 `numerical_recheck_top2_margin`。旧归档采集器、旧 P0、原始失败及独立 AR oracle 均不修改。首次失败是首轮第 5 个输出 token：正常 L8 packed BF16 验证在 `264` 与 `825` 之间并列，实际错收了 `264`；使用相同在线 KV 副本、把 query 缩短至该行时选择 `825`。这个观察支持尝试低间隔复核，不构成一般正确性的证明。

启用时，正常 `L+1` packed forward 仍先执行，且在其前保存在线验证前 KV 副本。只沿**当前可能提交的行**按顺序检查 top2 间隔；间隔不大于设定阈值且该行尚有后续提议时，再从同一在线 KV 副本执行一次截至该行的短 query。复核行的 BF16 argmax、完整 logits 和最终 hidden 一起替换为该短 query 的值，再决定接受还是首次拒绝。若它成为本轮有效决策行，下一轮沿用它自身的 KV，不裁剪数值不同的原长 query KV。全接收后的 bonus 行已是原 packed 查询的最后一行，无法缩短；若其间隔也低于阈值且确实需要 bonus，候选额外做一次独立完整前缀 Target 决策和一次同前缀缓存 refill，共 **2 次**额外 Target forward，并把该行来源标成 `canonical_full_prefix_numeric_recheck`。两种复核都不读取预存 AR 答案，也不窥视更晚草稿后缀；不需要 bonus 的 EOS/预算终止轮不复核 bonus。

运行身份为 `packed_with_numerical_recheck_candidate_v1`，不能叫单次 forward packed。每轮 trace 记录复核行、额外 Target forward 次数、每轮**必付**的在线 KV 快照 clone 同步墙钟时间、触发后的同步 forward 时间、两者及必要复制合计的总复核墙钟时间，以及提交 KV 的来源。若有效特征来自短 query 复核行，因果 provenance 标为 `numeric_recheck_query_forward`，并从另一份在线 KV 副本逐位重放该**同一短 query 的同一行**；短 query 在有效行结束，不含未确认后缀。若有效特征来自完整前缀 bonus 复核行，来源是 `canonical_full_prefix_numeric_recheck`，独立再次执行完整前缀检验。未复核有效行保留原 `original_packed_verify_forward` provenance 与后缀干预测试。

`numerical_recheck_extra_target_forwards` 只统计生成时为数值决策新增的 forward；P0/离线采集的因果 provenance 还会做独立重放或后缀干预，在 `diagnostic_feature_gate_target_forwards` 中另记，不能混入或隐去生成时成本。候选只有在新版本的真实 GPU 正确性 gate、逐请求长生成与重新标定成本完成后才能进入下一阶段。

此版本不是已通过的系统实现：

1. **阈值不是数值误差上界。** 仅凭 BF16 tie 样例，不能推断所有大间隔行都与独立完整前缀 AR 相同。任意长请求逐 token/EOS/截断不一致仍须立刻停止，不删失败样本。
2. 增加的 KV 复制、Target forward 和同步时间必须重新标定成本；原冻结的“每轮 0 次额外 Target forward”成本代理不能直接套用。不能从该候选宣称 WAN 或在线加速。
3. 有效 hidden/logits 若来自额外复核 forward，不能再写成“原单次 packed 验证免费给出”。原 P0 来源断言需要版本化更新并重跑完整真实 BF16 GPU gate；旧 P0 通过不继承。
4. 当前仅有针对性 CPU 形状敏感测试和语法检查，尚未部署 GPU，也未解锁 P1 训练或读取新验证集接受长度。

若未来真实 GPU 的长生成 exact gate 失败，需要保存其数值证据，针对实际根因做有界修复；不改变 Target Greedy 定义、BF16 精度或独立 AR 门槛。
