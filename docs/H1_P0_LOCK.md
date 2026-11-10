# H1 预实验 P0 封版前协议

原始研究设计是本目录的 `program.md`（SHA256 `75c4872896962a7c0e798bbc0ecde3e8579407c94fceb81aeb6564aaefef8ae5`）。本文件记录本次 4090 pilot 在看任何真实接受长度结果前作出的选择。P0 真实模型正确性通过后冻结评测语义、样本与数值配置。

## 模型与输入

- Target：官方 Qwen3-8B BF16，HF revision `b968826d9c46dd6066d109eabc6255188de91218`；模型文件逐一核对已有已验证资产清单的 SHA256。官方 ModelScope 镜像只作下载来源，必须与这些哈希相等。
- Draft：官方 Qwen3-0.6B BF16，权重 SHA256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`。两基座与 Draft 输出头冻结；不量化。
- Qwen chat template 使用 `enable_thinking=False`、`add_generation_prompt=True`，两个模型的 token ID、特殊 token 和实际模板必须相同。
- 数据源为固定 UltraChat revision `8049631c405ae6576f93f445c6b8166f76f5505a` 的首条 user 消息，不把既有 assistant 答案输入模型或用作监督。提示文件 SHA256 `f52acc89e67252bb2f8225c5c4bd1c6d4427a81a267a69053c8b3006ae45af1f`；1024 train、64 val、60 test。按归一化 prompt family 排重和拆分；字符 5-gram 跨 split 近重复阈值 0.85 的复核数为 0。正式 test 只在 val 决策后一次性解封。
- 这 60 条 test 曾作为另一独立项目的 prompt 来源，虽未用于本研究的模型/反馈训练和选择，最终报告仍需披露跨项目复用这一限制。

## 门槛与主比较

- 真实 P0：L=8，Target 与 Draft 同卡 BF16；固定首 4 条 train prompt 各最多 24 新 token；独立 Target greedy AR 与无反馈、随机未训练 8 字节反馈两种投机轨迹完整 token/EOS/截断逐项一致；固定首条 AR 重复结果一致。验证首轮无消息、8 字节往返、有效 Target 状态、消息晚一轮生效、冻结基座及编码器/码本/低秩修复器真实梯度。P0 不测加速。
- P1 正式采集：同一原始 Draft 轨迹，为每个 corrected prefix 固定 Target 有效状态和独立 greedy 未来；L=8，每个原始 prompt 最多 32 新 token。三 split 使用同一生成长度。先做小型 train smoke，仅测试管线正确性。正式 train/val/test 的原始 prompt 数分别为 1024/64/60，实际可用非终止 probe 窗口数由数据决定，不预设显著性。
- A raw；B edge-only；C 正常协议（至少显式纠正 token、接受数、拒 token）；D 在 C 信息上额外使用有效 Target hidden 所编码的 8 个原始字节；E 对同一 D 权重仅打乱消息。B/C/D 边端参数和码本查找次数匹配；D 额外云端参数单独统计。两本地控制码在所有三臂均参与修复，D 的云码只调制第二本地码，避免因多一个独立控制向量获得假收益。
- 主指标是同一 corrected prefix 上 Draft 真正逐 token 自回归生成的最多 8 token 连续 greedy 命中 `A_probe`。主集为未来至少 8 个 token 的非终止窗口；所有窗口（含 EOS）为敏感性分析。先在每个 prompt 内聚合，再做 prompt 分组配对 bootstrap 95% CI。主要差值是 D−C，同时报告 D−B、D−A、D−E；不少于 3 个训练 seed，训练曝光和优化步数一致。任何 CE/训练 loss 仅为辅助指标。
- 首个正式 B/C/D 候选按 `program.md` §5 P1-A 使用 CE-only，`prefix_weight=0`；后续改变训练目标必须作为单变量独立候选并完整重做对应三臂。
- P1 主门槛：D−C 的配对均值为正，跨 seed 方向基本一致，且 D 与 E 拉开；否则不宣称有效，更不进入在线加速或 WAN 结论。P1 尚未运行，所有相关数值待实测。

## 执行与限制

真实 4090 作业只在确认目标卡无其他计算进程、空闲显存至少 22000 MiB、连续三次利用率低于 10% 后启动。中断按同一源码/输入哈希与原子检查点恢复；失败保留原始证据，不删请求或放宽 gate。P1 训练步数与显存预算在真实 P0 通过后的硬件标定中一次性写入独立 run protocol，读取 val 结果前锁定。若修评测 bug，升级协议版本并重做所有受影响对照。

所有当前后端逐 token 重算完整前缀。由它们得出的接受长度只能称 fixed-prefix 诊断；没有 packed 验证、WAN 和净时延证据。
