# P1-B 目标隐藏态消融候选锁定

本文件在任何 P1-B GPU 验证集评测之前登记。原 `program.md`、P0/P1-A 数据、评测器和 Target greedy oracle 均保持冻结；这是 **诊断性消融**，不是改写 H1 成功标准，也不是第二轮更换训练目标。

## 单一干预

原 D 在每个窗口用有效 Target 验证状态 `h_T` 生成 8 字节。新对照保留完全相同的 D 编码器、码本、边端修复器、参数量、8 字节路径和 CE 损失，只在训练内存中将每条 `h_T` 替换为同形状、同 dtype 的全 1 向量。优化完成后，按恒等变换把云端线性投影的 `W·1+b` 折入 bias，并把 `W` 设为零。保存的 D 结构因此对任意 `h_T` 输出相同消息，推理仍走原有 8 字节与低秩修复路径。原始语料文件不修改；训练监督和正确前缀仍与 P1-A 相同。

若折叠前后恒定输入的码不同、任意两个不同 Target 状态产生不同码、云端投影/编码器或边端模块在训练时缺少非零有限梯度、消息不是 8 字节，GPU P0 必须失败，不能启动正式训练。

## 固定预算与数据

- 训练语料 1024 prompt、9366 固定窗口，SHA256 `dcdcde719da0dcd2071236f5eae50263dbdba83a330c9ac17535789a45e074e9`；验证 64 prompt、567 窗口，其中 427 个完整非终止窗口为主集，SHA256 `96e15f8b28cd873a71c7714ecd3769497271422b933cfbbcab25e6f4f36a09fd`。正式 test 封存。
- 复用 P1-A 的 seed `42/43/44`、每 seed `9366` 步、无放回单轮、LR `1e-4`、AdamW weight decay `0.01`、`prefix_weight=0`、Gumbel 温度 `1.0`、checkpoint 每 `256` 步。逐步 prompt/round 曝光哈希及初始边端参数哈希须逐 seed 与原 D 完全相同。
- P1-A 正式预算锁 SHA256 `9f7a046de99c14b950dc1df128d40ff1a85f3f68f1aad75ae8c60368179b91b2`；训练协议 SHA256 `5a0a4a699208094ac33e9b4731e817e005166e334a52b503f904dcdccfd3a5243`。原始 P1-A 结果与 checkpoint 不覆盖。
- Target 官方 Qwen3-8B BF16 revision `b968826d9c46dd6066d109eabc6255188de91218`；Draft 官方 Qwen3-0.6B BF16 权重 SHA256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`。Target 的逐文件 SHA 表由上述冻结训练协议的 `target_model_file_sha256_from_collection` 字段固定；不得替换模型或精度。
- 冻结评测协议 `fixed_prefix_v1`，原始基线代码提交 `8a52120b8bcdabec3a0678f702566c1fd2d00060`。本机 Windows checkout 的换行字节与远端 Linux 文件不同；部署时分别记录实际远端逐文件 SHA，不用本机 SHA 冒充远端 SHA。

## 预先指定的比较与决策

先在真实 4090 上运行新的 P1-B gate：两步恒定输入训练、所需梯度、折叠不变性、两个冻结 train 窗口各一轮提交 token 与独立 Target AR 一致、模型与输入 SHA。该小 gate 不保证覆盖 EOS 或截断情形；沿用冻结 P0 的 EOS/截断正确性证据，正式 val 再保留逐窗口与 EOS 敏感性检查。gate 通过后才训练三个 no-h_T D。验证只用冻结 val，真正自回归生成 L=8 的草稿，逐 prompt 聚合并作 2000 次配对 bootstrap 95%CI，随机种子 `20261031`。主比较为 **原 D − no-h_T D**（同架构下 Target 隐藏态的增量），同时给出 **no-h_T D − C**、原始 D−C、D−全局/分层打乱 E。所有窗口含 EOS 的敏感性分析同时保留。

若原 D 与 no-h_T D 接近，指向 Target 隐藏态没有被有效利用；若原 D 更差，说明该特征路径在本配置下可能有害；若原 D 优于 no-h_T D 但仍未优于 C/E，也只能说明存在不足以支撑 H1 的局部作用。**H1 原门槛保持 D−C 为正、跨 seed 同向且 D 与 E 拉开；任何 no-h_T 结果都不得替代此门槛。** 不解封 test，不推断在线验证轮数、WAN 或净加速。

## 本地实现与版本

新增 `scripts/no_hidden_control.py`、`scripts/check_no_hidden_gpu_p0.py`、`scripts/train_no_hidden_p1.py`、`scripts/evaluate_no_hidden_p1.py` 与 `tests/test_no_hidden_control.py`。没有修改 `feedback_codec/`、`scripts/evaluate_fixed_probe.py`、数据清洗、Target oracle 或指标聚合。新增脚本在训练/评测协议中记录自身、冻结源码、输入和模型的实际 SHA；每个 GPU 结果单独落盘，错误保留原件。新 gate 的完整报告和 SHA 必须写入正式训练协议，正式验证须再次核验它。

本候选完成后按 `program.md` §5 的 P1-B 顺序继续判断是否需要 F 高带宽上限诊断；不凭训练 CE、码本熵或本消融单独宣称 H1 成立。

## 本地源版本与离线检查

以下 SHA256 是本机新增文件在 GPU 运行前的字节哈希，部署后还须记录远端实际哈希并由 gate 核验：

| 文件 | 本机 SHA256 |
|---|---|
| `scripts/no_hidden_control.py` | `2d08e25ad55d19e470b4e942e9f4097d8ee27e059d6a840e0756c55e5fbf235e` |
| `scripts/check_no_hidden_gpu_p0.py` | `61995fb3d2c7a235bd6d90a7d6623fa4f82cdd51bc19e0835b8e783eed7aae3a` |
| `scripts/train_no_hidden_p1.py` | `bd73b99a18cacdc1fac66d71bc0fa10a01484f5d99d5c97bdf0df5ed0bd362fa` |
| `scripts/evaluate_no_hidden_p1.py` | `2fd524b35bfd2f204156a8f18908b81b7a08f34ddcda5c7227baa61fd5018044` |
| `tests/test_no_hidden_control.py` | `8f278d4f775c5a152fd67b50914a1dc222ae1d42d97af2ff681606295e5fdd81` |

本地完整测试在允许 loopback socket 的环境中通过 `81/81`，命令 `python -m pytest -q --basetemp=.pytest_tmp_noht_network`。这些测试不是实卡 gate。正式启动前还需三次 GPU 空闲采样；如果 gate 失败，归档失败输出和旧源码，再修复并重跑新 gate。评测部分输出重试前同样归档，不能覆盖原始失败记录。
