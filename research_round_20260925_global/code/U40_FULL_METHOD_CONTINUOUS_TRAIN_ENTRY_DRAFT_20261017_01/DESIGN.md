# U40 连续正式训练入口/工作体：普通未绑定草稿

状态：**SOURCE_ONLY_NOT_EXECUTED**。未导入或运行Torch/NumPy、构造模型、读取实际缓存/图像/GT/checkpoint/资格链，未读运行中的N28 SOURCE/H/RAW/Q/Ops。这里只写入口与工作体；没有binder、实际准入、训练启动或Root资格，不另加数值资格叶。

`run_train.py --spec ABSOLUTE_PATH --owner-execute` 对应spec `ROOT-U40-continuous-training-segment-spec-v1`，operation `GPU_U40_FULL_METHOD_CONTINUOUS_TRAIN_SEGMENT`。13项 `module_sources` 为原U39的12组件加 `u39_training_ops`；另 `entry_source/body_source/runtime_source/core_source`，共17项执行源。core固定为Root最新31062B、SHA `8f9b549a8383877aa6f0ae3b1d2a9786c44c0f486e45b447b06dea2e25f5a0c5`。原U39 `_update` 仍为唯一数值更新路径；本工作体只调用新core.train_segment，不复制训练循环。

完整FIT741/1661、首seed202610173101、PLAIN/FULL各50轮、每臂37,050次图更新。RGB400/FP32/B1/query1/save_on_cpu、native L与原完整G匹配监督、原图BCE+Dice及AdamW配方全部由core.RECIPE锁定；无VJP/AMP/重算checkpoint/scheduler/中途AP选型。正式段不是pilot，不能用失败后重跑或缩段隐去已经发生的更新。

## 入口与核心目录分工

外层 `spec.output_directory` 为Root预准入的新独占输出，工作体创建它以保存 `REPORT.json/probe.log`。`spec.segment_output_directory` 必须精确等于其 `SEGMENT` 子目录，工作体不先创建该子目录，由core自行独占创建。否则core会因输出已存在而拒绝。

core生成/复用 `ORDER.json`，实际写两份 `.trace.jsonl`、两份完整CPU `.pth`，双方完成后才写 `PAIR_COMMIT.json`。权重和完整训练状态属于私有科学输出，不发布GitHub。工作体不额外torch.load或重演checkpoint，只验证core返回的保存/roundtrip标志和元数据关系。原始失败、单边checkpoint、partial与实际更新计数全部保留；FAILED必须抛出，入口不写成功Q。

## Root必须事前锁定的字段

1. **一次冻结并全程复用的来源**：17项执行源、context与原数据身份。core identity包含core_source及13组件的完整path/bytes/sha，续段不能每次复制成新H路径再误称同一方法。body/entry/runtime也应复用原冻结版。修改核心需另立明确版本和恢复决定。
2. **固定完整图序**：CPython **3.8.17**；Root已用纯metadata生成的50轮完整order SHA为 `2afadef62eac9f797c125fa105a3a28f94cdbdd9db4b5b5a2b018d4f0d5f194e`。前后图序相同，每轮741图完整排列；续段只消费原ORDER与全局cursor，不重新shuffle。
3. **每段边界与硬预算**：`spec.segment`按core原字段显式给seed/start/stop/order_sha256/recipe/limits/core_source/order_ref/parent_pair_ref。本轮明确锁spec/pre.segment_length=256、145段、末段186，首段[0,256)；入口核start为256整数倍、stop=min(start+256,37050)，并令pre.segment==spec.segment。不能凭DEV/AP、loss或中途失败自动重定边界。每臂总37,050不变。
4. **真实父端点**：`parent_checkpoint_refs`精确PLAIN/FULL两键。初段两项None、order_ref/parent_pair_ref/resume_qualified均None。续段全部为冻结引用，`resume_qualified`须是Root已接收的配对端点，而不是worker pending文件。入口约定Root schema `ROOT-U40-continuous-training-segment-qualified-v1`、qualified/paired_prefix_advanced为True，result中pair_commit_ref/checkpoint_refs/order_ref/paired_checkpoint_prefix必须与本段start及对应引用一致；此新Root接收契约需后续私有助手实现。
5. **Source实际成员**：完整metadata741/1661与全部741 JPEG预登记；张量只登记本段固定order[start:stop]所需图的3类原carrier refs。`admitted_tensor_indices`为该span唯一ordinal排序表，工作体用core纯metadata函数/原ORDER核对；不要求其余缓存在本叶c.closed，不增补全FIT M100/S0。续段另准入原ORDER、PAIR_COMMIT、两checkpoint、两trace及Root接收JSON。原context保持U38的三refs。
6. **资源与存储**：每叶1800s、CPU0/1两线程、allocator4GiB/wholeGPU8GiB、RSS16GiB/freeRAM16GiB；原owned supervisor实际监督。`disk_budget`与`minimum_output_free_bytes`由Root保留规划明确给出、与pre一致，required_free_bytes必须相等；入口及调用core前分别核真实freeD，不把文件估计写成实测。初段规划为2GiB/paired checkpoint×145+32GiB=322GiB；后段32GiB+剩余段数×首个已接收paired checkpoint实际bytes，measurement引用与计算字段由binder闭合。只更新资源保留量，不改训练预算。冻结原件不由wrapper删除。

`U39_qualified`是实际短资源试点接收引用；其通过本身不自动授权主训练。Root另以spec.formal_main_training_authorized/pre.formal_training_authorized明确锁定已授权的正式计划。`parent_ROOT_qualified`为最新原执行账户父链，与续训端点resume_qualified分开，不因别的任务插入而误换模型父状态。

## prelock、结果与失败边界

pre schema为 `U40-continuous-training-segment-prelock-v1`：segment、segment_length、disk_budget、parent_checkpoint_refs、resume_qualified、admitted_tensor_indices、plan、minimum_output_free_bytes与spec逐项相等；formal_training_authorized=True、pilot_only=False、epochs50/images741/GT1661/total_updates_per_arm37050、no_AP/no_DEV/no_TEST=True、automatic_retry=False。

工作体按原dsconv/head依赖加载13组件与core，保持CUDA冷；cuDNNdeterministic=True/benchmark=False、TF32关闭。core拥有首次CUDA初始化和4GiB setlimit、fresh CPU skeleton/严格父状态恢复、所有训练/序列化操作。wrapper finally恢复外部Python/NumPy/Torch CPU RNG与模块别名；checkpoint里的各臂CUDA RNG保持core自己的保存/恢复合同，不能把外部CPU恢复写成整条GPU轨迹bitexact。

wrapper `REPORT.json` schema为 `U40-continuous-training-wrapper-report-pending-v1`，嵌 `core_report` 原 `U40-continuous-segment-report-v1`。完整原trace只读一次汇总实际行数、数据成员、GT visits、全部step资源峰和CPU保存事件，另保留最后实际allocator区间；不以最后小图峰代替全段峰。边界采样RSS与累计事件bytes不证明无慢泄漏或提速。正式attempted与main_started按Root最新core字段分别保留，0次optimizer进入的失败不能冒称正式更新已开始。

成功入口才写 `Q.json`（`U40-owned-pending-continuous-training-segment-v1`）。两个checkpoint/trace必须存在、相同stop、实际输入链一致，原PAIR_COMMIT仅pending。core/worker/Q的paired_prefix_advanced与qualified保持False；Root实际原terminal0、held、fresh FIRST和接收后才推进共同前缀。不能将一臂成功、函数正常返回FAILED或checkpoint错误当作成功。

成功result精确19项：report_ref/log_ref/seed/span/updates_per_arm/successful_optimizer_updates/pair_commit_ref/order_ref/checkpoint_refs/trace_refs/paired_checkpoint_prefix/paired_prefix_advanced/new_training/formal_main_training_started/pilot_only/qualified/AP/DEV/TEST。报告/返回的cursor是待接收建议，不是Root已经更新；所有AP/DEV/TEST与创新结论均False。目标是把50轮连续跑完，当前仍无实际正式段或恢复数值证明；首段与第二段分别在既定正式预算内检验真实保存/恢复，不另做资格叶。
