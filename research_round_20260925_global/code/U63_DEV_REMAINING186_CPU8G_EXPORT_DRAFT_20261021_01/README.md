# U63 剩余186图 CPU8G 导出 — UNBOUND / UNEXECUTED

本目录是独立新 profile，只新增 entry、3个有限标准库测试及本说明。U62 原件和 U59/DEV reader/RGB/geometry 均不修改。未调用 main、实际绑定、导出或科学库；没有读取真实 context、数据、Ops、权重或资格链，也不启动任何科学进程。

相对于 U62，**唯一运行行为变化**是 worker 的采样/协作式 RSS上限从16GiB改为8GiB（8589934592字节）。CPU0/1、两线程、可用 RAM下限16GiB、CUDA隐藏、D输出余量20GiB及1800秒完全保持。来源、FP32字节、尺寸、query顺序、原native字典及释放逻辑没有改变；不缩图、不跳图、不改匹配范围、不放宽门槛或自动补跑。

## 精确接口变化

调用仍为 `entry.py --spec ABSOLUTE_PRIVATE_SPEC.json`，精确14键及来源结构与 U62 相同：

`schema/operation/phase/expected_versions/limits/dataset_indices/source_roles/library_sources/context_ref/original_image_refs/input_sources/sources/output_directory/minimum_output_free_bytes`。

变动的标识和限制：

- spec schema：`U63-DEV-remaining186-full100-CPU8G-export-spec-v1`。
- operation：`CPU_U63_DEV_REMAINING186_FULL100_CPU8G_EXPORT`。
- report schema：`U63-DEV-remaining186-full100-CPU8G-export-report-v1`。
- `limits.maximum_RSS_bytes=8589934592`；原 `minimum_available_RAM_bytes=17179869184` 不变。

`minimum_output_free_bytes=21474836480` 固定值及其他13键的语义不变。版本仍 Torch2.0.1+cu118/NumPy1.23.5；五 source_roles、十库、U59/原reader/geometry pins 不变。报告中的 `limits` 反映8GiB；峰值仍是 `resource_observations.peak_RSS_bytes` 与 `cost.peak_RSS_bytes` 的采样值，`RSS_peak_is_sampled=True`。其他实际计数、export_entries、逐图summary、D盘观测和边界字段沿用 U62，不增加兼容旧schema的回退。

Root 新控制器的预准入选择为两次新鲜观测之间至少30秒、距绑定至多300秒，可用RAM至少25GiB，即16GiB原下限+8GiB worker预算+1GiB父进程开销预留。这是此次独立 CPU profile 的预算选择，位于 Root 控制器，不向 worker spec 增键，不改原 owner 的13字段 admission，也不改变原 GPU训练32GiB准入。

## 不能由此推出的结论

8GiB是采样/协作门槛，**不是 OS硬RSS cap**。原 Windows父driver没有新增针对RSS的硬终止，原16GiB free-RAM监督仍保留。短暂峰值可能未被采样观察，父进程、系统和其他应用也消耗内存；25GiB预准入不证明系统峰值安全。

已有单图观测和原metadata大小仅支持选择一次受控试验预算，不能证明全186图必然在8GiB或1800秒内完成。超限、失败或不支持的原ABI均应如实失败并保留记录；不得自动切回16GiB、跳过图像或据资源失败判断模型无效。

## 不变的科学及数据边界

固定原索引1..186，每图全部100候选；不重复 ordinal0。完整187图/18700候选只校验既有 metadata；先从context核验精确 selected-input union，才哈希所选 Sources，不递归读取teacher/annotation或额外 row0图像/张量。每图按原ordinal重建五ref，原 reader load → U59导出 → 原字段字节/独立紧凑storage核验 → 真实weights-only重载 → 清空别名/GC/弱引用释放，再进入下一图。

e/M/L仍自然query轴，query_ids仍原导出置换；RGB400、e、M、L的原FP32 shape和原native score/category/bbox/area/order/空mask/RLE均不改。仅保存 U59四字段和最小白名单，未知字段照常拒绝。历史 bundle 读取可能接触GT关联容器，不能声称上游从未接触GT。无新GT栅格化、教师/模型、训练、renderer校准、推理或AP。

输出仍为真实 `DEV_FULL100_1.pt`…`DEV_FULL100_186.pt` 和 REPORT，失败保留partial与小失败审计；whole_population_exported始终False。Root以后只能把实际接收的 U61 ordinal0真包与本profile真实186包汇总，不能把本次计划或局部成功写成187整体资格。

## 有限验证

`python -B test_metadata.py` 只验证新标识/14键/固定限额、8GiB采样边界以及free16/D20保留，并比较普通 U62/U63全部worker函数AST：仅将 U63→U62、CPU8G→CPU 的profile标签还原后，函数AST一致。数值导出、读写、释放和finally逻辑没有改写；这项静态一致性不等于运行轨迹或内存上界证明。没有重复执行旧整套元数据测试，没有导入科学库、读取实际资源/数据或调用 main。

所有真实执行、资源、字节等值、保存重载、终态和Root接收仍 **UNEXECUTED**。
