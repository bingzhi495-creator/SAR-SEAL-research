# 原生DiffSAR分段恢复：独立CPU比较器草案

2026-11-07。**尚未绑定或运行科学任务。** 已完成三项标准库元数据拒绝行为的red→green，以及源码AST检查；这些不等于Torch张量比较、真实checkpoint读取或实际续训验证。Root是唯一科学执行所有者。

本比较器验证预锁的C从0连续至8、在4写真实状态但不中断；S从这个同一C4恢复至8。两次训练共12次真实优化器更新。本CPU任务自身不训练、不写checkpoint、不运行模型、不计算AP，只读取Root已核验并固定字节身份的私有C4/C8/S8与两份原训练报告。

## 实际比较范围

1. 先独立CPU加载C4，调用**同一固定训练worker源码**的`component_digests`重算模型、grad、optimizer、scheduler、RNG和loader六个摘要，要求等于C报告中C4摘要和S实际恢复起点摘要；不另写摘要算法。C4释放后才同时加载C8/S8。
2. 严格核对C第5–8次更新与S全部四次更新的实际逻辑索引、image_id、映射shape/实例数、包含图像及mask/box/class的类型化联合字节摘要、LR与完成计数。相同shape不能代替相同增强；权重文件本身不能代替完整训练状态。
3. 对实际C8/S8重新计算摘要、与报告中的各自checkpoint摘要关联；结束RNG、MapDataset/sampler和native scheduler必须exact。
4. 最终model、gradients、optimizer逐张量核对shape/dtype；浮点张量用预锁 `max_abs(C−S)/(1+max_abs(C)) ≤ 1e-4`，报告每张量实测最大差、不同元素数和是否超界。optimizer的step即使保存为浮点tensor也按计数精确比较；非浮点张量、None/结构以及Python静态标量精确比较。model浮点buffer不未经核查全部视为常数，仍在预声明浮点范围内。原24项损失及总损失按相同归一sanity阈值逐步报告。

1e-4只属于预声明的非确定CUDA后续轨迹sanity阈值，不是机器舍入理论界，也不能称全程逐位复现。恢复起点、数据、RNG、整数计数不使用这个宽容。超界保持失败，不事后放宽；没有AP、收敛、完整作者复现或TGRS资格。

## Root接线接口

入口：`worker.py --spec <Root固定spec> --spec-sha <实际SHA256>`。顶层仅标准库；Torch/NumPy和固定训练helper仅在main的CPU科学生命周期内加载。私有Root产出的已固定SHA checkpoint才允许`torch.load(map_location='cpu', weights_only=False)`；不接受任意外部pickle文件。

常量：

- `SCHEMA = Root-DiffSAR-native-resume-compare-spec-v1`
- `LEASE_SCHEMA = Root-DiffSAR-native-resume-compare-lease-v1`
- `OPERATION = CPU_DIFFSAR_NATIVE_RESUME_C8_S8_COMPARISON_V01`
- `LIBRARY_MODULES = {'torch':'torch', 'numpy':'numpy'}`；`LIBRARY_NAMES`同两键。
- run_id前缀`DiffSAR-SSDD-resume-compare-v01-`。

spec精确键为`schema, operation, run_id, source_refs, library_sources, checkpoint_refs, training_report_refs, root_lease_ref, output_directory, prelock`。

`source_refs={worker,training_worker}`，固定后者与C/S使用的同一科学源码；`library_sources={torch,numpy}`；`checkpoint_refs={C4,C8,S8}`；`training_report_refs={C,S}`。所有ref严格为`{path,bytes,sha256}`。`spec_refs`枚举全部10项被读输入，Root parent须全部绑定、前后核验。`validate_spec`和`validate_lease`为标准库可调用接口，lease连接source_refs、两类input refs、operation/run_id/output/prelock，及single/exclusive/reaped/boot等Root资格。

CPU资源：CPU0/1，Torch/OMP/MKL/OpenBLAS线程2，GPU隐藏，PYTHONHASHSEED40244023，组RSS16GiB，可用RAM下限16GiB、启动48GiB，deadline1800秒。共享全局科学叶子独占，主训练先在自然结束处停下。父监督负责整组RSS、出生身份、回收及硬超时，worker检查自身RSS/可用内存/时间。比较分块计算浮点差，不拼接或克隆整套state；C4先释放，再保持两份最终CPU state，实际峰值由原任务报告，不能由估算认证。

## 报告与成功范围

REPORT schema为`DiffSAR-native-resume-CPU-comparison-report-v1`。实际成功时status为`CPU_RESUME_COMPARISON_COMPLETE_PENDING_ROOT`，`comparison_passed=True`、`selected_sources_and_inputs_unchanged_after=True`、`CUDA_initialized=False`、`worker_exit_code=0`。`qualified`始终False，`Root_external_admission_required=True`；Root仍须独立核验原工具/SSH/worker真实终态、来源及出生组/资源，不能用后核0升级原失败。

`comparison`包含：实际C4起点摘要核验、exact trace标志、结束rng/loader/scheduler exact摘要、model/gradients/optimizer的逐张量差异和聚合最大差/不同元素数、逐步各项loss差异、预锁阈值与sanity结论。`actual_library_refs`保留真实导入来源；资源字段为`peak_worker_RSS_bytes`、`minimum_available_RAM_bytes`、`elapsed_seconds`。该CPU任务`new_training=False`、`optimizer_steps=0`、`formal_training_credit=0`、`AP=False`、`checkpoint_written=False`。

退出1保留结构/来源/计数、数值超界或资源错误，不自动重试或改阈值。只有Root核验实际原任务后才可声称这次有限恢复比较完成；不能由本草案、元数据单测或恢复检查外推长期训练的完整可重复性。

## 本地标准库检查记录

- 缺失实际exact恢复起点必须拒绝：初红`1a8a88/1`，绿`19e805/0`。
- 真实输入联合摘要改变必须拒绝：初红`4ada3b/1`，绿`b7f6c3/0`。
- 未执行的续段更新不能通过：初红`a62a59/1`，最终三项绿`6501a4/0`、完成runtime后三项绿`0b9556/0`。
- AST检查`21462f/0`；没有实际Torch比较结果。一次空hunk补丁调用被工具拒绝，未写任何文件，不属科学执行。
