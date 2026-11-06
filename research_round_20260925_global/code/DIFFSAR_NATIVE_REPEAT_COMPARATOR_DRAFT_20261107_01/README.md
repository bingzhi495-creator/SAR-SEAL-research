# 同一C4恢复两次：只刻画S/R数值分歧的CPU比较器

2026-11-07。**尚未绑定或运行**；仅完成标准库元数据/结果分类行为测试和AST检查，没有SSH、科学库导入或真实checkpoint读取。Root统一实际执行。

旧C0→8与S4→8比较已真实失败，本文件不修改旧源码、旧1e-4判据、旧失败或原正式50轮的条件资格。新预锁只增加一次R：从同一真实C4恢复4→8，共4次新增更新，非独立训练种子。这个CPU任务只比较已结束S8/R8，回答两个相同恢复起点是否也会超出原阈值；单对结果不能估计总体方差或唯一识别原因。

## 不变的数学比较和严格条件

- `reference_comparator`明确固定为原16721字节、SHA256 `68a1c3e1c9ab097da0440aa85461eddf458d8141bf4f010e42097350545bf18a` 的旧CPU源码。新代码在runtime只读加载它，直接调用原`compare_tree`、`scalar_difference`，不重写或改动原数值差算法。S为reference，度量仍为`maxabs(S-R)/(1+maxabs(S))`，损失为对应标量形式，阈值仍1e-4。
- 先CPU加载实际C4，以同一固定训练worker的`component_digests`重算六组件，要求等于S与R各自实际恢复起点摘要；两报告的resume_checkpoint_ref也必须精确关联这个C4。释放C4后再持有S8/R8。
- 严格核对全局5..8的逻辑图像身份、实际增强后的图像/目标联合摘要、shape/实例数、LR和完成计数，以及结束RNG、MapDataset/sampler与native scheduler。来源、结构、dtype、shape、None、静态标量及有限性不享受数值宽容。
- 旧算法对非浮点tensor和optimizer语义step计数设置`exact_required=True`。新`classify_numeric_results`独立断言这些行必须passed，整数不一致真实退出1；不能把旧汇总passed=False的所有情况都当作合法方差。
- 只有其余浮点tensor/loss超1e-4是合法观察。报告所有逐tensor最大差、归一差、不同元素数及逐步loss差，`within_prior_threshold`可True或False；两者都可完成本次characterization并退出0。阈值不是执行成功门槛、机器舍入理论界或全程bitexact界。

S/R也超界，只说明同样的数值超差可发生于两个相同的fresh-restored起点，削弱“只有连续/冷启动差异才导致超差”的解释；不唯一证明atomicAdd、动态matching或其他机制。S/R不超界，也不能唯一断言有遗漏状态。任何结果都不升级旧C/S比较、不授予完整连续资格、不证明模型效用或TGRS就绪。

## Root接口

入口`worker.py --spec <固定Rootspec> --spec-sha <实际SHA256>`。顶层仅标准库；Torch/NumPy、旧比较器及训练helper只在CPU科学生命周期内加载。仅允许Root已核验且SHA固定的私有checkpoint使用`torch.load(map_location='cpu',weights_only=False)`。

常量：SCHEMA=`Root-DiffSAR-native-repeat-compare-spec-v1`；LEASE_SCHEMA=`Root-DiffSAR-native-repeat-compare-lease-v1`；OPERATION=`CPU_DIFFSAR_NATIVE_S8_R8_VARIANCE_CHARACTERIZATION_V01`；run_id前缀=`DiffSAR-SSDD-repeat-compare-v01-`。

spec精确键：`schema,operation,run_id,source_refs,library_sources,checkpoint_refs,training_report_refs,root_lease_ref,output_directory,prelock`。

`source_refs={worker,training_worker,reference_comparator}`，`library_sources={torch,numpy}`，`checkpoint_refs={C4,S8,R8}`，`training_report_refs={S,R}`；每项ref为`{path,bytes,sha256}`。`spec_refs`枚举全部11项被读文件供parent前后绑定。lease精确关联上述source/input roles和operation/run/output/prelock，及worker_ref、boot、single/exclusive/reaped通用flags，还要求`prior_failed_comparison_preserved=True`、`full_continuity_qualified=False`。

CPU0/1，线程2，GPU隐藏，PYTHONHASHSEED40244023，组RSS16GiB、可用RAM下限16GiB/启动48GiB、deadline1800秒。父监督负责整组资源和出生回收；不修改输入、不写checkpoint、不AP，CPU本身optimizer_steps=0。该任务没有额外训练信用。

## 报告判读

schema=`DiffSAR-native-repeat-CPU-characterization-report-v1`。完成时status=`CPU_REPEAT_CHARACTERIZATION_COMPLETE_PENDING_ROOT`、`characterization_complete=True`、`within_prior_threshold`为实测bool、`selected_sources_and_inputs_unchanged_after=True`、`CUDA_initialized=False`、`worker_exit_code=0`。

数值详情放在`characterization`，含`actual_C4_restored_start_exact`、`restored_start_component_digests`、`exact_trace_verified`、`strict_integer_state_match`、`mandatory_state_trace_RNG_finite_checks_passed`、`exact_endpoint_components`、`components`和`losses`。没有名为comparison_passed的连续资格标志。top和nested均保留`prior_failed_comparison_upgraded=False`、`full_continuity_qualified=False`；qualified始终False，Root_external_admission_required=True。Root仍需核验真实原工具/SSH/worker终态、选定来源和出生组/资源，不能用后核0改写原失败。

本地TDD记录：不同C4初值红9df9ee/1→绿027638/0；实际增强摘要改变红b3b105/1→绿978f68/0；浮点超界合法分类红f3066d/1→绿d51a02/0；语义整数超界必须拒绝红31f92d/1→四项绿b17c58/0。AST 8eabe2/0。以上只有控制逻辑证据，尚无实际S/R数值比较结果。
