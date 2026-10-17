# U39 跨图资源探针入口与工作体

状态：**SOURCE_ONLY_NOT_EXECUTED**。只准备普通未绑定源码，未执行Torch/NumPy、模型、数据读取或助手；没有binder、spec、准入或Root资格。运行中的N28 SOURCE/RAW/Q/H/SCI/Ops未读取。唯一数值owner为Root。

入口 `run_probe.py --spec ABSOLUTE_PATH --owner-execute` 沿U38原13字段准入/Control模式；spec为 `ROOT-U39-full-method-cross-image-resource-spec-v1`，operation为 `GPU_U39_FULL_METHOD_CROSS_IMAGE_RESOURCE_PROBE`。`module_sources`精确12角色，即原U38的11组件加 `u36_helpers`；另 `entry_source/body_source/runtime_source/probe_source`，共16可执行源码引用。`probe_source`固定为已审阅的resource_probe.py SHA `378c955877674a65f6b2180d4c269e01355245881229ed0d2ec555b184ee7e5d`，训练loop不复制到新工作体。

固定方案为 `[54,710,54]`，PLAIN/FULL各连续3次、共6次丢弃更新，seed202610173101公共104states及适配器同源；RGB400、FP32、query_chunk1、save_on_cpu(pin_memory=True)、AdamW3e-4/wd1e-4/betas0.9,0.999/eps1e-8/foreachFalse/fusedFalse。不用VJP、AMP、checkpoint或自动retry，正式训练预算记0；不计算AP/DEV/TEST。prelock沿原probe的 `U39-full-method-cross-image-resource-prelock-v1`，另引用同一plan快照。

未来Root spec显式提供原 `context` 严格引用，保持 `ROOT-U38-full-method-CPU-context-v1` 及 `FIT_context/N13_qualified/N13_report` 三ref；另 `U38_qualified/U38_report`，与实际U38接收的原report/context精确join，不猜路径。741 JPEG严格引用按原ordinal置于 `image_references` 并全部预准入，与U38已核完整有序refs摘要一致。

**完整metadata为741图/1661实例，实际tensor仅54与710两图。** spec明锁 `full_metadata_population_images=741/full_metadata_population_GT=1661/admitted_tensor_indices=[54,710]/tensor_images=2/image_order=[54,710,54]`。Source只需这两图各carrier/packet/native_L共6个张量引用，保留全部JPEG和原metadata/注释/资格JSON，不要求U38所有旧tensor refs入新Source。原reader.prepare枚举全人口refs但不打开其他739图缓存；probe只按该序列load，不重新授予全741像素资格，也不要求全FIT M100/S0。

所有原13ad、来源和动态parent计数核对在Torch导入前完成。CPU0/1、两线程、RSS≤16GiB、可用RAM≥16GiB、wholeGPU≤8GiB、allocator4GiB、1800秒，无自动retry。cuDNN deterministic=True、benchmark=False且TF32关闭；工作体以冷CUDA进入原probe，实际CUDA首次初始化及4GiB setlimit完全由resource_probe负责，不在入口预热或改配额。

工作体只source-closed加载12组件和probe，提供原dsconv/head模块别名，调用 `probe(torch,np,c,context,modules,source_refs,pre,report={},image_control=None)`。原函数原样执行读图、初始化、渲染/损失、反传、optimizer和清理。成功必须是原probe完整PASS、六个step实际返回、每臂3次及双臂/最终allocated与reserved均0；FAILED返回也必须拒绝，保存其真实partial。try/finally恢复模块别名和外部CPU RNG，report不包含Tensor对象或旧权重。

输出 `REPORT.json`（`U39-full-method-cross-image-resource-wrapper-report-pending-v1`）嵌套 `resource_probe` 原报告（其schema保留 `U39-full-method-cross-image-resource-report-pending-v1`），以及同时tee stdout/stderr的 `probe.log`。成功且最终清理与阶段峰值通过后，入口才写 `Q.json`（`U39-owned-pending-full-method-cross-image-resource-v1`）。峰值汇总遍历全部实际stage，不能用被重置后的最后一次allocator计数替代全程峰值。失败保留JSON/日志并抛错，不写成功Q；硬终止不能保证最终JSON必已写完。

成功result含 `report_ref/log_ref/pilot_optimizer_updates/successful_optimizer_updates/unique_images/new_training/pilot_only/formal_main_training_started/formal_training_budget_credit/AP/DEV/TEST/qualified/full_method_training_eligible`。`new_training=True`只指这6次disposable更新，main=False、budget_credit0、qualified和full_method_training_eligible始终False。Root仍须实际准入并单次启动，实际终态/held/FIRST与元数据接收由后续私有助手负责；短组三图不证明长期无泄漏、50轮可行或学术收益。
