# U38 全人口 CPU 入口草稿

设计初稿状态为SOURCE_ONLY_NOT_EXECUTED；截至文末2026-10-17实际补记，已完成一个原任务的CPU检查并经Root接收。本目录为普通源码/说明发布副本，不含私有binder、spec、数据或资格链。源码编写阶段没有读取图像、标注、缓存、权重或活动科学链，也没有执行Torch/NumPy；随后真实执行与本次文档补记区分记录。

`run_qualify.py --spec ABSOLUTE_PATH --owner-execute` 只接受一个原任务的全部 FIT741图/1661匹配实例，1800秒硬限、CPU0/1、两线程、CUDA不可见且始终未初始化、RSS≤16GiB、可用RAM≥16GiB。原13字段准入、当前parent计数、全部来源hash和Control闭合先于Torch导入。没有分片开关或自动重试。

spec schema：`ROOT-U38-full-method-CPU-reader-init-spec-v1`；operation：`CPU_U38_FULL_METHOD_READER_AND_COMMON_INITIALIZATION`。prelock schema：`U38-full-method-CPU-reader-init-prelock-v1`。源角色分为11个 `module_sources` 和4个基础源码，共15个严格引用：

- 组件11：`shared_initialization/rgb_matched_reader/full_legacy/full_dsconv/full_head/full_feature/plain_legacy/plain_feature/instance_adapter/matched_reader/image_geometry`，exact keys与qualifier一致。
- 基础4：`entry_source/body_source/runtime_source/qualifier_source`。未来Root必须Source闭合它们及全部实际直接输入、计划快照、context/prelock/parent。

`context` 是Root提供的严格JSON引用，schema为 `ROOT-U38-full-method-CPU-context-v1`，保留旧reader的 `FIT_context/N13_qualified/N13_report` 三项和它们的完整依赖。`image_references` 是Root预先登记的全部741个原JPEG严格引用，按原图ordinal排列；不得猜路径或遇缺失改控制器。真实引用清单应与qualifier返回的 `image_references_metadata_sha256` 一致。未来binder负责提前枚举/准入原缓存、注释、JPEG与context；本入口不会自行扩大Source或生成资格。

预锁固定images741/GT1661/dataset_indices0..740、branch400/canvas800、common_state104（68参数张量+36BNbuffer）、seed202610173101、native_L基底、全FIT M100/S0非必需、无NN forward/optimizer/train/AP/DEV/TEST、无自动重试/分片，并引用同一个plan快照。U34真实CPU构造/原位复制和所有未映射状态检查由原共享初始化实现；U37逐图来源、原matcher/GT/自然L、JPEG/RGB/整数几何检查由原reader执行。两者只组合调用，不另造解释相近的缓存解析或GT。

工作体 `run(torch,np,spec,c,pre)` source-closed加载依赖，临时准确提供原 `dsconv/head` imports，try/finally恢复所有模块别名、CPU RNG和CUDA lazy-init入口。CUDA初始化尝试会被阻断并记录。将空的可变qualification dict交给原qualifier，发生失败时保存其真实partial图数/GT数及原traceback，不能将部分进度当完整通过。

输出 `REPORT.json`（`U38-full-method-CPU-reader-init-report-pending-v1`）与 `probe.log`；成功入口另写 `Q.json`（`U38-owned-pending-full-method-CPU-reader-init-v1`）。report含完整qualifier结果、104state回执、实际源码导入、源/RNG恢复、CUDA冷检查和CPU资源。若原任务超时/资源检查失败，可保存失败report但不输出成功Q；外层进程被硬终止时能否写最终report不作保证，原日志与终态必须保留。

成功return字段为 `report_ref/log_ref/images/GT/CPU_reader_verified/common_initialization_verified/CUDA_initialized/qualified/new_training/AP/DEV/TEST/GPU_resource_gate_passed/full_method_training_eligible`。worker返回的GPU/训练资格及qualified始终False；CPU reader检查完成本身不等于原任务退出0或Root独立接收。实际启动另需binder、冻结发布、完整准入和Root单次执行；本轮已按这些条件完成，见补记，不允许仅用本目录自授资格。

静态依据为普通U30 CPU entry的Control/13ad模式、普通U38 qualifier、已发布U34共同初始化和U37 RGB组合。这里无400 GPU前向、反向、optimizer、稳态、DEV100或AP效用检验；CPU全量通过不解决这些后续问题。

## 2026-10-17 实际补记

原任务实际退出0、fresh FIRST通过并经Root接收：741图/1661匹配实例与U34公共104状态（68参数张量＋36BN buffer、seed202610173101）/实例适配器检查完成，CUDA始终未初始化，CPU0/1两线程。原任务289.953秒；worker峰值RSS824,074,240字节、最小可用RAM32,907,644,928字节，原任务终态另核最小可用RAM32,904,998,912字节，分别保留统计范围。没有NN forward、反向、优化器、旧权重读取、训练或AP；Root没有像素/NN重演。三图GPU序列仍只是计划，不授正式训练或GPU稳态资格。见[简明实际结果](../../U38_FULL_FIT_RGB_AND_COMMON_INITIALIZATION_ACTUAL_RESULT_20261017.md)。本补记不改任何.py或冻结执行原件。
