# U62 剩余 DEV186图/full100 CPU 导出 — UNBOUND / UNEXECUTED

本目录只新增 entry、有限标准库测试及本说明。基于已使用的 U61 普通 worker 最小扩展，不改 U61/U59、原 DEV/RGB reader、geometry 或 Windows owner。此草稿未调用 main，未导入 Torch/NumPy/PIL，未读真实 context、图像、tensor、权重、Ops 或资格链，未绑定或执行新任务。

固定导出原始 `dataset_indices=list(range(1,187))`：186图、每图原100候选，共18600候选；不重导已接收的 ordinal0。全187图/18700候选仍检查既有 metadata 的原 ordinal/image/qid 连接，但本任务不能自称实际完成187图，也不写虚假187包索引。Root 后续只能用已接收 U61 真包与本次真正完成并接收的186包汇总。

## Spec / 调用 ABI

`entry.py --spec ABSOLUTE_PRIVATE_SPEC.json`；恰14键，除了以下变化，其余承接 U61 精确接口：

- `schema="U62-DEV-remaining186-full100-CPU-export-spec-v1"`。
- `operation="CPU_U62_DEV_REMAINING186_FULL100_EXPORT"`。
- `dataset_indices` 严格为原序整数列表1..186，不接受0、重编号、重复、缺项、倒序或布尔值。
- 新键 `minimum_output_free_bytes=21474836480`，固定20GiB，不是可调门槛。

全部14键为 `schema/operation/phase/expected_versions/limits/dataset_indices/source_roles/library_sources/context_ref/original_image_refs/input_sources/sources/output_directory/minimum_output_free_bytes`。不加入 admission、父资格、账户或预算字段；这些由 Root 控制器单独维护。

`source_roles` 仍恰 `entry_source/export_source/dev_reader_source/dev_rgb_reader_source/geometry_source`；`export_source` 仍是 U59 package_io.py，20627字节/SHA256 `8fabc800e4d36e2da3988e42acf88058c80790f9c8b3d664a1a4fa9232233473`。三个原 reader/geometry pin 见源码。十个显式库角色和版本 Torch2.0.1+cu118、NumPy1.23.5 保持 U61。只核显式文件和真实模块规范路径，不声称整个框架/OS/binary 闭包。

`context_ref` 仍是原 `ROOT-N28-fixed24-dense-native-DEV-context-v1` DEV context；`original_image_refs` 必须完整187条、原 ordinal 顺序、distinct 的严格 JPEG refs。每个 ref 恰 path/bytes/sha256。完整列表是 metadata，不自动授权读取 ordinal0 图像或 tensor。

`input_sources` 是以下严格去重 union：

1. 原 context 的 `legacy_context/native_predictions/R_predictions/encoder_source/legacy_ROOT_result` 五个 refs。
2. 原索引1..186各行的 `original_capture_manifest/bundle/dense_packet`，及对应 `original_image_refs[index]`。

共享原 capture manifest 可作为所选行依赖去重保留；不另外加入 row0 的独立图像、bundle 或 packet。teacher/annotation ref 不递归闭合或加载。`sources` 恰五代码角色、十库、context_ref、input_sources 的去重 union；无额外控制或其他载荷。worker 先只读取已知 context 重建输入 union，拒绝误加引用后才哈希其余 Sources。

## 逐图执行与资源

保持原 CPU0/1、两线程、RSS≤16GiB、空闲 RAM≥16GiB、1800秒和 CUDA 隐藏。原 owner 负责原硬期限、实际进程退出和回收，其13字段 admission 不改。原 E盘 owner 的磁盘观测不冒充 D盘输出观测。

本 worker 另在每次 Control.check、后台资源样本及科学写入前查询实际 D输出父目录，要求 free≥固定20GiB。低于下限或读取失败后，Control 拒绝继续科学计算/包写入，失败不因后续空间恢复而清除。每次 U59 保存前通过同一 Control；写入中的取样是协作观测，不声称硬中断每个底层I/O。Root 绑定另需根据所选 shape 估算完整四字段逻辑字节及保守序列化开销，检查20GiB预留量加完整批预算；本 worker 不伪造预算，也不保证186图必能在1800秒完成。

原 reader 只 prepare 一次完整 metadata；随后严格按1..186逐图 load → U59 export_item → 原值/存储校验 → U59 save_export 的真实 CPU weights_only 重载 → 清空源/副本别名 → GC/weakref 检查，再进入下一张。只留标量审计、JSON、包 ref 和原 reader 的 manifest metadata，不累积各图 tensor。每图 U59 五个 input_refs 在循环内按该图原 ordinal 构造，不能沿用第一张或 row0。

每图原100 native 字典、score/category/bbox/area/顺序/空mask及opaque RLE 不删改；未知 native 字段按 U59 拒绝，不删除字段后重试。e/M/L 保持自然 query 轴，独立原 query_ids 置换用于导出。形状仍为 rgb400 `[1,3,400,400]`、e `[100,256]`、M `[100,200,200]`、L `[100,H,W]`。只输出 U59 四字段和最小白名单；F、targets、整个历史 bundle 不进入包。

原 bundle 反序列化可能接触历史 GT 关联容器；不能称整个上游没有接触 GT。本任务不创建教师/模型/优化器，不进行新 GT 栅格化、训练、renderer/RLE零残差校准、推理或 AP，不调用需要187个真包的 U59 全人口 prepare。

## 输出 / 报告 ABI

全新绝对 D输出目录下仅独占创建 `DEV_FULL100_1.pt` … `DEV_FULL100_186.pt` 以及 `REPORT.json`。失败保留 partial 包及小失败审计，不覆盖、跳过或自动重试。最终报告写前再次取样；若20GiB门槛已失败，仅保留小失败报告，不继续写科学包。

REPORT schema 为 `U62-DEV-remaining186-full100-CPU-export-report-v1`，终态 COMPLETE/FAILED。相对于 U61：

- `requested_batch_images=186`；actual_completed_images 为实际 save/roundtrip 返回次数0..186；actual_completed_queries 为其100倍，全部成功为18600。后续清理失败保留已返回次数，但不授 Root 资格。
- `export_entries` 是按原顺序追加的真实 U59返回 entry 列表，每项仍恰 dataset_index/image_id/query_ids/package_ref，替代 U61 单个 export_entry。
- `package_refs` 仅保存真实返回包，键为原索引字符串；`images` 按原1..186列阶段及原逐图 summary。成功图另有 `data_objects_released=True`，只在GC/weakref通过后设置。
- 顶层 `minimum_output_free_bytes=21474836480`；resource_observations 新增 `minimum_output_disk_free_bytes`（最低实际观测，初始None）、`output_disk_samples`（初始0）、`output_disk_sample_path`（D输出父目录），并保留 RSS/RAM/samples/violations/sampling_errors。D不足记作 `D output free-space boundary`，最终 COMPLETE 要求最低实际余量达标。
- `original_metadata_population={images:187,queries:18700,tensors_revalidated:False}` 仅为原 metadata 检查。`whole_population_exported=False` 始终保留；qualified/new_training/AP/TEST、新GT栅格化及模型/教师相关执行字段仍 False，DEV=True，历史GT容器接触 caveat=True。

其余 U61 来源/库路径、原13个 flags、RNG/dtype/grad/线程、临时模块别名、源文件前后身份与资源检查完整保留。每图 summary 的字段、U59 proof schema、原字典摘要和 ensure_ascii=True 的 canonical JSON SHA算法均不改变。COMPLETE 仍需各图完整释放、固定186/18600计数、来源/资源及最终清理通过，不等于原任务退出0或 Root 接收。

## 有限验证

`python -B test_metadata.py` 仅用合成 ref、行及标量测试新范围、来源 union、逐图 provenance 与 D不足拒绝。不会调用 main、读取磁盘余量或进程，不打开真实输入，不导入科学库。AST检查仅检查普通源码可解析/无顶层科学库导入。真实186图读取、张量等值/compact storage、weights-only重载、资源/磁盘余量、退出及整体资格均仍 **UNEXECUTED**，留给 Root 后续独立绑定的单叶任务。
