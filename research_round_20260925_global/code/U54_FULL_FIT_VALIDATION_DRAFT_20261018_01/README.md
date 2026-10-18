# U54 独立DGX全FIT便携输入CPU验证草稿

**SOURCE-ONLY / UNEXECUTED。** 仅标准库小型元数据行为测试和AST；未导入Torch/NumPy、执行worker/owner/body、读真实manifest/包/图像/GT/权重/Source/资格链/Ops或SSH。此目录不代表实际741图校验成功、DGX主训练准入或AP；未来Root仍须提供真实Linux refs、预锁、执行与独立接收。

`validation_body.py` 与 `entry.py` 调用已停止的原U54 reader，不修改其19618B/SHA `b65ed9ecb7a9a704899f3c75983b2cb9676e26b42d87f65336cfbcb16bd94557`。`linux_owner.py`为U52普通Linux owner的CPU克隆，保留原Linux birth/process-group/PDEATHSIG/实际wait-reap和资源守护，改为独立U54 CPU协议。原U52 owner固定GPU操作、CUDA0、旧源集合及equivalence完成门，不能不改便启动此CPU入口；旧源保持不动。

## 真实未来调用

```text
<DGX-python> linux_owner.py --spec /absolute/spec.json --spec-sha SHA256
```

owner只导入选定真实psutil，按原进程控制模式启动一次entry并记录实际终态；失败不自动重试。worker的CLI同为 `--spec/--spec-sha`，不能据源码准备或worker pending状态自行授Root资格。输出目录独占新建，不清除失败产物或覆盖旧尝试。

## 精确Spec与来源

Spec恰14键：`schema/operation/phase/entry_source/body_source/owner_source/source_refs/library_sources/manifest_ref/package_refs/prelock/output_directory/owner_output_directory/sources`。

- schema `ROOT-U54-DGX-full-FIT-CPU-validation-spec-v1`；operation `CPU_U54_DGX_FULL_FIT_PORTABLE_VALIDATION`。
- source_refs恰 `portable_reader/image_geometry`，分别为上述原U54 reader及原geometry SHA `65db107de80c5f1d4dade7cb2ffd3f241fd6bf7f9b233cb1aa9c90cbfd00abf7`。三执行ref为本目录实际entry/body/owner文件；body须与entry内固定的新字节数和SHA一致。
- library_sources恰 `psutil/torch/numpy` 的真实本地顶层模块文件ref。覆盖这三个选定模块文件，不宣称整个OS/Python或全部binary依赖闭包。
- manifest_ref为原U54 schema的真实私有全清单；package_refs为741个真实本地严格ref，按原dataset_index0..740排列，须与manifest rows逐项精确相等。Sources由三执行文件、两个reader/geometry、三个库、manifest与741包的完整引用集合计算并按POSIX路径排序，不把旧22源门改成猜测的新固定数量。输入不能放在输出目录下。
- 两个输出目录必须绝对规范POSIX路径、独立且不互相嵌套。worker写 `REPORT.json` 和 `IMAGE_CHECKS.jsonl`；owner沿原receipt/log/state文件布局保留实际启动和回收证据。

Root future manifest仍须按原U54合同精确合并三批：U51原0..31、U51原32..63、U53原64..740，各自原selected_indices和schema保持。现导出Q只有包ref和proof摘要，没有完整identity/query_ids；Root需额外从原已资格FIT metadata按index精确join records，不能凭摘要猜造。此目录不读取历史Windows Q/GT/缓存去补清单，也不假定U53已经完成。

## 固定CPU预锁

`schema='U54-full-FIT-CPU-validation-prelock-v1'`；environment为Python3.12.14、Torch2.10.0+cu130、NumPy1.26.4、Linux/aarch64。dataset_indices为0..740，images741、GT1661、CUDA_visible_devices空字符串。no_new_training/no_forward/no_matching/no_AP/no_DEV/no_TEST均True。

limits恰五键：CPU affinity `[0,1]`、threads2、RSS上限16GiB、可用RAM下限16GiB、deadline1800秒。此CPU任务不设置CUDA allocator、不初始化CUDA、不构造model/optimizer，也不访问GPU随机数状态；CUDA是否未初始化由真实Torch观察。没有将GB10 UMA观测冒称Windows整卡VRAM8GiB通过的行为。worker/owner资源检查和采样范围按实际日志报告，不承诺此次草稿必定满足时间或内存。

## Body与逐图释放

`run(torch,c,manifest,*,reader_module,image_geometry,source_refs,prelock,report,record_image)`；所有库和原模块由source-closed worker注入。report是调用者持有的同一个新字典，失败保留其partial状态。c沿owner原 `closed/check/summary` 接口。

原reader.prepare先核741条原序records及总G1661；随后0..740依次load，只保留当前图的五个CPU字段。每次原reader完成metadata、U51/U53包、五字段shape/dtype/finite/detached/storage与digest校验后，body核原记录身份与查询次序，生成scalar audit，释放当前item并实际核其Tensor弱引用均消失，才写当前审计并进入下一图。仅保留计数、SHA、shape、ref和原metadata摘要；不留全人口image/tensor数据。调用者仍需释放外部别名，资源监控负责实际RSS门。

`record_image(result)`由entry流式写私有JSONL，每图原index必须等于实际已写行数。body累计 attempted/validated/released/recorded_images、matched_instances和原reader visits，失败立即传播，不删图、跳包或重排/重匹配。完整metadata核验不等于741个包已经读过；只有741次实际load、全部释放/审计与总1661目标均完成，才可worker标记完成。

该过程确实读私有FIT目标Tensor及身份/ann/query元数据，不宣称“无GT读取”。不打开原COCO标注重新栅格化、不匹配teacher、不构建/加载主模型或前向。五字段与原冻结输入逐字节保持也不等于重新证明影像与标注的语义正确性。

失败证据区分尝试与成功返回：`FIT_GT_tensor_read_attempted` 初始False，在首次调用原reader.load之前置True；它只记录进入可能读取GT的路径，不证明文件或Tensor已实际读入。保留的 `FIT_GT_tensors_read` 初始False，仅在一次load成功返回并通过当前图身份核验后置True；该值为False不能证明失败路径从未读过GT。attempt标记不提前增加validated_images、matched_instances或审计计数。两者是body报告的证据字段，不改变owner终态、原成功门或数值路径。

## 报告与资格边界

body schema `U54-full-FIT-CPU-validation-body-report-v1`，成功pending状态 `COMPLETE_PENDING_ROOT`；完整计数741/1661、原visits0..740、失败load0和当前Tensor残留0都必须成立。body五flags为 CUDA_uninitialized_before/after、metadata_population_verified、all_field_proofs_verified、all_returned_tensor_objects_released。

worker schema `U54-full-FIT-CPU-validation-worker-report-v1`，成功状态COMPLETE；保留body_report、实际source/ref/environment、JSONL严格ref、audit_records_written741、实际CPU资源、cost与十flags：source_bytes_verified、CUDA_uninitialized_before/after、source_unchanged_after、module_aliases_restored、caller_CPU/NumPy/Python_RNG_preserved、dtype_grad_threads_preserved、child_resources_within_limits。所有False标志包括qualified/new_training/main_training/AP/DEV/TEST，formal_training_budget_credit为0。模型、optimizer、前向和匹配均未进入此协议。

新owner只在真实child退出0、wait/reap/birth/group条件与上述完整报告均满足时记pending Root审阅；不能把worker字面True或owner准备成功当Root接收。失败partial REPORT、标量审计及原终态保留，不修改U40 frontier、旧RAW/account或其它运行任务。

## 本轮测试范围

两项worker标准库行为只验证部分/失败计数不能报全覆盖，以及动态来源去重/冲突拒绝；另有一个owner元数据拒错/真实终态字段合同测试。fixture仅小字典和整数索引，不导入数值库、不模拟tensor或科学成功。AST及普通生命周期差分是静态证据；实际全人口输入、资源、文件、进程回收与Root资格均未执行。

另增一项标准库失败边界回归：在reader构造前中止与进入load后立即中止两种情形中，核attempt初值/置位、成功读取标志仍False及成功计数仍0。只有冷状态观察接口及原序整数metadata桩，不构造Tensor、不读包、不模拟成功科学载入。该回归先因缺少attempt字段失败，再在最小修正后通过。
