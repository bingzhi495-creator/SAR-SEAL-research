# U43 HRSID 全人口 CPU 检查入口草稿

状态：**未绑定、未执行科学 main；仅标准库测试及 AST 检查。** 不创建 admission、binder、训练队列或后台程序。本目录与人口核心目录独立，核心数值不在这里重写。

未来原 Root-owned 调用：`run_check.py --spec ABS --owner-execute`。入口 schema 为 `ROOT-U43-HRSID-population-pixel-mask-check-spec-v1`，operation 为 `CPU_HRSID_FULL_PIXEL_MASK_INPUT_CHECK`；body API 为 `run(spec,c,pre,context)`。

## 原资源接口及依赖

复用普通发布文件 `P/research_round_20260925_global/code/N23_ANCHORED_DOMAIN_TRAIN_SOURCE_ONLY_20261017_01/run_train.py` 的 `Control(spec)`，其中 P 为 `E:/SAR/seal/github_release_20261011`。SHA256：`9b17a7bdd9c44fe159583f4c82d38f1b42970662507986654900767e8ba5f383`。仅加载定义，不调用其旧训练 `run/main`。Control 构造只导入 psutil；`check/closed/ref/write_json` 不需要 Torch。

固定 CPU affinity [0,1]，OMP/MKL/OPENBLAS/NUMEXPR 线程环境变量均为2，CUDA_VISIBLE_DEVICES为空，RSS≤16GiB、系统可用RAM≥16GiB。沿原 Control 固定1800秒并把入口开始时间带入 `c.started`，不声称纯Python协作检查是独立硬终止器；未来原 owner 负责硬截止及真实句柄回收。原13字段 admission 保留整卡8192MiB/allocator4096MiB声明；本任务根本不导入Torch、不分配CUDA，不将这些声明当实际GPU资格。输出目录独占，1GiB日志/JSON存储余量；失败无重试。

`spec` 的代码角色精确为 `entry_source/body_source/population_source/checker_source/runtime_source`。U42 checker SHA固定 `cb247465226a77afc1d8c9bba00ad5a46ca1a10ffebf4931ba4a17a1c09ea69f`。人口核心严格ref由后续Root冻结提供，不用本草稿猜测未来路径或hash。

`library_sources` 精确12项：`PIL`、`PIL.Image`、`PIL.ImageFile`、`PIL.JpegImagePlugin`、`PIL._imaging`、`numpy`、`numpy.core._multiarray_umath`、`pycocotools`、`pycocotools.mask`、`pycocotools._mask`、`psutil`、`psutil._psutil_windows`。body在真实导入后核实际模块文件和字节，另核 Image.core / mask._mask 对象；这些是**选定依赖文件足迹**，不声称整个Python、操作系统或DLL环境已完全闭包。无CocoDataset/模型构造；原COCO polygon运算通过U42注入的官方mask API执行。

## context / prelock / Source

context 精确四键：`schema/image_root/annotations/image_refs_by_filename`；schema 为 `U43-HRSID-official-split-pixel-mask-check-context-v1`。annotations 精确 train/test 两个原JSON strictrefs，image_refs_by_filename为5604个原文件名→strictref，逐项对应原 image_root。context 只作为显式元数据入口，不发现文件、不猜路径、不用merged16951代替原split16969。

`spec.sources` 只允许上述5源码、12库、context/prelock/plan/parent_ROOT_qualified、2原splitJSON、5604JPEG的去重并集；不接纳额外cache/权重/模型。每项是 `path/bytes/sha256`。runtime实际核数据字节，人口核心才解析两个原split并调用U42完整解码/栅格检查。

prelock schema 为 `U43-HRSID-population-pixel-mask-check-prelock-v1`；绑定同一 context 和 plan，splits=['train','test']、split_counts={train:{images:3642,annotations:11047},test:{images:1962,annotations:5922}}、expected_images=5604、expected_annotations=16969、original_canvas=[800,800]；automatic_retry/automatic_sharding=False、population_filter=None；no_new_training/no_AP/no_model/no_prediction/no_DEV=True；TEST_annotations_read=True、TEST_used_for_model_selection=False。

spec使用同人口与资源字段，保留动态 owner_index/normal_RAW_count/total_failed_RAW_count/total_terminal_identity_pin_count/legacy_pin_count。总声明为normal+failed，legacy计数独立。parent_ROOT_qualified是实际账户父，与数据来源不同；入口只核原已接受父及计数，不能自行授予父资格。

## 工作体及产物

唯一人口调用：`population.run_population(Image,ImageFile,np,masks,context,checker.check_one_image,closed_ref=c.closed,check=c.check,record_image=callback,report={})`。传入U42函数本身而不是module；返回须为同一report对象。人口core负责全守恒和失败细目，不在本目录复制图像或mask数学。

私有 `IMAGE_CHECKS.jsonl` 每图原结果落盘并flush，保留原split/ordinal/filename/annotation细目，**不能作为公开摘要发布**。`REPORT.json` 包含完整人口聚合/失败状态、原来源refs、选定实际库足迹与资源观测；`probe.log`保留实际日志。`SAFE_SUMMARY.json`仅成功后输出，白名单仅COUNTERS对应的train/test与totals整数及固定范围布尔，不含filename、image/annotation ID、路径、refs、polygon或mask；仍是pending，须Root接受后才可发布。

完整成功要求人口schema `U43-HRSID-population-pixel-mask-check-report-pending-v1`、status `INPUT_CHECK_COMPLETE_PENDING_ROOT`、四项完成布尔为True、5604图/16969实例及5604回调一致。坏图或坏mask导致core完整失败状态时，body保留报告后抛错；资源/closed_ref/callback异常原样传播，partial report保持，不写Q。库加载前等早期入口拒绝由原终端日志保留，没有虚构已发生的数据检查。报告中TEST_annotations_read在metadata_validated之后才为True，更早失败留None；不把已授权读取误写成实际读取。

成功 result 精确包含 report_ref/log_ref/image_records_ref/safe_summary_ref、images/annotations/input_check_complete、qualified=False、Torch_imported=False/CUDA_initialized=False、new_training=False/AP=False、TEST_annotations_read=True/TEST_used_for_model_selection=False、semantic_alignment_proven=False。

只有入口完成后写 `Q.json`，schema `U43-owned-pending-HRSID-population-input-check-v1`，status `COMPLETED_PENDING_ORIGINAL_ACTUAL0_AND_ROOT_ACCEPTANCE`。Q明确TEST=True、TEST_annotations_read=True，**没有no_TEST声明**；不执行模型、不预测、不选超参数、不计算AP，不恢复盲测身份，也不证明JPG/PNG像素等价、语义标注对齐、跨数据集效用或完整方法资格。

## 已运行检查与未执行边界

三项标准库行为：固定人口/资源/计数拒错；完整5604+2输入并集及拒绝多余权重；故意资源异常后私有逐图记录和FAILED报告保存且无Q/安全成功摘要。各项先出现真实缺文件/缺函数RED，再GREEN。最后3项通过，测试中的 traceback 是预期异常传播，未发生真实库/数据执行。AST检查只证明Python语法可解析，不证明Pillow/NumPy/官方mask运行、人口成功或资源足够。

后续仍需Root冻结最终population/entry/body和选定实际库来源，构造真实完整context/prelock/admission并由原owner执行、接收。这里不做这些动作；不新增科学资格叶或改变U40。
