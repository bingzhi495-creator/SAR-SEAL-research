# U48：单张外部 TRAIN 图的原生100查询采集 body

状态：普通、未绑定、未执行数值；含函数及待执行单图 worker，没有控制器或人口遍历。
没有导入 ML 库、读取实际影像/权重/标注/Source/资格链，也没有执行教师前向。
已有 U47 原教师恢复、U41 采集及 U40 数值源码均不改动。

## 两个接口

```python
resolved = resolve_provenance(
    torch, c, model, context, fresh_restore_receipt, loading_receipt_ref,
    Config=..., capture_module=..., capture_modules=...,
    capture_source_refs=..., capture_source_ref=...,
)
# 精确返回 preprocessing / provenance / receipt，均为 JSON 对象。

result = capture_one(
    torch, np, c, context, image_metadata,
    selection={'dataset':'HRSID','split':'train','dataset_index':0},
    allocator_limit_bytes=...,
    restore_module=..., restore_modules=..., registration=..., restore_source_refs=...,
    capture_module=..., capture_modules=..., capture_source_refs=...,
    helper_source_refs={'restore':..., 'capture':...},
    loading_receipt_ref=..., report={},
)
# 精确返回 packet / report。packet 为本图 CPU 张量+原始导出元数据。
```

`c` 沿未来原执行者的 `closed/check` 接口，连续 RAM/RSS/整卡GPU/期限守护、
实际 Source/调用资格与输出保存均由外层负责。所有真实对象及 ref 显式注入，
不猜文件名或动态发现数据。没有新增 E50、残差头、指标或 AP 依赖。

## 真配置与加载证明

U47 按已核原 ABI 重新恢复本次 CPU 教师。`loading_receipt_ref` 必须指向外层
已真实写盘、预准入的 **原 `original_loading_proof` JSON**，不能是虚构绑定。
解析后的完整对象必须等于本次 fresh restore receipt 的同名对象，且 checkpoint
和 teacher_state 同时相等；文件存在或 `qualified=True` 不构成替代证明。

配置根与完整继承列表取自原加载证明
`original_initialization.official_config/official_config_closure`，每项先 closed。
真实 Config 定义文件也必须等于 U47 注入来源，之后原 `Config.fromfile` 解析。
mean/std/type 必须在原解析配置中显式存在，不从新数据计算或补默认值。

颜色转换优先取原配置；缺字段时只读取当前真实 `ImgDataPreprocessor.__init__`
签名里的同名默认值，该类实际源码按 U41 原 SHA 闭合。不存在默认参数、非 bool、
两个方向同时为真均拒绝。明确记录 `original_config` 或
`closed_constructor_default` 来源，不把缺项直接填为 False。

padding 从真正加载的 preprocessor/BatchFixedSizePad 读取；按原 pref 来源证明
核800画布、keep_ratio、固定补边和无增强，再由原 U41
`_verify_loaded_preprocessing` 与实际 mean/std buffer、channel flag、divisor、
pad_mask/pad_value/pad_seg 逐项核对。原覆盖项来自固定 pref.prepare_cpu，
不把官方 raw config 中未设置的固定补边配置误称为原配置本有。

返回的 provenance 精确沿 U41 五键：teacher_checkpoint、loading_receipt、
config_sources、teacher_state_proof、resolved_preprocessing。
解析 receipt 不代表实际预处理张量、全人口或模型效用已通过。

## 单图与资源生命周期

`image_metadata` 只有 image_id/image_ref/original_shape，明确800×800。
selection 固定为调用者在读像素前选定的 HRSID TRAIN ordinal0；不接入新图标注、
TEST、GT、matcher 或数据集对象。原历史教师 qualifier 仍可能校验历史数据文件
编码字节，不能因此写成“整个过程从未访问任何历史标注字节”。

U47 完成冷 CPU 恢复与配置核对后，才首次初始化 CUDA0。调用者必须明确指定
allocator_limit_bytes，必须大于0且不超过原4GiB；可明确给2GiB作更低显存窗口
的待验证运行，失败时不自动升限或重试。原8GiB整卡、RSS/freeRAM16GiB、
1800秒或外层更短有限窗口仍由未来原执行者守护。

原 U41 的8对象/10引用保持不变。一次 extract/head 原生前向加原 fusion 全字典
校准，保留自然查询轴0..99和真实 export 排列、全部100条S0、类别、RLE及空预测。
CPU packet 原形状：F[1,256,200,200]、e[1,100,256]、M[1,100,200,200]、
C[1,100,2]、L[100,800,800]。只 detach/cpu/clone，不重建 M、不重排自然轴、
不改原分数/阈值/类别，不做头训练。

只保留一个图的 CPU packet；不缓存全人口。结束时释放 GPU capture、将教师移回
CPU并删除内部引用，清 cublas workspace/cache；实际运行后必须 allocated/reserved
双0且 peak 不超过本次请求上限才能返回。报告中的这些检查仍只是待实际执行的
worker 证据，不自授 Root 资源或数值资格。失败保留部分 report，输出由 caller
保存，禁止把部分 packet 继续当作成功。

## 本轮验证范围

两个极小标准库元数据行为测试按 RED→GREEN：缺颜色方向必须有真实默认参数
证据，冲突方向拒绝；旧 proof JSON 与 fresh proof/状态不一致拒绝。
没有 fake NN、巨型 fixture 或数值成功伪造。真正教师恢复、配置解析、800图前向、
100查询/S0/RLE、显存与清理均未在本稿执行；未来原 owned task 才能验证。

## 待执行 worker 入口与输出

`entry.py --spec ABS` 仅由原 Root-owned driver 调用，未绑定或运行。
schema `ROOT-U48-external-native-capture-spec-v1`，operation
`GPU_U48_EXTERNAL_NATIVE_CAPTURE`；不写 Q、不自授资格，不自动重试。
显式来源字段：source_roles 四项 entry/body/restore/capture；teacher_sources
沿 U47；library_sources 为 psutil/torch/numpy；body_sources 为 U41 的八个
实际对象，capture_source_refs 为原十项，其中 PackDetInputs 对应 formatting。
所有 ref 与 input_sources/原image_metadata/selection_proof_ref/loading_receipt_ref
均须在完整 Source 内；排序不影响 query 轴。selection proof 只核其原引用/字节，
不猜未知 JSON schema；调用者提供且绑定的 selection 固定 HRSID/train/0、image_id0。

本 worker 固定2GiB allocator，CPU0/1两线程、CUDA_VISIBLE_DEVICES0、RSS/freeRAM
16GiB、180秒；原 driver 仍守护整卡8GiB。原 N26 GPU math 设置在首次 CUDA 前
显式应用并记录：matmul.allow_tf32=False、cudnn.allow_tf32=False、benchmark=False、
deterministic=True；结束 finally 恢复原调用者四设置。CPU RNG/default dtype/grad
mode/线程在调用 body 前后实际比较。GPU RNG 仅在首次 CUDA 初始化、设cap和
model transfer **之后**保存，原 capture 之后到 GPU 释放后实际比较；不声称 CUDA
恢复未初始化状态，不在 U47 冷 CPU 恢复前调用 CUDA RNG。

成功只保存一个 `NATIVE_PACKET.pt` 与 `REPORT.json`，CPU packet 不含教师权重、
新图标注或GT tensor。packet以独占新文件写入并流式hash；失败时不删除已写片段。
REPORT schema `U48-single-external-native-capture-report-v1`，成功 status COMPLETE，
qualified/new_training/AP/DEV/TEST=False。body_report 原样保留；packet_ref 为真实
输出 strict-ref，packet_summary 保留形状、自然/export查询序、100预测/S0/class计数。
native_RLE_calibration 明确是原 fusion masks/scores/labels 比较和原 exporter 编码，
`independent_RLE_decode_executed=False`，不谎称另行重新解码验证。

完整 Source 字节闭合可能打开版本链或历史标注文件做hash；报告只声明新图标注
记录未解析、GT未进入数值路径，不把“无GT推理”误写为“未打开任何标注文件字节”。
资源、随机数和实际状态字段必须未来真实运行后产生；本稿只有 AST 静态检查。


<!-- U48_ACTUAL_STATUS_20261018_01 -->

2026-10-18 实际补记：本 2GiB 版本已实际执行并退出 1，原生融合后处理申请 246MiB 时触及分配器上限；无完整 packet。真实失败已核验保留，退出后 GPU allocated/reserved 均为 0。独立 3GiB 资源修正版随后实际通过；未覆盖本版本、未改数学或原评分。

结果与范围见 [单图实测](../../U48_SINGLE_HRSID_NATIVE_CAPTURE_ACTUAL_AND_NEXT_STEP_20261018.md)。
