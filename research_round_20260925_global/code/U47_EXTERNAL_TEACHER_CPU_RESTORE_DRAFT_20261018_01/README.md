# U47：外部影像评估的原教师 CPU 恢复草稿

状态：普通、未绑定，**未执行真实权重加载、张量、模型或图像计算**。
本目录仅有恢复函数与待执行 CPU worker，没有数据发现、人口遍历、执行控制或资格接收器。
两个标准库测试仅检查注入的元数据调用及异常后的 helper 恢复，不模拟
神经网络或宣称真实恢复通过。现有 U44/U40/原 N26 文件均未修改。

```python
result = restore_teacher_cpu(
    torch, c, context,
    modules=modules, registration=registration,
    source_refs=source_refs, report={},
)
# result 精确三个键：model / receipt / teacher_state_proof
```

`model` 是原教师本体，返回时应为 CPU、FP32、完整 eval、冻结且无梯度。
`receipt` 是同一个调用者传入的 JSON 报告，成功状态 `RESTORED_CPU_ONLY`，
schema `U47-external-teacher-CPU-restore-pending-v1`，`qualified=False`。
`teacher_state_proof` 与报告中的同名对象相同，沿用原 `pref.state_proof`。
函数不写盘；外层必须真实保存并核定该报告，才能将其实际 strict-ref 填入
U41 `provenance.loading_receipt`，不能用本函数返回字典代替文件引用或资格。

## 准确注入边界

- `c` 提供原 `closed(ref)`、`check()`、`allowed`；输入、RAM、deadline 与执行
  来源资格由未来调用者闭合。所有 source ref 必须是实际 canonical 路径，
  不通过放宽字节/路径门接受 C/E junction 别名。
- `context` 是原 native 上下文的解析 JSON，必须等于
  `source_refs['context']` 经 `c.closed` 读取的原内容，不合成新训练上下文。
- `modules` 精确五项：`native/legacy/pref/initializer/common`，分别严格对应
  context 的 `native_loader_source/evaluation_source/micro_preflight_source/`
  `initializer_source/training_source`。调用者已经加载真实模块；这里不动态猜路径。
- `registration` 精确七项：`register_all_modules/init_default_scope/`
  `registry_module/MODELS/revert_sync_batchnorm/Config/CocoDataset`。
  `MODELS` 必须是该真实 `registry_module.MODELS` 对象。`CocoDataset` 只读取
  原 `METAINFO['classes']`；**不实例化数据集或 COCO**，不改成猜测的单类列表。
- `source_refs` 精确三项：`context` strict-ref，`modules` 的五项 strict-ref 表，
  `registration` 的六项 strict-ref 表（上述七对象除 `MODELS`）。后者通过
  `inspect.getsourcefile` 或实际模块 `__file__` 核对实际代码文件。

这只覆盖明确提供的来源文件，不是整个 Torch/MMCV/OS/DLL 二进制闭包。
实际来源、上下文和历史资格未由本轮读取或接收，因此此稿仍不能自称已具备
真实单图运行资格。Torch 为调用者实际注入的 `2.0.1+cu118`，Python 为原
`3.8.17`；函数始终要求 CUDA 冷态，不创建 CUDA 张量。

## 最薄加载路径及原始依据

路径以下均相对普通发布副本 `research_round_20260925_global/`，仅静态核对。

1. `code/N26_FIT_MATCHED_R_CAPTURE_SOURCE_ONLY_20261017_01/capture_setup.py:147–165`
   的教师构造段是本稿依据。**不调用完整 `setup`**，后续 strong PR、FIT、
   GT、缓存构建均不进入本函数。
2. `multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/`
   `spatial_position_preserving_common24_evaluation_source_20261009/evaluate.py:126–198`
   展示正确的单参数 `legacy.qualified_training(context)` 与返回 `native0` 字典。
   实际 legacy 来源仍按调用者的原 context/ref 核对，不把本文路径猜作实际绑定。
3. 同准备目录下 `native_finetune24_fresh_optimizer_source_20261010/train.py:171–216`
   是固定十参数恢复函数，45363B，SHA256
   `63ddb7783c24b06397476d178307c60eedee2cf751cc75236deb582999b0c7a2`。
   它严格重建原初始化并恢复全610状态/451参数，核原 E24 metadata 和独立状态证明。
   旧 checkpoint 的 optimizer/RNG 容器会被反序列化，但不创建/恢复 optimizer
   或恢复旧 RNG。48warm 版本额外需要 predecessor，**不属于本接口**。
4. `multimodal_data_pilot_20260928/native_coco_comparator_micro_preflight_preparation_20261008/`
   `run.py:62–69` 的 `digest_tensor/state_proof`：detach、CPU、contiguous、NumPy
   数值字节，并保留 shape/dtype/SHA，不含 device 字段；30547B，SHA256
   `7f7e5ce0eea8f0dae6e7fc01a4b54baafdc52fb3339b039ae50ef26b60870201`。

五模块的原单引用 `bound` 在调用期间暂时路由至当前 `c.closed`，异常时也恢复，
不改原文件。先闭合原训练 receipt，再调用原单参数 qualifier，保留其实际 held
历史检查；按原式 registration/scope、构造、library gate、插值来源门执行。
最后冻结全模型、关闭 `backbone.with_cp`，与独立 anchor 逐项 equal 并核原
loading.teacher_state。原 trainability policy 只保留作来源证明，**不是实际仍有
345个可训练参数**。独立 CPU anchor 在退出时释放，外部 Torch CPU RNG 保存后恢复。
失败报告保留异常，失败对象不返回，也不自动重试。

## 仍属调用者的实际验证

先在原资源/Source 准入下实际执行上述加载，确认原全部权重/BN、原加载证明、
610/451、CUDA 始终冷态和资源符合；再保留同一个 CPU 教师用于 U44 窗口。
原配置的 `resolved_preprocessing` 和 `config_sources` 仍需调用者明确闭合解析，
本稿不补默认值，也不假造 U41 provenance 的完整五键。

原 qualifier 会读取历史资格 JSON，并校验其训练来源文件；Source 字节闭合可能
包含历史图像/标注编码字节。因此本稿只承诺不构建/迭代数据集、不真实解码图片、
不解析标注记录、不执行 forward，不宣称“没有读取任何历史数据文件字节”。
成功也只表示该加载函数的 worker 回执，不授予外部人口、模型 AP、修复算子、
完整环境或科研效用资格。Teacher 释放/切换 CUDA、两臂 E50 恢复及资源守护仍由外层负责。

## 新增待执行 CPU worker

`entry.py --spec ABS` 仅供原 Root-owned driver 调用，本轮没有运行。
spec schema `ROOT-U47-external-teacher-CPU-restore-spec-v1`，operation
`CPU_U47_EXTERNAL_TEACHER_RESTORE`；没有 Q、自授资格、prepare、bind 或自动重试。
来源字段为 `source_roles={entry_source,restore_source}`、上述三块
`teacher_sources={context,modules,registration}`、
`library_sources={psutil,torch,numpy}`、`input_sources` 和完整显式 `sources`。
Root 提供原已资格上下文、原 CP/config/历史依赖引用，不推断新数据路径。
Source 的首次流式字节校验在 ML import 之前完成，后续每次 `closed` 重查
文件身份、大小、mtime/ctime；缓存只在本叶内且容量不超过显式 Source 表。
不是驻留后台或整个环境的字节资格。

资源与 U45 相同：CPU0/1、两线程、CUDA_VISIBLE_DEVICES 空、RSS16GiB、
available RAM16GiB、120秒；整卡8GiB观测仍由原 driver 控制。
worker 只导入原 N26 使用的真实框架注册对象，调用本函数，并在结束前释放
教师及模块引用。模型弱引用用于核对实际释放；不保留权重输出，不 forward。

无论成功或异常，已创建的独占 D 输出目录保留 `REPORT.json` 部分回执。
schema `U47-external-teacher-CPU-restore-report-v1`，成功 status `COMPLETE`，
`qualified/new_training/AP=False`。`teacher_load_receipt` 原样保存 helper 报告，
`teacher_state_proof` 保存全610状态摘要，另
`teacher_state_proof_canonical_sha256` 为其 `sort_keys=True`、
`separators=(',',':')`、`allow_nan=False` 默认 JSON UTF8 字节 SHA256。
`flags` 六项为 `CPU_only/source_bytes_verified/CUDA_uninitialized_before/`
`CUDA_uninitialized_after/module_aliases_restored/teacher_model_released`，
成功必须全部为真；`cost`、`resource_observations` 保留实际计时和采样资源。
该 worker 草稿只做 AST 检查，**不构成真实权重恢复已通过的证据**。


## Actual status, 2026-10-18

The original CPU job completed and was Root-accepted: all610model/BN state entries and451parameter tensors restored exactly; worker21.5s, sampledRSS1632632832B, coldCUDA, model/anchorreleased. No forward/image/AP. See [actual evidence and remaining work](../../U47_TEACHER_CPU_RESTORE_ACTUAL_AND_DGX_STATUS_20261018.md). Earlier unexecuted wording documents the authoring stage; numerical source bytes are frozen and unchanged. This function is ready in the checked local scope, not an executed external inference pipeline.
