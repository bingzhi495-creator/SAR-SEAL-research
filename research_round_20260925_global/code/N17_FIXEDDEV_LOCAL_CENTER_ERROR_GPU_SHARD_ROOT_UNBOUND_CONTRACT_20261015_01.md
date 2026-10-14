# N17 fixed DEV GPU shard outer：UNBOUND / STOP EDIT 合同

2026-10-15。仅普通源码构造和一项 tiny SYN；本代理没有读取实际 SCI、权重、旧 bundle、SPEC/RAW/SOURCE 资格链，没有执行真实 SAR、没有 Git。科学资格仍由 Root 在冻结后的原 owner 任务核验。

## 入口和 Root 必填字段

入口为 `N17_FIXEDDEV_LOCAL_CENTER_ERROR_GPU_SHARD_ROOT_UNBOUND_DRAFT_20261015_01.py`，standalone 复制既有 typed RAW9–17/carrier front、已审 N17 lock/cache validators 和旧纯共享验证函数；没有导入旧 N16 shard entry 或 monkeypatch。允许沿严格 Source gate 使用原 carrier `evaluation.load_heads(torch, context, gate)`，只取冻结 LOG_PR。

- 新 `schema = ROOT-N17-fixedDEV-local-center-error-GPU-shard-spec-v1`。当前首 `shard_id=0`，范围 `[0,47]`，normal/expected 122、failed 17、pins 139。后续预定范围 `[47,94]`、`[94,141]`、`[141,187]`；只有前序实际 0 后才能绑定下一 owner，失败后不猜计数、不重试。
- 继承已审 N17 GPU120/CPU121 的完整 Source、RAW、pins、carrier、protocol、N17 experiment lock、neighbor cache、init/scout/base factory refs；不要删除历史字段。`entry_source` 必须指当前入口自身，Root 指定 `GPU_admission_file`、`phase`、绝对且不存在的 `output_directory`，seed 202610150171、deadline 1800、CPU01/threads2、CUDA0、allocator 4096 MiB、whole GPU 8192 MiB、automatic_retry false。
- `ROOTtrain_result` 必须为 Root 原 GPU120/CPU121 双 actual0 的完整 qualified 结果；新 spec 显式单个 strict3ref 字段为 `training_GPU_source/spec/report/RAW/admission_refs` 与 `training_CPU_source/spec/qualification/RAW/admission_refs`。名称以源码为准，`admission_refs` 也是单个 strict3ref，不是列表。Root 结果中的 phase/bootstrap/details/init/qualifier/两 checkpoint/两 FIT endpoint 全部闭合进 Source；不把 immutable pending Q 改成 qualified。
- `legacy_qualified_result` 必须为旧 N16 全四 GPU、四 numeric CPU、全 DEV COCO 完整 qualified 结果；`legacy_shards` 精确等于其 final02 spec 的四项 `shards`，保留明确 `shard_owner_index`，不得用当前计数减 1/2 猜旧 owner。旧报告、manifest、capture bundle refs 及 native/R 全量预测/COCO refs 全闭合。
- `new_eval_body_source` 指已冻结 76d11… 的新 evaluate_all100 body；`legacy_eval_body_source` 指旧纯数值 helper；原 `labels_source`、neighbor/scout/base/frozen PR refs 不变。新 body 不由本入口改写。
- `capture_budget` 六个正整数字段：`maximum_process_RSS_bytes`、`maximum_perimage_raw_capture_bytes`、`maximum_total_raw_capture_payload_bytes`、`maximum_serialized_capture_bytes`、`minimum_free_output_bytes`、`auxiliary_output_budget_bytes`。RSS ≤16 GiB、单图 raw ≤1 GiB、serialized+aux ≤72 GiB，free ≥serialized+aux+16 GiB。每图新的保守 raw 上界为 `272*sumC + 192*sumK + 800*ceil(HW/8) + 16 MiB`，必须满足锁定预算。Root 另留序列化/JSON 开销余量；实际 raw/serialized/aux 仍逐次验证，不能将 raw 上界当作 pickle 大小保证。

## 实际共享输入与模型接线

新 owned GPU leaf 严格读取原 torch.save image bundle，核完整 binding/identity/query order、registry、full native L `[100,H,W]`、e0 `[100,256]`、F_unique `[U,256]`、CPU int64 union、全 100 个 C/K 与独立 C union。随后实际 CPU→GPU→CPU bitwise 检验。静态 ABI、旧 marker 和 Root 元数据闭合不能替代这一新载体数值门。

四模型由 qualified 原顶层 E8/E24 snapshot 的两 arms 注入，精确四 tensor/16803 参数；source/spec/ad/phase/epoch、E8 history/order 前缀绑定到 Root 训练资格。初态零扩展已经由该完整资格证明；训练后的新 8 列不要求零、非零或 arms 分化。原 LOG_PR 8 tensor/40497、无 buffers；PR 显式 zeros49，实际取 raw residual 及 FP32 `L+u`，不使用 gated head logits 替代 C action。

实际 `evaluate_all100` 使用 full L/e、union/F_unique、四模型和同 frozen PR；自然 q0..99，C+selected 同 GPU 重算。callback 同步 detach→CPU clone，并逐位核 F/e/L、独立 extra8 和两个 logits alias 后才去重 F/e/L。所有新预测恢复原 row.qids 顺序，保留 S0/class/query 非分割字段。主结果固定 E24 atmost；其余 E8/E24×两 arms×atmost/exact 共八 profile。native/R 由原完整 refs 绑定后复用，不重新训练或重新评 AP。

## 新 bundle 与 selected CPU core 解码

- 每图一个 `IMAGE_%03d_ACTUAL_GPU_BUNDLE.pth`；schema `N17-actual-GPU-selector-image-bundle-v1`。九个顶层键：schema、binding、input_carrier、dataset_index、image_id、members、member_registry、member_count、raw_tensor_payload_bytes。
- `input_carrier = {bundle: 原 strict3ref, shared_member: 原成员名字符串, original_GPU_binding: 原 binding}`。旧 shared 不复制进新 bundle；新的 Source/ref 门与输入逐位证明必须保留。
- 新 point schema `N17-actual-GPU-selector-point-chunk-v1`。保存 flat_indices 和原 callback 元数据；scout 保存 actual extra8、actual_GPU_logits 及 scout_logits alias；PR 保存旧六键中的 flat_indices、GPU_raw_residual、GPU_combined。sampled_F/e0/native_L 去重至 input_carrier，其余真实 GPU 值保持 CPU clone。
- 新 profile schema `N17-actual-GPU-selector-query-profile-v1`。字段含 query/epoch/arm/mode/K、`C_scout_chunks`、`common_C_PR_chunks`、`selected_PR_chunks`（都是内部成员名字）、actual_selected_flat_indices、actual_mask_packbits/shape/bitorder=little、actual_RLE、original_prediction_nonsegmentation_fields、PR_compatibility。
- CPUcore adapter 用 strict union searchsorted 从原 shared 恢复 F/e/L，给 `iter_scout_chunks(epoch,arm)` 升序 2048 六 tuple `(flat,F,e0,L,extra8,actualGPUlogits)`；给 `read_profile(epoch,arm,mode)` 四 tuple `(actual_selected,selectedPRchunks,actual_mask_bool,actual_RLE)`。mask 解 packbits little 后截取 HW 再 reshape；PR chunk 恢复原六键。共用实际 C logits 排名，不用 CPU 近似 logits 替换。class/S0/native export order/Source 由 outer/Root 核。

新 manifest schema `N17-fixedDEV-GPU-shard-image-manifest-v1`；新 report schema `N17-fixedDEV-local-center-error-GPU-shard-report-v1`，status `N17_FIXEDDEV_GPU_SHARD_CAPTURE_COMPLETE_PENDING_ROOT_ACTUAL0_AND_INDEPENDENT_CPU`。每 shard 只有完整局部分割预测与实际 captures；qualified、independent_CPU_proof_complete、AP_evaluated 全 false。四个实际 0 shard 和独立 CPU 后才统一 187 图/380 GT/18700 queries 全量 COCO12，禁止局部 AP 平均。

## 本次最小检查范围

一项 stdlib tiny SYN，header/member IDs 与 AST；RED 捕获 reference() 局部遮蔽，最小修正并显式释放末个 CUDA profile、按 8×100 mask 修正 raw 预算，GREEN 原 exit 0（chunk 0ab41e，0.038 秒）。此检查未运行 Torch/CUDA、真实模型、载体序列化或科研；不替代 Root 绑定、原任务终态、实际资源、逐点数值或全量 AP 资格。STOP EDIT，待 Root targeted review/freeze/首 owner122。
