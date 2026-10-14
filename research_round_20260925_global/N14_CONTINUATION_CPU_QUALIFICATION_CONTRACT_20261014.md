# N14 continuation CPU v2：UNBOUND 字段合同

仅开发草稿，未执行、未绑定、无新 Q。仅读指定普通 stage8 CPU、continuation GPU v2 与 memory module 草稿；未读取正在运行的 Source10014 或任何实际 RAW/SPEC/Q/cache/权重/输出。ROOT 仍须实际 GPU terminal/reap/birth absence、完整 SOURCE 审计、metadata/外部 ABI 审阅后绑定执行。

新脚本：`N14_CONTEXT_MEMORY_STAGE_CONTINUATION_CPU_QUALIFICATION_UNBOUND_DRAFT_20261014_02.py`。

## CPU spec

- `schema = ROOT-N14-LOG-E24-memory-paired-stage-continuation-CPU-spec-v2`。
- `(first_stage_epoch,last_stage_epoch)` 为 `(9,16)` 或 `(17,24)`；`fixed_stage_epochs=24`。本叶8轮／13288 paired rows；累计分别16轮／26576或24轮／39864 updates。
- `expected_normal_RAW_count == normal_RAW_count`，锁定完整最新祖先数。下一 stage16 CPU 为98；正常无额外叶时 stage24 CPU 为100。脚本保留完整最新 prefix，而非退回95。`terminal_identity_pins` 必须恰好 normal＋旧failed8；独立 `additional_preserved_independent_CPU_failure={RAW_ref,phase,pin,returncode:1}` 保持第九失败不合并。`total_failed_RAW_count=9`，`total_terminal_identity_pin_count=normal+9`。
- `sources` 完整闭包；`RAW_inputs`、旧failed8字段沿用。所有 normal、failed8、独立failed9 的实际 RAW 与 pin 在首 SHA／科学模块导入前检查。
- 当前 `training_GPU_source/spec/report/RAW/admission` 五项 refs。当前 GPU spec 的 RAW 为 CPU 的 `[:-1]`，末尾新增当前实际 GPU owner。
- 复制当前 GPU 的 `carrier_spec`、`carrier_evaluation_source`、`carrier_curve_qualified_summary`、`memory_source`、原训练 `protocol`、`seed`、`module_seed`、`order_seed`；不将新 CPU 数值协议替换原训练协议。
- 自身 `entry_source,phase,CPU_admission_file,Q_path,details_path,deadline_seconds=1800,automatic_retry=False`。
- `CPU_resource_fields` 的五个实际 admission JSON 非空键路径：`affinity->[0,1]`、`thread_limit->2`、`deadline_seconds->1800`、`automatic_retry->False`、`cuda_visible_devices->''`。真实环境 `CUDA_VISIBLE_DEVICES=''`、四项数值 thread env='2'、Torch保持 CUDA 未初始化；ROOT 独立控制 whole GPU inactivity。

继承实际绑定 continuation SOURCE 的 `parent_resume_gate(gpu_spec,closure)`，保留父 GPU/CPU Q、数字协议和原 `stage_origin_START`。CPU own_gate(cpu=True) 后独立核 parent GPU/CPU/current GPU 的 owner joins，索引 `normal-3 / normal-2 / normal-1`，GPU尾参`'0'`、CPU空串。再显式保留最初 GPU95/CPU96 的原 prefix 和 source/spec/admission joins。原 E24 endpoint_gate 只核共同暖起点 provenance，不将新42698状态送进旧40497断言。

## 新 CPU 数值协议

`CPU_endpoint_numeric_protocol == endpoint_numeric_source`，都是**同一个新 JSON ref**，必须在 sources 中，绝非 `.py` SOURCE。`CPU_numeric_fields={'atol':['atol'],'rtol':['rtol']}`，`endpoint_comparison={'atol':1e-4,'rtol':1e-4}`。

JSON 严格六字段：

```json
{
  "atol": 0.0001,
  "rtol": 0.0001,
  "locked_before_CPU_execution": true,
  "locked_after_GPU_training": true,
  "scope": "same cached input endpoint mean compatibility only, not AP/statistical equivalence",
  "bound_completed_GPU_report": "当前training_GPU_report的完整{path,bytes,sha256}对象"
}
```

最后一项实际值必须是 ref 对象，不是示例字符串。协议绑定**当前** stage16或24 report，不能沿用 stage8／父 report。明确 GPU完成后、CPU数值执行前锁定，不称 GPU 前预注册或训练 efficacy gate；不改冻结 GPU 或原 N13 门。

## Checkpoint、trace 与端点

当前 report/cp.meta 为 continuation v2：2个FP32 finite状态，base40497／addon42698；当前new trained state严格加载。保留原暖初态 digest、零addon初始化、`stage_origin_START`、original behavior/proofs。当前leaf `optimizer_reset_both=False`，cp仍记录stage1 `optimizer_reset_at_stage_start=True`，另 `optimizer_reset_this_leaf=False`。leaf开始 `resume_state_digests` 必须与**实际父 trained checkpoint**一致，不能与本叶结束状态比较。

每个实际 AdamW parameter step 精确等于1661*end，moments逐实际 module parameter traversal核 shape/dtype/finite/非负second moment，activity为累计 present1661*end、nonzero前缀单调且本叶增量至多13288。完整私有Generator从原 order_seed复现16／24个741排列、hash与最终private state。

`report.trace_chain` 恰为2／3个独立 refs，等于父chain＋本叶trace，路径唯一；cp.meta必须与report一致。每个trace恰好8epochs、13288**两臂同row**记录；全链epoch1..end、stage_update1..1661*end，逐真实缓存 image/GT/annotation/query identity核验。每epoch1661 paired rows，online math.fsum均值与对应GPU history exact。此检查不重放optimizer/gradient数值轨迹。

从同一qualified FIT741/1661 LOG缓存只测当前最终状态，分别核2个arm的完整FIT mean BCE。GPU无逐instance endpoint loss，不能把online trace当final-instance比较。保留已训练MEMORY首真实FIT实例的1e-4置换检查、非零projection与LOG外native exact；范围仅单实例。无DEV/AP/GT decoder、不重建teacher/backbone。

## Q ABI

`details` 是 `{path,bytes,sha256}` 文件ref；`parameters={'BASE_CONTINUATION':40497,'MEMORY_ADDON':42698}`；`new_stage_epochs=8`始终指本叶；`stage_epochs_completed=end`与`paired_updates_per_arm=1661*end`指累计。Q完整保留五项当前GPU refs、checkpoint refs、原训练protocol及独立新CPU协议ref/时序；`current_owned_CPU_terminal_self_certified=False`，ROOT还须核实际CPU RAW0及SOURCE。

stage16 必须输出：

- schema `N14-stage16-training-endpoint-CPU-qualification-v1`；status `ROOT_N14_MEMORY_PAIRED_STAGE16_ENDPOINT_CPU_QUALIFIED_V1`。
- `paired_updates_per_arm=26576`。
- `all_AdamW_parameter_steps_exact26576=True`。
- `complete_paired26576_trace_order_identity_online_means_exact=True`。
- `private_sixteen741_orders_and_final_state_exact=True`。
- `final_stage16_full_FIT_mean_BCE_CPU_GPU_compatible=True`。

stage24对应schema/status中的24、updates39864，以及 `all_AdamW_parameter_steps_exact39864`、`complete_paired39864_trace_order_identity_online_means_exact`、`private_twentyfour741_orders_and_final_state_exact`、`final_stage24_full_FIT_mean_BCE_CPU_GPU_compatible`。

继续保留 `per_instance_GPU_CPU_endpoint_losses_compared=False`、`optimizer_trajectory_independently_replayed=False`、`full_FIT_permutation_behavior_checked=False`、`capacity_confound_not_yet_controlled=True`、无AP／因果／部署／创新结论。当前稿完成不等于实际训练或资格已成功，也不属于 ROOT 当前同步六文件的完成证据。
