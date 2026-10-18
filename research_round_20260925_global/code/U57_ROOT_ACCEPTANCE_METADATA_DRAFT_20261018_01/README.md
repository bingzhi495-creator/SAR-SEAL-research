# U57 Root 接收元数据函数草稿

状态：**UNUSED / NOT_ACTUALLY_ACCEPTED**。只实现纯字典校对与摘要组装；本目录没有实际终端、数据、检查点、训练或接收结果。三个微型标准库测试只覆盖预算单位、终端拒绝和缺少字节证据的拒绝，不代表完整真实验收已通过。未修改 U57 已 STOP 的训练代码。

API：`accept_training(spec, original_terminal, owner_receipt, worker_report, postcheck) -> accepted_summary`。不读文件、不计算哈希、不加载模型/检查点，不导入科学库，不访问 SSH、Ops 或账本。任何字段缺失、身份错绑、非零终止、清理失败、源/输出核验缺失都会抛异常，不能将异常包装为正常资格。

Root 必须先真实观察原任务终止，并完成独立 postcollector。函数只核对 Root 显式传入的观测及 source-closed worker 的记录；无法从字典自行证明操作者是否实际做过这些工作，也不能替代原 owner、worker 与 Root 的执行规则。

## 输入合同

`spec` 是既有 U57 exact15-key spec，操作 `GPU_U57_DGX_ORIGINAL_UPDATE_TRAINING`。保留原 context、全部741 package refs、12组件、三个库、初段无父状态及续段显式 Root-accepted parent。不得用本机 U40、U56 或 SMOKE 父状态替换 FORMAL 父状态。

`original_terminal` exact8 keys：

- `root_observed_tool_exit_code`, `SSH_returncode`, `worker_returncode`：严格整数0，布尔 `False` 不可代替0。
- `reaped`, `owner_original_birth_absent`, `child_original_birth_absent`, `all_owned_processes_absent`：严格 `True`。
- `remaining_owned_group`：空列表。

这些观察必须对应原 owner/child 的真实出生身份；Root postcollector 应已核实际 boot、PID/start-ticks、会话及进程组，不能把数字 PID 不在列表或 owner 自述当作全部证据。

`postcheck` exact13 keys：

| 字段 | 实际 Root 证据 |
|---|---|
| `schema` | `U57-Root-training-postcheck-v1` |
| `spec_ref`, `worker_report_ref`, `owner_receipt_ref` | Root 实际核验的严格 path/bytes/SHA refs |
| `logs` | 原 owner 的 `stdout.log`、`stderr.log` 两个 refs |
| `owner_identity`, `child_identity` | 原 owner receipt 中的出生身份，须与真实终态观察相连 |
| `verified_source_refs` | 完整 `spec.sources`，保持其排序；所有输入、源和续段父状态都已实际核字节 |
| `verified_output_refs` | 两 checkpoint、两 trace、PAIR_COMMIT 共5个 refs，按 path 排序 |
| `source_and_logs_byte_verified` | 全部声明 sources 和两个原日志实际核验后为 `True` |
| `metadata_records_byte_verified` | spec、worker report、owner receipt 实际核验后为 `True` |
| `outputs_byte_verified` | 全部5个输出实际核验后为 `True` |
| `pair_commit` | 真正核过对应字节的原 PAIR_COMMIT JSON 字典；不是重新手填的替代内容 |

owner/worker/pair 的 phase、spec、mode/run、双臂 span、全部 refs、共同输入链、两完整 CPU state 序列化 roundtrip、恢复标志、RNG/backend/source 清理及 CUDA 双0须一致。原始报告不改写。

## 结果与预算单位

返回 `Root-U57-DGX-original-update-training-accepted-v1`，含 `root_accepted=True`、独立 `mode/run_id`、`paired_prefix_advanced=True`。`result` 恰为未来训练主体 `validate_parent_accepted` 消费的五键：

`pair_commit_ref`, `checkpoint_refs`, `trace_refs`, `order_ref`, `paired_checkpoint_prefix`。

body 的 `cursor.global_updates` 是**每臂**已完成图像优化器更新数，不是双臂合计。令 `d=stop-start`：

- 本段实际更新：每臂 `d`，双臂合计 `2d`。
- FORMAL 新增 credit：每臂 `d`，`formal_training_budget_credit=2d`，单位明确为 `aggregate_image_optimizer_updates_across_both_arms`。
- SMOKE 实际更新保留，正式 credit 恒为0；模式/运行身份与正式训练严格分开。
- 例如正式段7→9：每臂新增2次，双臂新增4次；不能按终点9计入18次新增预算。配对累计 cursor 为每臂9。

函数不写摘要、不做 CAS、不修改训练前缀、不去重账本；`ledger_modified=False`。Root 仅在真实验收完成后保存一次摘要并另行做原有唯一接收/计账，重复调用函数不能重复累加 credit。这里的 `paired_prefix_advanced=True` 仅是将来由 Root 保存的接受摘要字段，不代表函数已经修改任何实际前缀。

此接收范围限于实际终态、输入/输出字节及已闭合执行源码的状态/轨迹记录；`ROOT_neural_replay=False`。不证明整段轨迹 bitexact、AP、收敛、结构机制、速度提升或发表资格。
