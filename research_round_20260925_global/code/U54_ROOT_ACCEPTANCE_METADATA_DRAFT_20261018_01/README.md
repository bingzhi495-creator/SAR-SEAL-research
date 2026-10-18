# U54 Root 接收：纯元数据函数草稿

**SOURCE-ONLY / UNEXECUTED ON ACTUAL INPUTS。** 未用于实际 U54 任务，没有生成真实 Root 接收文件。只处理调用者显式给出的字典，不读取文件、Ops、资格链、数据包或权重，不哈希载荷，不导入 ML 库，不更改 U40 前沿或记账，也不运行训练。

```python
accept_validation(spec, original_terminal, owner_receipt, worker_report, postcheck)
```

调用者必须先独立完成真实原任务终态、进程出生身份、全部 Source 与输出字节核验，再传入相应原始字典。本函数只能检查这些证据的结构和关系，不能证明传入布尔值描述了真实外部观察。调用者必须传入实际解析的 spec/owner/worker，不能把测试数据、未运行输入或 worker 的完成标签当作 Root 证据。

`spec` 沿原 U54 的 14 项协议：三个执行源码引用、reader/geometry 两模块、三个明确库文件、manifest、原 ordinal 的 741 包、预锁、两个输出目录及精确 Source 列表。精确 Source 并集为 750 项；这里核引用元数据和成员关系，不重新核包字节，不声称完整 OS/二进制依赖闭包。

`original_terminal` 精确八项：

- `root_observed_tool_exit_code`、`SSH_returncode`、`worker_returncode`：必须是严格整数 0，未知值和布尔 False 不能充当返回码。
- `reaped`、`owner_original_birth_absent`、`child_original_birth_absent`、`all_owned_processes_absent`、`source_and_logs_byte_verified`：必须是 Root 独立确认的 True。

`postcheck` 精确十一项：

| 键 | 内容 |
|---|---|
| `schema` | `Root-U54-explicit-byte-postcheck-v1` |
| `spec_ref` | 真实核验的原 spec 严格引用 |
| `worker_report_ref` | 原 worker `REPORT.json` 严格引用 |
| `owner_receipt_ref` | 原 owner `OWNER_RECEIPT.json` 严格引用 |
| `private_audit_ref` | 原 `IMAGE_CHECKS.jsonl` 严格引用 |
| `logs` | 精确 `stdout.log`、`stderr.log` 两严格引用 |
| `verified_source_refs` | Root 已逐项核验的全部 750 引用，与 `spec.sources` 原列表完全相同 |
| `verified_output_refs` | 前述 spec/worker/owner/audit/两 log 的六项引用，按 POSIX path 排序 |
| `owner_identity` | 与原 owner receipt 中记录相同的原 owner 出生身份 |
| `child_identity` | 与原 owner receipt 中记录相同的原 child 出生身份 |
| `source_and_logs_byte_verified` | Root 已真实完成全部上述字节核验的 True |

出生消失、boot 与进程组消失由外部 Root 实际检查；函数只核其声明和原身份一致、原 owner 已 wait/reap、原 child 自有 session/group 及空组证据。日志可以是 0 字节；完整 741 行审计文件必须非空。所有引用只有绝对 POSIX 路径、字节数及 SHA256 元数据，函数不打开其路径。

完成合同核 worker COMPLETE、body COMPLETE_PENDING_ROOT、十项 worker flags 和五项 body flags、741 attempted/validated/released/recorded、1661 GT、零失败、0..740 原序访问和已释放对象数 0。原报告须明确实际读取 GT tensor；不宣称没有读 GT。还核原 no-model/no-forward/no-optimizer/no-rematch/no-training/AP/DEV/TEST 标志、资源采样、RSS/free RAM、1800 秒和 owner/worker 关系，不重放五字段 proof 或解码包。

只有全部检查通过，才返回 U57 使用的 14 项 `Root-U54-DGX-full-FIT-accepted-v1` 摘要，状态 `ROOT_ACCEPTED_ALL741_CPU_VALIDATION`，包含 `dataset_indices`、manifest/worker/owner 引用和原八项 terminal。`package_refs_sha256` 对 `spec.package_refs` **保持原 ordinal 顺序**，使用 `json.dumps(sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('utf-8')` 后 SHA256。这是引用 JSON 摘要，不是再次哈希包载荷。Root 仍须另行独占保存、核验返回的真实接收 JSON；此 helper 不写文件。

两个微型标准库测试仅覆盖未知/非零/未消失终态的拒绝、ASCII canonical 包引用摘要，以及拒绝无真实 Root postcheck 的 worker 完成声明。未构造完整人口成功资格 fixture，没有实际 U54 资格、像素、张量、资源执行或训练验证。
