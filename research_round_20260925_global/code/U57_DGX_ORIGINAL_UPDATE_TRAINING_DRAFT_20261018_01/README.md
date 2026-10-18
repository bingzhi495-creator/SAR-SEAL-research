# U57：独立 DGX 原更新训练草稿

**SOURCE-ONLY / UNBOUND / UNEXECUTED。** 本目录实现新的 DGX 训练协议，不延续本机 U40 的 2816 更新，不采用 U52 候选更新，也不把 U50 的短试点计入正式训练。全 741 图 U54 CPU 验证及其 Root 接收尚未完成；训练函数在任何模型构造前要求真实的完整输入接收摘要，不能用此说明、待执行计划或 worker pending 报告代替。

本目录 `training_body.py`、`test_metadata.py`、本说明由 n23 负责；`entry.py`、`linux_owner.py` 及其独立测试由 n22 负责。没有执行训练、加载权重或实际数据，没有生成实际资格、配对检查点或评价结果。

## 调用接口

```python
run_training(torch, np, c, *, modules, source_refs, context, segment,
             prelock, body_source_ref, output_directory, report)
```

外层注入真实库、已准入控制器、已加载的 12 个模块及其严格文件引用。模块为原 11 个数值组件加 U54 `portable_reader`；不会在函数内重新加载或替换数值模块。源码 SHA 固定于 `SOURCE_PINS`。`report` 必须是空字典，原对象用于保留成功或失败的实际进度。`output_directory` 是外层明确提供、尚不存在的独占 `SEGMENT` 目录；不由源码位置猜测，不覆盖旧输出。

`context` 精确六项：

- `manifest_ref`：全 741 个紧凑 CPU 包的原 metadata 清单。
- `order_ref`：用原 Python 3.8 生产函数新生成的正式 50×741 顺序。
- `order_generation_ref`：新生成行为的独立元数据回执。
- `U54_validation_ref`：实际 Root 接收的全输入验证摘要。
- `U54_validation_sources`：该摘要对应的 `worker_report_ref`、`owner_receipt_ref`。
- `parent_pair`：首段为 `None`；后段明确给出 `qualified_ref`、`pair_commit_ref`、两臂 `checkpoint_refs` 和两臂 `trace_refs`。

`segment` 精确为 `{start, stop}`。使用 `training_contract(mode, run_id)` 生成完整预锁，`validate_prelock` 不接受隐式默认或改过的配方。`backend_settings` 是入口和 worker 共用的真实设置观测函数；`completed_training` 仅核已执行 worker 的完成合同，不接收原任务终态，不推进 Root 配对前沿。

## 数值路径和独立预算

使用原 U34 `build_pair` 的共同 104 项初始化（68 个参数张量、36 个 BN buffer），保留不同的完整 FULL/PLAIN 主干；共同状态不表示两臂容量相同。每次输入直接调用原 U39 `_update`：FP32、单图 400 分支、完整匹配 G、query chunk 1、每 query 权重 1/G、原网格 BCE+softDice、每图一次共享 encode 和一次 AdamW 更新。仅训练 encode 使用原 `save_on_cpu(pin_memory=True)`。没有 AMP、VJP、重计算、scheduler、新模块或语义头训练。

AdamW 固定 lr=3e-4、weight_decay=1e-4、betas=(.9,.999)、eps=1e-8、foreach=False、fused=False。有效参数按名称排序映射至 optimizer；原声明失活参数保留分配及 `requires_grad`，其梯度仍必须为 None，旧语义头调用为 0，相关 BN buffer 保持。

FORMAL 从全新状态 0 开始，每臂 E50=37050 次更新。SMOKE 使用不同 `run_id`，只允许 0→1、1→2 两个接续片段，顺序固定 `[54,710]`，两臂共四次真实更新；不重放基线、不按 loss 选状态，也不计正式预算。54 的 G=1、710 的 G=27；全人口 metadata 最大 G=29 在 709，短 smoke 不能证明全量训练资源通过。

新环境明确锁定 Linux/aarch64、Python 3.12.14、Torch 2.10.0+cu130、NumPy 1.26.4。保留 TF32 关闭、cuDNN deterministic=True、benchmark=False，并明确 `torch deterministic_algorithms=False, warn_only=False`。环境要求 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，这不构成 strict 算法保证，也不宣称完整训练轨迹 bitexact。它与 U56 strict 诊断是不同协议。

CPU affinity 0/1、线程 2、allocator 4 GiB、RSS≤16 GiB、可用 RAM≥16 GiB、1800 秒不变。首次 CUDA 在 CPU 头构造完成后初始化，实际 allocator fraction 在模型和输入转入 GPU 前设置。GB10 共享内存没有被冒称为旧 Windows/WDDM 整卡 8 GiB 合同；whole-GPU 观测及准入仍需未来真实 owner 明确报告，不由此函数补造。

## 顺序、输入和连续恢复

顺序仍采用原 `U40-private-epoch-order-v1` **文件格式**，但必须是原 CPython 3.8 新生成的新文件。校验生产者 3.8、每轮完整 0..740、50×741 长度、生产者初末 RNG 结构和完整元数据摘要。消费端 3.12 不重新 shuffle，不要求生产者版本等于消费端；相同算法新生成的内容可能与旧顺序相同，不凭摘要相同否认新的生成行为。

生成回执为 `U57-independent-Python38-order-generation-v1`，精确九项：`schema/root_created_new/mode/run_id/producer/generator_source_sha256/order_ref/order_metadata_sha256/no_old_training_state_read`。FORMAL 的 run_id 与预锁一致；SMOKE 可引用正式新 order 的生产谱系，但实际只按预锁的两个索引执行。

U54 Root 摘要必须是 `Root-U54-DGX-full-FIT-accepted-v1`、`ROOT_ACCEPTED_ALL741_CPU_VALIDATION`，包含真实原任务终态、741/1661、零失败、`dataset_indices=list(range(741))`、manifest 与 741 包引用摘要和关联 worker/owner 引用。`package_refs_sha256` 对原 ordinal 顺序的包引用列表使用 `json.dumps(sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('utf-8')` 后取 SHA256；该单项与全局 order/state 摘要各遵循其已锁定规则，不混用。U54 reader 每次只加载一图并完成原包 proof；保留原 GT/query 顺序和 RGB/geometry。训练路径不需要 F，不把此紧凑下游接口称为原含 F reader 的完整替代。

每臂保存完整 CPU model（含 BN/失活状态）、完整 AdamW state 与设备映射、模块模式、Python/NumPy/CPU/CUDA RNG、共同初始化谱系、原输入和请求顺序双链、上一 checkpoint/pair。NumPy RNG 使用 primitive list，不把 NumPy pickle 放进包。保存后释放第一份 CPU payload，再以真实 `torch.load(..., map_location='cpu', weights_only=True)` 重载校验全状态摘要，然后才原子改为最终文件名。此 roundtrip 是状态序列化检查，不是独立重演优化器。

接续必须有同一 U57 mode/run_id 的真实 Root 接收摘要 `Root-U57-DGX-original-update-training-accepted-v1`，且精确关联 pair、两 CP、两 trace、order 和起始 cursor。先构造 CPU skeleton，再逐项恢复实际训练状态、optimizer 名称映射/步数/设备和四类 RNG，绝不清零已训练读出头或拼接本机 U40 状态。两臂在同一 span 使用完全相同实际输入链；每臂结束与最终都要求 CUDA allocated/reserved 双 0。

## 输出、失败及证据边界

每段独占输出两臂 `.pth`、两份逐更新 JSONL trace 和一个 `PAIR_COMMIT.json`。pair schema 为 `U57-DGX-paired-checkpoint-commit-v1`，仅 `PAIRED_CHECKPOINTS_WRITTEN_PENDING_ROOT_ACCEPTANCE`；direct `checkpoint_refs/trace_refs` 与完整 nested 记录一致。报告 schema 为 `U57-DGX-original-update-training-segment-report-v1`。完成状态为 `SEGMENT_COMPLETED_PENDING_ROOT_ACCEPTANCE`，顶层 `actual_trace_sha256` 与 pair 和两臂相同。

只有两臂完整返回、源检查、RNG 恢复和 GPU 清理均通过才写 pending pair。任一更新或保存失败会保留部分计数和错误，不写成功 Q，不自动重训、不复用单臂部分状态。`qualified=False`、`paired_prefix_advanced=False`、`formal_training_budget_credit=0` 始终由 worker 保持；正式原任务终态、资源接收与配对前沿推进均属于未来 Root。无 AP/DEV/TEST、权重发布或效率/收益结论。

当前仅三个微型标准库元数据测试通过：生产/消费 Python 版本分离、SMOKE 两段范围、拒绝跨旧协议/跨 run/profile 的模型父。另有 AST 检查；未导入 Torch/NumPy，没有真实完整顺序大 fixture、模型、梯度、optimizer、checkpoint 或数据验证。这些测试不能代替 U54 全人口真实 CPU 验证及后续 owned SMOKE/FORMAL 执行。
