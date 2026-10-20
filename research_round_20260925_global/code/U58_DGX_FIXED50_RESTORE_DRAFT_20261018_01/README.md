# U58：独立 U57 DGX 固定端点 CPU 模型恢复草稿

**SOURCE-ONLY / UNBOUND / UNEXECUTED。** 本目录仅完成标准库合成元数据、注入 tensor/model stub 及 AST 检查。没有导入 Torch/NumPy、读取真实数据或权重、执行实际网络恢复/forward、SSH、训练或评价；未创建执行资格。真实 DGX `strict restore`、模型构造、内存和运行完整性均待 Root 所有的实际单叶验证。合成测试通过不代表真实网络已验证，也没有 AP 或发表级结论。

这是独立 U57 的辅助函数，**不修改或替代 U44 的 Windows U40 恢复流程，也不修改 U57 训练**。运行环境仅为 Linux/aarch64、Python 3.12.14、Torch 2.10.0+cu130、NumPy 1.26.4、冷 CUDA/CPU FP32。不能把本地 U40 的 2816 次进度接入此独立运行，不能从当前 FORMAL 中途恢复并提前评价。

## 两个明确分开的入口

```python
receipt = restore_arm(
    torch, np, control, adapter, arm, endpoint_metadata, fresh_initialization_receipt,
    modules=modules, source_refs=source_refs, report={},
)
```

生产入口 `restore_arm` 仅接受实际 Root 接收、配对 cursor 为 `{global_updates: 37050, completed_epochs: 50, within_epoch: 0}` 的独立 U57 **FORMAL E50** 端点。末段起点不硬编码；接受记录、spec 和配对提交中的实际末段必须一致。单臂完成、worker pending 记录、单独 `qualified=True`、FORMAL 中途或 SMOKE 均不能代替该端点。

`restore_smoke_arm` 是独立技术入口：仅接受相同来源规则下实际 Root 接收的 **SMOKE2** 配对端点和原 `[54, 710]` 两次顺序。它拒绝所有 FORMAL 端点，包括 FORMAL 中途和最终端点；不提供正式训练额度、AP、模型选型或正式人口资格。不存在可将 FORMAL 中途放行的公开 mode 参数。

`validate_final_endpoint` / `validate_smoke_endpoint` 是上述两个端点的纯元数据门禁。`validate_loaded_checkpoint` / `restore_loaded_model` 是已经注入 CPU 容器的技术辅助接口，仍仅接受 FORMAL E50；它们不承担实际文件、库或模块闭合，不能替代生产入口或 Root 接收。

## 调用方必须提供的实际对象

- 真正且已准入的 Torch/NumPy 库、活动的资源 `Control`（`closed` / `check`），并闭合本新 helper 和实际运行库。Python 模块类型与版本检查不等于依赖二进制闭合。
- 原 `shared_initialization.build_pair` 在冷 CPU 上构造的 PLAIN/FULL 原 adapter，以及该次原构造返回的 fresh receipt。helper 不构造模型，不重新初始化或清零训练头。receipt 相等只验证元数据一致性，**不能证明调用方真的运行了原构造或注入了真实网络**；该事实仍须外层实际执行证据支持，返回值明确保留 `caller_build_pair_construction_proven=False`。
- `endpoint_metadata` 精确为 `refs={qualified,spec,pair_commit,order}` 加四个同名实际文档。全部 ref 仅接受严格绝对规范 POSIX `{path,bytes,sha256}`，拒绝 Windows 路径、反斜线、父目录跳转及归一化歧义。
- `modules/source_refs` 精确十二角色：`training_state/training_entry/u36_helpers/u39_training_ops`，以及八个原构造角色 `shared_initialization/full_legacy/full_dsconv/full_head/full_feature/plain_legacy/plain_feature/instance_adapter`。所有 ref 必须与实际已接收 spec 一致且满足固定原 SHA；实际注入模块的文件必须闭合为该 ref。八角色范围不允许加入替代构造模块。原 FULL 的 DSConv/CoordConv 依赖对象也须一致。

顺序为：调用方原 build_pair 冷 CPU 构造 → 两臂分别 CPU restore → 后续已准入的预测入口自行管理 GPU。现有 DEV 缓存路线不需要新增 teacher forward；新图捕获路线另行验证。成功后继续持有已恢复的 CPU 模型供未来已准入的窗口使用；helper 不替换、销毁或转移 adapter，不提供数据遍历、teacher、GPU、runner 或自动重试。

## 容器、签名与恢复边界

生产包装经 `Control.closed` 重核四个实际 JSON 的字节、大小、SHA 与给定字典相等，闭合原代码引用，并核同一 run/source/recipe/environment/backend/order、实际零退出和原 owner/child/reap 完整结束记录。它不读取 FIT 图像、标注、缓存、前一段 checkpoint 或 trace。

本臂 checkpoint 仅一次 `torch.load(map_location='cpu', weights_only=True)`。完整 U57 checkpoint **包含 optimizer 与四路 RNG 容器**；helper 如实报告这些 CPU 载荷已加载。它先用原 U57 `_signature` 及 U36 `_digest` 计算整个容器的 `complete_checkpoint_CPU_state_signature_sha256`，与配对提交的完整 CP 签名比较。模型单独计算 `model_CPU_state_signature_sha256`。二者不可互换，模型签名不能冒充完整 CP 签名。

检查包括原 schema、arm、实际 cursor、完整 identity、原顺序摘要、上一已接收 pair/arm ref、原初始化回执、optimizer/RNG 容器结构、模型完整名称、dtype/shape/CPU/finite、有效/失活参数名、无残留梯度及完整模块树。整数 BN 状态保持原 dtype，浮点模型状态须为 FP32。完整容器签名同时覆盖 optimizer、RNG、模式和血缘，但 **不创建 optimizer、不调用 optimizer.load_state_dict、不恢复任何训练 RNG**。

只执行 `adapter.load_state_dict(saved['model'], strict=True)`，随后原 `_equal` 和独立 model 签名核完整模型、BN、失活状态和训练头逐项相等。最后 `eval().requires_grad_(False)`，确认全部模块 eval、参数 frozen/no-grad、模型签名不变及 CUDA 仍冷；外部 Torch CPU、NumPy 和 Python RNG 须保持不变。完整模型树来自调用者的原实际 adapter，metadata/stub 检查不能替代这个事实。

成功返回同一个报告字典，status `RESTORED_CPU_ONLY`；`qualified=False`、`numerical_runtime_qualified=False`、`AP/DEV/TEST=False`、`population_qualification=False`。包含实际端点/CP/source refs、两种签名、参数及模块清单、环境、cursor、模式、冻结和 RNG 检查。它是 pending 技术证据，不能凭报告布尔值自行放行后续科学运行。

失败保留原错误和部分报告，不静默回退或重试。`strict load` 中途失败可能已部分修改 adapter，调用方必须丢弃该对象。finally 释放本地 checkpoint/RNG 引用并 GC，报告只说明 helper 释放了引用，**不保证 OS 已回收全部 RSS**。外层继续负责原资源上限、实际进程结束、产物接收和科学资格。

## 已完成的有限检查

10 项标准库合成测试覆盖固定 FORMAL/SMOKE 门禁、跨文档与来源替换拒绝、完整 CP/model 签名区分、重新签名的缺失 optimizer/RNG 拒绝、模型名称/有限性/模块树/血缘、strict CPU 恢复及训练头保留、冷 CUDA/无梯度门禁和两个包装入口。包装测试明确 mock 文件/模块闭合，只验证编排，不证明真实文件、实际 Torch 或网络有效。

本轮保留的实际 RED→GREEN 包括：缺独立 SMOKE 接口、缺 strict model 恢复、缺十二角色源门禁、缺生产包装，以及空 optimizer/RNG 容器被接受的两个拒绝缺口。全部测试始终只使用标准库、普通未绑定 U57 元数据函数和注入 stub；AST 检查不执行模型。
