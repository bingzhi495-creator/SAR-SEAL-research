# U44 固定 E50 单臂 CPU 恢复薄函数

**普通未绑定；checkpoint、Tensor、NN、资源数值均未执行。** 仅两项标准库合成元数据测试 RED→GREEN 和 AST；不授科学资格，不改变原训练/评价规则。没有 teacher 构造、人口、runner、模型构造、优化器、AP 或接收框架。

```python
receipt = restore_arm(
    torch, c, adapter, arm, endpoint_metadata, fresh_initialization_receipt,
    modules=modules, source_refs=source_refs, report={},
)
```

`adapter` 是调用者已经用原 U34 `build_pair` **冷 CPU 构造**的 PLAIN 或 FULL 原对象。顺序为冷 CPU build_pair → 逐臂 CPU restore → teacher 首次 CUDA；成功后继续持有两臂 CPU 模型供小窗口轮换，不能每图重读 checkpoint。本函数不 pop、替换或销毁 adapter，不转 GPU。原 `_partition` 在冻结前检查有效/无效边界；成功后设置 eval/无参数梯度，禁止再次拿同一已冻结 adapter 重走构造分区路径。

`endpoint_metadata` 为原 `prediction_core.validate_final_endpoint` 的四引用与四个实际 JSON 字典：`refs={training_qualified,pair_commit,preparation,order}` 及各同名文档。函数直接复用该原检查，不信任单独 callback 的 qualified=True；重新经 `c.closed`/原 `_json` 核文档字节与给定字典，检查完整 E50、每臂37050、配对提交、原17训练源和共享顺序。只闭合这些代码引用，不读取 FIT 图像/标注/载体。

`modules/source_refs` 精确四项：`prediction_core/training_state/u39_training_ops/u36_helpers`。实际模块文件、原 SHA 与最终训练 proof 的原 ref 相联，不能换同名辅助代码。调用方还须闭合本新源、真实 Torch/NumPy/依赖与资源控制。此函数沿原 Python3.8.17/Torch2.0.1+cu118、CPU FP32；原 pair runtime 的 NumPy1.23.5身份保留，真实 NumPy 运行库版本仍由外层验证。

恢复逻辑直接沿已发布 `U40_FIXED50_DEV_PREDICTION_DRAFT_20261017_01/prediction_core.py:258–288`：本臂 checkpoint 仅一次 `torch.load(map_location='cpu')`，核 arm/cursor/source-recipe-order、有效/无效参数名、原初始化回执、完整 CPU FP32 finite state（含 BN/失活状态），strict load，再用原训练 `_equal` 核逐项相等和完整模块树。保留训练过的残差头；没有清零末层、forward/backward 或优化器创建/恢复。

完整 checkpoint 会反序列化其 **CPU optimizer 字典**；本函数如实记录加载该容器，随后 finally 释放 `saved` 引用并 GC，不声称未加载该字典、不检查/恢复优化器或保证 OS 已收回全部 RSS。失败保留原错误文本，清除异常中的 tensor 引用；adapter 可能部分写入，调用方须丢弃，不得自动重试、静默回退或当作成功模型。原完整初始化回执只与 checkpoint 精确比较；回执返回其元数据 SHA，避免携带冗余完整记录。

返回同一 `report` 字典，schema `U44-external-fixed50-CPU-arm-restore-pending-v1`、status `RESTORED_CPU_ONLY`、qualified=False。包含 arm/seed/fixed_epoch/updates、endpoint四refs、本臂与配对CP refs、原17训练源/4恢复源、初始化回执SHA、`model_state_sha256`、有效/无效名与模块名、eval/frozen/no-grad标志、CPU RNG不变、CUDA未初始化、加载一次与payload释放标志。parameters manifest 是冻结前原 `_partition` 的记录，后置 all_parameters_frozen 明确描述恢复完成状态。

状态摘要直接用原 `prediction_core._state_digest(u36_helpers, adapter.state_dict())`。U36 `_digest:56–62` 先 detach/CPU/contiguous，再按 dtype、shape 与 NumPy数值字节计算；**不含 device**。未来窗口可用同一函数实际重算并与 `receipt.model_state_sha256` 比较，同时核 arm/endpoint/source refs与 eval/no-grad；不能仅凭该 pending 回执的布尔值或 qualified 字段放行。CPU/GPU往返数值一致性、实际模型恢复、内存峰与科学运行完整性仍待未来 Root-owned 单叶执行。
