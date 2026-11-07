# RF-DETR 旧检查点空 schema 补齐候选

2026-11-07。`complete_legacy_schema_state(torch, model, original_state)` 返回新的有序浅副本和 JSON 安全的说明。原模型仅缺顶层 `_kp_active_mask`，且关键点明确关闭、schema 为空时，函数补入模型中该缓冲区的独立 clone。原有 Tensor 对象保持同一引用，原映射及其 `_metadata` 均保留。返回映射的 `_metadata` 与原对象相同，调用方也应将其视为只读。

该函数验证缺键集合恰为 `{'_kp_active_mask'}`、无多余键，并检查现有 Tensor 的 CPU 设备、形状和 dtype 与模型匹配。目标必须由 `get_buffer` 返回，属于普通 Tensor，CPU、bool、精确形状 `(0, 0)`、元素数为零。关键点标志须为 `False`，schema 须为空 list，关键点头须为 `None`。其余情况抛出异常。调用函数需由外部提供 Torch 和已构造的模型，导入本文件只导入标准库。

官方 RF-DETR 固定提交 `639eac4555ceb89ef399ccda3fde716bbcfe34cb` 的配置版本为 `1.12.0`。`LWDETR` 在第 204 行注册该缓冲区，`_create_kp_active_mask` 第 253–256 行对空列表返回 `torch.zeros(0, 0, dtype=torch.bool)`。官方权重加载说明明确其为确定性的 schema 缓冲区，旧检查点可能缺少该键。 [缓冲区构造](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/models/lwdetr.py#L175-L204) · [精确空形状](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/models/lwdetr.py#L253-L256) · [官方版本兼容依据](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/models/weights.py#L195-L228)

实际 HTTP 获取上述两份固定提交源文件，其字节数和 SHA256 与普通静态副本一致。`lwdetr.py` 为 51,341 字节，`248faf15d673bc18f51f60bb55245f64eee40eb9c8adacea398aebdadad2243b`。`weights.py` 为 39,948 字节，`162a354b54d8e73ccdac77960bea1d9db46ade68847f1eaed289f541b34e0c14`。版本标签 `1.12.0` 的这两份原始文件也分别得到相同哈希。

未来独立探针必须对返回的 `completed_state` 使用 `load_state_dict(strict=True)`，随后逐键验证所有原检查点 Tensor 的值、形状、dtype、设备及有限性，并单独验证补入空缓冲区。调用方还须明确构造配置中的关键点设置。函数本身不加载模型，因此 `strict_original_state_load`、`strict_completed_state_load` 与 `qualified` 都保持 `False`。执行资格由外部流程判定。

原 v01 对原始 state 的 strict 加载失败继续保留。此候选用于窄范围的版本 schema 兼容，尚无真实 Torch 验证或 forward 结果，也不产生方法性能收益。

合成标准库测试覆盖唯一允许的补齐、原映射和 `_metadata` 保持、其他缺键及多余键、启用关键点、非空或错误设备及 dtype 的缓冲区，以及其他 Tensor schema 差异。测试使用显式 stub，以下命令仅运行合成控制逻辑。

```text
python -B -m unittest -v test_schema_compat
```
