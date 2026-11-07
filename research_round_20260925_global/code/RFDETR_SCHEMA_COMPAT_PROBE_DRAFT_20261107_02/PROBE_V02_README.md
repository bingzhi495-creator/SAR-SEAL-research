# v02 单张合成输入探针

`probe_v02.run_probe(checkpoint_path, seed=5201)` 为独立 CPU 候选，调用已冻结的 `schema_compat.complete_legacy_schema_state`。模块导入只依赖标准库和该同目录 helper，Torch 与 RF-DETR 在函数内按需导入。未来 worker 须先用已核验来源注册 `schema_compat` 同名模块，再加载本探针。

该候选沿用 v01 的 CPU 安全加载。`torch.load` 始终使用 `weights_only=True`、`map_location='cpu'`，至多在显式允许 `argparse.Namespace` 与 `types.SimpleNamespace` 的局部作用域内重试一次。函数将 CPU 线程数设为 2，仅固定 CPU 默认生成器的随机种子，并在 `finally` 恢复原线程数和 CPU RNG。

模型仍采用原始 COCO SegNano、90 类、312 分辨率和原分类头，关闭编译、CUDA graphs 与关键点。`use_grouppose_keypoints=False` 和 `num_keypoints_per_class=[]` 是官方配置字段。低层构造用 `force_no_pretrain=True`。 [配置字段](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/config.py#L733-L739)

探针先检查原 state 的每个 Tensor 为有限 CPU 数值，再通过冻结 helper 生成浅副本并补齐唯一允许的空缓冲区。它只对 `completed_state` 调用 `load_state_dict(strict=True)`。只有返回的 missing 与 unexpected 都为空，顶层 `strict_completed_state_load` 才设为 `True`。之后核对原映射的键顺序、Tensor 引用和 `_metadata` 引用保持，检查已加载状态中每个原始键及所补空缓冲区的形状、dtype、CPU 设备、逐值相等及有限性，全部通过才设 `original_tensor_equality=True` 和 `strict_state_equal=True`。验证后释放原检查点映射、浅副本与逐项引用。

随后模型执行原生 eager、eval、inference-mode forward。输入是一张全为 0.5 的合成图，经 ImageNet 均值和标准差归一化。输出须为 CPU float32 有限 Tensor，logits 为 `[1,100,91]`，boxes 为 `[1,100,4]`，masks 为 `[1,100,78,78]`。任何不符都会返回失败记录。返回结果仅包含形状、计数、布尔值、错误阶段和耗时等 JSON 安全内容。

`strict_original_state_load` 永为 `False`，`qualified` 永为 `False`。顶层 `schema_completion` 保留 helper 的完整元数据，其中 helper 自身的 `strict_completed_state_load=False` 只记录 helper 没有执行加载。真实加载是否成功由探针顶层同名布尔值记录。

v01 的原始 strict 失败继续保留。v02 尚无真实 Torch 或模型运行证据，版本兼容本身没有方法性能结论。独立执行、来源绑定、离线审计和资源资格由外部流程负责。

`test_probe_v02.py` 复用冻结 v01 `test_probe.py` 的纯标准库 stub，仅扩展空 schema 元数据。测试需要保留相邻目录 `RFDETR_CPU_MODEL_PROBE_DRAFT_20261107_01`。这四项合成测试覆盖补齐状态的 strict 成功路径、源映射保持、其他 schema 错误、加载后原 Tensor 等值或有限性失败、受限安全加载、错误输出拒绝与线程和 RNG 恢复。首项测试与等值拒绝项分别经过实际 RED → GREEN。测试中的 stub 不计算真实 Tensor 数值。

```text
python -B -m unittest -v test_probe_v02
```
