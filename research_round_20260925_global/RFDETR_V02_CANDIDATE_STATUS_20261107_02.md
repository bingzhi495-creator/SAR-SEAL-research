# RF-DETR schema 兼容候选：实现完成，真实验证待执行

2026-11-07。本页记录独立 v02 候选形成时的状态；原 v01 的实际失败记录继续保留。此项属于版本兼容与实验工程准备，不是实例分割性能提升或论文创新证据。

当前 PLAIN/FULL 两臂均已核验完成 33,792/37,050 次正式更新（预定训练预算的约 91.2%），后一段继续运行。尚未到完整端点评价，不能据此给出新的 AP。

## 已知问题与修正范围

v01 已实际安全读取官方 COCO 检查点并构造模型，但原始 state 严格加载缺少 `_kp_active_mask`，没有前向结果。它同时记录一次被拒 `socket.bind`，未记录实际调用栈；不能把静态解释写成已经证明该次 bind 的来源。[原失败与闭合事实](RFDETR_RAW_STRICT_PROBE_ACTUAL_RESULT_20261107_01.md)。

官方固定版本在关闭关键点、schema 为空时生成 `(0,0)` 的 bool 缓冲区，并说明旧检查点可能没有该确定性配置项。[官方构造函数](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/models/lwdetr.py#L253-L256)；[权重版本说明](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/models/weights.py#L195-L228)。

新候选只允许这一缺键，补入模型空缓冲区的独立 clone。其他原始 Tensor 的键、引用、形状、dtype、有限性以及加载后的值均需核查。原始映射及其 `_metadata` 保持。随后对补齐状态执行 `strict=True`，不使用 `strict=False`。

`strict_original_state_load` 始终为 False；只有未来真实检查通过，`strict_completed_state_load` 和 `original_tensor_equality` 才能记录为 True。合成测试中的同名字段是 stub 行为，不能替代真实 Torch 证据。

## 离线诊断边界

新审计候选仍阻止网络作用和子进程启动。只有精确 IPv6 `::1:0`、直接调用帧 `urllib3.util.connection._has_ipv6`、导出函数代码身份和固定源码引用全部一致的一次事件，才标记为“预期但仍被拒的能力探测”。它不会放行 bind。任何其他事件、证据不完整或重复探测都不通过。

实际集成必须校验 `observer.receipt()` 的安装状态及完整事件，不能用空事件列表自行证明已经安装。保障限于 CPython 审计覆盖范围，不声称约束任意 native I/O。

## 普通源码与测试状态

- [空 schema 补齐](code/RFDETR_SCHEMA_COMPAT_PROBE_DRAFT_20261107_02/README.md)：5 项标准库合成测试通过。
- [离线审计](code/RFDETR_SCHEMA_COMPAT_PROBE_DRAFT_20261107_02/OFFLINE_AUDIT_README.md)：5 项标准库合成测试通过。
- [独立 CPU 探针](code/RFDETR_SCHEMA_COMPAT_PROBE_DRAFT_20261107_02/PROBE_V02_README.md)：4 项标准库合成测试通过。

上述 14 项测试未导入真实 Torch、未读取权重或数据、未运行真实模型。普通发布副本在任何 v02 科学绑定之前独立保存。

下一步是在原长训完整段实际结束并核验资源空闲后，另行执行一次有时限的 CPU 312 合成前向检查，保留原始退出与失败，不自动重跑。即使通过，其范围也只是权重兼容与单次输出有限性；576 SAR 输入、真实 mask/类别映射、梯度、优化器、BF16、GPU 成本、完整训练、AP 和跨集均仍需各自证据。研究总目标继续进行。
