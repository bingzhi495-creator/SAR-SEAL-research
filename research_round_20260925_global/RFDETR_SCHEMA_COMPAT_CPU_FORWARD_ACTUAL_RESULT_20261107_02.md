# RF-DETR-Seg Nano：限定 schema 兼容与真实 CPU 前向结果

2026-11-07。Root 已核验真实原始调用、SSH、worker 均退出0，独立进程普查为空、选定来源复核通过，资源已闭合。此结果只支持下述 COCO Nano312 CPU 接口范围，未形成 SAR AP、训练效果或论文创新证据。

## 结果与范围

使用官方 RF-DETR 1.12.0、固定提交 `639eac4555ceb89ef399ccda3fde716bbcfe34cb` 和官方 Nano mask 权重，在 CPU 两线程、FP32、eval/inference_mode 下，对一张常数合成输入执行一次真实前向。没有读取 SAR 图像、GT、构造优化器或训练，也没有初始化 CUDA。

旧检查点仅缺官方空 `_kp_active_mask`。独立兼容层补入 CPU bool `(0,0)` 空缓冲区后，`strict=True` 加载通过；544个原始 state Tensor 逐项形状、dtype、值和有限性检查通过，补齐后为545个 Tensor，原 state 元素总数33,577,475。元素数包含 state 中的缓冲区，不能当作可训练参数数量。

`strict_completed_state_load=True`、`original_tensor_equality=True`；`strict_original_state_load=False` 继续保留，不能改称原始 schema 直接严格兼容。原映射、Tensor 引用与 `_metadata` 的保持检查通过。

| 原生输出 | 实际形状 | 设备与类型 | 有限性 |
|---|---|---|---|
| pred_logits | 1×100×91 | CPU FP32 | 全部有限 |
| pred_boxes | 1×100×4 | CPU FP32 | 全部有限 |
| pred_masks | 1×100×78×78 | CPU FP32 | 全部有限 |

单次前向计时0.8213359788秒，探针主体5.5179270061秒，资源父进程8.0636153899秒；父进程观测峰值group RSS为1,095,471,104字节、最低可用RAM为72,825,446,400字节，违规和终止信号均为空。这是一次312常数输入诊断，不是稳态吞吐、真实SAR或576训练速度基准。

## 离线行为与失败保留

审计实际安装且通过，记录1次预期但仍被拒的 IPv6 能力探测，unexpected和invalid均为0。实际首个调用帧为 `urllib3.util.connection._has_ipv6` 第127行，导出代码对象与固定源文件相合；bind作用仍被OSError阻止。该实际来源证据只属于本次v02，不能反推v01未记录的历史调用栈。

[v01原始state严格加载失败](RFDETR_RAW_STRICT_PROBE_ACTUAL_RESULT_20261107_01.md)完整保留。本次第一次元数据准备又因实际模块目录与预设目录不同而退出1，发生在lease/worker spec形成和科学original进入之前；[独立失败说明](RFDETR_V02_METADATA_SOURCE_PREFLIGHT_FAILURE_20261107_02.md)保留了原因。核实实际文件字节相同后，仅修正目录绑定、使用新准备记录叶，唯一一次v02科学original真实成功，未重跑模型以挑选成功结果。

## 后续工作

PLAIN/FULL当前各接受34,048/37,050次正式更新（约91.90%，45轮+703图），34,048→34,304原任务已接续。以上CPU检查不计主训练或Native训练次数，不产生AP。

新基线下一步仍需完整train/cli依赖、真实数据和类别映射、576/位置编码、单类头初始化政策、canonical导出与原GT对齐，以及真实梯度、优化器、BF16和GPU成本检查。候选[单类初始化对照](RFDETR_SINGLE_CLASS_TRANSFER_INITIALIZATION_REVIEW_20261107_01.md)和[独立预测导出](code/RFDETR_CANONICAL_EXPORT_DRAFT_20261107_01/README.md)仍属于未执行设计；不得从当前COCO90头检查推定其已验证。随后按共同预算训练强对照，再根据实际误差选择一条研究假设。

[官方空缓冲区定义](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/models/lwdetr.py#L253-L256)；[官方版本兼容说明](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/models/weights.py#L195-L228)。研究总目标继续进行。
