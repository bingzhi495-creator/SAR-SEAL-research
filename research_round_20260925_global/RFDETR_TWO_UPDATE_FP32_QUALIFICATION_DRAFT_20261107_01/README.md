# RF-DETR 两次原生 FP32 更新检查候选

原生 CLI、编解码和[真实 SSDD 两样本入口](../RFDETR_NATIVE_SSDD_DATA_ENTRY_ACTUAL_20261107_01.md)已通过。此处的自有 worker 已编译，等待独立 GPU 资源和已验收输入绑定。正式 E100 指标仍待完整训练。

检查采用真实 FIT741、固定 COCO 初始化和原生 RFDETRModelModule、DataModule、Trainer、AdamW、LambdaLR 与 EMA。保留原生每轮744个 loader 微批次、累积4和100轮18,600次更新的调度计划，其中558次是 warmup。诊断在实际完成8个微批次与2次更新后停止。运行使用严格 FP32 并记录 TF32 开关，完整基线的精度配方另行验收。

记录每个微批次的原生掩膜和分类等损失、两个累计窗口的梯度、实际 AdamW 计数、两个参数见证的最大绝对变化和 EMA 更新次数。首步调度的学习率为0，因此预期参数保持原值。第二步预期学习率为正，至少两个预定见证的参数产生有限变化。候选源码的编译状态与实际运行证据分别记录。[官方训练模块](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/training/module_model.py)
