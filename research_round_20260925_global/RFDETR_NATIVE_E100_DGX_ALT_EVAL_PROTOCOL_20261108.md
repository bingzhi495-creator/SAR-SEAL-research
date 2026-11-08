# 连续长训端点的同机 DGX 评估补充协议

[原 Windows 评估协议](RFDETR_NATIVE_E100_REMOTE_CLOSED_EVAL_PROTOCOL_20261108.md)需要从 DGX 复制三份含优化器的检查点，共约 1.63 GB。首份复制的实际速度约 40–50 KB/s，直接在 DGX 读取已闭合归档更快。本补充协议在任何三端点 AP 产生前固定。

- 独立旧 E24、连续长训 E24/E50/E100 全部使用同一 DGX 原生 eager FP32 评估入口。归档绑定原运行与闭合回执，用 weights-only 检查实际 epoch、微批次、优化器、EMA 和学习率轨迹，再逐键加载 EMA。另保留原本机训练 SSH 回执缺失的状态。
- 冻结同一 DEV187、整数掩膜、原生 576 输入及 top100 查询、类别映射和 COCO `maxDets=[1,10,100]`。逐端点先写完整预测，再用官方 COCO 计算全部 12 项统计。统一报告四个端点、相对同运行 E24 差值和实际前向成本，E100 保持固定长训终点。
- DGX 数值单列为同设备比较。Windows 参考与 DGX 的小差异按设备和数值路径报告。原 Windows 复制和评估状态继续记录。实验仅评价已训练模型，DEV 为已用开发集，TEST 封存。 [原生导出源码](code/RFDETR_NATIVE_E24_DEV_EXPORT_WORKER_CANDIDATE_20261107/e24_dev_export_worker.py)
