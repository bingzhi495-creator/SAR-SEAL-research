# D-FINE DEV705 裁剪前概率捕获

用原 D-FINE E24 权重与已使用的 HRSID DEV705 图像，按原作者入口完成 705 次模型前向。每图保留原前 100 个候选的 160×160 FP32 sigmoid 概率。70,500 份概率经作者原后处理逐像素复现二值掩膜，184,687 条候选的原预测和审计流逐字节一致。此次只为固定预测侧输入。未访问标注或官方 TEST，未计算 AP 或更新参数。 [原 DEV705 基准](DFINE_E24_HRSID_DEV705_ACTUAL_20261112.md) · [捕获协议](code/DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/dev705_probability_capture/PROTOCOL.md)

原捕获与独立 CPU 读回均自然退出 0。概率归档为 7,219,290,240 字节，原工作体 124.737 秒，独立读回 12.569 秒，CUDA 峰值已分配 683,898,368 字节。上述成本含离线捕获及验收。固定点头修正的评分和成本另行报告。 [安全汇总](results/DFINE_E24_DEV705_PROBABILITY_CAPTURE_ACTUAL_SAFE_20261112.json)

复现时将既有 FIT114 捕获实现按原摘要复制到该工作体目录的 `fit114_reference.py`。复用其作者概率观察和二值复现逻辑。 [共享实现](code/DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/fit114_capture/capture_fit.py) · [新捕获源码](code/DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/dev705_probability_capture/dfine_dev705_probability_worker.py)

