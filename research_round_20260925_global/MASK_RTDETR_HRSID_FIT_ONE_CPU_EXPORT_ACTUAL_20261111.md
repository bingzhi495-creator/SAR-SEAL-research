# Mask RT-DETR-S 单类 HRSID 原生导出实测

从已验收的 HRSID FIT2937 图像清单，在前向前按逻辑图像 ID 升序固定首张图。沿用 PaddleDetection v2.9.0 的固定 Mask RT-DETR-S 源码及原 CPU PaddlePaddle 2.6.2 环境。原作者加载器从官方 COCO 权重逐值复现 643 个名称和形状一致的张量，保留 3 个形状改变的单类头张量的种子初始化。

原生 EvalReader 和 TestReader 的变换序列一致。无标签 TestReader 对 800×800 原图生成一个 640×640 输入。模型在 CPU FP32 路径实际前向一次，由作者的 DETRPostProcess 输出 100 条框和 100 张 800×800 二值掩膜，其中 97 张非空。按原生类槽 0 对应 COCO 类别 ID 1 调用作者的 get_infer_results，100 条框与 100 条掩膜均沿用原分数和候选顺序。所有压缩 RLE 逐像素解码复现。输出分数范围为 0.01621–0.02292。

原工作进程及外层均退出 0，作业树关闭。工作体耗时 7.145 秒，原等待 7.625 秒。合成检查另核对了空掩膜保留、任意原始分数次序、类别映射和未裁剪的坐标转换。这是单类初始化后的单张 FIT 推理和文件格式资格，未读取标注、训练、计算 AP 或访问 GPU、DEV、TEST。CPU 结果不建立 GPU 数值等价。

[原生导出汇总](results/MASK_RTDETR_HRSID_FIT_ONE_CPU_EXPORT_ACTUAL_SAFE_20261111.json)
