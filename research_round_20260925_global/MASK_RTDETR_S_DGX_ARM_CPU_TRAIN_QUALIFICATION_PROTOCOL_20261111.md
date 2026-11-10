# Mask RT-DETR-S 的 DGX ARM CPU 两步资格

Windows 原生作者读取和 COCO 前向已通过。Windows 两步训练只观察到首批前向并受资源保护停止，原训练更新次数未验收。本轮预锁为独立 DGX ARM CPU 资格，沿用 8 张 FIT 图、16 条原始整数多边形、单类别、种子 20261111、原生四图批次及自然结束的一轮。作者 S 配方的原始 ImageNet B1 骨干加载方式、增强、匹配、损失、AdamW、学习率及 EMA 保留。Windows 与 DGX 具有不同运行时和资源，两个运行分别报告。 [Windows 记录](MASK_RTDETR_AUTHOR_CPU_FIT_QUALIFICATION_PROTOCOL_20261111.md) · [作者 S 配置](https://github.com/PaddlePaddle/PaddleDetection/blob/b25522a0f4bde8c80603f3ba5e3472059972e3b5/configs/mask_rtdetr/mask_rtdetr_hgnetv2_s_6x_coco.yml)

固定 DGX aarch64 CPython 3.12.3，PaddlePaddle 2.6.2 CPU wheel SHA-256 `8f1d922d0afad50352150f6caaf8781cb9850734ef5f15e96347be048fd46741`，OpenCV 4.5.5.64、pycocotools 2.0.8、NumPy 1.26.4 和作者提交 `b25522a0f4bde8c80603f3ba5e3472059972e3b5`。隐去 CUDA，CPU 固定为 2、3，双线程，低调度优先级。开始要求主机可用内存至少 52 GiB。进程树 RSS 限 12 GiB，运行中保留至少 40 GiB 主机可用内存，墙钟限 420 秒。先达到的保护条件与原等待退出分开记录，保护只作用于本次进程组。DGX 上的 CUDA 即时可用量属于独立 GPU 入训条件。 [官方 ARM CPU wheel](https://pypi.org/project/paddlepaddle/2.6.2/#files)

须观察每次真实的优化器、调度和 EMA 调用前及返回后的记录，独立复核两次有限前向、每图一次、单类槽、网络可训练分类与掩膜头张量变化、保存后在线与 EMA 状态、原生 AdamW 和调度张量，以及原进程等待和进程组清空。两步在作者预热段内，范围只限训练资格。模型 AP 与跨域效用仍由预锁的同预算长训判断。
