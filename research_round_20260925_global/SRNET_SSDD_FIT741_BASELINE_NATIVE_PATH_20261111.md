# SRNet 在固定 PSeg-SSDD FIT741 上的原生入口实测

SRNet 是船舶旋转框与实例掩膜联合建模的直接外部对照。作者 PSeg-SSDD 官方测试 mask AP、AP50、AP75 为 67.1、94.9、85.4。项目的 FIT741 与 DEV187 位于官方训练侧内。后续比较使用相同 FIT 数据和相同 DEV187 评分规则。此次原生作者源码固定为提交 `2e56c7208f217ff1824f1d4f3f3ea242a5f78bc6`，实际执行前后工作树干净。 [SRNet 原文](https://www.mdpi.com/2072-4292/15/5/1324) · [固定作者源码](https://github.com/XIDIAN212Lab/SRNet/tree/2e56c7208f217ff1824f1d4f3f3ea242a5f78bc6) · [PSeg-SSDD 官方划分](https://www.mdpi.com/2072-4292/13/18/3690)

## 原生数据入口

作者从原多边形经整数坐标、OpenCV 最小外接矩形和 `le90` 转换得到旋转框，原多边形仍进入掩膜。两类监督在同一索引中可复建。以原 FIT741 COCO 和图像布局替换作者配置的训练数据位置，保留其数据集类、重复倍数和增强。作者实际解析 741 张图及全部 1,661 条多边形，过滤图与标注均为 0。按元数据摘要预选的 8 张图在原栅格、等比缩放及最终填充后均保留 20 条实例。独立用 COCO 栅格化核对原尺寸位图逐像素一致。原 CPU 工作体 4.51 秒，退出 0。 [加载器](https://github.com/XIDIAN212Lab/SRNet/blob/2e56c7208f217ff1824f1d4f3f3ea242a5f78bc6/mmrotate/datasets/SSDD.py#L88-L135) · [作者增强配置](https://github.com/XIDIAN212Lab/SRNet/blob/2e56c7208f217ff1824f1d4f3f3ea242a5f78bc6/configs/_base_/datasets/SSDD.py) · [实测源码](code/SRNET_SSDD_FIT741_BASELINE_NATIVE_20261111/srnet_fit_native.py) · [汇总](results/SRNET_SSDD_FIT741_NATIVE_DATA_ACTUAL_SAFE_20261111.json)

Windows CPython 3.10.13 隔离环境使用 PyTorch 1.12.0 CPU、torchvision 0.13.0 CPU、MMCV Full 1.6.0、MMDetection 2.28.2、NumPy 1.26.4、OpenCV 4.11.0.86、pycocotools 2.0.11、Pillow 9.5.0 和 e2cnn 0.2.3。首次执行缺少作者注册树传递依赖 e2cnn，原进程退出 1。补装后保持同一源码重跑。零偏移可变形卷积与普通卷积逐值相同，三个框的旋转 NMS 保留两个预期框。 [PyTorch 官方包](https://download.pytorch.org/whl/cpu) · [MMCV 官方包](https://download.openmmlab.com/mmcv/dist/cpu/torch1.12.0/index.html) · [e2cnn](https://pypi.org/project/e2cnn/0.2.3/)

## 原生模型前向和 RLE

从 PyTorch 官方入口取得 102,530,333 字节的作者指定 ImageNet ResNet-50 初值，核对完整 SHA-256。原作者模型按本地该文件初始化，第一层卷积权重与官方文件逐值一致。FPN、框头和掩膜头按作者配置初始化。原作者 `simple_test` 在一张预选 FIT 图上执行一次，掩膜恢复到 317×501 原网格。为实际走过非空分支，固定 `score_thr=0`、`nms_pre=32`、`max_per_img=4`。得到 4 条有限框及 4 张非空布尔掩膜，原尺寸 COCO RLE 全部逐像素往返一致。CPU 工作体 9.33 秒，原进程退出 0。单独重放除时钟外的全部安全汇总字段一致，实际 Python 子进程的 Windows 峰值工作集为 1,060,839,424 字节。原启动器采样只覆盖 Windows venv 的转接进程，内存以子进程独立核验为准。本轮权重范围为 ImageNet 主干及新初始化的任务头。PSeg 训练、固定 DEV 评分和正式测试待执行。 [官方 ImageNet 初值](https://download.pytorch.org/models/resnet50-0676ba61.pth) · [实测源码](code/SRNET_SSDD_FIT741_BASELINE_NATIVE_20261111/srnet_native_mask_smoke.py) · [输出汇总](results/SRNET_SSDD_NATIVE_MASK_EXPORT_CPU_SMOKE_ACTUAL_SAFE_20261111.json) · [子进程资源核验](results/SRNET_SSDD_NATIVE_CHILD_RSS_ACTUAL_SAFE_20261111.json)

## 接续条件

- 原作者配置为 36 轮、每轮三次数据重复。同预算复现按 RF-DETR 的 FIT741 E24 实际访问量和更新量对齐。先锁定同一原标注训练范围及配方调整，按实际有效访问、优化器更新及显存核对预算。同预算 SRNet 比较用同一 DEV187 原图 RLE 和 COCO 评估规则，另记录作者默认的训练配方。
- CUDA 11.6 版 PyTorch 1.12.0 和相应 Windows MMCV Full 1.6.0 安装文件的头部均已实测返回正确 ZIP 标记。训练还需实际验证本机 CUDA 的作者可变形卷积、旋转 IoU 与原生损失反传。作者掩膜损失显式调用 CUDA。NVIDIA 的 Ada 兼容性说明支持较低次版本的同主版本 cubin，是否适用于整条依赖链要用真实短窗确定。 [PyTorch CUDA 包](https://download.pytorch.org/whl/cu116/torch/) · [MMCV CUDA 包](https://download.openmmlab.com/mmcv/dist/cu116/torch1.12.0/index.html) · [NVIDIA Ada 兼容性](https://docs.nvidia.com/cuda/ada-compatibility-guide/) · [作者掩膜损失](https://github.com/XIDIAN212Lab/SRNet/blob/2e56c7208f217ff1824f1d4f3f3ea242a5f78bc6/mmrotate/models/dense_heads/SRNet_head.py#L888-L913)
