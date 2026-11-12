# D-FINE E24 HRSID FIT114 原生掩膜和概率捕获

用已独立验收的 D-FINE-seg-S 第 24 轮 EMA，在旧 FIT114 的 109 张核心图和独立保存的 5 张压力图上各推理一次。核心图的原编码、RGB 内容和来源前缀各有 109 个不同值。各组按已验收 RF-DETR 5202 第 1 轮的首次访问顺序处理。未开启训练标注、DEV 或 TEST。 [冻结协议](code/DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/fit114_capture/PROTOCOL.md)

| FIT 图像组 | 图像 | 后 NMS 原候选 | 空掩膜 | 前 100 FP32 概率平面 | 原二值掩膜逐像素复现 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 核心 | 109 | 27,735 | 8,538 | 10,900 | 10,900 |
| 压力 | 5 | 796 | 1 | 500 | 500 |

作者原生推理、原箱 NMS 和原阈值保留。每个后 NMS 候选保存原 RLE、分数、框和查询 ID。前 100 候选另保存原解码器的 160×160 FP32 sigmoid 概率。用作者原生 CUDA FP16 上采样、掩膜阈值和框清理复现 11,400 个原二值掩膜。原工作体完成 114 次前向，用时 18.802 秒，CUDA 峰值已分配 1,063,377,920 字节。原始所有者与子进程自然退出 0，GPU 已释放。独立 CPU 读取核对了两组的全部文件摘要、原图顺序、查询与候选对应、概率数据格式和范围、原 RLE 面积。 [实际捕获](results/DFINE_E24_HRSID_FIT114_CAPTURE_ACTUAL_SAFE_20261112.json) · [独立读取](results/DFINE_E24_HRSID_FIT114_CAPTURE_INDEPENDENT_SAFE_20261112.json)

该产物提供后续面积误差试点所需的固定 FIT 输入。尚未拟合或评估面积修正，没有额外训练或评分。 [捕获源码](code/DFINE_SEG_EXTERNAL_BASELINE_FIT32_E24_20261112/e24/fit114_capture/capture_fit.py)
