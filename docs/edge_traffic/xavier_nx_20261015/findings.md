# 核查记录

- 本机 Python、torch、ultralytics、OpenCV、ONNX Runtime 与 ffmpeg 已安装；GPU 为 RTX 3090 Ti 24GB。
- 用户确认电动车是电动自行车，不要求当前连接 NX，使用公开测试。
- 已发现 Lily-Le/ebikedetection 声称提供电梯视角微调 YOLOv5s，可识别电动车、自行车、行人；是否实际含可下载权重、是否有汽车和公交车、道路泛化均待核查。
- 检索到的 NX YOLOv5 速度表有明显口径差异（纯推理/整流程、精度、输入尺寸），需回到原始来源确认。
- NVIDIA DashCamNet 的四类不足以直接满足全部要求，不能当作完整方案。
- 已确认 Seeed 的 YOLOv8 TensorRT 页面测试硬件为 **Orin NX 16GB**，不能引用成 Xavier NX。
- Lily-Le/ebikedetection 的 main 完整树未发现 .pt 或 .onnx 文件；README 给的是本地训练路径，不构成已取得权重。
- 默认 Python 的 torch 是 CPU 版且 ONNX Runtime 导入 DLL 失败；现已找到 seal_gpu 环境 torch 2.0.1+cu118，GPU 实际可用，无需更改环境。
- chub 已检索并读取 Ultralytics 官方维护者 API 摘要；结合官方 predict/export 页面核查接口。实际 GPU 环境 ultralytics 为 8.3.43。
- SunnyKing342/Electric-Bike-Detection-and-Tracking 的公开仓库仅 README/.gitignore/requirements，无实际训练源码和权重；其精度宣传不能作为本次可交付模型。
- nabang1010/YOLO_Object_Tracking_TensorRT 的 Xavier NX FPS 单元格是占位符“~”，不是测得数值。
- 官方 YOLOv8 COCO mAP50-95：n 37.3、s 44.9；这是整体 COCO 指标，不等价于本次路口五类准确率。
- 资产下载中：已取得 YOLOv8n、YOLO11n；YOLOv8s 与路口视频仍下载中。发现测试启动早于资产就绪并触发重复下载，已精确终止本任务测试进程，后续待文件就绪再启动。

## 最终已执行状态

三份权重已下载并实际加载；Pexels 5675618 十字路口 1080p/29.970FPS/719 帧，三模型各完整检测，无跳帧。RTX3090Ti 含画框落盘为 YOLOv8n 46.60、YOLO11n 46.64、YOLOv8s 48.58FPS；不是 NX 结果，没有统计速度排名。第479帧罐车/邻车合框误报bus已目视核实。完整条件和证据见 REPORT.md 与 comparison_results.json。

没有电动自行车独立类别、没有人工GT精度评价、没有NX实测；本次尚无全部条件合格的现成模型。接口集成检查实际通过。H.264演示已生成。

来源初查：https://github.com/Lily-Le/ebikedetection 、https://wiki.seeedstudio.com/YOLOv8-TRT-Jetson/ 、https://developer.nvidia.com/blog/implementing-real-time-multi-camera-pipelines-with-nvidia-jetson/ 。所有当前结论仍为候选核查。
