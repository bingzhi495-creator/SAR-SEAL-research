# Xavier NX 预训练路口检测选型与实际测试

日期：2026-10-15。用户要求：行人、自行车、**电动自行车**、汽车、公交车，高准确率，Xavier NX ≥30 FPS，尽量更快。用户确认先做公开样本测试，不连接 NX。

## 结论与当前交付

**本次尚未核实到同时满足全部要求的现成预训练权重。** 已取得并真实运行三种官方 COCO 预训练模型，生成完整城市十字路口演示；它们支持行人、自行车、汽车、bus 等类别，但都没有独立的电动自行车类别，不能把 motorcycle 改名来充当支持。

若现在要一个轻量部署起点，建议先试 **YOLO11n + TensorRT FP16 + 640×640 + batch=1**，并保留 YOLOv8n 作速度/兼容性对照，YOLOv8s 作较高 COCO 精度候选。这个建议是后续上板验证的起点，不是“已满足五类高准确率及 NX 30 FPS”的认证。

已经交付：三份可加载官方 `.pt`、719 帧路口视频逐帧推理、三份标注视频、逐帧检测记录、可复跑工具、权重/输入来源与 SHA-256、NX 部署和验收说明。没有产生或声称已经取得五类道路微调权重。

## 模型与权重

| 模型 | 官方 COCO val mAP50–95 | 这次下载大小 | 角色 | 电动自行车独立类别 |
|---|---:|---:|---|---|
| YOLOv8n | 37.3 | 6.55 MB | 轻量速度对照 | 无 |
| YOLO11n | 39.5 | 5.61 MB | 优先试部署的轻量候选 | 无 |
| YOLOv8s | 44.9 | 22.57 MB | 较高精度候选，NX 速度风险更大 | 无 |

这些是官方全 COCO 指标，不是五类路口准确率，更不是本次视频测得的 AP；39.5 也不能写成“识别正确率 39.5%”。这次没有自行训练，初始化为对应官方预训练权重，训练预算为 0。

官方模型卡：[YOLOv8](https://huggingface.co/Ultralytics/YOLOv8)、[YOLO11](https://huggingface.co/Ultralytics/YOLO11)。类别依据：[COCO 官方配置](https://docs.ultralytics.com/datasets/detect/coco/)。bus 表示泛化的巴士类别，不保证专门区分城市公交与长途客车。

可直接获取的官方权重：

- [YOLO11n.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt)
- [YOLOv8n.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt)
- [YOLOv8s.pt](https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8s.pt)

本地位于 `weights/`；精确校验值见 `weights_manifest.json`。没有提供从桌面 GPU 构建、却冒充适用于 NX 的 TensorRT 引擎。

## 十字路口实际执行结果

素材为 Erkan Avanoğlu 的 [Pexels 5675618：城市十字路口等红灯](https://www.pexels.com/video/cars-waiting-in-a-traffic-light-5675618/)，HD 1920×1080，29.970 FPS，完整 719 帧、约 23.991 秒。已经目视确认是真实城市十字路口；视角是等红灯车辆前方，不能代表中国路口高位固定监控或大量电动自行车场景。

实际硬件：**RTX 3090 Ti，Windows，PyTorch 2.0.1+cu118，Ultralytics 8.3.43，OpenCV 4.8.0**。三模型顺序执行，各预热 30 帧；网络固定 640×640、FP16、batch=1，置信度 0.25、NMS IoU 0.7、max_det=300。无增广、无跟踪、无跳帧，保留每帧结果。

| 模型 | 实际检测帧数 | 解码+前处理+检测+NMS+取回框 FPS | 再含画框/JSON/视频保存 FPS | 处理 p95 时延 |
|---|---:|---:|---:|---:|
| YOLOv8n | 719 | 94.42 | 46.60 | 11.91 ms |
| YOLO11n | 719 | 89.54 | 46.64 | 12.57 ms |
| YOLOv8s | 719 | 98.74 | 48.58 | 11.54 ms |

**全部是桌面 GPU 上的离线逐帧吞吐，均不是 Xavier NX FPS，也不是实时摄像头采集至显示延迟。** 单次测试没有锁定桌面 GPU 频率或隔离其他系统负载，不依据表中接近的速度断言大模型更快；本表用于验证代码确实跑通并记录本次执行。

每种模型均输出 person/bicycle/car/bus 等类别框，但“输出过某类”不等于这类识别正确。输入未提供人工框标注，**本次没有计算 precision、recall、AP 或宣称高准确率**。逐帧框数只是模型输出，不是独立目标数或真阳性数；各类出框覆盖率也不是召回率。

### 已确认的具体错误

第 479 帧（约 15.98 秒），YOLO11n 将横向罐车及邻近车辆合为 bus 框（置信度约 0.52），同时另有部分 truck 框；YOLOv8s 也将该组合判为 bus（约 0.75）。两者都能框出过街行人和多辆汽车，但这个实例足以反驳“该样本上公交识别已经很准”的说法。不能把全部 bus 输出自动算作正确公交检测。

演示保留这个失败，没有删去错误帧或把输出标签改成期待类别。自行车/摩托车出框尚未逐一人工核验；电动自行车区分能力因没有对应类别而不成立。该短片也不能替代夜间、雨雾、拥堵、遮挡和远距小目标评测。

主要产物：

- `demo/intersection_yolo11n.mp4`、`demo/intersection_yolov8n.mp4`、`demo/intersection_yolov8s.mp4`：便于播放的 H.264 标注演示，保留源视频时间轴。
- `results/intersection_*/summary.json`：完整条件和速度口径。
- `results/intersection_*/detections.jsonl`：每帧框、原类别、置信度和时延，仅本地保存。
- `results/intersection_*/frame_00479.jpg`：可复核的失败实例。
- `comparison_results.json`：三模型执行摘要，公开副本不含完整视频/逐帧记录。

辅助测试另使用 Roboflow `vehicles.mp4`，核看为高速公路，仅作基础检查和额外车辆推理，不能冒称十字路口。辅助 YOLO11n 完整 538 帧已执行，摘要单独存放。

## Xavier NX ≥30 FPS 的公开证据边界

1. [jario-jin 原始部署仓库](https://github.com/jario-jin/yolov5-on-nvidia-jetson)报告 YOLOv5s 在 Xavier NX、640、FP16、C++ 平均推理 25.5ms（倒数约 39.2 FPS）；Python 为 37.5ms（约 26.7 FPS）。这是旧模型/旧工具链的作者报告，表头是推理耗时，不能作为 YOLO11n/YOLOv8 或完整摄像头链路已达标的证明。
2. [Seeed YOLOv8 TensorRT 文档](https://wiki.seeedstudio.com/YOLOv8-TRT-Jetson/)的基准平台是 **Orin NX 16GB**，不是 Xavier NX。没有搬用其速度承诺。
3. [nabang1010 仓库](https://github.com/nabang1010/YOLO_Object_Tracking_TensorRT)的 Xavier NX 速度单元格为占位符 `~`，不构成可引用的数字。

因此，本次没有可靠依据承诺“YOLO11n 在你的 NX 端到端稳定 30/60 FPS”。TensorRT FP16 是合理待测路线，最终性能还取决于 JetPack、功耗/散热、图像尺寸、硬件解码、后处理和其他负载。完整待执行流程见 `NX_DEPLOYMENT.md`。

## 电动自行车专用候选核查

| 候选 | 实际可获取状态 | 本次判定 |
|---|---|---|
| [Lily-Le/ebikedetection](https://github.com/Lily-Le/ebikedetection) | README 提及微调 YOLOv5s 和本地 best.pt 路径；核查 main 文件树未见公开 `.pt`/`.onnx` | 未取得权重；电梯视角，不能认证道路效果或全五类 |
| [YOLO-DC 作者仓库](https://github.com/SunnyKing342/Electric-Bike-Detection-and-Tracking) | 核查只有 README、requirements、gitignore；权重和正文所示执行源码未发布，数据需向作者申请 | 宣传精度无法在本次复验，不作为可交付模型 |
| [AXERA-TECH/E_bike-axera](https://huggingface.co/AXERA-TECH/E_bike-axera) | 厂商实际提供自行车/e-bike 两类 `.axmodel` | 是 Axera NPU 专用格式、主要电梯场景；没有可直接加载到 NX 的 PT/ONNX，也不覆盖全部五类 |

上述仅说明本次检索和核查状态，不宣称整个互联网不存在满足要求的权重。

## 当前状态和下一步

**已完成：选型核查、三份官方权重下载和校验、公开十字路口实际逐帧推理、失败例复核、可复跑工具、演示与部署草案。**

**未完成：合格的五类道路权重获取、电动自行车独立识别、带真实标注的高准确率验收、NX 上 TensorRT 构建和 ≥30 FPS 验收、真实摄像头低延迟长时测试。** 用户本轮明确允许暂不连接 NX；没有暗示已经完成这些环节。

要严格满足原需求，需要继续获得可信的道路五类已训练权重，或用预训练初始化微调一个明确含电动自行车类别的轻量检测器。后一条需要训练数据与新实验，本次没有用“已经预训练”来包装待训练方案。

所有权重、第三方视频、逐帧记录、下载分块和缓存仅保留本地；研究仓库同步自有源码、来源/校验清单、条件与结果摘要。仓库记录与文件同步不代表模型达标。
