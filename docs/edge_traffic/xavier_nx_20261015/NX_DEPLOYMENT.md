# Xavier NX 部署与 30 FPS 验收（尚未在 NX 执行）

日期：2026-10-15。以下是待上板验证的部署方案，不是已经达到 30 FPS 的证明。本次实际执行环境是 Windows + RTX 3090 Ti。

## 建议起点

优先试 YOLO11n 的 TensorRT FP16 / 640×640 / batch=1；保留 YOLOv8n 作速度和兼容性对照，YOLOv8s 作精度上限候选。若只接受所有五类都现成支持，则本次三个权重均不合格：缺电动自行车独立类别。

Xavier NX 支持 JetPack 5，不要套用仅支持 Orin 的 JetPack 6/7 指南。官方版本说明：[JetPack 5.1.5](https://docs.nvidia.com/jetson/jetpack/5.1.5/release-notes/index.html)。实际已有 JetPack 版本和 NVIDIA PyTorch/torchvision/CUDA/TensorRT 配套需先检查，不要直接用普通桌面 PyTorch wheel 覆盖设备环境。

本次本机 API 实测版本为 ultralytics 8.3.43，torch 2.0.1+cu118；这不是 NX 的安装配方。设备上应使用与自身 JetPack 匹配的 NVIDIA PyTorch，再安装匹配的 Ultralytics 及其依赖。

## 在目标 NX 上生成引擎

将原始 `.pt` 权重放入 `weights/`，然后在已配置好的目标 NX 上运行：

```bash
yolo export model=weights/yolo11n.pt format=engine imgsz=640 batch=1 half=True dynamic=False device=0 workspace=2
```

FP16 作为第一选择。不要从 RTX 3090 Ti 生成 `.engine` 再复制到 NX；序列化引擎受 TensorRT 版本、平台和设备兼容性约束。本次没有伪造或提供未在 NX 构建的引擎。

若改用 INT8，需要有代表性的路口校准图和独立验证集；校准覆盖日夜、远距两轮车、遮挡及雨天。没有标注复核，不声称量化没有精度损失。

## 对同一个本地视频做整流程复测

```bash
python3 traffic_benchmark.py --model weights/yolo11n.engine --source input/intersection_pexels_5675618.mp4 --output results/nx_fp16 --frames 719 --warmup 50 --imgsz 640 --device 0 --half
```

脚本输出每帧检测、标注视频和两种速度口径：不含画框/编码的处理速度，以及包含画框/编码的落盘速度。支持 `--source 0 --show` 取摄像头，也可以指定自己有访问权限的 RTSP URL。脚本不跳帧、不用跟踪结果替代检测；摄像头输入的取帧/显示延迟还需另测。实际部署建议用 DeepStream/GStreamer 硬件解码、缩放和低缓冲输入，处理速度与输入源帧率分别记录。

## 验收规则

1. 单路、batch=1、固定输入分辨率和网络尺寸；确认功耗模式、散热、频率和版本；连续至少 10 分钟，覆盖拥堵时段。系统其他负载也需记录。
2. 30 FPS 必须指每秒完成至少 30 次真实帧检测。若摄像头只有 25 FPS，只能证明持续处理这路输入，不能据此宣称 30 次/秒现场检测。不能用跳帧、插帧、跟踪或 batch 吞吐代替。
3. 同时报告平均帧率、p50/p95 处理时延、排队/丢帧、采集至显示延迟、温度和功耗。30 FPS 是约 33.3ms/帧的预算；纯网络推理过线仍需留解码、前后处理和显示余量。
4. “准确度很高”需要先约定每类 precision/recall 或 AP 目标，并在独立路口标注上测量；每类都要出现，电动自行车与普通自行车、摩托车分别定义。COCO bus 也不保证区分城市公交与长途客车。
5. 最终高准确率五类权重尚未取得。若无法找到公开可复验权重，可用本次预训练作为初始化做五类道路数据微调；这是后续方案，本次未训练、未产生五类新权重。

API 依据：[Ultralytics Predict](https://docs.ultralytics.com/modes/predict/)、[TensorRT 导出](https://docs.ultralytics.com/integrations/tensorrt/)、[NVIDIA Jetson 指南](https://docs.ultralytics.com/guides/nvidia-jetson/)。
