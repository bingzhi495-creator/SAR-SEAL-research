# Mask RT-DETR 官方 CPU 读取与前向资格

本项资格用于将先前的隔离几何算子复验接到完整官方 Mask RT-DETR-S。源码为 PaddleDetection v2.9.0 固定提交 `b25522a0f4bde8c80603f3ba5e3472059972e3b5`，采用官方 S 配置、原生 Dataset、TrainReader、TestReader 和 DETR。首先验证 Windows CPython 3.10.13 的 PaddlePaddle 2.6.2 官方 CPU wheel。这是一项实际兼容性测试。公开导出报告与运行本试验是不同的证据。CPU 限制单线程、隐藏 CUDA、子进程低优先级、可用主存至少 16 GiB。正式 HRSID 的 GPU 原进程具有优先权。 [官方模型说明](https://github.com/PaddlePaddle/PaddleDetection/blob/b25522a0f4bde8c80603f3ba5e3472059972e3b5/configs/mask_rtdetr/README.md) · [CPU wheel](https://pypi.org/project/paddlepaddle/2.6.2/)

- 固定沿用前轮 SSDD 几何清单中最前的 8 个不同 JPEG 内容组，16 条项目整数实例标注。输入清单 SHA-256 为 `8a0099846bdb530f2250d8c8473014b934a9107bc82f6fd053074b71a77a746a`。保留 JPEG 原字节、原生图像高宽、整数多边形和类别 ID 1。由官方 COCO Dataset 映射为单类槽位 0。训练读取器保留作者的四图批次、原始随机增强和两批预算。Windows worker 为 0，共享内存关闭。需实际观察每张恰一次、掩膜图框尺寸和类槽位。
- 另在首张固定 FIT 图上使用官方无标签 TestReader 的 640×640 输入、80 类 S 模型及作者完整 COCO 权重。权重 58,728,806 字节，MD5 `110e973d6d533cdf48f3d848f8372791`，SHA-256 `d71ea6590442d1c3d6118fe0558b4a9fe84283a4832b12dc6d7b5c8cf3e09d9f`。核对原生模型和权重的键、张量形状及值，以及可变形注意力的 Paddle CPU 路径。原生后处理输出最多 100 个候选和原图网格掩膜。仅验证前向和张量契约，评价不读标签。 [官方 S 权重](https://paddledet.bj.bcebos.com/models/mask_rtdetr_hgnetv2_s_6x_coco.pdparams) · [CPU 回退实现](https://github.com/PaddlePaddle/PaddleDetection/blob/b25522a0f4bde8c80603f3ba5e3472059972e3b5/ppdet/modeling/transformers/deformable_transformer.py#L61-L74)
- 资格范围为 FIT 输入与原生 CPU 前向。原进程退出、输入校验、读取器与前向输出分别记录。后续如进入两步训练，单类初始化、官方骨干预训练来源和优化器及 EMA 的真实计数另外锁定。完整目标域基线仍须统一训练预算、固定开发集和独立评价。

## 原生读取器与前向实测

完整 `ppdet` 由官方固定源码导入。原生 Windows CPU PaddlePaddle 2.6.2 已确认 CUDA 未编译。原生 TrainReader 实际处理 8 张 FIT 图，每图恰一次，原 16 条标注经增强保留 15 条，均有非空掩膜且映射到船类槽位 0。两批的输出网格为 512×512、672×672。配置和作者权重均在读取前核对。80 类模型的 646 个权重与缓冲张量在键、形状、类型及加载值上全部一致。官方 TestReader 的一个 640×640 输入经过 CPU 原生可变形注意力，输出 100×6 候选和 100×319×503 的原图网格二值掩膜。实际绑定的计算函数来自 PaddleDetection CPU 回退模块。日志中的 CUDA 字样来自尝试导入前的固定打印。原生前向仅验证结构、运行及张量输出。 [实际汇总](results/MASK_RTDETR_S_NATIVE_CPU_COCO_FIT8_QUALIFICATION_ACTUAL_SAFE_20261111.json)

首个进程退出 1。Windows Git 策略将 YAML 的 39 个 LF 改成 CRLF，触发文件字节预检，在 FIT 读入及模型执行前退出。随后从同一提交重取官方 LF 字节，原始进程实际等待退出 0，耗时 16.375 秒，作业容器已清空。原监督的内存统计只观察到虚拟环境启动器，实际前向峰值尚无有效记录。独立 CPU 环境的依赖一致性检查通过，源码导入包含作者已有语法和可选追踪依赖提示。模型前向次数为 1，训练更新及正式 DEV、TEST 和 AP 均为零。 [入口和依赖](code/MASK_RTDETR_AUTHOR_CPU_QUALIFICATION_20261111/)

## 两步训练资格的预锁

复用同一 8 图、16 条原始标注、原作者四图批次与固定种子 20261111。目标为单类模型，采用作者 S 配方指定的 ImageNet PPHGNetV2 B1 骨干资产，25,457,962 字节，MD5 `91eee8263a2992db5b068587f336aa18`，SHA-256 `a37a27dfab3fe923ee4d1095f2f84c48d67af836d4dab2596eadeba6a1213408`。以作者匹配逻辑核对实际从骨干加载的张量，任务头保留同种子原生初始化。输入、初始化和训练预算与前述 80 类前向资格分别记录。 [作者训练配置](https://github.com/PaddlePaddle/PaddleDetection/blob/b25522a0f4bde8c80603f3ba5e3472059972e3b5/configs/mask_rtdetr/mask_rtdetr_hgnetv2_s_6x_coco.yml)

CPU FP32 单线程，实际观察启动器和子进程树，设可用主存至少 16 GiB、进程树 RSS 不超过 6 GiB、墙钟不超过 300 秒。只改为一个自然结束的训练轮、FIT 路径、单类别及 Windows 读取资源配置。增强、匹配、损失、AdamW、学习率预热和 EMA 走原训练器。两步结束须核对各次真实样本、前反向更新与 EMA 计数、有限损失和终点档案。两步均在 2,000 步预热段，只能验收训练路径。尚待执行。

## 两步训练的资源停止

真实原训练器解析了 8 个合格 FIT 图块并加载作者 B1 ImageNet 骨干。已观察到第一批原生模型前向，损失为 259.0564。随后原进程在 118.015 秒受控退出，退出码 70，作业容器已清空。资源回执为 `EXCLUSIVE_RESERVATION_CHANGED`。运行耗时小于 300 秒，现有回执未区分树 RSS 超限或宿主内存低于下限。没有生成 E1 检查点，也没有完成优化器或 EMA 次数验收。保留本次终态，作为资源中断归档。原生读取器和先前 COCO 推理的资格仍由其独立关闭回执支持。 [训练停止记录](results/MASK_RTDETR_S_NATIVE_CPU_FIT8_TWO_UPDATE_RESOURCE_STOP_SAFE_20261111.json) · [实际训练入口](code/MASK_RTDETR_AUTHOR_CPU_QUALIFICATION_20261111/train_two_steps.py)

进一步资格需在独立资源条件下重新准入，并分别记录资源保护触发分支。验收时应逐项核对掩膜预测头的实际可训练张量，避免以掩膜相关缓冲区代替参数更新。优化器读回应核对 AdamW 张量和调度器状态，EMA 与在线归档分开。两步资格与充分训练的模型基线仍分别记录。
