# Mask RT-DETR-S 在 HRSID DEV705 的固定 E24 配对

固定 PaddleDetection v2.9.0 作者模型、训练种子 20261111、共同 COCO 初值及 FIT2937。两臂各 24 轮、70,560 次访问、17,640 次 AdamW／调度／EMA 更新。B 在作者批次掩膜缩放使用整数中心规则，原生与中心任一掩膜为空时沿用原生掩膜。验收核对非空监督支持一致。端点使用各自 E24 EMA，作者 FP32 CUDA 读入 640²、输出 800²、掩膜阈值 0.5，固定每图前 100 条预测。 [完整训练验收](MASK_RTDETR_HRSID_FIT2937_E24_PAIR_ACTUAL_20261112.md) · [固定评价协议](MASK_RTDETR_FIXED_ENDPOINT_STREAM_FIT_ACTUAL_20261111.md)

## 原栅格主结果

705 张开发图、1,782 条原标注、每臂 70,500 条候选，按 COCO maxDets 1／10／100。表中 A、B 为百分数，差值为百分点。

| 指标 | 原生 A | 中心 B | B−A |
| --- | ---: | ---: | ---: |
| AP | 50.304 | 56.043 | +5.739 |
| AP50 | 88.530 | 89.536 | +1.006 |
| AP75 | 55.373 | 65.717 | +10.344 |
| AP90 | 0.431 | 2.346 | +1.914 |
| APs | 49.414 | 55.760 | +6.345 |
| APm | 58.526 | 59.804 | +1.278 |
| APl | 47.033 | 32.995 | -14.038 |
| AR100 | 56.639 | 61.510 | +4.871 |

排除预锁四条空栅格后，AP 为 50.429→56.180（+5.752 点），AP75 为 55.456→65.899（+10.443 点），AP90 和 APl 逐值不变。大目标损失 14.038 点须保留，既有同 DEV705 标注支持核验中的大目标只有 16 个。正式 TEST 继续隔离。 [原始与空栅格敏感性汇总](results/MASK_RTDETR_HRSID_DEV705_AUTHOR_PAIR_ACTUAL_SAFE_20261112.json) · [同标注支持核验](HRSID_DEV705_CONTINUOUS_AP_SENSITIVITY_20261111.md)

## 候选、成本与结论范围

原生 A 的非空候选 23,448，中心 B 为 39,281，固定总数下增加 15,833。原生端点允许模型自己的候选和分数变化。该差值提示需结合候选覆盖、排序和固定匹配几何理解 AP。两臂 705 张图的同步前向计时为 22.123／21.998 秒，逐图推理及流式导出为 136.578／136.419 秒，原子进程等待为 146.390／142.656 秒。 [CUDA 导出验收](results/MASK_RTDETR_HRSID_DEV705_GPU_EXPORT_CLOSURE_SAFE_20261112.json)

单个固定种子和相同 HRSID 开发集上，RF-DETR 已观察到的中心监督正向 AP 效应在第二个作者实现出现，主要增益落在小目标；两套配对都观察到大目标下降。Mask RT-DETR 的回退保留规则与 RF-DETR 的共同非空实例规则不同。连续真值敏感性、第二库及跨种子证据分别决定可推广范围。显式精确最近邻已在 615 条真实 FIT 掩膜上复现整数中心算子，应作为单步先例。 [RF-DETR 固定共同实例复验](HRSID_MASK_PHASE_COMMON_SUPPORT_AND_SSDD_TRANSFER_DESIGN_20261109.md#共同行-e24-的固定-dev705-实测) · [OpenCV 对照](MASK_RTDETR_OPENCV_AXIS_ENVELOPE_ACTUAL_20261112.md) · [连续口径预锁](code/MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/PROTOCOL.md)

## 原执行与修复记录

第一次真实 FIT 烟测在前向前退出，原读取器返回 CUDA 页锁定主存张量。修复明确将作者读取器的张量搬入 CUDA，实际 FIT 烟测中六个张量共 4,915,228 字节逐字节复核一致。第二次 FIT 烟测通过，CPU 图像暂存在 Windows 上触发大小写等价的目录冲突。第三次用独立的数据暂存路径，FIT 烟测、705 图暂存、两臂独立 CUDA 导出和 CPU 计分均原样闭合，原始模型、阈值、图像及评分口径相同。发布入口依赖同一预锁执行环境和私有输入，闭合汇总可直接审阅。 [两次原失败](results/MASK_RTDETR_HRSID_DEV705_PRIOR_ATTEMPTS_SAFE_20261112.json) · [实测修复源码](code/MASK_RTDETR_DEV705_WINDOWS_DISTINCT_STAGE_20261112/p392_windows_dev705_owner.py) · [设备搬运检查](code/MASK_RTDETR_DEV705_WINDOWS_DISTINCT_STAGE_20261112/test_reader_transport.py) · [路径检查](code/MASK_RTDETR_DEV705_WINDOWS_DISTINCT_STAGE_20261112/test_distinct_stage_paths.py)

