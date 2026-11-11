# Mask RT-DETR 训练尺寸的最近邻坐标实测

先前在 96 个 HRSID FIT 视图中，OpenCV 5 显式精确最近邻与整数中心规则在 615 条真实掩膜上逐像素一致。本次用合成索引独立测量固定作者配置的 11 个不同输出边长。输入边长从 1 至 3,200，覆盖原始 800 像素边长与默认最多四倍扩展、随后可裁剪的保守范围。输出边长从 480 至 800，每隔 32。每个源索引用两个单通道 uint8 数组编码，分别通过真实的水平与垂直 OpenCV 路径还原索引，两个轴的全部结果相同。参考取整为 $\lfloor(2j+1)N/(2M)\rfloor$，落在整数处的半格平局选较大索引。 [作者训练读取器](https://github.com/PaddlePaddle/PaddleDetection/blob/b25522a0f4bde8c80603f3ba5e3472059972e3b5/configs/mask_rtdetr/_base_/mask_rtdetr_reader.yml#L1-L14) · [作者扩展算子](https://raw.githubusercontent.com/PaddlePaddle/PaddleDetection/b25522a0f4bde8c80603f3ba5e3472059972e3b5/ppdet/data/transform/operators.py) · [既有真实掩膜对照](MASK_RTDETR_HRSID_FIT_CPU_GEOMETRY_PROTOCOL_20261111.md#opencv-5-在真实-fit-栅格上的直接对照)

| OpenCV 5.0.0 路径 | 不同尺寸对／35,200 | 不同索引／22,528,000 | 半格处差异 | 非半格处差异 | 最大源索引差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 默认最近邻 | 35,019 | 17,868,616 | 34,880 | 17,833,736 | 4 |
| 显式精确最近邻 | 347 | 9,722 | 9,722 | 0 | 1 |

上述每种模式的计数按单轴报告，另一轴逐值相同。固定源边长 800 时，显式精确路径在 11 个目标尺寸的全部位置与整数中心规则一致。比例按合成尺寸等权，训练中实际尺寸和正像素的频率需另计。程序以 OpenCV 5.0.0.93 Windows x64 wheel、NumPy 2.2.6、单 CPU 线程和关闭 OpenCL 实际运行，耗时 3.506 秒，原等待退出 0。官方 wheel 的 SHA-256 与本地归档一致。未读取实际 FIT、DEV 或 TEST 影像与掩膜，未训练和评分。 [程序](code/OPENCV_MASK_AXIS_ENVELOPE_20261112/measure_mask_axis.py) · [实测汇总](results/MASK_RTDETR_OPENCV_AXIS_ENVELOPE_ACTUAL_SAFE_20261112.json) · [官方 wheel](https://pypi.org/project/opencv-python/5.0.0.93/) · [OpenCV 5 实现](https://github.com/opencv/opencv/blob/5.0.0/modules/imgproc/src/resize.cpp#L1266-L1277)

将显式精确模式作为单次缩放的直接先例。Mask RT-DETR 的 E24 配对还包含原非空监督的回退保护，当前设计无法从配对总效应单独识别回退的贡献。后续若主张超出库内单步算子，须在等预算、同实例和同评分下检验额外组件，例如多步几何统一后一次栅格化，并同时复核连续多边形口径。 [原配对协议](MASK_RTDETR_HRSID_FIT2937_E24_AUTHOR_PROTOCOL_20261111.md)

