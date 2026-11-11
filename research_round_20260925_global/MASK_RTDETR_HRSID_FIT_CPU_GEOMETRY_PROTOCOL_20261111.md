# Mask RT-DETR 的 HRSID FIT 掩膜采样对照

研究问题是第二框架的最近邻掩膜缩放是否出现同方向的纵向几何偏差。固定 PaddleDetection v2.9.0 提交 `b25522a0f4bde8c80603f3ba5e3472059972e3b5`。CPU 试验复用官方类别的 NumPy、OpenCV 路径，保留多边形栅格化、颜色、扩展、裁剪、翻转及每批共用的尺寸和图像插值。官方掩膜支路 A 用默认最近邻，B 仅在同一预缩放栅格上替换为整数中心映射 `floor((2j+1)N/(2M))`。官方配方的非整数缩放只有一步。读取器类别边界与原生 PaddleDataLoader 的导入、注册及模型训练分开记。 [官方读取器](https://github.com/PaddlePaddle/PaddleDetection/blob/b25522a0f4bde8c80603f3ba5e3472059972e3b5/configs/mask_rtdetr/_base_/mask_rtdetr_reader.yml#L2-L21) · [掩膜缩放](https://github.com/PaddlePaddle/PaddleDetection/blob/b25522a0f4bde8c80603f3ba5e3472059972e3b5/ppdet/data/transform/operators.py#L1010-L1022)

预先锁定封存 FIT114 中 24 个不同核心文件名前缀和不同编码内容，按 `HRSID_FIT_MASK_RTDETR_STOCK_A_B_20261111_V1` 与原 PNG 摘要拼接后的 SHA-256 排序。每张用四次不同增强，六个四图批次在四轮上合计 96 个视图。批次种子为 `712100 + 100r + b`，r 从 0 到 3，b 从 0 到 5。Python 和 NumPy 随机过程共用。源数据为已验收的 FIT114 训练子包，包摘要 `af12a4e86ea36e9d7fe6ac47b17c684a4812baba6aa9be019f889f833e0f577c`。不披露图像、实例标注、成员名或逐图索引。病例与阈值锁 SHA-256 为 `3d78297dd49a99be208a91f4bf45d2d56a5d0f1ae55b80328dbb4a42de6102ed`。版本固定为 OpenCV 4.5.5.64、pycocotools 2.0.8 和 NumPy 1.26.4。 [既有 FIT 与机制协议](HRSID_MASK_PHASE_COMMON_SUPPORT_AND_SSDD_TRANSFER_DESIGN_20261109.md#第二框架的原生掩膜取样入口)

参考为**同一预缩放栅格的前景像素单元连续缩放后的覆盖**。它用于测量取样相位，并保留栅格化与整数裁剪的既有效果。主要人口为参考连续面积小于 1,024 个输出像素、两臂共同非空的监督行。先按来源图聚合，再以 24 个来源代理组中有有效行的组等权。主要指标为 A 减 B 的绝对纵向质心误差，换回预缩放坐标的像素单位。PCG64 31991 做 2,048 次组重采样，晋级条件为至少 16 个有效组、主要差值至少 0.10 像素且分位区间下界大于零。全部小目标的组等权连续面积 IoU 差须至少 −0.002，非空率差须至少 −0.01。后两项计入参考有效但输出为空的行。方向、横轴、裁剪、缩放方向及图像是否抽到最近邻单列。图像最近邻与 A 的取样坐标关系可能改变图像和监督的一致性，独立参考几何的收益不支持图像对齐或 AP 结论。

保留无效、多边形或掩膜为空的行数及 A/B 各自和共同的保留数。只有 A/B 成功共用图像、框、行顺序及随机过程，才能解释其几何差值。此次试验限定 FIT、CPU 和掩膜坐标。实际数据、读数及关闭证据将在执行后补录。

## 真实 FIT 视图结果

按预锁顺序完成 24 个四图批次，共 96 个真实 FIT 视图。36 个视图发生裁剪，47 个发生扩展，42 个发生翻转，16 个图像侧抽到最近邻。652 条输入监督经整数变换保留 615 条。四图共享每批一次原生尺寸及图像插值抽取。所有原生输出与独立 OpenCV 最近邻逐值一致，A/B 共有 594／615 个掩膜不同。有与无编号见证的重放逐值比较了图像、框、类别和掩膜，原生保留编号顺序一致。随机性核对以同种子实际增强选择和输出为依据。此次使用原版几何类别的隔离调用与独立的 COCO 入口字段构建，未执行完整 Paddle DataLoader。 [实际汇总](results/MASK_RTDETR_HRSID_FIT_STOCK_CPU_GEOMETRY_ACTUAL_SAFE_20261111.json)

| 小目标读数 | 原生 A | 中心 B | 组等权差值 |
| --- | ---: | ---: | ---: |
| 纵向绝对质心误差，预缩放像素 | 0.914 | 0.191 | 减少 0.723，分组分位区间 0.602–0.855 |
| 纵向有符号质心偏差，预缩放像素 | +0.911 | −0.005 | −0.916 |
| 横向绝对质心误差，预缩放像素 | 1.005 | 0.273 | 减少 0.731 |
| 连续栅格足迹 IoU | 0.7850 | 0.8739 | +8.890 个百分点 |
| 输出非空率 | 100.000% | 99.960% | −0.040 个百分点 |

共同非空质心分析覆盖 578 条，全部 24 个来源代理组的纵向误差均减少。小目标总计 581 条，3 条仅原生非空，均发生在图像也抽到最近邻的裁剪路径。原始逐行非空率下降 3／581，即 0.516 个百分点。总体达到预锁几何门槛。裁剪与未裁剪的纵向绝对误差分别减少 0.614、0.764 像素。排除图像最近邻的 23 个组中，纵向误差减少 0.755 像素且无新增空行。尺度不变的一个组含 9 条小目标，两个结果逐值相同。

复验原 96 个视图后，除耗时外全部安全字段一致。四张源图在四次重复中始终共处一个批次块，组重采样区间条件于这个固定安排。额外逐一去掉六个四图块，每次余下 20 个组，纵向改善范围 0.675–0.752 像素，IoU 增益 8.302–9.310 点。两次原始 CPU 进程均实际等待退出 0，耗时 21.938 与 21.422 秒。主运行峰值 RSS 2,372,190,208 字节，复验为 1,950,232,576 字节。未打开 DEV 或 TEST，模型前向、更新和 AP 评分均为零。 [固定批次块复核](results/MASK_RTDETR_HRSID_FIT_BATCH_BLOCK_SENSITIVITY_ACTUAL_SAFE_20261111.json)

结果表明，当前 RF-DETR 诊断中的取样相位现象也出现在固定版本 Mask RT-DETR 的 OpenCV 掩膜缩放。效用限于源栅格的连续几何，仍需用等预算训练检验模型 AP 与图像对齐。三个空行提示应记录相同实例保留或单列极小目标的取样代价。方法的主线证据仍取决于正在运行的 RF-DETR 共同行实验与 SSDD 外部验证。

## 复现入口

在独立 CPU 环境中固定 Python 3.10、NumPy 1.26.4、OpenCV 4.5.5.64、pycocotools 2.0.8、Pillow 12.3.0、PyYAML 6.0.3。将官方三个源文件下载到单独目录，保留文件名。源码加载时逐一核对摘要。使用已授权取得的相同封存 FIT114 子包，调用 `scope.zip_digest`、`scope.read_catalog` 和 `scope.make_plan`，以 `scope.canonical` 写出私有预锁清单。然后以 `CUDA_VISIBLE_DEVICES=-1` 和单线程运行 `replay.py --bundle <FIT114包> --plan <清单> --upstream <作者源码目录> --output <新结果> --protocol-commit <预锁提交> --plan-sha256 <清单摘要>`。合成测试读取同代码目录下的 `UPSTREAM`。数据、原始清单与外部作者代码未随结果发布。 [回放代码和测试](code/MASK_RTDETR_HRSID_FIT_CPU_GEOMETRY_20261111/) · [作者算子](https://github.com/PaddlePaddle/PaddleDetection/blob/b25522a0f4bde8c80603f3ba5e3472059972e3b5/ppdet/data/transform/operators.py) · [作者批次算子](https://github.com/PaddlePaddle/PaddleDetection/blob/b25522a0f4bde8c80603f3ba5e3472059972e3b5/ppdet/data/transform/batch_operators.py)

[同视图的原始多边形参考复核](MASK_RTDETR_HRSID_FIT_POLYGON_AND_SUPPORT_PROTOCOL_20261111.md#实际结果)保留方向。连续多边形 IoU 提高 6.874 点，纵向绝对质心误差减少 0.424 像素。三条新增空行仍有可见面积，当前二值输出网格下各自最高连续 IoU 为 0.289、0.344、0.335。

## OpenCV 5 原生与精确最近邻实测

在独立 Windows x64 CPU 进程中核对 PyPI 的 OpenCV 4.5.5.64 和 5.0.0.93 wheel，分别使用 NumPy 1.26.4、2.2.6。按整数像素中心规则 `floor((2j+1)N/(2M))`，精确半格取较大索引。实际比较 4,115 组一维尺寸、145,248 个坐标位置和 12 组二维合成二值掩膜。尺寸含 800→576、800→512、800→640 和 640→576。关闭 OpenCL，分别开启和关闭 CPU 优化，所有对应输出摘要相同。 [复现脚本](code/OPENCV_NEAREST_VERSION_CPU_20261111/qualify_nearest.py) · [合成实测结果](results/OPENCV_NEAREST_VERSION_CPU_ACTUAL_SAFE_20261111.json)

两个版本的默认 `INTER_NEAREST` 在全部测试输出逐字节一致，均有 58,037 个坐标位置偏离整数参考，53,857 个不在半格平局位置。800→576 的 384 个偏离全部为非平局。固定版本迁移说明对默认模式的描述与所测 wheel 表现存在出入，5.0.0 源码的两个入口仍分别派发。显式 `INTER_NEAREST_EXACT` 在 OpenCV 5 所测非平局位置全部与参考一致，半格位置仍有 1,090 个差异。OpenCV 4 精确路径有 2,950 个半格和 13 个非半格差异。合成结果支持把 OpenCV 5 显式精确模式纳入后续同输入比较，原生默认、显式精确和整数规则各作为独立对照。实际掩膜与模型指标仍需同条件评估。 [固定迁移说明](https://github.com/opencv/opencv/wiki/OpenCV-4-to-5-migration/8ea769425c7697403a1483ac5e7afba2990e726e#nearest-neighbor-resize) · [5.0.0 默认与精确派发](https://github.com/opencv/opencv/blob/5.0.0/modules/imgproc/src/resize.cpp#L4002-L4011) · [5.0.0 精确取样实现](https://github.com/opencv/opencv/blob/5.0.0/modules/imgproc/src/resize.cpp#L1266-L1277)

## OpenCV 5 在真实 FIT 栅格上的直接对照

按此前固定的 24 个来源代理组和 96 个 FIT 视图重跑 OpenCV 4.5.5.64 原增强。24 个原批次全部完成，原报告的所有非计时字段一致。把同一预缩放栅格送给独立进程中的 OpenCV 5.0.0.93。图像、框、类别、增强轨迹和监督行均沿用已核对的原回放。5.0.0.93 和 NumPy 2.2.6 的 Windows wheel 摘要与 PyPI 发布相符。 [OpenCV 5.0.0.93](https://pypi.org/project/opencv-python/5.0.0.93/) · [NumPy 2.2.6](https://pypi.org/project/numpy/2.2.6/)

| 同一预缩放输入的比较 | 不同掩膜数，合计 615 条 | 不同像素数 |
| --- | ---: | ---: |
| OpenCV 5 默认最近邻与 OpenCV 4 默认最近邻 | 0 | 0 |
| OpenCV 5 显式精确最近邻与整数中心 | 0 | 0 |
| OpenCV 5 显式精确最近邻与 OpenCV 4 默认最近邻 | 594 | 未汇总 |

581 条小目标行的组等权连续足迹 IoU 为 0.7850、0.8739、0.8739，顺序为旧版默认、5.0.0 显式精确、整数中心。三臂共同非空行的组等权绝对纵向误差分别为 0.9140、0.1907、0.1907 个预缩放像素。新版显式精确与整数中心之间的新增空行和挽回空行均为 0，两者共享此前相对原生的三条新增空行。合成半格的差异仍需按其原测试范围报告。 [实测与原等待摘要](results/MASK_RTDETR_CV5_FIT96_COMPARISON_ACTUAL_SAFE_20261111.json)

两个独立 CPU 进程均退出 0，作业树清空，整段耗时 26.09 秒和 6.11 秒。此耗时分别包含原增强与保存、读回与算子比较，各阶段工作量不同。数据仅来自原锁定 FIT，模型前向、优化器更新、AP、DEV 和 TEST 均未执行。单次掩膜缩放在本组真实输入的几何收益可由库内显式精确模式复现，应纳入直接对照。裁剪与多步变换统一后再单次栅格化的额外价值，继续用已锁定的 B/C 同预算长训与固定 DEV705 检验。 [实际比较源码](code/MASK_RTDETR_HRSID_FIT_CPU_GEOMETRY_20261111/real_cv5_mask_compare.py)
