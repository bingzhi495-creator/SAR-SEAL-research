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
