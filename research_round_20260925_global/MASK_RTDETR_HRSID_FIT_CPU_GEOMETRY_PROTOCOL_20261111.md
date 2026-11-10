# Mask RT-DETR 的 HRSID FIT 掩膜采样对照

研究问题是第二框架的最近邻掩膜缩放是否出现同方向的纵向几何偏差。固定 PaddleDetection v2.9.0 提交 `b25522a0f4bde8c80603f3ba5e3472059972e3b5`。CPU 试验复用官方类别的 NumPy、OpenCV 路径，保留多边形栅格化、颜色、扩展、裁剪、翻转及每批共用的尺寸和图像插值。官方掩膜支路 A 用默认最近邻，B 仅在同一预缩放栅格上替换为整数中心映射 `floor((2j+1)N/(2M))`。官方配方的非整数缩放只有一步。读取器类别边界与原生 PaddleDataLoader 的导入、注册及模型训练分开记。 [官方读取器](https://github.com/PaddlePaddle/PaddleDetection/blob/b25522a0f4bde8c80603f3ba5e3472059972e3b5/configs/mask_rtdetr/_base_/mask_rtdetr_reader.yml#L2-L21) · [掩膜缩放](https://github.com/PaddlePaddle/PaddleDetection/blob/b25522a0f4bde8c80603f3ba5e3472059972e3b5/ppdet/data/transform/operators.py#L1010-L1022)

预先锁定封存 FIT114 中 24 个不同核心文件名前缀和不同编码内容，按 `HRSID_FIT_MASK_RTDETR_STOCK_A_B_20261111_V1` 与原 PNG 摘要拼接后的 SHA-256 排序。每张用四次不同增强，六个四图批次在四轮上合计 96 个视图。批次种子为 `712100 + 100r + b`，r 从 0 到 3，b 从 0 到 5。Python 和 NumPy 随机过程共用。源数据为已验收的 FIT114 训练子包，包摘要 `af12a4e86ea36e9d7fe6ac47b17c684a4812baba6aa9be019f889f833e0f577c`。不披露图像、实例标注、成员名或逐图索引。病例与阈值锁 SHA-256 为 `3d78297dd49a99be208a91f4bf45d2d56a5d0f1ae55b80328dbb4a42de6102ed`。版本固定为 OpenCV 4.5.5.64、pycocotools 2.0.8 和 NumPy 1.26.4。 [既有 FIT 与机制协议](HRSID_MASK_PHASE_COMMON_SUPPORT_AND_SSDD_TRANSFER_DESIGN_20261109.md#第二框架的原生掩膜取样入口)

参考为**同一预缩放栅格的前景像素单元连续缩放后的覆盖**。它用于测量取样相位，并保留栅格化与整数裁剪的既有效果。主要人口为参考连续面积小于 1,024 个输出像素、两臂共同非空的监督行。先按来源图聚合，再以 24 个来源代理组中有有效行的组等权。主要指标为 A 减 B 的绝对纵向质心误差，换回预缩放坐标的像素单位。PCG64 31991 做 2,048 次组重采样，晋级条件为至少 16 个有效组、主要差值至少 0.10 像素且分位区间下界大于零。全部小目标的组等权连续面积 IoU 差须至少 −0.002，非空率差须至少 −0.01。后两项计入参考有效但输出为空的行。方向、横轴、裁剪、缩放方向及图像是否抽到最近邻单列。图像最近邻与 A 的取样坐标关系可能改变图像和监督的一致性，独立参考几何的收益不支持图像对齐或 AP 结论。

保留无效、多边形或掩膜为空的行数及 A/B 各自和共同的保留数。只有 A/B 成功共用图像、框、行顺序及随机过程，才能解释其几何差值。此次试验限定 FIT、CPU 和掩膜坐标。实际数据、读数及关闭证据将在执行后补录。
