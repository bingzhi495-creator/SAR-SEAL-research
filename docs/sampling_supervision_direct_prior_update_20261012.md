# 监督与采样的定点查新补充（2026-10-12）

用途：为下一项设计选择直接先例与必要控制。本文是有限查新，不是完整综述，也不是新颖性已成立的证明。本轮只补核能改变设计的缺口；现有 Mask2Former、PointRend、MaskTransfiner 等记录继续保留。

| 直接先例 | 已核验范围 | 对本研究的约束 |
| --- | --- | --- |
| Zhang & Zhang，HTC+，Remote Sensing 2022 | 出版社正文3.7的索引内容：按末端分类/回归/掩膜损失排序，选择困难负样本；512 ROI、正负1:3。全文直接打开失败，本轮没有独立执行作者方法 | SAR中的背景稀疏、困难负样本与掩膜监督均已有先例；ROI采样不等于当前匹配实例的原图像素配比 |
| Xu等，DiffSARShipInst，ISPRS JPRS 2025 | 出版社摘要/介绍和作者仓库：扩散框候选、FIoU困难样本权重、动态实例掩膜；仓库存在训练配置和模型目录，尚未执行或核验其全部源码 | 不能把“困难样本加权+动态实例条件”直接当新贡献；FIoU的具体作用层级需核对loss源码，不能自动等同像素前景平衡 |
| Gao等，Dynamic Key Points Information Enhancement，JSTARS 2024 | 作者机构库摘要和书目信息：动态关键点编码、实例级二维高斯注意力，用于密集实例重叠与杂波问题。方法全文本轮未成功打开 | 几何条件、强散射关键点及实例空间支持已有直接先例；新方向须指出可实验区分的运算/信息差异 |
| Li、Yin、Chen，YOLO-Argus，AIMS Mathematics 2026，7月29日 | 出版社摘要与图注：并行上下文、多尺度边缘融合、注意力自适应变换、通道解耦；HRSID/PSeg-SSDD作者结果 | 不能把普通边缘增强、多尺度融合或噪声抑制的组合当未经查新的结构创新；摘要所报AP50/AP75不能与本地不同协议主AP直接排名 |
| Yang等，DACF-Net，IJDE 2026，9月8日在线 | 出版社索引摘要、数据/代码声明和当前作者仓库；论文具有几何与多尺度精修主题，方法全文直接打开失败。仓库本次仅见README、LICENSE、gitignore，README说明代码仍在整理 | 这是近期直接相关先例，后续结构查新应补全文/补充材料；当前不能标成已复现，也不能把其缺少代码解释为结果不可信 |

上述作者结果不是本项目的独立复现证据。不得用不同划分或评价数字拼SOTA表。本轮不重复摘录性能数字。

## 当前可作出的设计判断

- 类别平衡BCE、困难样本采样、一般梯度投影和边缘精修均有明确既有机制。即使配方有效，也需另有具体技术差异，才能支撑核心创新。
- 必要简单控制取决于假设：纯原生续训、均匀附加监督、仅改变全局强度，以及最接近的采样/条件建模替代。不要用同一控制回答所有机制问题。
- 当前局部梯度诊断仍须独立CPU资格后解释。负夹角本身不证明有害漂移，局部门槛未过也不否定所有监督设计。
- 后续结构方向应先量化一对一匹配中的漏检、重复、粘连与轮廓误差，确认哪个错误占主导，再确定唯一主要改动。仅凭“船细长/背景多”尚不能决定新模块。
- 可探索的假设（新颖性未确认）：若密集目标的错误主要来自实例间空间支持混淆，比较显式实例间竞争的空间支持机制与独立高斯/关键点注意力。必须先确认错误存在，并以普通mask NMS、直接关键点/高斯先例及容量控制检验；不能把这个方向写成已提出有效新方法。

## 一手入口

- [HTC+，出版社](https://www.mdpi.com/2072-4292/14/10/2395)
- [DiffSARShipInst，出版社](https://www.sciencedirect.com/science/article/pii/S0924271625000887)；[作者代码](https://github.com/XiaowoXuSAR/DiffSARShipInst)
- [Dynamic Key Points，作者机构库](https://napier-repository.worktribe.com/output/3687229/sar-ship-instance-segmentation-with-dynamic-key-points-information-enhancement)
- [YOLO-Argus，出版社](https://www.aimspress.com/article/doi/10.3934/math.2026923?viewType=HTML)
- [DACF-Net，出版社](https://www.tandfonline.com/doi/full/10.1080/17538947.2026.2696657)；[作者仓库](https://github.com/yang199809/DACF-Net)

本轮查新支持必要先例/控制的选择，不支持查新已穷尽或本项目胜过以上方法。

