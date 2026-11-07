# RF-DETR E100 已启动与等预算修边诊断方案

2026-11-07。DGX 上一次全新 RF-DETR Nano BF16 连续 E100 原任务已启动。UTC 15:28:47 的独立同启动会话进程检查识别到唯一原 worker，GPU 计算进程列表确认同一进程。原进程最终退出、实际完成轮数、档案写完及独立闭合均等待后续核验。沿用 COCO Nano 初始化、SSDD 固定 FIT741、seed 5201、576 输入和预先固定的 E100 调度。预定 74,400 个微批、18,600 次优化器更新和三个原生归档 E24/E50/E100。E100 EMA 是主要端点，较早档用于学习曲线。沿 E24 的实测时间线性外推约 6 小时 8 分钟，实际耗时以原任务为准。本次训练不读取测试集，不计算 AP。 [原生连续训练实现](code/RFDETR_NATIVE_UNINTERRUPTED_E100_WORKER_CANDIDATE_20261107/e100_worker.py) · [同方案的已验收 E24](RFDETR_NATIVE_CONTINUOUS_E24_STATUS_20261107.md)

## 冻结修边诊断

独立于 E100 的下一项诊断沿用 E24 原查询、固定分数、原图网格和已验收的两版标注。沿原 377 个固定配对，取同一整数 GT 最高 IoU 原始查询，同分取较小编号。主要条件人口沿用已定位的 224 个双原始最高 IoU 均低于 0.90 的配对。另列全 377 中该子集新增跨阈值的份额。三个原无配对实例单列。对 153 个其余配对若做扩展统计，单列为补充，并明确相同整数最优查询在浮点 GT 上的原始状态。缺少原始查询可达性的三例或其他配对不替换成最邻近实例。

在原有限网格的 2 像素共同 GT 边界带内，只允许补入两版共同认可的漏分，或删除两版共同认可的溢出。设可补、可删像素数为 F、P，各版 GT 与同一预测的交集、并集为 Iᵥ、Uᵥ。补 a 个、删 b 个后的 IoU 为 (Iᵥ+a)/(Uᵥ−b)。共同正确编辑使两版 IoU 单调不降。固定 k=a+b 时，先补再删同时最大化两版 IoU，因为 k≤F+P≤Uᵥ−Iᵥ。使用整数条件 10(Iᵥ+a)≥9(Uᵥ−b)，避免浮点临界点差异。双通过所需编辑计数满足 10a+9b≥max(0,9Uᵢ−10Iᵢ,9U𝒻−10I𝒻)。令 a 为 F 与上述差额除以 10 向上取整的较小者，再取满足条件的最小 b。若 b>P，带内无法达到双通过。仅补漏、仅删溢出分别置另一种容量为 0。Iᵥ、Uᵥ须从原选中查询与两版 GT 逐像素重算，不能由汇总数反推。

固定展示 K=0、1、2、4、8、16、32 像素，并保留无预算上限的带内可达数。逐对保存最少编辑数以画完整阶梯曲线。分别报告整数、浮点和双版跨 IoU 0.90 的数量，带内单方向、双方向及同预算带外对照，以及各 GT 的原始基线达标状态。各曲线的尺度依赖通过按整数 GT 面积预先分为 COCO 小、中、大描述。每个实例的预测选择、分数和标注固定。算法先验证精确整数计数与小规模穷举相符。这里的上限允许离散像素和不同实例独立编辑，含 GT 信息，不测排序、实例冲突、AP、真实可部署性或伤害率。当前未执行实际诊断。 [边界定位结果与完整范围](RFDETR_E24_BOUNDARY_LOCALIZATION_ACTUAL_20261107.md) · [BPR 原文表 1 的预测边界邻域真值上限](https://arxiv.org/html/2104.05239) · [BPR 作者实现](https://github.com/chenhang98/BPR)

上限计算的独立数学实现和四项合成测试已完成，包括四像素所有可实现计数与暴力枚举对照。尚未绑定或读取真实预测与标注。 [精确整数源码](code/RFDETR_E24_SIGNED_EDIT_ORACLE_MATH_CANDIDATE_20261107/signed_edit_oracle.py) · [测试](code/RFDETR_E24_SIGNED_EDIT_ORACLE_MATH_CANDIDATE_20261107/test_signed_edit_oracle.py)

## 进入模型训练前的对照

先从同一预测的边界带检查候选支持，使用 FIT 内预留图像估计带可覆盖的共同错误量。用同一个修正头、预算、无 GT 候选集和训练设置，比对均匀取点、PointRend 式负绝对 logit、简单形态学和输入灰度线索。候选灰度统计可用两侧邻域的平均秩或成对比较比例，例如 2U/(mn)−1，平灰度按半票。严格递增灰度映射保持次序和并列关系，量化、裁剪和局部增强可能改变它。SAR 邻域像素相关，秩统计仅作特征，不输出独立同分布假设下的 p 值或恒虚警声明。以同等预算的条件统计检查可修复、过度修改和未覆盖实例，预先固定模型比较的主要端点后才进入训练。历史同 K 行动选择已出现理想上限而实际选择器无稳健增益，本次分开报告两级证据。 [PointRend 作者源码](https://github.com/facebookresearch/detectron2/blob/main/projects/PointRend/point_rend/mask_head.py#L26-L46) · [秩与 Census 原文](https://courses.cs.duke.edu/spring06/cps296.1/handouts/Zabih%20Woodfill%201994.pdf) · [SciPy 平均秩与 U 统计源码](https://github.com/scipy/scipy/blob/v1.17.0/scipy/stats/_mannwhitneyu.py) · [scikit-image 形态学灰度梯度源码](https://github.com/scikit-image/scikit-image/blob/v0.26.0/src/skimage/filters/rank/generic.py) · [历史行动选择结果](N17_FULLDEV_QUALIFIED_FIXED24_RESULT_AND_STOP_20261015.md)

雷达方法以 [数学、物理与雷达信号处理原始证据表](RADAR_MATH_PHYSICS_SIGNAL_PRIMARY_REVIEW_20261107_01.md) 和 [经典雷达处理原文与源码补充](CLASSICAL_RADAR_SIGNAL_SOURCE_SUPPLEMENT_20261107_01.md) 为入口。原数据为显示灰度，优先验证可观测的灰度秩特征。跨模态、SLC 相位、极化及物理恒虚警标定在可配准或可标定输入得到验证后进入独立数据方案。
