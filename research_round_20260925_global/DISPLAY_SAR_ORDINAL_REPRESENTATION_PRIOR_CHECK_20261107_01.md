# 8-bit 显示 SAR 的局部秩表示查新

**2026-11-07 · 文献与源码静态核验 · 待检验假设。** 单独使用局部秩、局部 CDF 或保留并列灰度，均不足以构成论文核心贡献。值得保留的低优先级假设是，已证实对显示映射敏感的实例分割模型，能否利用并列比例，在同容量和同增广条件下减少可定位的掩膜错误。项目尚无这项误差证据。

## 六组最直接的先例

| 先例 | 核验结果与对本候选的约束 |
|---|---|
| **SAR 非参数秩和检测** | Xiangwei Meng，IEEE JSTARS 18，2025-02-13。Wilcoxon CFAR 合并待检和背景参考单元排序，使用待检样本的秩和阈值。原文的零假设模型依赖同分布、独立且连续的参考与待检样本。Meng 2026 年又研究了多脉冲、多目标参考窗中的修正秩和 CFAR。两项作者实现均未核实。单个待检像素的局部经验秩与之紧密相关，实例掩膜网络不能继承其虚警率结论。常规 OS-CFAR 中乘常数的顺序统计量阈值也不等价于对任意严格单调映射保持不变的秩比较。 [2025 正刊](https://doi.org/10.1109/JSTARS.2025.3533140) · [作者稿第 II 节](https://arxiv.org/pdf/2402.18579) · [2026 修正方法](https://doi.org/10.1186/s13634-026-01336-w) |
| **SAR 局部背景门控** | CaSA-DETR，Remote Sensing 18(21), 3799，版本记录为 2026-11-05。它在检测器的步长 8 和 16 特征上，用参考窗均值与标准差归一化后做乘法软门控。已覆盖“局部背景统计辅助小船识别”的泛化表述。这里拟考察输入图像上的固定邻域秩、并列比例与实例掩膜错误，属于不同的具体机制和任务。正刊未给出可核实作者代码入口。 [正刊方法](https://doi.org/10.3390/rs18213799) · [版本记录](https://www.mdpi.com/2072-4292/18/21/3799/notes) |
| **Rank、Census 与完整秩** | Zabih 和 Woodfill，ECCV 1994。Rank 计较暗邻点数，Census 保留中心与各邻点比较的空间分布。Demetz 等，BMVC 2013，Complete Rank Transform 对邻域每点编码严格小于它的数量，保留含相等关系的完整弱序。用三值中心比较编码也可还原两端点，且保留空间位置。Meister 等 AAAI 2018 的 UnFlow 已把 Census 风格比较用于神经网络训练。其作者代码 `ternary_loss` 使用软归一化强度差，幅值随非线性单调变换改变，适合作为软 Census 参考。 [Rank/Census 原文](https://courses.cs.duke.edu/spring06/cps296.1/handouts/Zabih%20Woodfill%201994.pdf) · [CRT 官方存档](https://www.bmva-archive.org.uk/bmvc/2013/Papers/paper0050/abstract0050.pdf) · [UnFlow 原文](https://arxiv.org/abs/1711.07837) · [UnFlow 作者源码固定提交](https://github.com/simonmeister/UnFlow/blob/bf0c09972dca0bc139c5ed39f25cc25c86f79037/src/e2eflow/core/losses.py#L83-L113) |
| **AHE 与 CLAHE** | Pizer 等 1987 已研究自适应直方图均衡。scikit-image `v0.25.2` 的 `rank.equalize` 内核 `_kernel_equalize` 对滑动足迹中的直方图累加到中心灰度，含同值样本，乘输出尺度后取输出类型。相同足迹和边界条件下，缩放及舍入前的统计量就是候选右端点。该版 `_clahe` 另外对分块直方图截断、再分配并插值，`map_histogram` 用累积频数构建 LUT。 [AHE 原文](https://doi.org/10.1016/S0734-189X(87)80186-X) · [滑窗等化源码](https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/filters/rank/generic_cy.pyx#L40-L56) · [CLAHE 源码](https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/exposure/_adapthist.py#L94-L290) |
| **全局秩预处理 RBT** | Chen 和 Nordling，Signal Processing: Image Communication 141，2026 年 2 月卷期。作者 MATLAB `RBT` 用 `unique` 与 `ismember` 对全图不同灰度等距映射，忽略各灰度频数，与滑窗经验 CDF 有明确差异。原文用 Cellpose 展示视觉收益，不能作为 SAR 实例掩膜证据。静态核对同提交 Python `rbt` 发现零起点 `return_inverse` 后又减一。两版在单一灰度时分母为零。复现应明确使用 MATLAB 逻辑与常量输入约定，并验证边界值。仓库固定提交许可证为 Apache-2.0。 [正刊](https://doi.org/10.1016/j.image.2025.117432) · [作者 MATLAB](https://github.com/nordlinglab/RBT/blob/3d4736c428dbba49b3d8bb6c8cbb3d8f6c898619/RBT.m) · [作者 Python](https://github.com/nordlinglab/RBT/blob/3d4736c428dbba49b3d8bb6c8cbb3d8f6c898619/RBT.py#L4-L21) · [许可证](https://github.com/nordlinglab/RBT/blob/3d4736c428dbba49b3d8bb6c8cbb3d8f6c898619/LICENSE) |
| **学习式 IA-CLAHE** | Otsuka 等，CVPR Workshops NTIRE 2026。轻量 CNN 从输入亮度图估计分块截断阈值，通过可微分 CLAHE、合成退化与干净图像的 L1 训练。已覆盖“学习局部直方图变换以改善下游识别”。论文实验为可见光增强、分类与检测。作者实现未核实，上述 scikit-image 是普通 CLAHE 的独立实现。 [会议论文](https://openaccess.thecvf.com/content/CVPR2026W/NTIRE/html/Otsuka_IA-CLAHE_Image-Adaptive_Clip_Limit_Estimation_for_CLAHE_CVPRW_2026_paper.html) · [作者方法第 3–5 节](https://arxiv.org/html/2604.16010) |

定向检索覆盖 2025–2026 年 SAR rank-CFAR、SAR ordinal/rank invariant、Census neural 与 adaptive CLAHE，以及方法直接引用的早期先例。检索范围内未核实同一 8-bit SAR 实例掩膜设置中的同构对照。此项检索空缺无法建立首创性。源码仅作静态核验。

## 可检验的技术差异与输入边界

对固定且非空的参考足迹 $W$，中心灰度为 $v$。令 $r_- = \#\{x_i<v\}/|W|$，$r_+ = \#\{x_i\le v\}/|W|$。前者为经典局部秩，后者为未截断的局部 CDF，$t=r_+-r_-$ 是观测的同值比例。中秩 $m=(r_-+r_+)/2$ 与 $t$ 是两个端点的等价表达。右端点作为单通道预处理等价于滑窗均衡的基本映射。保留原图并将 $(m,t)$ 并入固定容量适配器，技术上可检验的是并列比例的条件效用。

同一足迹应用同一个严格递增点映射时，大小和相等关系都保持。应用非降的量化或截断映射时，原先相等的值仍相等，原先不同的值可以合并。原排序区间被新排序区间包含，因为严格小于的集合只能缩小，小于等于的集合只能扩大。这是给定映射和足迹上的计数关系。$t$ 同时受平坦纹理、显示量化及饱和影响，不能用于识别损失来自哪种因素。空间变化映射、重采样、极化、成像几何与斑点变化可改变局部排序。显示灰度不能还原定标幅度、复相位或推导 $P_{fa}$。即使秩支路保持不变，融合模型的原图路径仍可能敏感。

## 是否值得进入实验

1. **先验证误差机制。** 在同一冻结模型上配对比较同一影像的原始视图和预先规定、保留浮点精度的严格递增映射，另设量化或截断，以及几何与噪声对照。256 个输入级到 256 个输出级的严格递增整数 LUT 只有恒等映射。固定图像坐标、足迹、边界规则和插值顺序。分别观察小船和近岸船的漏检、实例合并与掩膜边界漂移，单独做匹配实例的掩膜比较，并保留全流程 GT mask AP，以区分检测和掩膜误差。稳定性需与 GT 质量一起评估。
2. **出现可复现误差后再做最小归因对照。** 固定骨架、初始化、数据划分和训练预算，所有组使用完全相同的 LUT 增广。对照包括原生模型、同容量原图适配器、同架构的普通局部秩或 CLAHE/Census，以及原图加中秩时真实 $t$ 对固定零值或保留边际分布的对照。报告计算量和洁净输入性能，并按基线纹理和饱和分层。进一步在同一影像配对中比较量化前后的 $t$ 变化，减少将背景纹理误当作量化效应。真实跨传感器结论需要传感器及场景来源可追溯的独立测试，合成 LUT 只识别给定干预。
3. **预先设定降级条件。** 若原生强模型无实质性掩膜敏感、同 LUT 增广消除差异，或普通秩/CLAHE/Census 已解释全部收益，该路径保留为诊断或普通预处理。若 $t$ 的作用只来自明显的饱和区域，应继续检验其能否超过饱和指示等简单替代量。只有可定位误差、同容量归因、重复种子和来源独立检验共同成立，才评估为研究贡献。
