# 掩膜残差读出与校准的直接先例核查

本轮文件名沿用 20261012；实际核查时间为北京时间 2026-10-13。状态：针对性文献核查与静态源码对照已完成；下述机制假设未确认，新对照尚未执行。此文档由 ROOT 整理后同步 GitHub，不据此宣称研究目标完成。

## 核查范围与当前依据

本轮只检索原论文、作者代码及作者机构存档，并只读独立发布副本中的普通源码文本。没有读取、哈希、导入或执行原科研目录中的科学载荷、权重、GT、RAW 或 SPEC。本轮不是领域穷尽查新，也没有完成外部方法复现。

ROOT 提供的当前已核验事实：同一冻结 Mask2Former/R50 E24，内部 DEV 原生 mask AP 为 67.071057；C 两个下游种子均值为 68.166136，约提高 1.095079 个 AP 百分点；添加原图 JPEG patch 的 R 未胜 C。FIT-only 单一偏移控制的 DEV 评价仍有实现一致性问题，没有可引用的 AP。本文不独立重算这些实验数字，不将 DEV 与外部论文的官方测试 AP 横向排名。

独立发布副本核对结果：

- `cache_fit.py` 的被动 hook 从 `pixel_decoder` 第一输出取得 F，从最后阶段 `mask_embed` 输出取得 e。F 是 256 通道、200×200 的原生 mask feature；e 是每实例 256 维的最终 **mask embedding**，并非原始 query token。源码断言原生 M 与 `einsum(e,F)` 相同。
- 原生 F 和 L 经既定渲染路径投影到原图。C 的输入为 LN(F)、LN(e)、截断缩放的 L 与零 patch，使用共享 `562→64→64→1` MLP；输出在原图 `|L|≤2` 内相加到 L，带外保持原值，末层零初始化。C 和 R 均列出 40,257 个参数，但 C 的零 patch 输入使相应权重不能自动视为与 R 等效活跃容量。
- `evaluate.py` 保留 100 个原生候选、类别、顺序及 S0 分数，仅改变 segmentation，不增加质量评分头。`|L|≤2` 是预测不确定带，不能不经检验称为真实船体轮廓带。插值后的粗 F 不能称为新增原图细节；JPEG patch 也不是复数 SAR 或物理散射测量。

核对对象均为独立发布副本：`research_round_20260925_global/multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N07_frozen_E24_original_detail_point_refinement_source_20261012_01/` 下的 `cache_fit.py`、`point_heads.py`、`evaluate.py`。仅源码静态核对不能替代实际运行资格。

## 锁定四个直接先例

| 原始工作与证据状态 | 已有技术与本项目的关系 | 本项目当前可描述的差异及必要对照 |
|---|---|---|
| **PointRend**，Kirillov、Wu、He、Girshick，CVPR 2020，已发表。核查原文方法段与官方 StandardPointHead。 | 结合空间特征和粗掩膜，由共享 MLP 做点级预测；训练偏向不确定点并保留随机覆盖，推理逐级细化。实例特定信息并非本项目首次引入。 | C 使用冻结 Mask2Former 的 F/e/L，在原图不确定带进行加性残差而非重建完整掩膜。冻结、残差、换载体及带约束只是具体差异，尚不能推出新颖性；应与相同冻结表征、点监督和预算下的普通 PointRend 式适配比较。[原文](https://arxiv.org/html/1912.08193v2)、[CVPR 记录](https://openaccess.thecvf.com/content_CVPR_2020/html/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.html)、[作者点头](https://raw.githubusercontent.com/facebookresearch/detectron2/main/projects/PointRend/point_rend/point_head.py) |
| **Pointly-Supervised Instance Segmentation / Implicit PointRend**，Cheng、Parkhi、Kirillov，CVPR 2022，已发表，arXiv v2 与作者代码可核查。 | 每实例生成点头函数参数，输入空间特征和相对坐标，以单一点级掩膜损失训练；实例条件点函数与点监督均已有直接先例。原文也讨论完整掩膜监督，不应因标题含 point supervision 而排除。 | C 拼接最终 mask embedding，使用共享残差 MLP；它不是作者动态参数头。这个结构差异需效用证明，不能把“query-aware 点头”本身写成创新。优先建立同协议实例条件点函数适配；说明它是结构适配而非完整作者复现。[原文 §4](https://arxiv.org/html/2104.06404v2)、[CVPR 记录](https://openaccess.thecvf.com/content/CVPR2022/html/Cheng_Pointly-Supervised_Instance_Segmentation_CVPR_2022_paper.html)、[作者实现](https://raw.githubusercontent.com/facebookresearch/detectron2/main/projects/PointRend/point_rend/point_head.py) |
| **Mask Scoring R-CNN**，Huang、Huang、Gong、Huang、Wang，CVPR 2019，已发表；核查论文摘要、作者仓库及后处理源码。 | 用实例特征和预测掩膜回归 mask IoU，乘以分类分数重新形成 mask score。AP 可以因候选排序更符合掩膜质量而改变，不能只凭 AP 宣称几何修复。 | C 当前固定原生 S0，未新增 IoU 回归或评分重排。已通过的固定分数约束可排除该次比较中的直接重新评分，却不能说明唯一因果机制，也不能说明自适应评分无价值。未来开放评分时需保留固定分数主比较。[论文记录](https://arxiv.org/abs/1903.00241)、[作者仓库](https://github.com/zjhuang22/maskscoring_rcnn)、[评分代码](https://raw.githubusercontent.com/zjhuang22/maskscoring_rcnn/master/maskrcnn_benchmark/modeling/roi_heads/maskiou_head/inference.py) |
| **SAR Ship Instance Segmentation With Dynamic Key Points Information Enhancement**，Gao、Han、Wang、Sun、Hussain、Zhou，JSTARS 17:11365–11385，2024，已发表。作者机构存档明确为 Accepted Manuscript，并给出出版社 DOI。 | 将几何与强散射区域信息编码到动态 mask head 参数，并生成实例级二维高斯注意图；已有 SAR 实例条件几何建模，不能把“实例感知”“关键点”“强散射”当作未出现过的贡献。 | C 没有该关键点编码、散射监督或高斯实例注意，且未取得新的物理测量。两者的机制差异明确，但公平比较尚未完成。本轮核查到机构摘要与稿件来源，未验证完整官方代码对应关系，未独立复现收益。[作者机构存档](https://figshare.le.ac.uk/articles/journal_contribution/SAR_Ship_Instance_Segmentation_with_Dynamic_Key_Points_Information_Enhancement/25507168)、[出版社 DOI](https://doi.org/10.1109/JSTARS.2024.3383779) |

这里的“直接先例”是需要面对的竞争性解释，不表示已经完成同预算强对照。作者论文报告及仓库表格属于作者证据，不等于本项目复现；外部方法的收益和本项目收益均适用同一证据标准。

## 校准控制应怎样解释

置信校准与几何修复需要区分。Guo、Pleiss、Sun、Weinberger 的 ICML 2017 工作研究分类概率校准，报告单参数 temperature scaling 的效用；它不是 SAR 掩膜边界修复的证明。[原始论文入口](https://proceedings.mlr.press/v70/guo17a.html)

以下是数学推论，不是该论文的结论：正温度缩放 `L/T` 在同一数值网格、同一零阈值下不改变 logit 正负；加性偏移 `L+δ` 可以改变支持域。若偏移只作用在 `|L|≤b`、`|δ|<b`，并在**同一网格直接阈值化**，它的二值支持与统一阈值 `L>−δ` 相同。若先做带门控偏移，再插值、裁剪或作其他非线性处理，不能未经核查套用这个等价关系。控制必须明确偏移的施加位置、阈值规则和实际渲染路径。

本项目 FIT-only 单 δ 控制是普通决策偏移的实用替代解释检验，不是新方法，也没有已经证实的 DEV 效用。FIT 点 BCE 降低只能支持该点采样分布上的目标下降；它不是全图概率校准、AP 提升或边界机制成立的证据。该控制若接近 C，可保留低成本效用，但应收缩 C 的复杂机制主张；若不接近 C，也只排除了这个特定单偏移控制。

## 一条可否定假设与后续顺序

**候选假设 H，尚未确认：在冻结原生表征、候选和 S0 下，C 的开发集收益需要由 F 与最终 mask embedding e 支持的空间变化残差，而不能主要由一个 FIT-only 全局决策偏移解释。**

可描述的技术差异是“固定原生点积表征与候选评分后，仅在原图预测不确定带进行共享的实例条件加性适配”。“实例条件点预测”和“掩膜精修”已存在；H 即使得到支持，也不单独证明这组差异优于直接先例或达到 TGRS 要求。

先完成现有评价一致性诊断，再按已锁规则评价单 δ 控制，不能修改阈值或从多种控制事后挑最好者。如果普通偏移达到既定近似替代门槛，停止把空间条件机制当作已有收益的必要解释。若没有达到，再检验下列最少对照，而不是立即添加模块：

1. 以相同数据、初始化、点采样、监督、预算、候选和固定 S0 比较 C 与同载体普通 PointRend 式 F/L 头，以及必要的 Implicit PointRend 式实例条件适配。预先声明适配差异、训练成本与判据；仅匹配参数总数不足以匹配活跃容量、信息和计算。
2. 检查相同候选的像素 FP/FN、支持面积及轮廓指标。只有 AP 提高但没有轮廓改善时，收缩为当前协议中的效用；不得称已解决近岸杂波、相邻船舶或细长船体问题。要主张空间修复，应先出现能够区分局部增删与统一收缩/膨胀的真实误差证据。
3. 若普通适配足以解释收益，承认核心创新仍未形成。只有确证某类实际误差还未被强替代方案处理，再围绕那一类误差设计一处必要的结构改动，预锁能否定它的实验。

不在此文档预先宣称三个贡献已经成立，不在读取结果后新增有利停止规则。重复种子、正式测试、跨数据集及成本证据仍须由 ROOT 按完整目标推进。

## 最新相关工作的状态提醒

O2Former 最初有 2025 arXiv 版本；截至本轮核查，作者所属机构记录已列为 **JSTARS 2026，19:11633–11645，DOI 10.1109/JSTARS.2026.3677088**，不应再仅写为未发表预印本。它在 Mask2Former 上引入多尺度查询与方向嵌入。若后续转向方向、多尺度或骨架改造，应重新核查这一直接邻近先例。本轮仅核验机构发表记录和摘要，未比对期刊版全部公式或独立复现。[机构发表记录](https://pure.kfupm.edu.sa/en/publications/o2-former-direction-aware-and-multiscale-query-enhancement-for-sa/)、[最初预印本](https://arxiv.org/abs/2506.11913)

LFG-Net 作者仓库已列出 SAR 上的 PointRend、Mask Scoring R-CNN 等比较，并采用图像级到实例级的低层特征增强。本轮不据其不同训练与数据协议的数字宣称本项目领先。它提醒我们“补低层特征用于精修”本身也不是空白；R 未胜 C 后，不能继续把 JPEG patch 写成已验证贡献。[作者代码与报告](https://github.com/Evarray/LFG-Net)

结论状态：C 的当前固定协议效用值得保留；创新机制仍未确证。优先用已有单偏移控制和直接先例适配区分必要性，避免把模块名称、参数数量或局部开发集收益替代创新论证。
