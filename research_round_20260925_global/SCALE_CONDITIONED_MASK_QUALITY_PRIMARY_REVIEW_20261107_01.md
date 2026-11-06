# 尺度条件下的Mask质量排序：四项直接先例与最小诊断短记

检索日期：2026-11-07。状态：**有界原始来源研究；未实现、未运行新诊断、未训练、未上传。** 本文只读取指定普通既有综述与结果报告；没有读取科学SOURCE、实际Ops、GT、预测、权重，没有SSH、改旧文件、下载论文全文或打包。FULL/PLAIN锁定E50仍在运行，本文不修改其设置、停止规则或预告端点结果。

## 已有证据与此次新增内容

先读了发布副本 `research_round_20260925_global/` 中的 `MASK_QUALITY_RASTERIZATION_PRIMARY_RESEARCH_NOTE_20261107_09.md`、`MASK_QUALITY_ANNOTATION_CONVENTION_PRIMARY_REVIEW_20261107_01.md`、`N26_FROZEN_DUALGT02_ACTUAL_RESULT_20261107.md` 及有限复核安全JSON。旧综述已覆盖Mask Scoring R-CNN、边界度量与标注约定；本轮增加2024年实例分割后校准先例、2025年class-mask解耦及small/medium折衷先例，并核到更早的尺度条件校准。不是重新验证项目原始实验。

项目已报告：同一已暴露DEV187图/380实例，R掩膜、全部非score字段与顺序相同，只消费三个已冻结下游质量头的历史最终分数；不重新计算qhat或乘分。整数GT：AP 68.099385→69.537367，平均+1.437982；候选浮点GT：67.219522→68.821633，+1.602111，三个头各自均正。AP_small均值分别67.743992→69.408816与66.854138→68.570202；AP_medium分别69.742423→69.451422与69.590658→69.196038，三头在两版本medium项都下降。AR10微降，AR100各GT内部不变。

这些事实支持“固定几何的总体排序效用在两套评价约定内均为正”，同时存在medium代价。两GT的area分组成员未逐项核同；这不是物理船长效应，也不是尺度偏差已经证实。三个头共享同一教师/R，不能当三个完整模型种子。

## 直接先例差异表（保留四项研究，含各自作者实现）

| 原始研究与实际访问范围 | 已处理的问题 | 本项目与其具体差异及创新约束 |
|---|---|---|
| Huang等，**Mask Scoring R-CNN**，CVPR 2019。[CVF记录](https://openaccess.thecvf.com/content_CVPR_2019/html/Huang_Mask_Scoring_R-CNN_CVPR_2019_paper.html)本次直接访问403；实际核读[作者固定提交的后处理](https://github.com/zjhuang22/maskscoring_rcnn/blob/0e8fae677b8ab987e4cdd3e894f17c25410388d8/maskrcnn_benchmark/modeling/roi_heads/maskiou_head/inference.py#L32)，未通读论文全文。 | 分类分数与mask质量不一致，作者实现选取类别对应的预测MaskIoU，并与bbox score相乘。 | S0×qhat具有直接原则先例。N26的冻结R、两GT诊断和具体质量头是不同实验设置；不能把乘分或IoU质量头改名当创新，也不擅自把本项目S0等同该论文分类分数。 |
| Küppers、Haselhoff、Kronenberger、Schneider，**Confidence Calibration for Object Detection and Segmentation**，2022年Springer开放章节。[出版方HTML](https://link.springer.com/chapter/10.1007/978-3-031-01233-4_8)、[arXiv元数据](https://arxiv.org/abs/2202.12785)。实际访问HTML正文，重点§1、§5.1–5.2；未核其代码。 | 检测校准已条件化于位置、宽高/形状；实例分割部分校准的是像素置信度，并使用归一化像素位置/边界距离。单调分数校准与改变排序须区分。 | “加入尺度条件做校准”不是新原则。其检测框尺度条件和像素校准不等于本项目整张实例mask的IoU排序，也未据此证明N26存在尺度偏差。 |
| Kuzucu、Oksuz、Sadeghi、Dokania，**On Calibration of Object Detectors: Pitfalls, Evaluation and Baselines**，ECCV 2024。[arXiv](https://arxiv.org/abs/2405.20459)、[HTML（见附录D.4）](https://arxiv.org/html/2405.20459v1)、[论文链接确认的官方仓库](https://github.com/fiveai/detection_calibration)。实际访问HTML正文/附录相关节及仓库README；未审实现源码或复现。 | 用廉价Platt Scaling/Isotonic Regression做后校准，强调校准与准确性分开评价；附录D.4已用mask IoU作为目标，测试HTC、QueryInst、Mask2Former。README明确支持`eval_type=segm`。 | “不改几何、冻结模型后校准mask score”已有直接先例。该实例实验报告校准误差/LRP，没有在所核节中建立本项目式small增、medium降的尺度诊断；不能用AP增益直接声称校准误差下降。 |
| Lin、Lu、Shen、Zhu、Zhang、Cao、Ji，**Pseudo-Label Quality Decoupling and Correction for Semi-Supervised Instance Segmentation (PL-DC)**，2025，**本轮只核到arXiv预印本，未确认正式venue**。[摘要](https://arxiv.org/abs/2505.11075)、[HTML §3.1/§4.3](https://arxiv.org/html/2505.11075v1)。实际访问HTML相应方法、Table 3及附录E；未核作者实现，未把同名仓库当证据。 | class×mask单分数不能分别约束两种伪标签质量，改为双阈值。Table 3在COCO 1%、70k消融中：用耦合阈值代替DDTF，mAP 21.6→21.1，AP_small 7.0→7.3，AP_medium 21.4→20.4。 | 已有明确small/medium折衷先例。它改变半监督伪标签筛选和训练；mask质量为前景像素概率均值，与N26学得IoU头、冻结推理重排不同。单类别DEV不能直接套用其多类混淆解释；其机制解释仍是作者在该设置的解释。 |

四项合起来否定三个过宽候选：“质量乘分是新方法”“尺度条件校准原则无人处理”“class-mask质量折衷及small/medium方向相反是首次发现”。**不否定**SAR特定误差机制或经过公平验证的工程整合可以有贡献；只是现在尚没有那个差异证据。本轮未在保留的四项中确认一项已完整覆盖“同一SAR冻结候选、两GT约定、下游三头、尺度分组一致性”的研究，也不能据这个有界结果宣称领域空白。

## 最强替代解释

**总体有效的质量重排，可能只是把TP/FP和重复候选的相对次序重新安排；medium项的损失来自有限实例上的排序及COCO匹配/area-ignore交互，而非质量头存在可迁移的尺度条件偏差。** 这与固定几何、总体AP上升及AR100相同相容；AR10微降另外说明前十名预算内可召回情况有代价，不是mask变差的证据。AP不等同于预测IoU校准；相同AR也不能保证TP身份、重复候选和排序完全相同。

该替代解释不是已证机制。尤其不能把两GT之间area成员差异当作已查实的原因；在每一套固定GT内部，N26相对R的medium下降是真实需保留的评价结果。此前“全靠整数约定才有排序收益”的简单解释已被两GT均正收缩，但不能由此跳到“尺度偏差就是剩余原因”。

## 一个可否定的诊断假设，而非新方法立项

现有证据不足以确定有差异性的新方法假设。只保留一个更窄的候选H：**排除重复匹配竞争后，在同一组具有唯一GT对应的固定候选上，N26相对R的mask质量排序一致性在medium组下降、small组不下降，而且三头及两套GT均同向。** 它检验尺度相关的候选内排序失配是否还存在；不预设新增尺度分支、分组阈值或校准器能改善AP。

未来如Root在既定主实验端点之后另行授权并满足原执行资格，可先做一个不训练、不拟合参数、不改历史分数的诊断：

1. 逐GT ID核对两GT的area分组，仅保留在两套约定中均为small或均为medium的共同成员，公开保留/排除计数；若无法形成足够样本，报告H不可判定。
2. 在每套GT中，用原冻结mask的IoU建立与分数无关的候选—非crowd GT关系；仅保留IoU≥0.50关系图中两端度数均为1的对应，并对两GT取共同候选ID。这样排除该阈值以上的重复争抢，不调用新模型或生成新分数。该子集不覆盖背景FP和漏检，不能替代canonical AP。
3. 对该子集，用每个候选的真实mask IoU定义质量档位：在原COCO十个阈值0.50:0.05:0.95上通过阈值的比例。分别按small/medium，比较R最终冻结score与三个N26最终冻结score对不同质量档位候选的配对排序一致率；相同质量档位不比较，score并列计半。报告实际GT/图像数，不能用大量相关候选对伪装独立样本量。
4. 只检验预先写定的方向：若任一头/GT的medium一致率未降，或small一致率下降，则这个“六组均同向”的H被否定；样本不足/不确定时记未确认，不选择有利头或GT。即使H成立，也只说明所选唯一对应子集的排序失配，尚不能把总AP损失唯一归因于尺度，更不能证明新增容量有效。

这里没有设置H的新颖性评分或正收益门。若H被否定，应优先回到重复候选、FP排序及area-ignore的有限诊断，而不是继续堆尺度模块；若H保留，下一阶段仍须与已知条件校准和廉价后处理比较，并在独立数据、可比预算与完整模型重复中检验效用。

## 检索边界与未确认事项

检索先围绕2024–2026的`instance segmentation / score calibration / mask quality ranking / scale bias / class-mask mismatch`组合，再沿直接原始链接核对；最后保留四项，未扩展backbone或模块清单。检索不是系统综述，2026年以及最近六个月的定向查询没有提供足以替换上述直接先例的已核来源；这不是“不存在近期先例”的结论。arXiv HTML访问属于相关全文节实际阅读，但没有全文下载/本地保存。除2019固定后处理外，没有审计其余作者实现或执行外部代码；2024仓库只确认官方归属和README所宣称接口，不能当实现正确性核验。

指定Codex MCP审阅后端不在本次可用工具中，**[pending Codex review]**，未冒称跨模型复核。两GT的area成员一致性、唯一匹配子集、H方向、误差来源和物理尺度解释均未验证；FULL/PLAIN E50结果仍未知。本轮实质进展是补充直接先例并否定过宽创新候选，不是新增实验成功或TGRS就绪证据。
