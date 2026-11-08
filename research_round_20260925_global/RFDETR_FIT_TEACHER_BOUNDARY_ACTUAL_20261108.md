# FIT 训练增强上的冻结教师边界实测

沿用[预注册协议](RFDETR_FIT_TEACHER_BOUNDARY_PROTOCOL_20261108.md)。本机 CUDA 实际处理 24 个 FIT JPEG 组，每组两次真实增强，同一输入分别前向 E24 EMA 与短程 A EMA。48 个视图中有 2 个增强后没有目标。裁剪前后分别计 94 和 78 个 GT 视图实例。原进程退出 0，实际诊断用时 36.80 秒。从 48 条闭合视图记录独立复算全部门槛。

| 固定检验 | 实测 | 预设门槛 | 判定 |
| --- | ---: | ---: | --- |
| 教师 IoU≥0.90 的 GT 次数 | 26 | ≥15 | 通过 |
| 至少一个教师达标实例的 JPEG 组 | 17 | ≥8 | 通过 |
| 两次增强都各有教师达标实例的组 | 8 | ≥4 | 通过 |
| 教师达标后，A 可配对且原始查询几何仍未达标 | 3/26，11.54% | ≥25% | 未通过 |
| 可配对子集带内教师独对 / A 独对像素 | 229 / 106 | 教师独对更多 | 通过 |
| 可配对子集中教师带内严格正净收益的组 | 2 | ≥4 | 未通过 |

3 个可配对实例在该子集中共有 1,914 个带内像素，教师误差为 398，A 为 521。教师在全部 78 个配对上有 4,282 个带内独对像素，A 有 4,417 个，说明窄子集的方向不能外推到全部视图。两端点的一对一严格达标均为 26 次，两次重复按 JPEG 分组计数。3 个 A 严格回落案例均有配对 IoU≥0.50，且 100 个原始 A 查询均未达到 0.90。该试验使用已经过训练的 FIT 图、固定两次增强和第一查询组推理前向，衡量筛选条件的可用性。 [安全汇总](results/RFDETR_FIT_TEACHER_BOUNDARY_ACTUAL_SAFE_20261108.json) · [诊断源码](code/RFDETR_FIT_TEACHER_BOUNDARY_20261108/teacher_boundary_v2.py)

预锁决策为停止这条固定软教师配方。现有信号足以在局部定向补边，出现频率和跨图组分布没有达到训练准入标准。两臂未经过新的软监督训练，未读取 DEV/TEST，也未计算新 AP。优先完成同一连续训练的三个端点，随后才决定是否研究优化器重建造成的几何退化。

## 与最接近先例的差异

| 原始工作 | 已有机制 | 与当前筛查的关系 |
| --- | --- | --- |
| BDKD-Net（2026） | 在冻结教师掩膜构造的形态学窄带内蒸馏软概率 | 已覆盖一般的边界软蒸馏。本文筛查使用 GT 公共边界带和一对一实例配对。 |
| COIN（ICCV 2025） | 以模型与 SAM 的实例 IoU 筛选伪标签，再做递归自蒸馏 | 已覆盖实例 IoU 筛选。作者实现中 SAM 由模型实例质心提示。本文使用公开 GT 和固定阈值。 |
| Li 等（CVPRW 2024） | 独立将教师和学生匹配到同一 GT，再蒸馏实例输出 | 已覆盖同 GT 的跨模型配对。本文只诊断几何可靠性、分配竞争与同带双向错误。 |

该筛查及其停止结论用于约束后续训练。 [BDKD-Net 原文及损失定义](https://pmc.ncbi.nlm.nih.gov/articles/PMC13412844/) · [COIN 原文](https://openaccess.thecvf.com/content/ICCV2025/papers/Jo_COIN_Confidence_Score-Guided_Distillation_for_Annotation-Free_Cell_Segmentation_ICCV_2025_paper.pdf) · [COIN 作者评分源码](https://github.com/shjo-april/COIN/blob/master/step2_score_instances_with_SAM.py#L50-L76) · [同 GT 配对原文](https://openaccess.thecvf.com/content/CVPR2024W/Vision4Ag/papers/Li_Knowledge_Distillation_for_Efficient_Instance_Semantic_Segmentation_with_Transformers_CVPRW_2024_paper.pdf)
