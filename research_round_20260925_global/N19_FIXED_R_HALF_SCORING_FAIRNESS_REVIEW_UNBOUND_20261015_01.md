# UNBOUND：N19 固定代价读出的边界审查

固定A、独立逐点动作且并集U>0时，I/U≥r等价于I−rU≥0。按原P=0/1区分的两式，分别是Δ(I−rU)的条件期望；概率需对应部署分布并校准。该可加代理不等于期望IoU最优，更不等于AP最优。旧N13已有校准反例129/200<2/3及r=1/2对照，不能重称新推导。[旧说明](https://github.com/bingzhi495-creator/SAR-SEAL-research/blob/main/research_round_20260925_global/N13_FIXED_K_BINARY_IOU_ORACLE_AND_PROBABILISTIC_PROXY_LIMIT_20261014.md)

r=.5令原背景点偏向添加、原前景点更谨慎删除，改变两组阈值及跨组排序；它不是单一全局阈值，但仍是普通固定代价偏置，无新增信息。N18只覆盖374个原native匹配，不能从其help/harm总数推断所有自然query的最佳r。原N16 STOP保留；看过AP/N18后再选择旧候选，仍是DEV再利用的探索性检验。

质量回归也有直接先例：Mask Scoring R-CNN由实例特征及预测mask回归IoU并校准实例分数；本候选若保持S0、仅加权像素行动，具体用途不同，但不能把质量估计本身当创新。[作者论文页](https://openaccess.thecvf.com/content_CVPR_2019/html/Huang_Mask_Scoring_R-CNN_CVPR_2019_paper.html)

建议仅另锁一次E24 atmost读出：共享已冻结N16 Utility概率/输入/初始化，唯一比较r=1和r=.5；同C/K、strict score>0、原ties、全187×100、S0与PR不变。不训练、不扫r/阈值、不改预算、不补seed；exact/E8不替代主结果。先闭合实际完整C概率载体；无资格不得假称可复用。机制差看同头两读出的AP及加删修复/损害；实用价值还须超过已合格最强ERROR/强R，成本含原概率头和PR。无收益即结束此单次候选，有收益也只支持该载体的固定非对称评分，不能归因质量估计、期望IoU最优或AP因果。以后若研究质量信息，需同一质量头的实例r与常数/预锁打乱r对照，本轮不实施。

仅普通文档与公开原文核查；无科学读写、运行、训练、Git或锁定协议变更。数学边界按formula-derivation技能核对，沿用旧说明，不新增理论主张。
