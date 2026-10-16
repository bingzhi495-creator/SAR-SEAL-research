# N26 固定质量头训练与评价源码

2026-10-17。源码内容已按原冻结绑定协议实际执行并完成Root原终态/产物核验；不是只通过静态检查。

三个预锁完整种子202610172601/2602/2603已各完成24轮、624更新、batch64尾61及完整18700query评价；AP69.423613/69.637142/69.551345，均值69.537367。保持原R掩膜、类别、query及顺序，仅S0*qhat；qhat1原COCO12逐组exact，评价batch64尾12。模型299521参数，无DEV选轮/评分函数搜索；样本SD0.107449，平均+1.437982点。

这是已有MaskIoU思路的query–mask适配，非完整Mask Scoring R-CNN复现或已成立创新。输入、训练和评价资源及准入由原执行协议控制；不能从该源码副本直接推得独立复现成功。

本目录SOURCE_ONLY，不含数据、全量标注、权重、缓存或资格链，仍需生成原协议要求的输入和身份绑定。不同捕获阶段特征工厂的路径差异已通过各组内部完整引用一致、实际绑定源码同bytes+SHA256处理。

[完整实测与限制](../../N26_FIXED_R_QUALITY_THREE_SEED_RESULT_20261017.md) · [逐种子完整COCO](../../results/N26_FIXED_R_QUALITY_THREE_SEED_AGGREGATE_20261017.json) · [执行前主计划](../../../refine-logs/N26_QUERY_MASK_QUALITY_BASELINE_20261017_01/EXPERIMENT_PLAN.md)
