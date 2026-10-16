# N26 独立输入核验源码

2026-10-17。源码内容已按原冻结绑定协议实际执行并完成Root原终态/产物核验；不是只通过静态检查。

全量CPU核验任务实际正常结束118.672秒并经Root接受。FIT741图/1661真配对IoU及特征逐位一致，DEV187图/18700query特征逐位一致；DEV GT不读，GT仅作FIT原matcher目标，未匹配query不伪标0，empty/K0保留。共享同一插值实现再提取，不能称独立插值算法或R前向重演。

这是已有MaskIoU思路的query–mask适配，非完整Mask Scoring R-CNN复现或已成立创新。输入、训练和评价资源及准入由原执行协议控制；不能从该源码副本直接推得独立复现成功。

本目录SOURCE_ONLY，不含数据、全量标注、权重、缓存或资格链，仍需生成原协议要求的输入和身份绑定。不同捕获阶段特征工厂的路径差异已通过各组内部完整引用一致、实际绑定源码同bytes+SHA256处理。

[完整实测与限制](../../N26_FIXED_R_QUALITY_THREE_SEED_RESULT_20261017.md) · [逐种子完整COCO](../../results/N26_FIXED_R_QUALITY_THREE_SEED_AGGREGATE_20261017.json) · [执行前主计划](../../../refine-logs/N26_QUERY_MASK_QUALITY_BASELINE_20261017_01/EXPERIMENT_PLAN.md)
