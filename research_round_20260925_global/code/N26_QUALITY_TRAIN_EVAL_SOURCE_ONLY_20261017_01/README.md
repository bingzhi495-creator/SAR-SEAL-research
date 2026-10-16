# N26 质量头固定训练与评价源码

2026-10-17，SOURCE_ONLY，未执行的实现。纯标准库元数据检查和静态审阅不能替代真实数据执行。当前第一片371图/681匹配实例已完成实际0退出及Root文件/人口接收，第二片捕获仍在运行；独立CPU核验和质量头训练尚未运行。

训练：冻结强R，仅1040→256→128→1质量MLP，299521参数；FIT741图/1661原始GT匹配query，MSE回归完整原图IoU，24轮×26批=624更新，batch64、尾61，AdamW 3e-4、wd1e-4，无调度/早停/DEV选轮。仅支持预锁完整种子202610172601、202610172602、202610172603，是否复验由主计划门槛决定，不自动启动全部。

评价：E24固定端点，完整187图/18700query，64批、尾12；先用qhat=1精确恢复R预测与COCO12，再只评价S0*qhat，保留原mask/class/query/顺序，无qhat-only、NMS或分数函数选择。GT只在预测及CUDA释放后用于原官方COCO。保留训练状态与优化器/RNG。

已完成3项纯元数据种子测试RED→GREEN及独立静态接口复核；未发生新训练或AP结果。这是已有MaskIoU思想的query–mask适配，不是完整Mask Scoring R-CNN复现或已成立创新。

本目录不能直接独立训练：仍需按原协议生成并绑定数据、源文件、输入资格与资源准入。缓存、权重、标注、私有RLE和执行资格未发布；resource_control.py与quality_head.py来自此前已批准的普通发布内容缓存，本次没有为同步读取运行中的绑定SOURCE。

[主计划](../../../refine-logs/N26_QUERY_MASK_QUALITY_BASELINE_20261017_01/EXPERIMENT_PLAN.md) · [数据阶段记录](../../N26_DATA_CAPTURE_PROGRESS_20261017.md)
