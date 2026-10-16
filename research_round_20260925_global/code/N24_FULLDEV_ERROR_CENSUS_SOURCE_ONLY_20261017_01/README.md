# N24 完整DEV匹配普查：执行前源码

状态：源码、3项NumPy合成检查及独立静态审阅已完成；真实执行尚未开始。测试含64种2x3二值图的独立穷举对照、增广重匹配、零边/空GT和NaN拒绝，不代表真实COCO已核。

固定五组既有预测，完整187图/380GT/18700query，沿原官方COCO12并观察其原对象，不改评分/GT/matcher。原指标必须精确复现；再统计全部IoU阈值、原score稳定top10/100的贪心与最大基数一对一关联、每GT最佳重叠，保留零/低质/ignore/crowd。从实际贪心计数重建AR10/100核原stats。最大匹配不先删依赖原关联的dtIgnore候选。

最大匹配是离线关联诊断，既不是AP上界，也不是可部署GT重排；几何无候选、竞争和贪心机会分别报告，不唯一归因score，不预选新结构。逐GT记录中的official_top100_matched_dt_ids明确只对应原top100，即使旁列best-IoU top10结果也不冒充其匹配身份。

入口复用resource_control.Control，只在CPU0/1、线程2、空CUDA、RSS/剩余RAM各16GiB阈值内运行。不importTorch、不加载模型/权重/F/L缓存，不训练，不增加新策略AP；现有五组AP仅作原指标复算校准。资源模块的训练入口不被调用。

运行`python -B -m unittest discover -p 'test_*.py' -v`检查纯合成接口。真实执行仍需未发布的原预测、标注和私有绑定；这不是完整一键复现包。私人逐GT/逐query细节不作为公开数据发布，后续仅公开范围内聚合与结论。

[执行前计划](../../../refine-logs/N24_FULLDEV_ERROR_CENSUS_20261017_01/EXPERIMENT_PLAN.md)、[N23已停止结果](../../N23_FIXED24_FULLDEV_RESULT_AND_STOP_20261017.md)。
