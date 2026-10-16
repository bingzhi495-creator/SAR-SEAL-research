# N26 数据阶段实际进展

日期2026-10-17。当前没有质量头真实训练或新模型AP。

DEV特征CPU任务已实际退出0并完成Root终态/文件及完整人口接收：187图、18700原ordinal自然query，FP32[18700,1040]。两份合格R原预测完整字典精确校准；GT/target/matcher/COCO未用于特征。耗时84.797秒，最低可用RAM32618864640字节，CUDA未初始化，整卡环境基线观测1577MiB。旧整bundle连带反序列化F/L，只有e0用于输入，不宣称完全没加载F/L。

Root接收检查了原任务/Source/文件refs、完整身份、e0/特征摘要、empty/crop元数据；尚未独立重建全部实际feature数值。因此这是artifact intake，quality_training_eligible仍为False；不能用已接收替代独立数值验证。当前236正常+28保留科学失败=264，legacy pins244。

完整FIT捕获源码也已冻结；最终同团队静态审阅无确定fatal，5项纯标准库检查通过。两项具体ABI错误已在未执行阶段修正，没有据此重跑模型：Control admission字段，以及旧N07 packet不存在的GPU_admission字段。

下一步固定两片[0,371)、[371,741)获取全1661真实匹配的完整强R mask，独立CPU复核后，才以唯一合格小包训练固定E24质量头。GPU捕获未开始，没有虚构捕获时长或loss。

[DEV源码](code/N26_DEV_QUALITY_FEATURES_SOURCE_ONLY_20261017_01/README.md)、[FIT源码](code/N26_FIT_MATCHED_R_CAPTURE_SOURCE_ONLY_20261017_01/README.md)、[数据计划](../refine-logs/N26_QUERY_MASK_QUALITY_BASELINE_20261017_01/DATA_CAPTURE_PLAN.md)。N25的+5.297021仍是不可部署GT oracle，不是本模型收益；N20/N23 STOP不变。

## 2026-10-17 后续进展

第一片371图、681个原匹配实例的完整R捕获实际正常结束，约232.672秒，观察整卡峰值3217MiB；Root已完成原终态/绑定源码/产物/人口接收，资格明确不覆盖独立数值核验。第二片370图已启动，不能预报成功或补写结果。

独立全量CPU核验、固定24轮624步训练及固定R完整COCO评价源码均已形成并通过静态接口复核，尚未实际执行。三种子支持由纯元数据测试核查；实际是否开展重复仍按主计划停止规则。两个捕获阶段特征函数路径差异已在新核验实现中明确处理，旧冻结文件不改写。
