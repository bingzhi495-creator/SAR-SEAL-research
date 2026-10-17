# 未绑定研发草稿，未完成数值资格

2026-10-17。调用方提供实际archive/old_code/Unet++/model1.py的UnetPlusPlus(num_classes,deep_supervision=False)。保留原64/128/256/512/1024主体、BN和十条独立转置卷积，返回CONV0_4，跳过4语义头；注册同一legacy实例，无新增参数。其与full更贴近，但不冒充历史clean32/GN，也不称参数/计算配平。唯一真实plain64特征测试由make_real_plain_suite(torch,legacy_class)产生；Root先缺模块RED（68a9f9，无Torch），实现后仅AST，尚无真实数值GREEN。

代码不包含本机spec/资格/原数据/权重。主线范围见[完整方案](../../../refine-logs/UNETPP_FULL_METHOD_20261017_01/EXPERIMENT_PLAN.md)。
