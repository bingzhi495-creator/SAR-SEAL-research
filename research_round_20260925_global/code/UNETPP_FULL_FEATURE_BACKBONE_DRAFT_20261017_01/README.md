# 原完整UNet++的特征返回接口：未执行草稿

2026-10-17。仅AST和静态源码核对，未实际导入Torch或运行原full。调用方提供实际原UnetPlusPlus与Torch，make_feature_backbone返回仅增加包装的原模块图；无新参数、不改模式或requires_grad。显式关闭且不分配x2/x4 Transformer；单类、无深监督；图像batch必须1，固定input_size至少32且16整除。

保留原编码、H/4语义/独立普通输入横纵DSConv融合、所有原嵌套解码有效路径；直接返回CONV0_4，跳过语义输出头。原BN旁路及DSConv实现均保留。旧DSConv多图索引问题因此仅由单图合同避开，没有修复。inactive_parameter_names只报告已静态确认的原失活参数，不冻结或选择优化器；其名字在wrapper下为legacy.*，外层adapter下需加backbone.。

接共享适配器时feature_module_name='legacy.CONV0_4'、feature_channels=64；原有hook捕获该模块，但wrapper真正停止在特征处，不继续执行旧语义头。

测试工厂make_real_legacy_suite(torch,legacy_class)要求实际闭合原类，计划用64尺寸、eval、no_grad，临时将第四语义头替换Identity获得原forward特征，再与wrapper逐元素精确比较，最后恢复原头。测试不会把tiny假主干当成原结构；尚未执行，不证明梯度、真实影像、400资源、训练或AP。原历史模型及其依赖与权重不在本目录重新发布。

进一步执行入口见[真实full CPU64检查](../U30_FULL_FEATURE_CPU_DRAFT_20261017_01/README.md)，方法计划见[完整方案](../../../refine-logs/UNETPP_FULL_METHOD_20261017_01/EXPERIMENT_PLAN.md)。
