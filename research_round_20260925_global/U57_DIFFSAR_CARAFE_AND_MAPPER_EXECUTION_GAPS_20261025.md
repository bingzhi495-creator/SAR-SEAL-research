# DiffSARShipInst：短增量执行核查

2026-10-25。**只读静态核查、未执行、无新AP。** 按research-lit一手来源流程，先读已有U57_DIFFSAR_AUTHOR_CONFIG_EXECUTION_GAPS_20261025.md，不重复配置综述。作者当前main仍为 **bdc3362372213bc8c3a1f6d5b7b74dc224d1b871**。未SSH、科学导入、构建、读取GT/权重/活动SOURCE或资格链；未改发布目录及当前训练。

## 1. 新闭合的硬依赖：mmcv._ext

作者fpn.py:26–30通过自带ext_loader要求carafe_forward/backward；**detectron2/modeling/ext_loader.py:13实际导入mmcv._ext**。setup只构建detectron2._C，install_requires及README未列MMCV。仅安装或编译Detectron2不能满足此依赖。

这不是未调用的备用类：选定build_resnet_scje_fpn_backbone在fpn.py:600传入FirestLevelUpSample；后者:241–244构造并调用CARAFEPack；:687/:718进入扩展前/反向。不能换普通插值后仍称作者原方法。

DGX aarch64/Torch2.10/cu130兼容性仍未执行核验。下一项最小工作是独立锁构建链，并真实检查mmcv._ext CARAFE、作者detectron2._C modulated deform-conv前后向及所用RoIAlign/NMS。成功import、最低版本声明或静态阅读都不证明能训；也不能由旧代码推断必然不能运行。不重装主环境，不改变主训练资源。

来源：[扩展实际加载器](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/detectron2/modeling/ext_loader.py#L13)、[选定FPN/CARAFE](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/detectron2/modeling/backbone/fpn.py#L584)、[setup](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/setup.py)。

## 2. 新闭合的实际mapper选择链

train_net.py:22导入DiffSARShipInstDatasetMapper；diffsarshipinst/__init__.py:3明确导出自有.dataset_mapper中的类；Trainer.__init__:44调用自身loader；Trainer.build_train_loader:103–105实例化此类并通过mapper参数传给build_detection_train_loader。

最终 **diffsarshipinst/dataset_mapper.py:114–120** 排除crowd训练实例、变换原实例标注，显式annotations_to_instances(mask_format="bitmask")，再filter_empty_instances。普通detectron2/data/dataset_mapper.py的self.instance_mask_format不是这一训练入口。YAML写polygon不能单独证明内部使用PolygonMasks；输入仍须保留逐实例polygon/RLE，不能合并为语义mask。

因此最少数据适配是独立名称register_coco_instances，显式注册真实COCO路径与image_root，在train_net setup前完成注册并覆盖DATASETS。保留原图高宽、原浮点polygon及身份/类映射；记录crowd排除、无效polygon与裁剪后空实例数。原loader还对ignore!=0断言，须按实际输入处理并报告，不能静默改GT。现有整数FIT741/DEV187仅可明确称同协议适配，不能自动当作者原标注或官方TEST。

附带输入口径修正：自定义mapper使用保持比例的ResizeShortestEdge。MIN_SIZE=MAX_SIZE=512时非方图通常是最长边512后padding，**不是统一拉伸512×512**；训练另有0.5分支的400/500/600预缩放与crop。与RGB400方形主实验比较时须记录这一差异，不改当前配方。

来源：[真实loader](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/train_net.py#L103)、[导出关联](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/diffsarshipinst/__init__.py#L3)、[实际自定义mapper](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/diffsarshipinst/dataset_mapper.py#L114)、[缩放实现](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/detectron2/data/transforms/augmentation_impl.py#L134)。

## 3. 初值剩余边界

本次补读下游detectron2/modeling/backbone/resnet.py:614–694：R101构造[3,4,23,3]阶段，按配置选择DeformBottleneckBlock；stem/block有MSRA等构造初始化，**该构造函数未自动下载或装载ImageNet/COCO**。结合既有入口调查，空WEIGHTS不能解释为隐式预训练。

完整空路径行为仍须锁定实际fvcore版本及运行日志；论文实际初始化仍未知。出版社全文直开403，只核到摘要及章节预览，未核其实际GPU/初值。若未来选合法预训练，应明示来源、加载键覆盖与DCN新增参数初始化，作为独立实现选择，不能伪称作者确切初值。

来源：[ResNet构造](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/detectron2/modeling/backbone/resnet.py#L614)、[原论文](https://www.sciencedirect.com/science/article/abs/pii/S0924271625000887)。

## 下一项具体任务

Root在合格间隙先核隔离环境中这两个真实扩展及其精确版本/构建链；缺失时只针对真实缺口准备。通过真实算子前后向后，再进入明确数据注册、单批完整模型训练/原网格导出、统一COCO校准和预先锁定的完整外部长训。相同短预算仅作诊断；充分训练、可核初值、共同标注/划分和成本证据不能由本静态核查替代。

本文仅自写增量与原始链接，没有第三方全文/源码复制，没有新测试框架。当前项目research-wiki不存在，未做wiki扩展。其余架构、schedule与数据协议沿用已有调查，不另造综述。
