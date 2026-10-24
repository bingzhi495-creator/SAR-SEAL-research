# U57教师身份与数据接触的证据范围

2026-10-25（北京时间）。本次首先审阅独立普通发布副本中的文档、安全JSON和公开源码；随后对指定官方初始化文件做一次无代理流式字节核对。未访问本地权重、原工作区绑定SOURCE或资格链，未SSH、导入机器学习库、执行模型或测试；远端权重未落盘或反序列化。本文记录来源证据，不改变当前训练、配方或数值。

## 当前教师应如何命名

按现有公开构造和恢复合同，可称为**项目Mask2Former/R50 E24原生教师**；这里的模型族依据不是checkpoint文件名猜测：

- [既有校准审阅第9行](CALIBRATION_PRECEDENT_REVIEW_20261012.md#L9)明确将同一原生67.071057 mask AP载体记为冻结Mask2Former/R50 E24。
- [公开恢复函数第176–188行](multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/native_finetune24_fresh_optimizer_source_20261010/train.py#L176)调用`pref.prepare_cpu(..., 'mask2former', ...)`，要求`meta.kind == 'mask2former'`、`mode == 'native0'`和固定E24元数据；第190–203行检查610个模型/BN状态条目及451个**parameter tensors**，不是451个标量参数。
- 该构造经[原preflight第107–129行](multimodal_data_pilot_20260928/native_coco_comparator_micro_preflight_preparation_20261008/run.py#L107)读取绑定配置、要求PyTorch风格ResNet-50并经registry构造。本次未取得当前实际有效配置的`cfg.model.type`字符串，不把其他历史carrier配置冒充该配置。

**此前把这一固定原生输出称为“DiffSARShipInst项目E24版本/已复现DiffSARShipInst”，缺乏上述证据支持，应更正。** DiffSARShipInst仍可作为外部文献及强对照候选；不能用项目67.071057 AP替代它的作者结果或同协议复现。更正身份说明不改变已经保存的项目原生输出数值。

## 当前可核的权重及输入谱系关联

[U47实际恢复摘要](results/U47_EXTERNAL_TEACHER_CPU_RESTORE_ACTUAL_20261018.json#L43)记录教师E24 checkpoint完整SHA256：

`376f725c42a0d522700add19548943328064426835061282d95d510f27d1ec4e`

该摘要报告610个状态条目、451个parameter tensors及原模型/BN精确恢复；它不是本次重新加载权重或独立神经网络重放。[初始化关联安全摘要](results/M2F_COCO_INITIALIZER_ENDPOINT_FINGERPRINT_LINKAGE_20261017.json)记录当前FIT教师五处元数据角色引用相等，链为：已接收U40 context → FIT/N13元数据 → N16 GPU报告 → 原START加载回执 → 原initializer引用；其E24 SHA与U47一致。[准备度文档第37–43行](N16_FORMAL_TEST_AND_CROSS_DATASET_EVIDENCE_READINESS_20261014.md#L37)说明该次历史核查的实际范围。

[U54输入记录](U54_ALL_INPUTS_AND_UPDATE_CONTROL_20261018.md)及[U53安全汇总](results/U53_ALL741_CPU_EXPORTS_ACTUAL_20261018.json)描述原FIT741图/1661匹配实例的五字段便携导出及逐位保持；这支持缓存和便携输入的来源连接，不能额外证明教师的上游训练数据从未包含DEV。

## FIT监督合同与未闭合范围

[SPATIAL24公开训练源码第35–38行](multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/spatial_position_preserving_common24_source_20261009/train_common.py#L35)限定`kind='mask2former'`、FIT741图/1661 GT、24轮；[第130–140行](multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/spatial_position_preserving_common24_source_20261009/train_common.py#L130)及[第432–436行](multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/spatial_position_preserving_common24_source_20261009/train_common.py#L432)要求每轮完整覆盖741索引。恢复/训练接口[第583–588行](multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/native_finetune24_fresh_optimizer_source_20261010/train.py#L583)要求实际dataset roster等于原FIT图像列表。

这些是可核源码合同。**本次没有重新核对该E24原实际训练注释引用、逐轮实际图像清单及它们与当前DEV187的对应。** [当前内部协议核查](U57_INTERNAL_DATASET_PROTOCOL_TRACE_20261024.md)已证明当前FIT/DEV的划分一致性，但“当前两个集合不相交”不自动证明所有上游教师训练阶段均排除了DEV。[静态输入合同第11行](TEACHER_REAL_IMAGE_INPUT_STATIC_EVIDENCE_20261017.md#L11)提到FIT与DEV共用图像根目录；共目录本身也不是泄漏证据。

上游m2f_r50_coco.pth的旧安全摘要记录多项指纹关联，当时完整远端SHA比较未完成。**本轮2026-10-25新增核对已读到官方响应体EOF，共176,432,372字节，完整SHA256为41b088b62838c986004f94bab2870395b669ec6bd11b6590369b838b0de06879，与先前公开绑定initializer摘要完全一致。** 见[本轮完整摘要安全记录](results/U57_OFFICIAL_COCO_INITIALIZER_FULL_STREAM_AUDIT_20261025.json)。旧摘要的未完成字段保留为历史记录，不回写。

该新增证据闭合的是指定初始化文件的完整字节身份关联。ImageNet-1K/COCO train2017范围仍是发布方声明；independent_training_data_ancestry_verified=false的范围没有因文件相等自动改变。它不重建全部上游训练历史，也不验证E24实际微调清单与DEV187的交集，因此既不能保证冻结teacher从未接触DEV，也不能把未取得该证明写成已发生监督泄漏。

## 最小后续核查

在合法的历史元数据核查窗口，由Root把当前E24的实际有效配置引用、原训练注释strict-ref及各阶段完整实际训练图像身份清单，与当前FIT741/DEV187严格连接并核交集；另行保留全部上游初始化/训练阶段的来源边界。该待办不要求读取运行中载荷来同步，也不要求重新训练或重放神经网络。

当前FULL/PLAIN仍可继续既定同载体、同预算的内部结构效用比较。最终报告须区分refiner的FIT监督范围、teacher的项目微调范围与更上游预训练范围；当前证据不支持未见DEV/盲测、外部SOTA复现或完整训练谱系认证。
