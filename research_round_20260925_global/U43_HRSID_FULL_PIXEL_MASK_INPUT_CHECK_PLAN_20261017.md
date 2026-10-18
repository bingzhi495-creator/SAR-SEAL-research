# HRSID 全集像素与实例掩膜输入检查：固定执行计划

**当前实际状态（2026-10-18）：** U43规范路径修复已实际执行并完成失败保留。十二库引用检查通过，完整5604图RGB解码成功、16969实例均尝试官方栅格，16956项非空通过、13项为空（train9/test4）；原每实例非空预锁未满足，不能称全量输入通过。未删图、未改标注；TEST标注已实际读取，未用于模型选择，没有模型/AP。见[实际失败](results/U43_CANONICAL_POPULATION_EMPTY_RASTER_FAILURE_ACTUAL_20261018.json)及[COCO规则与后续协议](U43_EMPTY_RASTER_COCO_PROTOCOL_REVIEW_20261018.md)。后文保留形成时历史状态。


**后续实际状态（2026-10-17）：** 首次原任务已退出1并完成失败保留，在PIL库路径/哈希引用门处提前结束，未像素解码、split记录解析或COCO栅格；0逐图记录。入口已核编码Source字节。旧冻结来源保持，后续仅按[独立路径修复计划](U43_CANONICAL_LIBRARY_REFERENCE_REPAIR_PLAN_20261017.md)另立尝试，尚未重新执行。以下为首次执行前的预登记计划。

日期：2026-10-17。状态：planned / not bound / not run。本文用于一次完整输入检查；不训练、不预测、不计算AP，不改变当前U40主方法、配方、图序或50轮端点。

## 目的与固定人口

验证当前本地HRSID JPG与对应原split标注能否逐图完整解码，并按官方COCO polygon规则逐原实例栅格化。固定train3642图/11047实例、test1962图/5922实例，共5604图/16969实例；不以merged16951版本替代，不更改划分，不按结果筛图。裸image/annotation ID仅在各自split内唯一，跨split复用合法。

输入引用从此前完整encoded/header记录提取，属于引用候选，不是新的像素资格。实际任务须核原JPEG及两个原JSON的严格字节引用，保留原文件名、800×800声明、实例与多组件分组、crowd及空标注图；不将多边形外接框当真值。

## 唯一执行与资源

采用已发布[人口函数](code/U43_HRSID_POPULATION_PIXEL_MASK_CHECK_DRAFT_20261017_01/README.md)、[CPU入口与工作体](code/U43_HRSID_POPULATION_CHECK_ENTRY_DRAFT_20261017_01/README.md)，逐图调用原[U42检查器](code/U42_UNBOUND_HRSID_PIXEL_MASK_CHECK_DRAFT_20261017_01/README.md)。精确接入entry/body/population/checker/runtime五个代码角色及十二个选定真实库文件；它们是明确依赖足迹，不代表整个Python/OS/DLL环境闭包。

Root统一拥有一个科学任务，沿原受控driver与真实句柄；CPU[0,1]、四项线程环境变量2、CUDA不可见，全程不导入Torch/不构造模型。进程RSS≤16GiB、系统可用RAM≥16GiB、原owned总时限1800秒，整卡观测上限8GiB；不放宽现有门。输出独占目录、保留完整私有逐图JSONL和partial；仅一次固定全集任务，不自动重试、分片、换库或改限额。观察超时只复查原句柄。

## 检查、失败与接收规则

- JPEG必须真实Image.load解码，RGB/uint8/800×800；禁用容忍截断的执行环境，不做转色、补图、旋转或缩放修复。
- EXIF无orientation标签按存储坐标记录不施加变换；显式1允许；其他值、非法值或读取异常保留为失败。无标签不等于已经证明语义方向一致。
- 每个原annotation的全部polygon组件由官方frPyObjects→merge→decode合为同一实例，保留crowd。二值非空、面积计数等按U42原规则检查；原continuous area/bbox与离散raster差值仅诊断，不强制两者相等。
- 有限越界坐标只统计，不预先裁剪、删除或修标。原U42已处理的图像/标注/栅格失败保留并继续；source、资源、回调及未处理异常直接传播并保留partial，不伪造成功。
- 访问全集、全部检查成功、独立Root接收是三个不同状态。只有5604图真实解码通过、16969原实例栅格通过、零失败/未尝试且5604逐图审计写入，才可申请原actual0及Root接收。存在失败不静默缩小分母；Root接收只核原终态、来源、产物及计数，不重复像素或栅格计算。

## 结果范围与发布

允许读取两个split的原标注，包括TEST；这是输入兼容检查，不用于模型、阈值或超参数选择，不声称no_TEST，也不能恢复历史盲测身份。保持已记录跨split及test内相同编码图，不删除或择标。成功仍不能证明语义真值正确、JPG与PNG像素相等、母场景独立、模型未接触目标数据或跨数据集有效。

只发布Root核验后的聚合状态、计数、资源与失败原因摘要；原图、全量标注、逐图/逐实例明细、运行Source与资格链不发布。尚无实际结果时保持planned状态；代码和元数据测试通过不计为真实人口通过。来源与历史暴露范围见[跨集准备度](N16_FORMAL_TEST_AND_CROSS_DATASET_EVIDENCE_READINESS_20261014.md)及[本地JPG核验](HRSID_LOCAL_JPG_METADATA_AND_VERSION_BOUNDARY_20261015.md)。
