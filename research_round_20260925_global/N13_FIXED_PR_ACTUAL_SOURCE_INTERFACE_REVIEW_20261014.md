# 固定PR诊断的真实源码接口核对

2026-10-14。状态：**已完成ROOT源码文字核对；真实SAR诊断未登记、未执行。** 本核对发生在原703 E08 GPU和独立CPU实际结束、下一701 E24准备及ROOT完整来源读回通过之后，且在下一GPU准入之前。没有加载权重、图像或GT，没有导入科学模块，也没有修改冻结源码。本说明是后续实现的接口依据，不是适配器真实任务资格或效用证据。

## 权重与残差

[实际工厂](multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N13_sameK_region_qualified_relation_training_source_20261013_02/point_heads.py)和[独立算术](multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N13_continuous48_adopted_endpoint_training_source_20261013_04/qualify_train_cpu.py)一致使用以下顺序。现有[未绑定适配器](code/N13_original_raster_fixed_PR_adapter_draft.py)的角色标签与源码键一一对应；未来加载器仍需核验实际端点来源、种子、臂、轮次、张量类型和参数证明，不能仅凭键名加载任意权重。

|适配器角色及实际源码键|形状|
|---|---|
|hidden1.weight / hidden1.bias|[91,257] / [91]|
|hidden2.weight / hidden2.bias|[91,92] / [91]|
|hidden3.weight / hidden3.bias|[91,92] / [91]|
|final.weight / final.bias|[1,92] / [1]|

共8张量、40,497标量。F256逐点作无仿射LayerNorm，eps=1e-5；c=clamp(L,−8,8)/8在四个线性层前重新拼接，前三层ReLU。e和patch仅保留旧ABI验证，不参与残差计算。工厂forward的logits仍含旧|L|≤2门，**实际N13评价取其未门控residual，再由调用者在精确区域写L+u**；诊断扩域时不能使用工厂的旧logits输出替代这个规则。独立CPU公式同样由外部flag控制写入。

## 原栅格和二值规则

[原渲染函数](multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/frozen_feature_linear_readout_diagnostic_source_20261010/diagnose.py)对batch1 FP32字段执行：双线性到800×800、align_corners=False；按actual_img_shape从左上裁切；再双线性到original_shape、align_corners=False。几何来自原生元数据，不能由GT或ROI重新定义。

[original_feature](multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N10_PR_FL91_direct_precedent_control_source_20261013_01/evaluate.py)按16通道渲染并写入F_original[256,H,W]。[N13 refine](multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N13_sameK_fixed_DEV_evaluation_source_20261013_02/evaluate.py)使用原栅格flat索引、F_original.flatten(1)[:,idx].T及L[q].flatten()[idx]，每次最多2,048点。原区域由np.flatnonzero形成升序索引。候选域扩展时旧域/新增域各自固定分块，是新诊断为保持旧候选不因并域而重新分块的约束，尚未获得真实大栅格运行资格。

实际固定分数导出明确使用**refined_logit > 0**，即适配器strict_gt_zero；L=0属于背景。inclusive_ge_zero仅为合成API测试分支，不能在本科学协议中任选。只更改segmentation，原类别、分数、query次序保持，空掩膜也不重新置零分数或过滤。候选计算使用已锁exact GPU L；CPU近似渲染不能代替该L形成选区或计算指标。

## 固定配对与完整AP

[原matching_images](multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N07_frozen_E24_original_detail_point_refinement_source_20261012_01/qualify_evaluation_cpu.py)读取真实COCO area-all / maxDet100数组，按分数稳定mergesort核对顺序、忽略状态和一对一互反关系。[fixed_native_pairs](multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N13_sameK_fixed_DEV_evaluation_source_20261013_02/qualify_evaluation_cpu.py)从原生指定阈值配对保留原query_id，不随细化结果重配；FN/ignored_GT保留在全部380条记录中。

下一诊断仍预锁701 LOG E48，使用原生.50下的374固定query/GT对；6个原生未匹配GT单列，不能把374配对均值叫作全数据AP。未来正式策略评价必须使用自然187图/18,700 query及完整380 GT重新运行COCO匹配；GT不能参与部署query或像素选择。固定候选的oracle只解释条件余量。

## 仍待完成

还需登记真实诊断的端点与输入引用、exact L/域重构、width2的具体原实现与边缘规则、.50实际配对身份、资源及独立CPU验证。不能用普通距离阈值近似width2，不能把本次文字核对视作这些资格已通过。适配器已有非零/非连续CPU合成检查，但科学运行时版本、真实渲染与导出联接、大栅格成本及真实候选行动仍未验证。完成剩余三条预锁E24曲线后，按[诊断方案](N13_FIXED_PR_COVERAGE_REPAIRABILITY_DIAGNOSTIC_DRAFT_20261014.md)登记并执行，不新增训练轮次或选最好端点。
