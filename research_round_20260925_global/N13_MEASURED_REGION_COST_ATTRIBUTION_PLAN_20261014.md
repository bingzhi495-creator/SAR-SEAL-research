# N13区域阶段成本：静态归因边界与最小补测

2026-10-14。**仅静态映射独立发布副本，未执行、未补测、未取得加速结果。当前运行中的N13源码、资格链、区域、K、训练和评价配方均不改。** 没有读取原工作区科学载荷、运行记录或绑定配置。

[首个合格E08曲线](../refine-logs/N13_SAMEK_PR_20261013_01/findings.md)记录全187图区域阶段566.770秒、LOG/GEO头各约60秒。区域是该表最大的单列阶段，但这一事实尚不能定位排序、GEO或者模型前向为主因。此前[共享投影公平性审阅](N13_SHARED_PROJECTION_COST_FAIRNESS_REVIEW_20261013.md)已分析头的共享投影；本补核只补实际计时边界，避免继续以MLP算量解释整个开销。

## 已计入区域时间的操作

下表文件均位于 `multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/`。

|静态依据|计时包含的内容及边界|
|---|---|
|[评价入口246–257行](multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N13_sameK_fixed_DEV_evaluation_source_20261013_02/evaluate.py#L246)|每图100个query的两臂区域构建、K/回退记录、两臂flat索引提取及int64转换、K检查、两个完整布尔掩膜的序列化与SHA。它不是只测区域选择公式。|
|[区域函数439–449行](multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N13_sameK_region_point_refinement_source_20261013_01/region_common.py#L439)|NumPy有限值检查、P/B/absL/flat构造、K统计和前景状态检查；非回退query执行两次SciPy chessboard距离变换及距离合并/检查，回退query填充无穷距离。函数名为gpu_regions，但这里接收的是CPU NumPy数组。|
|[同函数450–453行](multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N13_sameK_region_point_refinement_source_20261013_01/region_common.py#L450)|全域(D,absL,flat)字典序排序、所选K索引的升序排序与G掩膜构建；还执行第二次全域(absL,flat)排序，构造uncertain掩膜并核对它与B相等。第二次排序用于当前实现的身份校验；返回的LOG区域仍是B。不能把两次排序都说成GEO的必要算法成本。|

这些子操作没有各自计时。566.770秒不能按静态调用次数分摊，也不能因看到全域排序就认定它占多数。两臂共同计时不提供LOG与GEO各自的区域增量成本。

## 当前阶段表未单列或容易混淆的成本

- **另一次GPU→CPU的L复制在区域计时之外。** [246行](multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N13_sameK_fixed_DEV_evaluation_source_20261013_02/evaluate.py#L246)先完成 `L.detach().cpu().contiguous().numpy()`，再启动region计时。这项时间进入每图总elapsed，却未单列到当前报告的五类costs。
- **selector I/O包含资格性工作。** [170–198行](multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N13_sameK_fixed_DEV_evaluation_source_20261013_02/evaluate.py#L170)另有一次L复制、值摘要、写入或读取、逐字节摘要及精确一致性校验。首次produce与后续reuse走不同路径，41.360秒不能直接当作后续所有端点的相同成本。
- **head_seconds并非纯神经头。** [201–216行](multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N13_sameK_fixed_DEV_evaluation_source_20261013_02/evaluate.py#L201)含全L克隆、每query索引上传、特征/logit取点、零patch分配、分块头计算、有限值检查、写回、域外布尔掩膜构建和精确不变检查；[263–264行](multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N13_sameK_fixed_DEV_evaluation_source_20261013_02/evaluate.py#L263)末端同步后才结束计时。仅优化Linear可能无法解释这一完整阶段的变化。
- **导出已局部计时，但没有汇总进当前costs表。** [266–279行](multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N13_sameK_fixed_DEV_evaluation_source_20261013_02/evaluate.py#L266)记录export_seconds，包含掩膜导出及像素变化诊断；[415–417行](multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N13_sameK_fixed_DEV_evaluation_source_20261013_02/evaluate.py#L415)未对它求和。输入packet校验在每图begun之前，packet读取/文件摘要在evaluate_image之外；全局资格、加载、输出及COCO评估也未形成相互排斥的完整成本表。不能把阶段和与原任务墙钟的差全部称为I/O。
- **原生阶段来自缓存字段。** [225–240行](multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N13_sameK_fixed_DEV_evaluation_source_20261013_02/evaluate.py#L225)包含字段上传、eF投影检查、原域掩膜渲染、质量/分数及RLE核对；[410行](multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N13_sameK_fixed_DEV_evaluation_source_20261013_02/evaluate.py#L410)明确没有加载原生模型或调用其forward。5.888秒不能解释完整模型延迟。

## 最小后续测量方案（尚未实施）

1. 先完成预锁E48主评价。原任务实际结束并按原执行资格规则闭合后，由所有者汇总已有逐图elapsed和两臂export_seconds，不新增训练或改变选择器。这一步只能补成本账，不能反推出未记录的子阶段时间。
2. 如区域成本值得继续定位，再设独立新路径和资格，锁定一次诊断用的计时边界。将CPU区域阶段拆为：基础P/B/K及有限值检查；距离变换/合并；GEO全域排序与K选择；LOG身份校验排序；两臂索引提取；掩膜摘要。另单列GPU→CPU L复制。记录H/W、K和回退状态，避免把图尺寸及退化query差异当作算法差异。先作有限诊断，只有需要质量—成本主张时才扩大为完整187图的公平重复测量。
3. CPU子阶段用单调墙钟并验证它们与父阶段的差额；GPU子阶段须明确同步边界和计时工具，另报同步/检查成本。warm-up、硬件、CPU亲和性/线程、chunk、I/O和重复次数预先锁定，报告波动与峰值存储。不得把不同计时器的数值直接相加为端到端时间，也不把缓存重放称为完整模型性能。
4. 计时版本先核对原生query/分数/顺序、exact K、两臂支持索引及预测一致性。当前资格性第二排序、掩膜SHA和域外不变检查先保留；未来若区分“算法执行”与“审计开销”，必须分别报告并在独立方案中核验，不能静默删除后与原表宣称加速。没有这些实测前，不改变排序、跨设备实现或宣称工程创新成立。

当前可行动结论是：**先补完整成本边界，再定位区域内的真实子阶段；暂不根据这次聚合时间选择优化方向。** 该计划不授权修改正在运行的N13任务，也不提供质量改善、稳定加速或TGRS结论。
