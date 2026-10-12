# N10 固定全DEV评价：GPU与独立CPU已闭合

2026-10-13。固定8轮、四头各13,288更新，训练GPU/CPU与完整DEV评价GPU/CPU均实际正常退出并由ROOT核验。评价人口为DEV187图/380个原始实例，每图100查询，native及四头各18,700预测；无TEST、NMS、分数筛除或GT选query，不据DEV选最好轮次。

C均值mask AP **68.1661355**，PR结构适配均值 **67.9273781**，原生 **67.0710566**。两组C−PR为+0.1250985/+0.3524163，均值+0.2387574 AP点，未满足结果前锁定的两组均正且均值≥+0.30优先条件。因此保留两臂收益，不自动优先延长该结构；当前预算不证明充分收敛或方法总体失效。完整AP/AP75、公平边界、固定配对反例及成本见 [结果摘要](../docs/N10_fixed8_C_PR_fullDEV_qualified_result_20261013.md)。

GPU复用原N07 F/e/M/classes来源，natural all100渲染/排序/quality/S0以及全部18,700条原生输出必须精确复现合格E24参考，才输出细化。C/PR同一原生abs(L)≤2闭带、点块2048、F通道块16、原渲染器、固定S0/类别/顺序及空掩膜规则。GPU耗时339.50秒，整卡观测峰5,491MiB。

独立CPU实际1128.937秒、CPU0/1两线程、CUDA冷态；以独立函数公式复算四头各4,573,049带内像素/19,439块，五组官方COCO12、官方一对一像素及固定原生query–GT像素分别核对。预锁1e−4实用支持容差不是CPU/GPU逐位等同或认证误差界；近似CPU掩膜没有替换GPU指标输入。没有教师CPU前向、F/e→M投影CPU重放、JPEG解码或完整优化器轨迹复算。

ROOT先确认原CPU进程held exit0、父回收及原身份消失，闭合33正常祖先与3个保留实际失败，再核对Q、来源、真实五导出指标和配对汇总，最后执行预锁决定。CPU Q为206,415B / SHA256 343e8137be64f09eded2f7c1822e77538a471f832a45b63c205cdb0d193e73b2；ROOT决定记录12,718B / dfbd54dca53648306d02284f36fa9d5b672bde865ea2537a6511b65b2dd42937。原资格链和科学载荷不加入公开范围。

四臂在原生IoU≥.90固定配对上的平均IoU均小幅下降，整体AP增益不能代表全部轮廓改善。下一问题是可编辑区域的实际误差覆盖与修复上限；[N11诊断](N11_EDITABLE_REGION_COVERAGE_STATUS_20261013.md)已通过独立静态审阅并启动GPU，CPU尚未执行，无新合格数值。见 [误差依据](../docs/N01_N06_N07_N09_ERROR_EVIDENCE_FOR_NEXT_DECISION_20261013.md)、[定向查新](FIXED_BAND_SAMPLING_RISK_PRECEDENT_MEMO_20261013.md)、[协议](../refine-logs/N10_PR_FL91_20261013_01/EXPERIMENT_PLAN.md)、[跟踪](../refine-logs/N10_PR_FL91_20261013_01/EXPERIMENT_TRACKER.md)及 [评价ABI](multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N10_PR_FL91_direct_precedent_control_source_20261013_01/EVALUATION_SOURCE_ABI.md)。

当前C条件偏置已有直接先例，PR是同载体结构适配而非完整PointRend作者复现；参数接近不等于有效容量严格相同。核心创新、外部SOTA优势、充分收敛、正式TEST及跨数据集验证仍未成立。
