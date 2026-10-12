# N11：原生可编辑区域的误差覆盖与条件修复上限

2026-10-13。**GPU诊断所有者已启动，尚未获得实际完成回执；独立CPU尚未执行，无新数值结果。** 新run.py/qualify.py/common.py和ABI已完成一次独立静态审阅；ROOT核对33个正常祖先及3个历史失败，提取已合格原生匹配锚点并建立新冻结输入后，启动单次受限GPU任务。

依据 [N10固定8轮结果](../docs/N10_fixed8_C_PR_fullDEV_qualified_result_20261013.md)：C/PR都提高AP，但原生IoU≥.90固定107配对的平均IoU在四臂均小幅下降；C−PR均值未过预锁优先线。当前尚没有原生abs(L)≤2带内外误差覆盖数字。因此本项只回答固定修改区域是否限制几何修复，不训练新头，不自动延长旧头，不宣称候选方法成功。

诊断复用已冻结的M200/classes及原渲染器。必须先精确重现原生自然100渲染、查询重排、完整quality/S0和全部18,700导出字典，之后才对既有固定原生query–GT配对做像素诊断。原生.50/.75/.90槽位0/5/8分别保留374/348/107对，三组各保留全部380 GT及6/32/273原生未匹配；不同门槛不假设同一query。没有GT选推理query、评分重排、NMS、截分或TEST。

在GT面积A、原生二值P与固定可编辑B上，分别统计TP/FP/FN的带内外数量。自由二值修改带内的乐观上限为 **IoU*=(A−FN_out)/(A+FP_out)**，并核对直接构造的二值支持。它是所选原生固定query的条件上限；有限点头可能达不到，374配对不代表全部380的上限，不是可部署AP。

独立CPU源码将重跑原生官方匹配并逐行核对compact anchors，再用保存的实际GPU band RLE重算像素及上限；CPU原渲染器支持/近2带核对采用预锁1e−4实用容差，不冒充GPU逐位认证或用CPU近似band替换指标。GPU与CPU各事前限时1800秒，CPU0/1两线程，GPUallocator4096MiB/整卡观测8192MiB，无自动重试。只有原GPU held退出、父回收与原身份消失后，ROOT才读取产物并安排独立CPU；Q也不自证其运行正常结束。

公开的是启动前保存的独立SOURCE副本及普通状态；运行中的科学原件和绑定SOURCE不为同步读取、复制或哈希。[源码与详细ABI](multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N11_native_editable_region_coverage_source_20261013_01/SOURCE_ABI.md)里的UNBOUND/UNEXECUTED文字是编写时历史状态，当前执行以本文为准。科学数据/完整GT/权重/运行缓存/compact anchors不随仓库提供，源码不是下载后即一键重跑的发行版。

后续由完成资格的覆盖证据决定改目标函数还是改可编辑区域。区域细化、MIS和控制变量均有直接先例；[定向查新](FIXED_BAND_SAMPLING_RISK_PRECEDENT_MEMO_20261013.md)中的几何目标仍是未执行候选，核心创新、充分收敛、正式TEST和跨集均未成立。
