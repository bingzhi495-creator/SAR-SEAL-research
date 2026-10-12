# N13 同K区域训练：准备备忘录

2026-10-13。状态：**PREPARATION_DRAFT_NOT_PREREGISTERED_OR_EXECUTED**。N12已核验主槽位的弱正诊断通过原锁跟进条件，下一项检验真实训练效用。本文件记录静态可行性与ROOT选择，不冒充已冻结执行协议或新结果。

## 实际依据及研究边界

[N12限定结果](../docs/N12_sameK_region_qualified_result_20261013.md)：主固定374配对中同K错误覆盖22.53%→24.67%，条件oracle IoU只+0.000728650；.75槽位覆盖下降。主门通过允许跟进，不支持实际AP提升、收敛或新颖性。几何排序作为强常规对照，不能改名当创新。

## 静态源码盘点

独立发布副本的N07/N10源码表明：FIT741图、1661实例的旧缓存，每GT固定1024点（旧B中有放回768＋全域均匀256；B空时均匀1024），只保存采样位置F256/L/targets/point_indices及e/C/query与GT身份等，不保存完整原图GPU L。不是“全部只在旧B”，也不足以重建新的几何支持。DEV缓存保存全100 query的coarse F/e/M/C，可重放原图L，不必重做DEV教师前向。

N07 C和N10 PR头内部按 |L|≤2 门控。**只改采样和外层区域而继续使用其返回的logits，会导致geometry新增像素无法改动。** PR另返回未门控residual，可保持参数结构不变，由统一调用者计算 L'=L+editable_flag×residual。训练、终点损失、独立CPU核验及全100 query推理必须一致；两臂都走同一调用路径，验证old_B行为一致。

静态出处：

- [N07缓存生产](multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N07_frozen_E24_original_detail_point_refinement_source_20261012_01/cache_fit.py)、[旧缓存资格边界](multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N07_frozen_E24_original_detail_point_refinement_source_20261012_01/qualify_cache_cpu.py)。
- [N10 PR头](multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N10_PR_FL91_direct_precedent_control_source_20261013_01/point_heads.py)、[训练框架](multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N10_PR_FL91_direct_precedent_control_source_20261013_01/train.py)、[评价器](multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N10_PR_FL91_direct_precedent_control_source_20261013_01/evaluate.py)。

## 下一单一检验

ROOT选择同一PR_FL91骨架的LOG_PR/GEO_PR两臂，各40,497参数，作为直接常规点细化骨架下的效用检验。旧8轮C略高于PR不能证明其总体优劣，也不复用旧8轮权重冒充新长预算基线。

- 冻结同一E24，重新生成匹配FIT缓存；保持原有训练匹配/GT顺序/query身份。GT只用于共同训练匹配及监督，不决定区域。保存matched query真实原图GPU FP32 L、P/两区RLE/K、各臂采样索引、采样F/L/targets/regionflag及来源关联；独立CPU从保存的真实L/P重建区域、K、采样RNG及标签。
- 每query两区相同K；几何排序、tie与无异类fallback沿用N12原定义。拟各区有放回768点＋共同全域均匀256点，保留重复点；共同均匀点需独立RNG流确保真实相同。
- 至少三个预锁配对种子，六头全部冷初始化；每对逐权重相同、零输出层、相同实例顺序/优化器/预算/评价和分数。两臂实际采样坐标不同，因此检验**支持域与监督分布的联合方案**，不能单独归因推理区域。
- 较长固定预算草案为48轮，即每头79,728次更新；唯一主端点48，不按DEV挑best或淘汰种子。8/24仅曲线。分段须连续保留optimizer、shuffle与RNG，不能重置并表述为连续训练。具体种子、配方和叶分段/时限仍需在真实缓存可行性与资源计数确定后一次冻结，不直接扩大旧诊断时限。
- 每轮记录全FIT BCE与可编辑点BCE；草案检查37–42与43–48窗口的可编辑BCE，下降仍超过1%则标记端点未稳定。48是预算选择，不自动等于充分训练。若另增加预算，全部种子和两臂共同延长并另登记。
- 新DEV预测对全部100 query构造各臂区域，保持原native class/score/order；原始18,700字典精确一致性先通过。正式新预测AP需独立COCO匹配与评价，N12固定配对只作机制附表。

## 执行与结论要求

科学执行仍由ROOT单一所有者维护，原始真实结束后资格闭合再解释。资格、非有限值或资源失败保留为未完成，不因短期胜负中断或筛选种子。完整报告三组配对AP及均值、波动、固定配对IoU、成本和全部下降结果。

DEV已参与多轮选型，三个新种子不能恢复盲测性质。只有完整公平效用得到支持后再做必要机制对照，并冻结方法开展正式测试/跨公共数据集验证；尚未建立核心技术差异或TGRS投稿依据。下一最小实现是新冻结FIT缓存和统一门控，当前未开展N13训练。
