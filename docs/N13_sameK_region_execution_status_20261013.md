# N13 同区域预算点细化：执行状态

记录日期：2026-10-13。此文件为持续更新的普通状态摘要；冻结方案和实际科学工件另行保留。

## 当前状态

- 固定48轮/三个配对种子方案已预登记，冻结GPU关系与独立CPU缓存资格已通过；原full-L CPU allclose在6/741图失败，保留为诊断。
- 701原GPU在完整E08产物保存后报告处实际退出1；新CPU恢复实际0并经ROOT采用，两臂13288更新和原九产物保留，缺失报告不补造。
- 702两臂首段E01–08 GPU0及独立CPU0已ROOT闭合：639.406/184.609秒，43正常/7失败、50身份固定值、8776来源。没有新AP或效用结论。
- 第三种子703已启动单次冷GPU首段，原任务48721；尚未实际终止或CPU资格。全部首段合格后交错接续至48轮。
- 连续48轮及完整187配对评价SOURCE均静态审阅通过，评价尚未绑定或执行；共享真实GPU L的实际存储需求仍需核查。

[第二种子实际结果](../research_round_20260925_global/N13_SEED702_E08_ACTUAL_QUALIFIED_RESULT_20261013.md)、[评价源码](../research_round_20260925_global/multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N13_sameK_fixed_DEV_evaluation_source_20261013_01/EVALUATION_SOURCE_ABI.md)、[定向查新](../research_round_20260925_global/N13_GEOMETRIC_POINT_REFINEMENT_TARGETED_PRIOR_UPDATE_20261013.md)。下文为时间顺序历史记录，以本节为当前状态。

## 可支持的结论

N12 在固定原生匹配群体中显示弱的、依阈值变化的同预算覆盖收益；它还没有证明 AP 提升。N13 要检验相同 PR 头、相同参数量、相同监督点数及训练预算下，更换可编辑区域与对应采样位置是否有用。该设计同时改变支持区域和监督位置，不能解释为纯推理门控效应。48 轮是预先固定的预算，是否趋于稳定仍需实际曲线判断。

未建立核心创新、正式测试或跨数据集有效性，不宣称达到 TGRS 投稿水平。原始数据、全量标注、特征载体、权重和运行缓存不发布，按 [发布范围](../PUBLICATION_SCOPE.md)提供数据获取说明。

## 研发源码

[缓存及训练源码目录](../research_round_20260925_global/multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N13_sameK_region_point_refinement_source_20261013_01/)：cache_fit.py / qualify_cache_cpu.py / region_common.py / CACHE_ABI.md，以及 point_heads.py / train.py / qualify_train_cpu.py / TRAIN_ABI.md。缓存四文件已进入实际运行绑定；训练四文件仅完成静态审阅。原训练工件、终止状态及来源链按每个执行任务单独核验。

## 首个CPU核验失败与根因调查

原v01源码、GPU缓存和失败运行冻结保存。实际检查错误为 Practical CPU native L replay differs; never substitute CPU L for inputs。静态核对的插值与裁剪路径一致：200→800、裁剪有效图像、再恢复原图；尚不能区分平台浮点差与错误载体。历史N07的1e-4近零/近带 allowance 不等于全幅L逐像素allclose保证，N13新增的更强检查此前未经实际校准。下一步以固定第4图做一次有限诊断，报告原判据和差异位置，不后验放宽容差。诊断本身不使缓存合格，不作为方法负结果。

## 有限诊断准备

[诊断登记](../research_round_20260925_global/N13_NATIVE_L_REPLAY_DIAGNOSTIC_PLAN_20261013.md)和[数值验证审阅](../research_round_20260925_global/N13_NATIVE_L_NUMERIC_VALIDATION_REVIEW_20261013.md)已形成。[独立单图诊断源码](../research_round_20260925_global/multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N13_cache_L_replay_diagnostic_source_20261013_01/ABI.md)已经停写，AST/编译检查通过，由ROOT绑定；原持有任务已真实退出1并完成ROOT失败记录核对，104.718秒，已保存诊断报告。诊断固定第4图，原FP32检查仍失败时保存报告后退出1，不能生成缓存Q。核心点包反序列化会带入未使用的监督缓冲，但诊断不访问标签或标注。后续资格方案需单独登记和实际核验。

固定单图已得到[限定范围的调试观察](../research_round_20260925_global/N13_NATIVE_L_REPLAY_DIAGNOSTIC_OBSERVATION_20261013.md)：FP32在536571值中违例1个、FP64违例3个，该图P/原边界区域没有变化。原判据仍失败，未生成Q，没有训练提升或缓存资格。后续登记同设备数值关系重放及独立CPU选择器核验，不按观测调容差。

## 2026-10-13 新版关系复核准备

原 CPU v01 与 index3 诊断的实际退出 1 均保留。新协议先对全部741图、1661匹配query做原两段GPU渲染的FP32逐字节核验，再由新CPU v2独立核验冻结GPU L上的区域、点索引和监督。原full-L CPU allclose记为诊断，不能写成通过。当前关系SOURCE独立静态审阅通过，ROOT准备绑定；关系尚未执行，缓存未获CPU资格，训练未开始。

新协议见 [资格修订](../research_round_20260925_global/N13_CANONICAL_GPU_RELATION_AND_CPU_QUALIFICATION_AMENDMENT_20261013.md)。canonical100未匹配通道是dummy零，不是原all100 M恢复；静态kernel分析不能替代全部实际字节检查。

## GPU 关系叶已实际启动；v02 训练源码待资格

ROOT 已启动单次有限关系叶，等待实际终止。六份新 v02 源码完成独立静态审阅 PASS 并保存；修复了关系 SPEC/report 的来源键映射错误。审阅不代表数值通过。CPU 资格与训练尚未开始；原失败和固定48轮配方保留。

## 全体关系实际通过（157.109秒）

原owned leaf实际exit0，ROOT闭合来源/全部741图与1661query，canonical100和subsetG均逐字节一致，mismatch0，GPU cleanup0。关系资格不能代替CPU缓存Q；CPU v2待执行，训练未开始。[实际观察与范围](../research_round_20260925_global/N13_CANONICAL_GPU_RELATION_OBSERVATION_20261013.md)。

## CPU v2 已实际启动

关系ROOT独立复核实际通过后，新CPU v2已经启动单次复核：完整39正常/6失败/45身份pin，8712项来源，CPU0/1两线程、1800秒、无自动重试。当前尚未获CPU缓存Q，训练未开始；原持有执行句柄持续等待同一任务，不超时重启。

## CPU v2实际资格通过；种子701首段训练已启动

ROOT已确认CPU原leaf实际exit0（834.391秒），全部741图/1661实例的区域、draw、GT监督、alias及原sampled-F1e-4强门通过。完整来源与原PR factory字节相同核验完成。原full-L CPU1e-4在6/741图仍失败，仅按新v2协议记诊断，不宣称原门通过。

种子701 LOG_PR/GEO_PR各1–8轮已实际启动，原owned session38050，固定48总端点/三个配对种子。还未产生合格训练端点或AP。见[CPU完整资格与范围](../research_round_20260925_global/N13_CACHE_CPU_V2_ACTUAL_QUALIFIED_RESULT_20261013.md)。

## 首段8轮保存后REPORT失败，待独立恢复资格

原38050 CLOSED actual1，收尾变量NameError；两臂13288更新记录和E08权重均保留。完整40正常/7失败RAW及8719来源已ROOT核验。失败不属于模型效用结果，原目录不补造REPORT；新CPU恢复任务尚未执行。见[故障与恢复登记](../research_round_20260925_global/N13_SEED701_E08_REPORT_FAILURE_AND_RECOVERY_PLAN_20261013.md)。

## 独立CPU恢复协议已登记，未执行

[恢复采用协议](../research_round_20260925_global/N13_COMPLETED_EPOCH_CPU_RECOVERY_ADOPTION_PROTOCOL_20261013.md)锁定新任务40正常/7失败/47身份pins、原9项产物与原数值判据，不补造原GPU报告、退出0或临时运行测量，不额外训练。新的纯CPU入口正在实现；只有实际新CPU0并由ROOT闭合才可采用端点。后续保持702、703首段后再进入701第9轮的交错顺序。

## 恢复SOURCE停写与静态审阅通过

[新v03纯CPU恢复入口及ABI](../research_round_20260925_global/multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N13_completed_epoch_artifact_recovery_source_20261013_03/ABI.md)已停写，通过AST3.8/编译与一次独立静态审阅；资源记录精确关联的两处收紧已核对。原v02六文件与九项产物不变。新SOURCE将由ROOT单次绑定执行；当前尚未执行，不能宣布采用端点或训练提升。

## 新CPU恢复叶已实际启动

ROOT单次owned session56178已启动；40正常/7失败/47pins和8740项完整来源均已绑定，CPU0/1两线程、1800秒、无CUDA初始化、无优化器更新、无自动重试。原GPU1及9产物仍保留，原REPORT继续缺失。当前尚无恢复采用Q/真实终止资格；等待同一原句柄，观察超时不重启。源码与协议在执行前已经同步。

## CPU恢复实际0、ROOT端点采用资格通过

原56178 CLOSED actual0，181.719秒；完整配对初始化、13288更新记录、状态/AdamW步数/私有RNG、全FIT最终BCE均通过独立检查。ROOT all41正常/7失败及8740项来源闭合，原GPU1没有改为成功，原REPORT仍缺失。完整E08两臂检查点可保留用于后续明确采用协议。见[实际结果与范围](../research_round_20260925_global/N13_SEED701_E08_CPU_RECOVERY_ACTUAL_QUALIFIED_RESULT_20261013.md)。下一步702/703冷启动；目前尚无新AP。

## 后续冷启动SOURCE已停写，未执行

[新v03 cold-only入口及ABI](../research_round_20260925_global/multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N13_qualified_endpoint_training_source_20261013_03/ABI.md)通过AST3.8/编译和一次独立差异审阅；修复报告作用域、继承七失败链、仅允许702/703首8轮冷启动，拒绝所有resume。训练配方/数学函数与原D02相同，48/三种子计划不变。本入口尚未执行，701第9轮仍需明确的新采用分支。


## 702首段GPU已启动；连续48轮源码静态通过

2026-10-13 08:06快照：ROOT原持有任务41567已启动cold702两臂1–8轮，41正常/7失败历史、8753项来源，尚无真实终止及CPU端点资格。新v04连续48轮SOURCE语法与一次独立静态审阅通过，未执行；优化器、私有RNG与累计历史连续恢复。完整187图/18,700查询评价接口已形成，评价源码正在编写，尚无新AP。见[详细状态](../research_round_20260925_global/N13_SEED702_COLD_TRAINING_AND_CONTINUOUS48_SOURCE_STATUS_20261013.md)和[完整评价接口](../research_round_20260925_global/N13_FIXED_DEV_EVALUATION_INTERFACE_CONTRACT_20261013.md)。


## 702已完整资格，703已启动，完整评价SOURCE静态通过

2026-10-13 08:27快照：原702 GPU/独立CPU均actual0并经ROOT闭合，8776来源，原701GPU1保持失败。703单次冷GPU首段已启动，未终止。完整187/18700评价四文件语法和一次独立审阅通过、尚未执行。新增先例memo已保存，边界点精修本身不能作为新颖性，当前无新AP。


## 2026-10-13 / 第三个种子首段GPU实际完成，独立CPU已启动

原703 GPU任务48721已经CLOSED actual0，646.985秒，观察到的整卡峰值2111MiB；两臂首8轮各13,288次更新，报告来源及完整8,792项继承来源经ROOT元数据准备核验。独立CPU原任务35181已单次启动（44正常/7失败、51身份固定值，1800秒），尚未实际终止或获端点资格。当前没有新的AP结果，固定三种子48轮预算不变。

新增[预测修补收益的设计](../research_round_20260925_global/N13_LEARNED_EDIT_UTILITY_NEXT_HYPOTHESIS_20261013.md)已审阅，仍是待检验假设：共享冻结修补器和容量匹配的错误预测对照，分开报告训练覆盖、支持分配与全图排序成本，不把像素修复等同于AP或创新成立。


## 2026-10-13 / 三个首段已资格化；701第9–16轮接续已启动

703 GPU48721和独立CPU35181均CLOSED actual0，分别646.985/181.797秒；ROOT all45正常/7失败及8802项来源检查通过，三seed E08端点现均可按明确协议接续，701原GPU1和缺报告保留。v04接续元数据21396实际0后，701第9–16轮原任务18190已启动，尚未终止。完整187图评价载荷实际约10.954GiB，加2GiB预留需12.954GiB，元数据检查72247实际0、空间足够；没有新AP。[实际结果与完整范围](../research_round_20260925_global/N13_ALL_THREE_E08_ENDPOINTS_AND_CONTINUOUS48_STATUS_20261013.md)。


## 2026-10-13 / 701第9–16轮GPU实际0；独立CPU核验已启动

原18190 CLOSED actual0/648.875秒，整卡观察2379MiB。ROOT全46正常/7失败与8818项来源检查后，CPU准备94470 actual0；两臂v4 E16报告及累计26,576更新引用通过核对。独立CPU原64759已单次启动，尚未终止，E16端点尚未资格化，没有新AP。原701首段GPU1、缺失报告和固定48/三seed保留。[完整实际状态](../research_round_20260925_global/N13_ALL_THREE_E08_ENDPOINTS_AND_CONTINUOUS48_STATUS_20261013.md)。


## 2026-10-13 / 最新：701第16轮已核验，702第9–16轮任务已启动

701原GPU18190与独立CPU64759均已CLOSED actual0，分别648.875秒和186.234秒；ROOT按全部47正常/7失败RAW先行门核对54项原进程身份与8,825项来源，下一接续准备原91032真实退出0并闭合报告、CPU资格、状态与累计26,576更新/臂。701当前合格端点为E16；原首段GPU1与缺失报告继续保留，不改为成功。

702仅恢复其自身正常v3 E08检查点及优化器、私有RNG和累计历史；第9–16轮单次原持有任务62175已启动，尚未实际终止或获得本段独立CPU资格。702与703当前已核验端点仍为E08，固定三配对种子/48轮、逐8轮块交错顺序和评价规则保持不变。全DEV评价尚未执行，没有新AP、创新或TGRS结论。观察超时继续等待同一原任务，不自动重启。以上为最新状态，此前各段为历史执行快照。
