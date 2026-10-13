# SEAL：公共 SAR 船舶实例分割研究

论文、方法源码、协议和研究记录的工作快照，更新至 **2026-10-13**。研究仍在进行；当前没有可直接投稿的最终论文，也没有经过正式测试和跨数据集验证的 SOTA 结论。


**当前状态（N13）：** 701、702两臂的固定48轮端点已通过独立CPU及ROOT核验，各臂累计79,728次更新；703合格端点为第40轮，每臂累计66,440次更新。703最后第41–48轮单次原GPU任务已启动，尚未实际终止。固定三配对种子/48轮及逐8轮块交错顺序不变，原701首段GPU1和缺失报告保留。完整DEV评价尚未执行，没有新AP、核心创新或TGRS结论。[已核验端点与接续状态](research_round_20260925_global/N13_ALL_THREE_E08_ENDPOINTS_AND_CONTINUOUS48_STATUS_20261013.md)、[完整历史记录](docs/N13_sameK_region_execution_status_20261013.md)。

新[CPU恢复入口](research_round_20260925_global/multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N13_completed_epoch_artifact_recovery_source_20261013_03/ABI.md)已实际通过独立端点检查；原GPU失败与缺失报告保留。

后续[冷启动训练源码](research_round_20260925_global/multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N13_qualified_endpoint_training_source_20261013_03/ABI.md)已静态审阅并保存，702/703首段训练及独立CPU核验已通过；当前入口只允许702/703首段。后续[连续48轮接续源码](research_round_20260925_global/multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N13_continuous48_adopted_endpoint_training_source_20261013_04/ABI.md)已通过静态审阅，三个种子第9–16轮、第17–24轮、第25–32轮及第33–40轮均已通过核验，701、702固定E48已通过独立CPU和ROOT核验，703最后第41–48轮运行中。见[训练与接续状态](research_round_20260925_global/N13_SEED702_COLD_TRAINING_AND_CONTINUOUS48_SOURCE_STATUS_20261013.md)及[完整DEV评价接口](research_round_20260925_global/N13_FIXED_DEV_EVALUATION_INTERFACE_CONTRACT_20261013.md)。

## 从这里开始

- [HRSID官方协议与发布范围核查](research_round_20260925_global/HRSID_OFFICIAL_PROTOCOL_AND_RELEASE_SCOPE_CHECK_20261013.md)：原文公开文本支持裁片比例，场景独立性与数据再发布许可仍待核验；未访问本地数据包。
- [Poly-BRBLE 原论文全文核查](research_round_20260925_global/N13_POLY_BRBLE_FULLTEXT_CHECK_20261013.md)：直接先例已包含边界buffer、不确定选点与点精修；明确当前sameK差异、联合处理限制和后续强对照。
- [SAR提示式基础模型全文核查](research_round_20260925_global/SAR_PROMPTED_FOUNDATION_MODELS_FULLTEXT_REVIEW_20261013.md)：核对监督范围、指标与可复现性，并明确它与当前残差点头实验的可比边界。
- [共享投影与成本公平性审阅](research_round_20260925_global/N13_SHARED_PROJECTION_COST_FAIRNESS_REVIEW_20261013.md)：稠密强对照应同样共享特征投影；列明计算、内存和数值等价的边界，实际加速与质量收益仍待检验。
- [当前研究目标与执行要求](RESEARCH_GOAL.md)：问题、公平比较、重复实验、机制验证和停止规则。
- [修补行动收益与计算分配的直接先例](research_round_20260925_global/N13_ACTION_UTILITY_AND_ALLOCATION_DIRECT_PRIOR_ADDENDUM_20261013.md)：六项原始工作、近期SAR有限补核、像素收益与IoU的区别，以及待检验的质量与成本方案；出版核验限制明确保留，当前训练不改。
- [论文版本索引](docs/PAPER_INDEX.md)：各代正文、补充材料与历史稿的关系。
- 可阅读历史修订稿：[正文 PDF](paper_tgrs/revision_tgrs_20260905/main.pdf)、[补充 PDF](paper_tgrs/revision_tgrs_20260905/supplement.pdf)、[说明](paper_tgrs/revision_tgrs_20260905/README_CN.md)。它们属于旧融合主线，未重写成 10 月当前研究；PDF 与当前源码的对应关系未重新编译核验。
- [数据获取与版本说明](docs/DATA_ACCESS.md)、[上游数据索引](docs/upstream_data_index.json)：PSeg-SSDD 与 HRSID 的官方入口。
- [掩膜残差读出与校准先例核查](research_round_20260925_global/CALIBRATION_PRECEDENT_REVIEW_20261012.md)：PointRend、Implicit PointRend、质量评分及 SAR 直接先例；机制假设仍待检验。
- [完整实验计划](refine-logs/EXPERIMENT_PLAN.md)、[实验跟踪](refine-logs/EXPERIMENT_TRACKER.md)、[下一研究批次](refine-logs/NEXT_RESEARCH_BATCH.md)。这些记录保留历史合同；有日期的最新状态优先。

## 当前证据和决定

所有下表结果均来自历史内部 **DEV187 / 380 GT**，不是官方 TEST。AP 使用百分制；“+1 AP 点”表示 67 到 68。没有把本项目的内部指标与外部论文数值直接排名。

| 固定比较 | mask AP | 能支持的结论 |
|---|---:|---|
| 预选强 E24 原生参考 | 67.071057 | 同协议固定参考 |
| A：原生模型 + 同样点监督，24 轮再追加 48 轮 | 64.686168 | 当前续训配方的端点 |
| B：联合残差读出，同样预算 | 66.797068 | B 高于 A，但低于强 E24；结束该配方扩展 |
| C：冻结 E24，粗 F/e/L 点头，两个下游种子均值 | 68.166136 | 相对 E24 +1.095079 AP 点的开发集效用 |
| R：相同点头加真实 JPEG 7×7 patch，两个下游种子均值 | 68.153211 | 与 C 几乎持平；未支持新增 patch 的作用 |

详见 [固定长预算 A/B 结果](docs/fixed_endpoint_joint_native_E24warm48_result_20261012.md)与 [N07 全部种子及固定端点评价](docs/N07_fixed8_fullDEV_point_refinement_result_20261012.md)。N07 每头固定 8 轮、13,288 更新；两个下游种子共享同一个教师初始化，不能替代多教师重复或充分收敛证据。R−C 配对均值为 −0.012924 AP 点，未满足结果前锁定的跟进规则；保留效用，停止当前 patch 配方扩展。

N09已完成简单全局logit偏移控制：仅在FIT缓存点拟合、只在原生 |L|≤2 内应用、保持query和原生分数。首个DEV评价在原生精确一致性检查处退出1；已保留[历史失败状态](docs/N09_global_offset_runtime_state_20261012.md)，完整修复与合格结果如下。

2026-10-13 已修复原始 all100 评分路径；完整 DEV 的 GPU 评价和独立 CPU 核验均真实退出 0，原生 18,700 候选输出精确复现。全局偏移 AP 为 **67.1466401**，比原生仅高 **0.0755834** 个百分点、比 C 均值低 **1.0194955**；未达到预登记近似替代规则。这只限定本次全局偏移，不证明新颖性或 F/e 因果作用。[完整核验结果](research_round_20260925_global/N09_FIXED_GLOBAL_OFFSET_RESULT_20261013.md)。已进入 [PointRend 式公平结构对照](research_round_20260925_global/FAIR_POINT_HEAD_CONTROL_DESIGN_20261013.md)，实际执行状态见下文 N10 更新。

当前核心创新、正式 TEST、跨集收益和 TGRS 投稿依据仍未成立。失败结果和停止原因保留，不将局部提升包装为最终成功。

## 目录导航

| 目录 | 内容 |
|---|---|
| `paper/`、`paper_tgrs/` | 多代论文 LaTeX / BibTeX、作者说明、历史正文和补充 PDF |
| `carrier/`、`mne/`、`seeding/`、`losses/` | 历史模型扩展、配置与候选组件 |
| `protocol/`、`benchmark/`、`diagnostics/` | 数据转换、评价和误差诊断源码 |
| `research_round_20260925_global/` | 当前及历史训练、固定评价和独立核验 SOURCE |
| `docs/`、`refine-logs/` | 审阅、先例分析、实验计划和研究范围 |
| `autodl/`、`tools/` | 历史环境准备与辅助工具 |

最新 SOURCE 位于 `research_round_20260925_global/multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/` 下：

- `N07_frozen_E24_original_detail_point_refinement_source_20261012_01/`：FIT 缓存、点头训练、GPU 评价、独立 CPU 核验。
- `N09_FIT_only_global_logit_offset_control_source_20261012_01/`：全局偏移数学、FIT 求解及独立 CPU 核验；历史首个评价失败保留。
- `N09_FIT_only_global_logit_offset_control_source_20261013_02/`：恢复原始 all100 评分路径的完整 DEV 评价器。
- `N06_fixed_E24_one_to_one_error_audit_source_20261012_01/`、`N01_uniform_point_supervision_gradient_probe_source_20261012_01/`：固定误差和监督诊断。

## 数据与复现范围

本仓库包含数据获取说明和转换/审计源码，**不包含原始数据、全量标注、权重及运行缓存**，不能称为完整数据镜像。官方 SSDD 的 PSeg 多边形与 BBox/RBox 检测框版本不同；内部 FIT741/DEV187 不等于官方 TEST232。HRSID 的 JPG/PNG、场景协议和纯背景补充版本也必须分别记录。数据及其许可入口见 [DATA_ACCESS.md](docs/DATA_ACCESS.md)。

历史环境记录为 PyTorch 2.0.1+cu118、torchvision 0.15.2、MMEngine 0.10.4、MMCV 2.1.0、MMDetection 3.3.0，并依赖 NumPy、SciPy、OpenCV、Pillow、pycocotools。最新本机资格环境与历史环境不可直接混用；没有声称全仓库在新机器完成重跑。

历史入口示例：`python protocol/pseg_to_coco.py`（需配置数据路径）及 `python protocol/seal_eval.py --preds ... --gt ...`。当前科学入口采用 `--spec ABS --owner-execute`，依赖本地冻结数据、初始化和来源资格链。本次保存源码，不包含全部绑定输入，**尚不能下载后一键复现最新实验**；不得删除检查以冒充已验证的可移植版本。



**2026-10-13 / N10 已闭合：** 四头各固定8轮/13,288次更新，完整DEV GPU评价及独立CPU核验均正常结束并完成ROOT核验。C均值AP **68.166136**，PointRend结构适配PR均值 **67.927378**；C−PR两组均正，平均 **+0.238757 AP点**，未达到预锁+0.30开发优先线，不自动延长此结构。两臂均高于原生67.071057，但不能据此确认创新或充分收敛。原生IoU≥.90固定配对上四臂平均IoU均小幅下降，下一步检验可编辑区域的误差覆盖与修复上限。见 [完整合格结果](docs/N10_fixed8_C_PR_fullDEV_qualified_result_20261013.md)、[全DEV状态](research_round_20260925_global/N10_FULLDEV_STATUS_20261013.md)、[固定协议](refine-logs/N10_PR_FL91_20261013_01/EXPERIMENT_PLAN.md)、[实际状态](refine-logs/N10_PR_FL91_20261013_01/EXPERIMENT_TRACKER.md)及 [研发代码](research_round_20260925_global/multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N10_PR_FL91_direct_precedent_control_source_20261013_01/)。

当前C实例条件偏置属于已知原语，不能单独宣称创新；定向先例与最小反证方案见 [查新备忘录](research_round_20260925_global/C_CONDITIONED_POINT_HEAD_NOVELTY_MEMO_20261013.md)。

下一方向依据 [既有误差证据](docs/N01_N06_N07_N09_ERROR_EVIDENCE_FOR_NEXT_DECISION_20261013.md)与 [固定区域/采样风险定向查新](research_round_20260925_global/FIXED_BAND_SAMPLING_RISK_PRECEDENT_MEMO_20261013.md)。区域监督、MIS和控制变量均有先例；候选全图几何目标尚未执行或确认，不能用新名称替代实证。

**2026-10-13 / N11：** GPU及独立CPU均已实际退出0并完成ROOT资格。原生.50固定374对中，当前区域只覆盖 **22.53%错分像素**；仅改带内损失/容量无法覆盖多数排除错误。见 [限定范围的完整结果](docs/N11_original_editable_band_qualified_result_20261013.md)，下一方向优先强几何区域对照与可否证的新选择规则；尚未建立新方法效用或创新。见 [当前状态与范围](research_round_20260925_global/N11_EDITABLE_REGION_COVERAGE_STATUS_20261013.md)及 [诊断SOURCE](research_round_20260925_global/multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N11_native_editable_region_coverage_source_20261013_01/SOURCE_ABI.md)。

## 持续同步

本仓库是后续研究的持续记录入口。总结、综述、方法设计、研发源码与配置、论文更新及实质对话研究总结形成后，在当前工作任务内及时提交；实验成功、失败和未完成的验证均按实际状态保留。只有核验远端提交与目标文件后才记为已同步，失败则标为待同步；应用关闭或离线时不承诺自动上传。具体范围与流程见 [持续同步规则](docs/CONTINUOUS_SYNC.md)。

## 发布范围与许可

首次发布快照及完整性验证见 [publication_manifest_20261012.json](docs/publication_manifest_20261012.json)（限定于其中记录的提交；后续版本以 Git 提交历史为准），历史工作区盘点见 `docs/workspace_inventory_20261011.csv.gz`（元数据，非原始数据）。[发布范围](PUBLICATION_SCOPE.md)明确哪些内容未随仓库提供。

作者、单位、日期和其他提交元数据保留占位符。自有项目尚未指定统一开源许可；第三方实现使用原始来源链接与各自许可证，不把第三方代码改授为本项目许可。外部下载的文献全文、第三方参考仓库、凭据、运维配置、无关工程以及整个工作区 Git 历史不随本快照发布。

**以下为历史形成记录；当前训练进度以顶部和最新状态段为准。**

当时下一项仅作为待检验假设：同一冻结模型、点头、编辑与训练预算下，先比较几何边界选择与不确定性选择，完整训练采用至少三个配对种子。强先例、反证条件及执行边界见 [区域选择设计备忘录](research_round_20260925_global/N11_REGION_CHOICE_NEXT_HYPOTHESIS_20261013.md)；尚未新增训练或确认创新。

**N12 实验前登记：** [同预算区域覆盖诊断协议](refine-logs/N12_REGION_COVERAGE_20261013_01/EXPERIMENT_PLAN.md)已锁定主比较与反证规则；[执行跟踪](refine-logs/N12_REGION_COVERAGE_20261013_01/EXPERIMENT_TRACKER.md)记录实际状态：首个GPU运行在依赖文件来源核验处退出1，尚未开始区域计算；修正实际包树绑定后，GPU及独立CPU均真实退出0并完成ROOT核验。失败源码与修正版均保存，见[运行状态](docs/N12_region_diagnostic_runtime_status_20261013.md)。只有真实GPU及独立CPU核验完成后才报告新覆盖/距离结果；并非新方法效用。


**N12 已核验诊断：** 相同K下，主固定374配对的错误覆盖22.53%→24.67%，条件oracle IoU仅增加0.000728650；.75固定集合覆盖略降，两臂oracle≥.90/.95主槽位计数未增加。通过预登记的有限跟进条件，将登记匹配FIT缓存与至少三个配对种子的公平训练，尚无新训练提升或创新。见[完整限定结果](docs/N12_sameK_region_qualified_result_20261013.md)。


**历史快照（首轮缓存核验失败时，后续进展见顶部当前状态）：** 下一项[N13训练准备](research_round_20260925_global/N13_REGION_TRAINING_PREPARATION_20261013.md)已完成缓存/门控静态盘点：旧FIT采样缓存不足以构造新区域，旧头内部门控必须统一外移。采用相同PR骨架和三个冷初始化配对种子，已在新缓存计算前登记48轮固定预算、评价和停止规则；新缓存和训练源码均通过一次独立静态审阅；GPU缓存已正常退出并完成生成阶段ROOT核验；首个独立CPU核验在第4图原生L重算检查处退出1，正在调查；缓存尚未资格，未开始训练。失败原件保留，不改判据追求通过。缓存源码已经保存，见[执行状态与代码入口](docs/N13_sameK_region_execution_status_20261013.md)。见[固定协议](refine-logs/N13_SAMEK_PR_20261013_01/EXPERIMENT_PLAN.md)和[实际跟踪](refine-logs/N13_SAMEK_PR_20261013_01/EXPERIMENT_TRACKER.md)。

N13核验失败后，已在新的数值执行前保存[固定单图诊断登记](research_round_20260925_global/N13_NATIVE_L_REPLAY_DIAGNOSTIC_PLAN_20261013.md)及[原生L数值验证审阅](research_round_20260925_global/N13_NATIVE_L_NUMERIC_VALIDATION_REVIEW_20261013.md)；新[诊断源码](research_round_20260925_global/multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N13_cache_L_replay_diagnostic_source_20261013_01/ABI.md)仅用于定位重算差异，不改变容差或放行未资格缓存。

**N13有限诊断已结束：** 第4图原FP32重算有1/536571个值超过原容差，FP64有3个；该图P与原边界区域未变。原判据仍失败，未产生缓存Q或开始训练。见[限定调试观察](research_round_20260925_global/N13_NATIVE_L_REPLAY_DIAGNOSTIC_OBSERVATION_20261013.md)；不以单图数值差或源码审阅宣称方法效用、创新或全缓存资格。

- [N13 全样本 GPU 关系与 CPU v2 资格修订](research_round_20260925_global/N13_CANONICAL_GPU_RELATION_AND_CPU_QUALIFICATION_AMENDMENT_20261013.md)：GPU关系与CPU v2资格实际通过，原失败及固定训练配方保留。

- [N13 CPU v2 与固定48轮训练源码接口](research_round_20260925_global/multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N13_sameK_region_qualified_relation_training_source_20261013_02/SOURCE_ABI.md)：CPU v2实际资格通过，种子701配对训练1–8轮已启动。

- [N13 全体冻结 M→L GPU 关系实际通过](research_round_20260925_global/N13_CANONICAL_GPU_RELATION_OBSERVATION_20261013.md)：741图/1661query逐字节一致，原执行157.109秒；已获CPU v2资格并启动首段配对训练。

- [N13 直接先例定点更新（六项）](research_round_20260925_global/N13_POINT_REGION_DIRECT_PRECEDENT_UPDATE_20261013.md)：含2026年SeSAM近邻；同K几何区域是待检验差异，已有采样/点头/带外继承不算独立创新。当前训练配方保持不变。

- [N13 CPU v2实际合格结果与训练准入范围](research_round_20260925_global/N13_CACHE_CPU_V2_ACTUAL_QUALIFIED_RESULT_20261013.md)：全部区域/监督/采样强门通过；原full-L CPU容差在6图失败保留为诊断。种子701首段配对训练运行中。


**本轮研发更新：** [完整DEV评价源码](research_round_20260925_global/multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N13_sameK_fixed_DEV_evaluation_source_20261013_01/EVALUATION_SOURCE_ABI.md)已静态审阅，尚未实际评价；[几何与点精修定向查新](research_round_20260925_global/N13_GEOMETRIC_POINT_REFINEMENT_TARGETED_PRIOR_UPDATE_20261013.md)补充五项论文及更接近的Poly-BRBLE技术披露，创新主张需以具体支持分配差异和强对照效用为依据。

- [后续候选：预测实际修补的正负收益](research_round_20260925_global/N13_LEARNED_EDIT_UTILITY_NEXT_HYPOTHESIS_20261013.md)：与同容量错误概率选择器比较；待登记、未执行，不改变当前三种子48轮实验。像素收益不等于IoU/AP提升。

- [N13三个首段端点完整核验及连续48轮进度](research_round_20260925_global/N13_ALL_THREE_E08_ENDPOINTS_AND_CONTINUOUS48_STATUS_20261013.md)：701原报告失败保留；701、702已核验固定E48（各79,728更新/臂），703合格端点为E40；703最后第41–48轮运行中。完整DEV实际载荷约10.95GiB，尚无新AP。
