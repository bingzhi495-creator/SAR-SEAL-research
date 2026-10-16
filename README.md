# SEAL：公共 SAR 船舶实例分割研究

论文、方法源码、协议和研究记录的工作快照，更新至 **2026-10-16**。研究仍在进行；当前没有可直接投稿的最终论文，也没有经过正式测试和跨数据集验证的 SOTA 结论。



**自主设计与作者基线归属补充：** 团队自定义UNet++按完整结构评估；历史D目录的单个FPN文件与DiffSARShipInst作者固定版本统一换行后完全一致，SCJE-FPN及其中采用的CARAFE/ACmix须按作者基线引用。两份本地R535配置声明启用SCJE，不能仅改builder构造普通FPN对照；这不认证当前E/N20调用。[归属、实际路由与对照要求](research_round_20260925_global/UNETPP_OWN_DESIGN_AND_DIFFSARSHIPINST_BASELINE_ATTRIBUTION_20261016.md)。

**当前进行（N20）：** 第二种子202610150202的固定24轮训练、端点核验和四段DEV元数据接收历史保持。CPU32第5批[24,30)原任务191实际退出0，Root来源及保存数值证明摘要的独立元数据接收已通过；目前5/32批、30/187图、3000/18700自然query完成，十组对照保留。第6批[30,36)原任务192、supervisor26042已启动，活动产物不读取。当前192个正常原任务终态、26个保留科学失败、合计218；活动任务不计完成。全187图数值资格与统一COCO仍待完成，尚无第二种子AP。首201主差值仍+0.197741 AP点；重复效用、正式TEST、跨集、机制与创新结论待证。 [结果范围与当前状态](research_round_20260925_global/N20_REPEAT202_TRAINING_COMPLETE_ENDPOINT_CPU_STATUS_20261016.md)。

**复核开销测量：** 第二批外部接收实际完成分段计时，共297.468秒，其中原来源及保存证明核验293.453秒；不能进一步归因于某一具体检查，也不作为模型FPS或加速收益。当前32范围、科学入口与核验规则保持。[实测分项与后续工程候选](research_round_20260925_global/N20_CPU_REVIEW_COST_AND_FUTURE_PROTOCOL_20261016.md)。

**首种子201的全量汇总状态：** 已按32份固定范围执行全187图统一COCO，并完成独立post与Root实际终态核验；静态审阅仅保留为执行前历史。[静态审阅记录](research_round_20260925_global/N20_FINAL32_ENTRY_STATIC_READINESS_REVIEW_20261016.md)。

**跨集数据准备：** 本地5604个JPG的train/test实例标注，与已核官方PNG对应split按文件stem、ID及全部标注字段精确相等；完整文本SHA不同。5604文件编码哈希/头部检查实际0，均800×800三分量JPEG，尚未解码像素/mask。发现两组跨train/test相同编码图、一组test内部重复，其标注多重集均不同；原协议保留，未来另预锁去重敏感性补充。合并标注16951实例与split合计16969不等，不能互换。官方JPG归档身份、掩膜空间对齐和母景独立未确认，不恢复盲TEST；不继续等待PNG下载，不改变N20。[本地数据完整补充](research_round_20260925_global/HRSID_LOCAL_JPG_METADATA_AND_VERSION_BOUNDARY_20261015.md)。

**增量查新：** [O²Former与DACF-net先例核查](research_round_20260925_global/DIRECT_SAR_INSTANCE_SEGMENTATION_INCREMENTAL_PRIOR_CHECK_20261015_01.md)分别标记全文和官方域索引证据，不与当前DEV跨协议排名；方向、多尺度和上下文模块已有直接先例；已补记v1实现需澄清的形状与对齐边界，不将自行适配称作完整作者复现。

**采样与精修直接先例补查：** LIIF已包含原生3×3、相对坐标及解码后局部集成；I-MedSAM展示先插值到目标网格再作连续精修的另一条路线。条件rank上界不是本项目实测瓶颈；原生邻域本身不能算新原语。七项有限证据区分代码、正文与索引摘录，不宣称穷尽2026研究。[有限补查与设计边界](research_round_20260925_global/N20_NATIVE_GRID_VS_RENDERED_F9_DIRECT_PRIOR_REVIEW_20261016_01.md)。

**自主UNet++与历史资产扩展审阅：** 按用户说明记录自主设计，已还原13组实际前向结构；重点保留选择性跨尺度传输、H/4语义与方向互补、采样重建协同三条候选。扩展登记1009个源码/配置/示例项；D内260个模型候选中222个关联数值训练记录，另20个pth是评价缓存。Mamba100轮/MSCK160轮等检测成绩与实例mask AP分开；旧full骨架与clean50轮不能混同，来源未接上不等于未训练。[扩展总评与下一步](research_round_20260925_global/UNETPP_EXPANDED_MODEL_ASSET_REVIEW_AND_NEXT_DESIGN_20261015.md)、[自主结构还原](research_round_20260925_global/UNETPP_SELF_DESIGNED_STRUCTURE_RECONSTRUCTION_20261015.md)、[全部源码目录](research_round_20260925_global/UNETPP_COMPLETE_MODEL_SOURCE_CATALOG_20261015.md)、[训练资产目录](research_round_20260925_global/UNETPP_COMPLETE_TRAINED_ASSET_CATALOG_20261015.md)、[有界先例核对](research_round_20260925_global/UNETPP_SELF_STRUCTURE_BOUNDED_PRIOR_CONTEXT_20261015.md)。三条均尚未确认学术新颖性或当前实例效用，不改N20端点。

**UNet++模块内部改造补充：** 已核到复合残差注意力、门控细节上采样及真正进入输出的抗混叠下采样，不能只按组件名字当作原版。另定位full的未使用BlurPool结果、前置BN旁路和DSConv批次索引问题，目前为静态证据，尚未张量回归；不改正在运行的N20。[具体实现、问题与验证范围](research_round_20260925_global/UNETPP_MODULE_INTERNAL_CHANGES_SOURCE_CHECK_20261016.md)。

**采样候选直接先例补查（10月16日）：** 已核BlurPool、FADE、DySample、FreqFusion，以及2025遥感双向重采样和2026抗混叠分割先例；模块内部改造已确认，联合机制与实例效用待验。记录聚合/相位、普通残差容量、外部skip及共享BN的替代解释；不改当前N20。[有限查新与可否定假设](research_round_20260925_global/UNETPP_SAMPLING_DIRECT_PRIOR_TARGETED_REVIEW_20261016.md)。

**自主传输路由的近年先例补查（10月16日）：** 新读ICUnet++（2023）、RPAU-Net++（2025）及两篇遥感分割作者稿，补充直接门控位置对照；不以“UNet+++CBAM”称新。Root确认原model_new1为先门控后上采样，后续需区分运算顺序与被门控支路，所列对照尚未执行。[有界查新与源码顺序校核](research_round_20260925_global/UNETPP_UPSAMPLE_TRANSPORT_RECENT_DIRECT_PRIOR_20261016.md)。

**旧自主结构的权重来源补查：** 222已定位writer/reader代码声明；111/222/两份ALL_NEW完成只读ZIP与pickle操作码元数据登记，未加载模型或tensor。不同前向、开关与完整训练身份仍需核对；部分旧AP50/AP75实际是整幅掩膜IoU阈值通过指标，不能用于COCO mask AP排名。[来源与结构线索](research_round_20260925_global/UNETPP_LEGACY_PROVENANCE_AND_CONTAINER_EVIDENCE_20261015.md)、[有界只读工具](tools/legacy_weight_metadata_intake.py)、[两项合成回归](tests/test_legacy_weight_metadata_intake.py)。

**已完成（N16）：** 同24轮配对训练及完整187图/380GT/18700自然query/12策略GPU、独立数值CPU、统一全量COCO与Root资格均完成。主Utility AP **67.666732**，同训Error **67.895932**，强参考 **68.099385**；Delta **−0.432653** AP点，按预锁STOP，不补旧162/163。原生67.071057的局部收益不能替代强参考。原受控截止/接口失败均保留，不改主端点；单首初始化内部DEV，没有跨集、核心创新或TGRS结论。[完整结果和范围](research_round_20260925_global/N16_FULLDEV_QUALIFIED_FIXED24_RESULT_AND_STOP_20261015.md)、[独立质疑](research_round_20260925_global/N16_RESULT_TO_CLAIM_AND_LOCAL_CONTEXT_INDEPENDENT_CHALLENGE_20261015.md)。

**最近已完成的单模块结果（N14）：** 固定新stage24训练与完整DEV187/all100 GPU、独立CPU及ROOT资格均通过（每臂39,864更新，全18,700 query同GPU输入sign/RLE/COCO12一致）。BASE AP67.686612、MEMORY67.683479；模块−基线−0.003132点、相对原较强E24端点−0.415905点，C1未通过，按预锁规则停止该配方追加seed/预算和融合。训练BCE下降约1.41%没有转为AP收益；不推断方法总体失效或充分收敛。[完整合格结果与停止决定](research_round_20260925_global/N14_FIXED24_FAIR_COMPARISON_QUALIFIED_RESULT_AND_STOP_20261014.md)、[首次AP前规则](research_round_20260925_global/N14_FIXED24_FIRST_SEED_DECISION_CLARIFICATION_20261014.md)。下一轮[固定强E24行动机会诊断](refine-logs/N15_FIXED_ACTION_OPPORTUNITY_20261014_01/EXPERIMENT_PLAN.md)已于UTC04:31:14预登记，[真实GPU、独立CPU与ROOT诊断现已全部合格](research_round_20260925_global/N15_FIXED_E24_ACTION_OPPORTUNITY_QUALIFIED_RESULT_AND_NEXT_STEP_20261014.md)：同K的C行动oracle相对Omega平均+1.461603 IoU点，286/374>=.5点，高IoU实例增加39，预锁机会门通过。当前是GT理想行动选择上界，没有学习选择器的AP结果，[N16唯一公平scout协议现已锁定](research_round_20260925_global/N16_FAIR_UTILITY_SELECTOR_LOCKED_TRAINING_DESIGN_20261014.md)，已核原matcher负query不受mask监督，双方共同扩展背景代理target并纠正分层采样比例；完整缓存已通过GPU、独立CPU和ROOT核验，两scout已完成相同固定24轮训练及E8/E24端点资格；两个实现失败和原641/741受控时限失败均保留。固定无GT候选C、相同K预算和374匹配条件人口，区分实际PR可修补机会、避免伤害及理想GT上界。GO门提前锁定，没有新AP或核心创新结论。原[下一假设](research_round_20260925_global/N15_ACTION_UTILITY_SUPPORT_SELECTION_NEXT_HYPOTHESIS_20261014.md)保留提出时状态。

**当前研发（N17已完成）：** 同预算24轮LOCAL/CENTER、全187图/380GT/18700query、五GPU与七数值CPU、统一全量官方COCO及post实际退出后Root资格均完成。主E24-atmost LOCAL **67.786531**、CENTER **68.103370**、强R **68.099385**，Δ **−0.316840 AP点**，按预锁STOP当前配方，不补172/173、不融合。正常136/失败21/157终态身份；没有运行中的N17科学任务。N18诊断已实际完成并核验，见下一段；没有创新/TGRS就绪结论。[完整结果](research_round_20260925_global/N17_FULLDEV_QUALIFIED_FIXED24_RESULT_AND_STOP_20261015.md)、[全部10端点原始stats](research_round_20260925_global/results/N17_FIXED24_ALL10_COCO_STATS_20261015.json)、[claim判定](research_round_20260925_global/N17_RESULT_TO_CLAIM_20261015/findings.md)。

**N18已核诊断：** 原374固定配对，LOCAL可修选中4574/有害3651，强Ω6564/8528；LOCAL同时减少纠正和误改，漏选率58.140%高于Ω39.929%。PR可修11146/65423（17.037%）与N15一致，不能据此认定主要AP瓶颈。原CPU实际0/472.219秒，正常137/失败21/158身份，无科学任务运行。下一项先研究FP/FN的非对称修补取舍，再锁定公平强对照；没有新训练/AP/创新结论。[完整统计和下一假设](research_round_20260925_global/N18_QUALIFIED_ROUTING_CAPABILITY_RESULT_AND_NEXT_HYPOTHESIS_20261015.md)、[像素数与AP边界](research_round_20260925_global/N18_PIXEL_COUNTS_VERSUS_AP_INTERPRETATION_20261015.md)。

**最新完成（N19）：** 固定1/2方向评分DIR AP **67.726509**，普通偏置67.816800、反向偏置67.789369、强R68.099385；主Δ **−0.372876点**，当前方案STOP，不补评分参数/质量头/种子。全187图/18700query及Root资格完成，正常138/失败21/159身份，无科学任务运行。后续真实feature邻域同容量/同预算检验N20已完成缓存资格并启动原训练任务，见顶部当前状态。[全部结果与范围](research_round_20260925_global/N19_FIXED_HALF_FULLDEV_QUALIFIED_RESULT_AND_STOP_20261015.md)、[原始stats12](research_round_20260925_global/results/N19_FIXED_HALF_ALL6_COCO_STATS_20261015.json)。

**机会与指标的边界：** [条件IoU机会不能保证全量AP](research_round_20260925_global/N17_CONDITIONAL_IOU_OPPORTUNITY_VERSUS_FULL_AP_20261015.md)给出结果后最小诊断：区分实际选点漏掉可修位置与固定PR本身不可修；N18统计已完成，不改变主端点/停止规则，不把未匹配query当真实背景。

**新增直接先例（GLDet）：** 用户提供的TGRS2025论文含实例分割，原文SSDD/HRSID mask AP71.3/58.6、整网500轮。已核全文/表格；完整作者代码尚未确认。空间扫描/邻域卷积/方向pool需要新空间载体，当前点头不能原样接入；N14无空间memory首试已按负结果停止，GLDet的完整空间结构迁移仍未执行，不直接比较不同划分AP。[原论文综述](research_round_20260925_global/GLDET_PRIMARY_PAPER_REVIEW_AND_SEGMENTATION_TRANSFER_20261014.md)、[代码现况和单项局部信息检验设计](research_round_20260925_global/GLDET_CODE_RECHECK_AND_MINIMAL_LOCAL_INFORMATION_TEST_20261015.md)。其局部信息检验已演变为N17：完整FIT邻域缓存和固定24轮训练已合格，DEV全量验证完成且主结果负收益停止；这是低成本信息检验，尚非完整GLDet复现或新创新。不能使用DEV捕获训练。

**局部特征与细化的直接先例：** 新增[五项定向核查](research_round_20260925_global/N20_DIRECT_REFINEMENT_PRECEDENT_FOCUSED_REVIEW_20261015_01.md)，覆盖RefineMask、Mask Transfiner、SegRefiner、SAMRefiner（ICLR 2025）与UnfoldCRF（2026预印本）。普通3×3真实邻域即使涨点，也先支持当前配方的信息效用；稀疏点交互、多阶段细化和通用条件修正均有先例，尚不能宣称结构创新或SOTA。

**当前已核验结果（N13）：** 三种子固定E48 LOG/GEO均值67.768660/67.579076 AP，配对差−0.189583点，三组均负，已停止该GEO联合配方扩展。全部E24曲线已核验，LOG/GEO均值67.992117/67.781883。真实诊断CPU失败9保留；image113四路探针实际0并ROOT核验，仅支持此例CPU/GPU特征渲染差异，同GPU F的CPU PR满足原容差，全374对尚未qualified。[局部探针结果](research_round_20260925_global/N13_IMAGE113_NUMERICAL_PROBE_RESULT_20261014.md)。无正式TEST、跨集或核心创新/TGRS结论。


新[CPU恢复入口](research_round_20260925_global/multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N13_completed_epoch_artifact_recovery_source_20261013_03/ABI.md)已实际通过独立端点检查；原GPU失败与缺失报告保留。

后续[冷启动训练源码](research_round_20260925_global/multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N13_qualified_endpoint_training_source_20261013_03/ABI.md)已静态审阅并保存，702/703首段训练及独立CPU核验已通过；当前入口只允许702/703首段。后续[连续48轮接续源码](research_round_20260925_global/multimodal_data_pilot_20260928/native_geometry_paired_mask_readout_preparation_20261009/N13_continuous48_adopted_endpoint_training_source_20261013_04/ABI.md)已通过静态审阅，三个种子第9–16轮、第17–24轮、第25–32轮及第33–40轮均已通过核验，三个固定E48均已通过独立CPU和ROOT核验，原固定DEV评价失败保留，修复版GPU/CPU及ROOT已通过，701 E08曲线及三条E48主端点均已合格；三个E08曲线已完成独立CPU和ROOT核验，全部三条E24已完成独立CPU/ROOT核验，真实固定PR诊断GPU原59604实际0，独立CPU原4404实际1，数值故障定位中。见[训练与接续状态](research_round_20260925_global/N13_SEED702_COLD_TRAINING_AND_CONTINUOUS48_SOURCE_STATUS_20261013.md)及[完整DEV评价接口](research_round_20260925_global/N13_FIXED_DEV_EVALUATION_INTERFACE_CONTRACT_20261013.md)。

新[原栅格固定PR适配器草案](research_round_20260925_global/code/N13_original_raster_fixed_PR_adapter_draft.py)实现已知F/L算术及独立候选分块；无GT、K或最终选集合输入。已有CPU合成验证；新[非零残差/非连续栅格逐点检查](research_round_20260925_global/code/N13_original_raster_fixed_PR_nonzero_strided_check.py)实际通过，真实权重映射、渲染/导出及固定配对的源码规则已完成[文字核对](research_round_20260925_global/N13_FIXED_PR_ACTUAL_SOURCE_INTERFACE_REVIEW_20261014.md)，真实输入/端点绑定、域定义、科学运行时及大栅格资源资格仍待核验。见[诊断方案和验证范围](research_round_20260925_global/N13_FIXED_PR_COVERAGE_REPAIRABILITY_DIAGNOSTIC_DRAFT_20261014.md)。 已补入条件IoU诊断进入完整AP检验的两个必要条件：自然全部query的GT无关策略，以及明确固定候选/exact-K或拒绝写入的行动范围；当前未执行新方法评价。新增[纯诊断统计](research_round_20260925_global/code/N13_fixed_PR_diagnostic_statistics_draft.py)及[独立手算检查](research_round_20260925_global/code/N13_fixed_PR_diagnostic_statistics_independent_hand_check.py)已通过host合成检查，仅SYNTHETIC_ONLY。 已核对外部作者width2函数的严格邻域计数规则及空/满/孤点静态含义；本地作者函数及N12 v02调用/计数规则已在任务间隙完成ROOT文字核对，真实诊断输入/域重建和运行资格仍待核验。

新[候选—原LOG重建—统计连接层](research_round_20260925_global/N13_FIXED_PR_PIPELINE_SYNTHETIC_INTEGRATION_CHECK_20261014.md)已保存；72项首次合成检查与独立106项halo/域外补查均实际退出0。新增域真实翻转、非空域外保持和完整C冒充原参考的统计前拒绝已验证，仅SYNTHETIC_ONLY，未取得真实SAR诊断资格。

新[显式区域装配接口](research_round_20260925_global/N13_FIXED_PR_NATIVE_DOMAIN_ASSEMBLY_SYNTHETIC_CHECK_20261014.md)连接native L、CPU作者回调与互斥旧域/halo；49项字面回调检查及13项拒绝实际通过，尚未验证真实作者数值、来源、CUDA或SAR域。

新[诊断接口补齐](research_round_20260925_global/N13_FIXED_PR_PRECOMPUTED_POPULATION_AND_DISTANCE_INTERFACE_CHECK_20261014.md)复用同一人口汇总，允许保留逐对JSON后释放栅格，并在原LOG重建后透传显式距离分层。59项/10拒绝和35项有限检查实际通过；不认证预计算数值、距离来源或真实SAR效果。

## 从这里开始

- [GLDet附件定向复核与loss边界](research_round_20260925_global/GLDET_ATTACHMENT_TARGETED_RECHECK_AND_LOSS_SCOPE_20261015_01.md)：确认实例分割直接先例；式(7)10/3/10属于标签分配cost，不能写成联合训练loss权重。
- [HRSID官方PNG编码文件获取试点](research_round_20260925_global/HRSID_OFFICIAL_PNG_ENCODED_IMAGE_BYTES_PILOT_20261015_01.md)：两文件真实字节/SHA/ZIP及PNG容器CRC/IHDR通过；未解码像素、未取得完整数据集或评价模型，原header解析失败保留。
- [HRSID官方PNG同包TRAIN/TEST实核](research_round_20260925_global/HRSID_PNG_OFFICIAL_ARCHIVE_DIRECTORY_AND_TEST_METADATA_20261015_01.md)：Root独立核TRAIN3642图/11047实例、TEST1962图/5922实例，文件名无交集且覆盖5604 PNG目录；数值ID跨split复用，不能裸ID合并。仅元数据，尚无影像/mask/AP资格。
- [N20渲染F9条件性信息分析](research_round_20260925_global/N20_RENDERED_F9_CONDITIONAL_INFORMATION_AND_CONTROL_SCOPE_20261015_01.md)：明确±1为最终栅格像素；仅在同一有效双线性单元等条件下有四基表示，尚无实际秩/覆盖率或效用结论。
- [HRSID作者固定版本标注实文件核验](research_round_20260925_global/HRSID_OFFICIAL_ANNOTATION_INTAKE_20261015_01.md)：取得3573图/9870实例的Git候选JSON并独立核对，215图无标注关联；该候选的配套TEST及母景独立未认证，与同包PNG的具体差异见上方实核，不能混配。

- [YOLO11n路口微调实测](docs/edge_traffic/xavier_nx_20261015/training_20261015/TRAINING_REPORT.md)：完成1022/128/128、30轮训练；四类mAP50 17.53%→48.12%，行人AP50仅16.33%；公开视频仍有小车误报公交/广告牌误报卡车，不具备五类高精度部署结论；NX与电动自行车未验证。

- [N20真实F邻域实验](refine-logs/N20_TRUE_FEATURE_NEIGHBOR_20261015_01/EXPERIMENT_PLAN.md)：全741图缓存资格完成，固定24轮E8/E24保存模型已独立CPU/Root核验；原GPU清理失败保留，完整DEV AP尚未执行。[恢复核验](research_round_20260925_global/N20_SAVED_E8_E24_CPU_RECOVERY_QUALIFIED_STATUS_20261015.md)、[执行状态](refine-logs/N20_TRUE_FEATURE_NEIGHBOR_20261015_01/EXPERIMENT_TRACKER.md)。

- [N20当前32份核验合同](research_round_20260925_global/code/N20_FIXEDDEV_FEATURE_ERROR_CPU_FRONT_UNBOUND_CONTRACT_20261015_02.md)与[全187统一评价合同](research_round_20260925_global/code/N20_FIXEDDEV_FULL187_COCO_CPU_FRONT_UNBOUND_CONTRACT_20261015_02.md)：前七份独立CPU已实际0并通过Root核验，第八份已启动；final/post完整entry02未执行；旧八份合同保留为历史。主端点与继续规则不变，合成检查不是实际AP或发表依据。

- [N20真实F邻域结构原型与训练设计](research_round_20260925_global/N20_TRUE_FEATURE_NEIGHBOR_SCOUT_DESIGN_AND_IMPLEMENTATION_STATUS_20261015.md)：18,851参数零初始化空间适配器；设计文档保留提出时状态，最新固定24轮保存模型已核验。普通DW结构仅为信息基线，不称新创新，实际AP仍待完整DEV。

- [路口五类检测数据集核查（2026-10-15）](docs/edge_traffic/xavier_nx_20261015/DATASET_SHORTLIST.md)：Intersection-Flow-5K、MIO-TCD、BDD100K与电动自行车专项候选；明确类别/视角/许可/获取状态及部分标注问题。

- [Xavier NX 路口预训练检测选型与公开实测（2026-10-15）](docs/edge_traffic/xavier_nx_20261015/REPORT.md)：三模型各完整719帧；权重来源、可复跑工具、误报失败例与NX待验收条件。尚未满足电动自行车独立识别/五类高准确率/NX30FPS。

- [N19固定方向代价探针](refine-logs/N19_FIXED_HALF_DIRECTIONAL_READOUT_20261015_01/EXPERIMENT_PLAN.md)：只用已有同头概率、固定1/2评分及普通/反向偏置对照，协议已事前锁定；body手算与outer接口检查通过，源码及一次独立定向审阅完成。原CPU实际0/426.812秒，三份完整AP及Root资格完成，主DIR67.726509，Δ−0.372876点，按预锁STOP；[完整结果](research_round_20260925_global/N19_FIXED_HALF_FULLDEV_QUALIFIED_RESULT_AND_STOP_20261015.md)。[查新](research_round_20260925_global/N19_IOU_DIRECTIONAL_ACTION_RISK_TARGETED_PRIOR_20261015.md)判为低新颖性/效用待检验，保持旧STOP，不称新理论或质量估计效应。

- [N18最小机制诊断计划](refine-logs/N18_FIXED_PR_ROUTING_CAPABILITY_20261015_01/EXPERIMENT_PLAN.md)：固定374配对复用已存动作，区分路由漏选与PR纠错能力；纯计数手算及单项接口检查已通过；新读取源码完成一次独立静态审查。真实CPU诊断实际退出0并通过Root整数/来源核对；结果见[N18诊断](research_round_20260925_global/N18_QUALIFIED_ROUTING_CAPABILITY_RESULT_AND_NEXT_HYPOTHESIS_20261015.md)。

- [N17局部粗mask、错误估计和选点直接先例](research_round_20260925_global/N17_LOCAL_NATIVE_L_ERROR_SCOUT_TARGETED_PRIOR_CHECK_20261015.md)：SEED/QuBER/MagNet/PointRend重叠与具体差异，现阶段只做信息探针，不确认创新。

- [N16预锁主结果解释器](research_round_20260925_global/code/N16_FIRST_SEED_RESULT_INTERPRETER_UNBOUND_DRAFT_20261014_01.md)：仅E24主端点触发复验/停止；6项合成检查通过，真实AP解释待完整独立资格。

- [N16四CPU证明后的最终完整核验合同](research_round_20260925_global/code/N16_FIXED_E24_ALL100_SCOUT_CPU_NUMERIC_SHARDS_QUALIFIER_CONTRACT_20261014_01.md)：代码/最小合成/静态审阅已准备，尚未实际执行，不替代完整资格。

- [N15下一单一公平选择器草案](research_round_20260925_global/N15_SINGLE_FAIR_UTILITY_SELECTOR_MINIMAL_UNBOUND_DESIGN_20261014.md)：相同容量/输入的修补效用与错误概率，两者共有真实目标/负query及成本规则；ROOT指出均匀all100采样的稀释风险，尚未正式登记或训练。

- [N15完整机会诊断合格结果与下一步](research_round_20260925_global/N15_FIXED_E24_ACTION_OPPORTUNITY_QUALIFIED_RESULT_AND_NEXT_STEP_20261014.md)：实际固定头存在扩域修补机会；GT oracle、学习能力和部署AP分开，3个提前门通过。

- [N15行动效用的五项直接先例查新](research_round_20260925_global/N15_ACTION_UTILITY_TARGETED_PRIMARY_PRIOR_CHECK_20261014.md)：PointRend、MagNet、Mask Transfiner、PromptMoE与SAM-Veteran；已知效用监督与新的工程假设分开，当前N15机会诊断已合格，不能等同AP或选择器效用。
- [N15固定E24行动机会预登记](refine-logs/N15_FIXED_ACTION_OPPORTUNITY_20261014_01/EXPERIMENT_PLAN.md)：同C动作、同K、原强预测核对、实际GPU输入CPU回放及预锁GO门。

- [真实诊断CPU数值失败与单图定位](research_round_20260925_global/N13_REAL_FIXED_PR_CPU_COMPATIBILITY_FAILURE_AND_SINGLE_IMAGE_PROBE_20261014.md)：第9实际失败保留，不放宽原容差，探针区分特征重渲染与算术，原GPU33452单次执行中。

- [全部三种子E24曲线与真实固定PR诊断启动](research_round_20260925_global/N13_FIXED24_THREE_SEED_CURVE_RESULT_AND_REAL_DIAGNOSTIC_START_20261014.md)：全部固定曲线核验完成，当前701 LOG E48的374＋6真实诊断GPU原59604实际0，独立CPU原4404实际1，数值故障定位中，尚无核验后新效用。

- [合格曲线与三种子主结果](refine-logs/N13_SAMEK_PR_20261013_01/findings.md)：三条E48已核验并执行当前联合配方停止决定；三个E08仅作固定学习曲线，全部三条E24已完成独立CPU/ROOT核验，真实固定PR诊断GPU原59604实际0，独立CPU原4404实际1，数值故障定位中。

- [固定DEV执行错误与独立修复版](research_round_20260925_global/N13_FIXED_DEV_SERIALIZATION_REPAIR_STATUS_20261013.md)：原实际失败及未完成产物保留；修复版已通过GPU、独立CPU和ROOT核验，完整三种子主结果及当前曲线见最新状态。

- [固定48轮训练资格与稳定性结果](research_round_20260925_global/N13_FIXED48_TRAINING_QUALIFIED_RESULT_20261013.md)：六头各79,728更新，全部端点核验通过；六头43–48轮平均可编辑BCE较37–42轮均下降超过1%，尚不支持充分收敛；训练资格与实际AP评价分别核验。
- [HRSID官方协议与发布范围核查](research_round_20260925_global/HRSID_OFFICIAL_PROTOCOL_AND_RELEASE_SCOPE_CHECK_20261013.md)：原文支持65/35裁片协议；补核固定作者提交与未回应许可issue，场景独立与数据再发布许可仍未确认；未打开数据包。
- [Poly-BRBLE 原论文全文核查](research_round_20260925_global/N13_POLY_BRBLE_FULLTEXT_CHECK_20261013.md)：直接先例已包含边界buffer、不确定选点与点精修；明确当前sameK差异、联合处理限制和后续强对照。
- [SAR提示式基础模型全文核查](research_round_20260925_global/SAR_PROMPTED_FOUNDATION_MODELS_FULLTEXT_REVIEW_20261013.md)：核对监督范围、指标与可复现性，并明确它与当前残差点头实验的可比边界。
- [学习选择器的监督目标与完整模型边界](research_round_20260925_global/N13_LEARNED_SELECTOR_TARGETS_AND_FINE_FEATURE_PRIOR_CHECK_20261014.md)：Mask Transfiner的GT缩放损失目标与固定作者重实现、RefineMask的可训练P2/SFM及Simple版本；明确后续强对照与创新措辞。
- [固定K二值IoU oracle与概率代理边界](research_round_20260925_global/N13_FIXED_K_BINARY_IOU_ORACLE_AND_PROBABILISTIC_PROXY_LIMIT_20261014.md)：更正真实标签与预测概率的适用范围，给出证明、人工反例及固定权重强对照；不是新SAR实验。
- [固定预算行动选择的两项直接先例查漏](research_round_20260925_global/N13_FIXED_BUDGET_ACTION_UTILITY_DIRECT_PRIOR_GAPS_20261014.md)：MagNet候选置信排序、PromptMoE执行质量监督，以及待登记的错误预测/实际翻转、MagNet式排序和固定权重强对照；当前实验不改。
- [预计算人口与距离分层接口补齐](research_round_20260925_global/N13_FIXED_PR_PRECOMPUTED_POPULATION_AND_DISTANCE_INTERFACE_CHECK_20261014.md)：保留原数组路径、避免重复逐对统计；公开有限复现，科学资格仍未认证。
- [固定PR显式区域装配及有限检查](research_round_20260925_global/N13_FIXED_PR_NATIVE_DOMAIN_ASSEMBLY_SYNTHETIC_CHECK_20261014.md)：固定CPU FP32作者回调接口和标签1域，不实现或替换作者width2；只有合成装配证据。
- [固定PR连接层及新增域/域外独立合成补查](research_round_20260925_global/N13_FIXED_PR_PIPELINE_SYNTHETIC_INTEGRATION_CHECK_20261014.md)：必需原预测参考、只写旧域的完整重建和统计前拒绝；公开便携源码及四路径复现命令，尚无真实SAR效用。
- [固定PR候选的覆盖、纠正能力与选择余量诊断草案](research_round_20260925_global/N13_FIXED_PR_COVERAGE_REPAIRABILITY_DIAGNOSTIC_DRAFT_20261014.md)：四项嵌套oracle、原输出重建、独立候选及exact-K强制伤害门；纯布尔代码人工穷举已通过，真实SAR适配/执行尚未登记。
- [固定48轮三种子完整主结果与下一诊断](research_round_20260925_global/N13_FIXED48_THREE_SEED_PRIMARY_RESULT_AND_NEXT_DIAGNOSTIC_20261014.md)：配对均值为负，按原规则停止当前联合配方；保留所有指标、成本、失败与适用范围，下一行动oracle只是草案。
- [实测区域成本的归因边界](research_round_20260925_global/N13_MEASURED_REGION_COST_ATTRIBUTION_PLAN_20261014.md)：聚合计时含两臂选择、排序校验和摘要；列明遗漏的传输/导出成本及主评价后最小补测，未认定子阶段瓶颈或取得加速。
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

- [N13三个首段端点完整核验及连续48轮进度](research_round_20260925_global/N13_ALL_THREE_E08_ENDPOINTS_AND_CONTINUOUS48_STATUS_20261013.md)：701原报告失败保留；三个固定E48均已核验（各79,728更新/臂），首个固定DEV评价执行失败待修复。完整DEV实际载荷约10.95GiB，尚无新AP。

单图探针元数据原44062实际0（9,975来源、94成功、原8失败及新增1失败、总103身份）；ROOT准入后，原GPU33452已单次启动。比较原容差不变，不生成Q／AP，数值原因尚待实际结束和ROOT读回。

最新执行状态：[N16全FIT缓存已实际完成、待独立CPU核验](research_round_20260925_global/N16_FULLFIT_CACHE_EXECUTION_STATUS_20261014.md)。正常退出0，741图/3035136点/611.937秒；完整独立CPU核验及ROOT最终准入均已实际通过；固定24轮配对scout训练已原实际退出0，独立端点核验与ROOT最终资格已通过，尚无新增AP。

[独立机制与结果解释矩阵](research_round_20260925_global/N16_FIXED_PROTOCOL_MECHANISM_RESULT_INTERPRETATION_20261014.md)：净纠错不等于AP；背景代理监督、dense-flip强对照及完整部署成本的主张边界。训练端点与部署选点的独立CPU核验源码已保存为未绑定草稿，合成通过不等于实际资格。

**N16正式训练准入：** GPU和独立CPU全部741图/3035136点实际通过，combined最大绝对误差1.52587890625e-5、实际修正符号精确；ROOT闭合通过后已启动固定24轮Utility/Error配对训练任务。E24主评价尚未执行。

**固定24轮已完成：** 两臂各17784更新、72843264点暴露，原GPU任务正常退出0；端点独立CPU核验已启动，完整DEV AP尚未执行。训练完成不等于方法涨点或投稿资格。

**固定端点完整资格通过：** E8/E24模型、实际AdamW状态与计数、私有图序、完整训练记录和独立CPU前向/固定损失均通过，原GPU/CPU/ROOT均实际0。当前准备完整DEV187/all100的固定E24主评价；尚无新AP、部署选点或SOTA/TGRS结论。

[固定24轮完整DEV评价协议与状态](research_round_20260925_global/N16_FIXED_E24_FULLDEV_EVALUATION_PROTOCOL_AND_STATUS_20261014.md)

[完整DEV固定分段执行补充](research_round_20260925_global/N16_FIXED_FULLDEV_EXECUTION_PARTITION_SUPPLEMENT_20261014.md)


[正式TEST与跨集证据准备度](research_round_20260925_global/N16_FORMAL_TEST_AND_CROSS_DATASET_EVIDENCE_READINESS_20261014.md)：定位既有测试暴露与HRSID旧TEST选型历史，区分固定配方外部评价与独立确认性证据；实际包、真mask、ID/场景与许可覆盖仍须执行前闭合。本次仅普通历史/元数据审阅，未新增科学验证。


[scout条件重复的初始化编号登记](research_round_20260925_global/N16_SCOUT_INITIALIZATION_REPEAT_PROSPECTIVE_LOCK_20261014.md)：在ROOT查看本轮完整AP前，固定后续202610140162/163；只改变配对scout初始化，原停止门不变，没有开始新增训练，结论限固定载体下初始化稳健性。

[后续最少必要实验与验证边界](research_round_20260925_global/N17_MINIMAL_SCIENTIFIC_EVIDENCE_NEXT_ACTIONS_20261015.md)：先完成主AP，有收益按锁复验，无收益先分析已有全量错误；只在新改动、失败或具体疑点出现时重复工程检查。

[统一COCO后的独立后置ROOT入口](research_round_20260925_global/code/N17_FIXEDDEV_FULL187_POST_ACTUAL_ROOT_CPU_UNBOUND_DRAFT_20261015_01.py)：已冻结/接口fixture通过，独立静态审阅READY、真实执行未开始；不重复计算COCO。


[保留第21次失败的统一COCO03](research_round_20260925_global/code/N17_FIXEDDEV_FULL187_CPU_COCO_FINAL_ROOT_UNBOUND_DRAFT_20261015_03.py) · [独立后置ROOT02](research_round_20260925_global/code/N17_FIXEDDEV_FULL187_POST_ACTUAL_ROOT_CPU_UNBOUND_DRAFT_20261015_02.py)：均已冻结/接口检查通过，待五段独立CPU实际证据和原终态后执行，不是新AP。

[独立CPU重复计算的静态观察](research_round_20260925_global/N17_CPU_REDUNDANT_COMPUTATION_STATIC_OBSERVATION_20261015.md)：两个只读缓存候选，未测耗时、未实现，不改变当前核验。


N20后续未执行源码：[独立CPU端点核验入口](research_round_20260925_global/code/N20_PAIRED_TRAIN_CPU_ENTRY_UNBOUND_DRAFT_20261015_01.py)、[真实邻域全候选评分](research_round_20260925_global/code/N20_ALL100_LOCAL_CENTER_FEATURE_ERROR_SELECTOR_EVAL_BODY_UNBOUND_DRAFT_20261015_01.py)、[分段DEV入口前门](research_round_20260925_global/code/N20_FIXEDDEV_FEATURE_ERROR_GPU_FRONT_UNBOUND_DRAFT_20261015_01.py)。CPU入口只通过语法检查；评分body通过语法和一次独立静态核对，尚未数值执行；DEV前门尚需接真实legacy/prior carrier hook，不可直接启动。已有common-C精修仅作固定参考，新选中点仍fresh计算并保留原数值/符号门，不把缓存读取当部署计时。

原正常GPU0路径下的CPU141/23和DEV142前门保留为未执行历史草稿；收尾失败24后当前改用CPU恢复140/24/164，原GPUactual0保持False。后续DEV需消费新恢复资格，不能直接套用旧前门或补写缺失GPU报告。

已另存[消费恢复资格的DEV前门v02](research_round_20260925_global/code/N20_FIXEDDEV_FEATURE_ERROR_GPU_FRONT_UNBOUND_DRAFT_20261015_02.py)：GPU失败只保留为失败记录，四个模型从原快照取，后续首DEV141/24；仍需真实legacy/priorcarrier hooks和CPU恢复实际资格，未绑定、未执行。
