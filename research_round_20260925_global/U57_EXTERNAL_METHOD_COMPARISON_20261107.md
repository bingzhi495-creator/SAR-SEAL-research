# 当前与其他方法的比较（2026-11-07）

[本次用户问答简表](OTHER_METHODS_COMPARISON_USER_BRIEF_20261107.md)：可报告内部原生/R/N26与SAM基础默认版实测；完整SAR外部强方法仍未完成，不能声称已胜过GLDet或DiffSAR。

新增源码准备：[DiffSAR完整端点预测候选](U57_DIFFSAR_ENDPOINT_PREDICTION_SOURCE_STATUS_20261107.md)已形成、仅3个有限metadata/I/O测试，真实模型与AP未执行。[GLDet原始来源补核](U57_GLDET_AUTHOR_REPRODUCTION_PRIMARY_REVIEW_20261107.md)确认论文500轮/800×800等条件；本轮检查未取得可认证作者可训练实现/初值，不能扩大为空仓库即全互联网无代码。

2026-11-07当前状态：主PLAIN/FULL各严格完成31232/37050（84.30%，42轮+110图），下一31232→31488已实际运行。最新段首次严格进程快照UNKNOWN保留；同算法单次独立新观察及原接受器通过，无训练重跑。DiffSAR仍正式512，无完整端点AP。下表方法AP没有因训练计数、控制接口、数据I/O或静态检查新增数值。

尚未建立新FULL结构相对SAR外部完整强方法的性能优势。现有数字包含内部模块结果、一个SAMRefiner基础默认版推理结果和一个PointRend风格适配实验，不能合并成同预算SOTA排名。本轮补齐已有冻结输出的8个COCO诊断案例和候选浮点AP，没有中途主模型选型。

## 同一内部DEV、原整数GT的已完成结果

187张图、380个实例、每图100候选；指标为COCO mask AP@[.50:.95]，下表百分制，差值为百分点。DEV已用于分析和选型，不能充作未接触官方TEST。

| 方法 | AP | AP50 | AP75 | 相对原生AP | 证据范围 |
|---|---:|---:|---:|---:|---|
| 项目原生Mask2Former/R50 E24 | 67.071 | 97.009 | 83.843 | 0 | 冻结教师原生参考 |
| 固定R精修 | 68.099 | 96.999 | 85.213 | +1.028 | 相同候选和原评分S0 |
| 固定R＋N26质量排序 | 69.537 ± 0.107 | 97.464 | 87.127 | +2.466 | 三个下游质量头各24轮/624更新，AP50/AP75列为均值 |
| SAMRefiner基础默认版 | 42.771 | 95.134 | 27.947 | −24.300 | ViT-H FP32、作者5轮默认，同粗候选/S0，无GT提示 |
| 当前PLAIN/FULL固定E50 | 尚无最终AP | — | — | — | 不用训练进度或损失代替端点效果 |

N26的±0.107是三个下游头AP的样本标准差，三个AP为69.423613、69.637142、69.551345；共享冻结教师和R，不是三个完整模型种子。相对固定R均值+1.438点，主要证据是S0×qhat的质量排序效用，并非掩膜几何或新骨架提升。这些方法的额外训练、预训练、模块和计算不同，不能称为同训练成本的结构消融。SAMRefiner此结果只约束基础默认配置，不能代表SAMRefiner++、最优域适配方案或方法普遍失败。

原始普通结果：[原生/R完整评价](results/N28_FIXED24_COMPLETE187_COCO_ACTUAL_20261017.json)、[N26三头种子](results/N26_FIXED_R_QUALITY_THREE_SEED_AGGREGATE_20261017.json)、[SAM统一评价](results/SAMRefiner_FULL187_NATIVE_R_SAM_COCO_ACTUAL_20261025.json)。

## 必须同时报告的标注敏感性反例

对同一批冻结预测改用本地候选浮点GT后：

| 评价GT | 历史原生AP | 本次固定R AP | N26均值AP | N26−R |
|---|---:|---:|---:|---:|
| 固定整数GT（主协议） | 67.071 | 68.099 | 69.537 ± 0.107 | +1.438 |
| 候选浮点GT（事后诊断） | 68.444 | 67.220 | 68.822 ± 0.104 | +1.602 |

新02已实际完成R与三个N26质量头各两GT的8个COCO案例，四个整数AP精确重现（固定1e−8百分点门），候选浮点N26均值68.821633±0.104069，相对同版本R +1.602111，三个头均为正。原生数值来自此前合格记录，本次8例未重跑原生；对其浮点68.443523的均值差为+0.378110。候选浮点GT未认证官方整包或物理真值；主整数协议不改。R几何的相对收益反转仍成立，N26收益来自固定几何上的排序，不能写成新骨架或几何修复。[新实际诊断](N26_FROZEN_DUALGT02_ACTUAL_RESULT_20261107.md)。

## PointRend风格适配：另外一组同短预算实验

| 方法 | seed701 AP | seed702 AP | 均值AP |
|---|---:|---:|---:|
| C条件点头 | 68.083734 | 68.248537 | 68.166136 |
| PR-FL-91适配头 | 67.958636 | 67.896121 | 67.927378 |
| 配对C−PR | +0.125098 | +0.352416 | +0.238757 |

两臂各FIT741/1661、8轮/13288更新，两个头种子、共享教师和图像/实例顺序及S0；初始化张量不同。未完整复现PointRend动态采样/逐级细分，有效容量未严格匹配。局部平均+0.239点不能写作击败完整PointRend，也未达到当轮预设+0.30开发优先线。短预算结果不证明充分训练后总体无效。[原8轮实验报告](../docs/N10_fixed8_C_PR_fullDEV_qualified_result_20261013.md)。

## 真正的外部完整方法对比缺口

| 外部方法 | 现有证据 | 本项目同协议完整AP |
|---|---|---|
| GLDet（TGRS） | 用户原PDF核对作者报告SSDD mask AP 71.3、HRSID 58.6 | 未完成；不能与本地DEV187直接相减 |
| DiffSARShipInst | 固定作者代码启动后，新v2原生公开代码参考已从新0正式训练512/37050；健康/恢复验证另列，数值变异披露 | 尚无完整端点AP；当前初值与本项目教师上游成本未匹配，不能称论文强复现 |
| SegS-YOLO | SAR实例分割直接先例，实现与数据版本需核实 | 未完成 |
| SAMRefiner++ | 含域内IoU适配，与上述基础默认版不同 | 未完成 |
| SegRefiner / RefineMask | 通用CV掩膜精修相关强方法 | 未完成 |
| 完整Mask R-CNN / 完整PointRend | 历史微型检查/适配不等于完整基线 | 未完成 |

GLDet引用的是作者原PDFTable IV/VI，不代表统一数据划分、GT、初始化、训练长度或评价；数字只提供背景尺度。[GLDet原PDF指标核查](U57_GLDET_ORIGINAL_PDF_METRICS_AUDIT_20261025.md)、[直接精修先例核查](U57_HIGH_IOU_REFINEMENT_PRIOR_UPDATE_20261025.md)、[DiffSAR限定观察](U57_DIFFSAR_MASK_GRADIENT_DIAGNOSTICS_20261107.md)。

## 当前执行和结论边界

PLAIN/FULL固定目标每臂50轮/37050更新，现严格接收31232（84.30%，42轮+110图），下一31232→31488运行，尚无中途或端点AP。最新每臂+256更新/627匹配实例访问；自30464各+768/1805。本机2816独立，FULL约53.12M与PLAIN47.20M未容量匹配。[最新严格安全标量](results/U57_FORMAL31232_ACTUAL_SCALARS_20261021.json)。[原UNKNOWN及新独立观察](U57_STRICT_CENSUS_UNKNOWN_AND_FRESH_OBSERVATION_20261107.md)分开保留，不把元数据失败当方法失效。

下一证据顺序：保持锁定E50至端点并按预定规则统一评价原生/R/N26/PLAIN/FULL；在公开统一数据协议和可核初始化下完成DiffSAR等完整对照，报告收敛与预算而非只对齐epoch标签；随后补容量/计算、完整训练重复种子与HRSID跨集。有限预算筛选要给两臂同预算，最终方法判断仍需充分训练依据。需要验证排序收益能否跨GT、跨种子和跨集保持，不能只选择有利的整数GT或开发集结果。

可确认的是特定内部协议下的局部精修/排序收益；尚未证明完整新骨架有效、超越SAR SOTA或达到TGRS投稿证据要求。

本次用户追问“与其他方法对比”后的补充：当前确有同一内部整数DEV人口上的原生/R/N26/SAMRefiner基础默认版实测，但额外训练、预训练与计算不同，不能把这个表称作同成本结构消融。GLDet作者表值和未完成DiffSAR端点不填入该实测排名。PointRend风格只支持该两臂8轮配方下的相对结果。

[直接先例综述](MASK_QUALITY_ANNOTATION_CONVENTION_PRIMARY_REVIEW_20261107_01.md)确认Mask Scoring R-CNN已有分类分数乘预测mask IoU的质量重排原则；N26是已知强参考和效用证据，不以公式宣称核心创新。新两GT实测相对R分别+1.438/+1.602点，只约束同一暴露DEV上的三个冻结下游头，不是整网训练或跨数据集证明。

最新真实诊断状态：[首次原调用](N26_FROZEN_DUALGT_ATTEMPT01_FAILURE_STATUS_20261107.md)因可选bbox检查错误退出1、COCO未开始；该失败保持。独立[02最小修复](code/N26_FROZEN_RANKING_GT_DIAGNOSTIC_DRAFT_20261107_02/README.md)完成真实8案例后原始tool/SSH/parent-child/worker均0，Root限定接收冻结输出诊断，旧失败不升级。源码/输入/库、资源、真实exec与组闭合均后核；[有限独立复核](N26_DUALGT02_LIMITED_RESULT_CLAIM_REVIEW_20261107.md)不冒称完整跨模型审计。

最新研究取舍：[mask-IoU后校准/尺度折衷的直接先例](SCALE_CONDITIONED_MASK_QUALITY_PRIMARY_REVIEW_20261107_01.md)和[官方COCO排序/ignore竞争解释](COCO_SCALE_GROUP_RANKING_ALTERNATIVE_REVIEW_20261107_01.md)已补核。当前分组AP方向不足以证明一个新的尺度条件误差机制，不据此加模块。仍按[原效用与重复政策](U57_PROSPECTIVE_UTILITY_AND_REPEAT_POLICY_20261025.md)完成主固定端点；N26综合参考强制并列、固定评分G不事后改写。[单一可选后续诊断与决策](N26_POST_DUALGT_RESEARCH_DECISION_20261107_01.md)尚未执行。


[主E50端点接口候选](MAIN_E50_ENDPOINT_RESOURCE_CLOSURE_SOURCE_STATUS_20261107.md)和[原参考数据桥](MAIN_E50_REFERENCE_ORDER_ROLE_BRIDGE_STATUS_20261107.md)补齐执行准备；前者未真实执行，后者不计算AP，均不替代外部效果对比。
