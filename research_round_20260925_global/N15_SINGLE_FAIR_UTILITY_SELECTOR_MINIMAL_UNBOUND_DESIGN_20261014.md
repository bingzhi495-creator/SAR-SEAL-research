# N15 单一公平 selector：UNBOUND 最小设计

ROOT已闭合机会GO；**仅374匹配域**meanΔ+.0146160337，非AP/可学性证据。C约97.12%noop，always-noop accuracy无效。本稿未登记、未训练，原GPU/CPU配方不变；下轮ROOT统一锁新protocol及宽FIT缓存资格。

1. **唯一学习对照。** 固定701 LOG E24 PR、C=Ω∪P∪dilate8(P)、K=|Ω|。两scout同原域LN(F256/e256,非affine,eps1e-5)+L/8输入，513→32/ReLU→3，均16547参数、同初始state/点集/无权重CE/AdamW/24epochs。utility类help/harm/noop，score=p_help−p_harm；error类FP/FN/correct，score=2(p_FP+p_FN)−1。u=δ(2e−1)是同GT经真实PR flip重编码，无新增GT；非IoU/AP最优理论。读出flip条件PR/help召回/损害率，不用overall accuracy，不在DEV重加权/调阈值。

2. **全100监督与有限缓存。** 必须有原正统matcher/ignore规则的全100 target assignment证明；未分配query空GT由该规则产生，不能从旧1661组伪造。每FIT图最多4096点，均匀query→其C合法点、S0不筛、无GT正负/utility重采样；空C/K0跳过。一次private seed采样，两臂同点/次序、每图一paired update、空有效集同跳，不套旧1661步数。最多741×4096点；标签额外PR calls、已见FIT及复用曝光披露。固定8/24曲线，完整all100/18700官方COCO与strongE24比，不DEV挑epoch。

3. **选择锁定。** 主atmostK仅score>0最多K；补充exactK可含真实C零/负分点，不造noop。统一(score降、|L|升、flat升)；K0直接native、空P则C=Ω。不读GT、不改S0/类别/query顺序。另报AP75/AR/FP/FN、支持点与实际翻转，374 IoU不替代AP。

4. **强控与公平计算。** 同error scout加dense实际δ：δ(2p_error−1)；另二类MagNet式(1−|2σ(L)−1|)|2σ(L+u)−1|，无平滑、同tie/预算，标适配非作者复现。两者全C PR成本单列。双方scout在all100真实C并集共享W_F·LN(F)，W_e·LN(e)每query一次；稠密/所选PR同享首层优化，各自权重不混用，训练更新后不复用旧投影。FP32数值/sign先资格，LN不移粗网格。按**全100含负query**统计M=Σ|C|、T=ΣK、U，计构域/F渲染/投影/scout/排序/PR/IO及标签生成时间/峰值；不从374 C/K推效率，不以MAC代实测。

5. **必要ABI。** make_scout(torch,seed)/forward(F,e,L)→FP32[N,3]；4 tensors=(32,513),(32),(3,32),(3)，无buffers。绑定E24源码/spec/RAW/admission/Q、qualified FIT及assignment证明、新label-cache source/spec/RAW/Q、新protocol；记录image/query/assignment/valid-target、C_count/K/S0/原域ref、flat/nativeL、实际GPU sampledF/e0、u/combined/δ、两3class labels、sampler/order/实际steps、8/24 checkpoint。样本raw界1032ΣU_sample+25S+741×102400字节，S≤3,035,136；exact全100GPU_L若无父ref另加400ΣN。逐图流式/双读回，metadata/序列化另计，ROOT按真实meta锁磁盘/RAM/1800s单次准入及训练前quality/cost门。

依据：[假设](N15_ACTION_UTILITY_SUPPORT_SELECTION_NEXT_HYPOTHESIS_20261014.md)、[primary查新](N15_ACTION_UTILITY_TARGETED_PRIMARY_PRIOR_CHECK_20261014.md)、[共享投影公平性](N13_SHARED_PROJECTION_COST_FAIRNESS_REVIEW_20261013.md)。已有行动效用先例，不称新原语。本轮只读普通文件及ROOT许可数字，未读actual SCI/新Source/Q/GT/权重，未运行训练或Git。

## ROOT审阅：不能直接把草案采样规则当最终配方

已有374匹配条件诊断不保证自然all100策略可学。均匀采100 query可能将稀少有益/有害动作进一步稀释；本稿的“无GT正负重采样”只是待验候选，未获正式锁定。先依据FIT的真实target assignment统计matched/unmatched及flip标签支持；若正例严重稀释，考虑按合法正负assignment分层，两臂共享同一采样与重要性/校准规则。GT只参与FIT监督和训练采样，部署C/query/排名不读GT；不能虚构未匹配query的空目标，也不使用DEV挑抽样比例或收益阈值。

24轮是两scout的相同新训练预算，并非与旧PR的24轮更新计数相等；实际image/query/points/steps、初始化成本与标签生成成本均需登记。原强E24引用保持，主效用仍需完整AP、重复种子和后续跨集。概率读出不是IoU/AP最优保证，类不平衡、真实全100成本和简单替代仍可能否定此方案。先冻结最小公平协议和缓存资格，再启动训练；截至本轮该模型未实现/登记/训练。
