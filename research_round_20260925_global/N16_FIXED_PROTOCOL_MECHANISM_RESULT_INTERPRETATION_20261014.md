# UNBOUND — N16既定协议内机制与结果解释矩阵

仅审阅已发布普通设计/状态及普通helper/evalbody；没有读运行中CPU73717、科学Source/Q/数据/权重，没有新增结果、改协议或追加实验。以下都是未来结果的条件解释。

**机制边界。** 对共同目标G、native位P和冻结PR动作A，定义δ=1[P≠A]、ε=1[P≠G]，净纠错标签g=+1(help)/−1(harm)/0(noop)=δ(2ε−1)。若δ由实际观察X确定且p_error是真实同目标分布的条件概率，则E[g|X]=δ(2p_error−1)；否则不能把两个条件期望直接相乘。固定PR残差是F/e/L和冻结执行路径的确定函数，Utility虽不显式输入残差，仍有足够输入信息；它学习动作与错误的耦合，dense-flip显式计算该耦合。两者没有新增GT信息，不能把既有期望效用关系称新原语。有限32宽scout的容量、校准、标签复杂度和优化均可能改变近似质量；同16547参数/输入/数据/初态/更新预算并不证明两种目标同样难学。g是净纠错点数，不是IoU/AP目标：例如TP=FP=FN=1时，去FP使IoU从1/3到1/2，加FN则到2/3，两者g均为+1。

**监督风险。** FIT的65279/72439背景query具有native max-mask-IoU≥.5，约90.1%，只提示重复或潜在TP风险；它们不是已确认的eval FP/TP。COCO还受分数、原S0、排序及一对一匹配影响。背景→emptyset是两arm共同的新代理监督，不能称原生mask GT；共享代理偏差可同时伤害两者，赢Error不能消除该风险。N15约97.12%noop只属374匹配对的C，不能外推全100 FIT采样。低CE/高整体accuracy不足证明help召回、harm抑制或概率校准；本轮保留既定重要性加权mean(wCE)和完整自然query评价，不据DEV调先验、阈值或标签。

| 固定E24 atmost端点的可能结果 | 可支持的范围 | 不能主张／仍缺的证据 |
|---|---|---|
| Utility>同训Error，但≤dense-flip | 本配方直接动作标签相对简单error选择有效；质量强控能解释收益 | 不能声称超过已知分解或新原语；若主门过，只保留原协议结论。低成本解释须完整all100独立成本证据 |
| Utility超过所有质量强控且主门过，但外集未验 | 单seed、固定预算、内部DEV上当前选择配方有效 | 不能称跨数据集优越、SOTA或TGRS充分；未来需协议允许的重复、针对性机制与外部可比评价 |
| Utility与dense-flip质量相近且合格独立完整成本更低 | 学习近似固定动作效用的质量/成本取舍 | 不能以固定K、参数少或matched C/K推断部署效率，也不能据此认定数学创新 |
| FIT loss下降，主AP平或降 | 优化了缓存目标，但该配方未转成主指标收益 | 不证明Utility总体失效；背景代理、目标与AP不一致等只能列假设，不能无证据归因；执行原stop |
| 只赢Error仍≤强E24，或仅E8/exactK胜 | 仅对应弱比较／补充预算的观察 | 不能救主端点、挑epoch或更换预算；原主门与stop不变 |
| 两arm都差于强E24，或Utility不胜Error | 当前容量/监督/24轮配方未支持预定收益 | 不能推广到收敛训练或所有效用选择；共享背景代理是否致因仍未证明 |
| 数值/严格符号/排序资格失败 | 对应执行路径尚未合格 | 不自动等于GPU AP或方法失效；不后验放宽门、替换动作或重试救结果 |

**执行结论仍按原门。** E8只报曲线；E24主atmost、score>0、原S0/类别/自然100query/精确ties不改。GO仍为Utility−max(Error_same24,68.09938494745545)≥1 AP且胜配对Error；非正停止、正不足1仅按原锁定额外seed规则。dense-flip/MagNet为质量强控，不后验改写GO。

**成本与未来证据。** 统计全部自然18700query，纳入原模型/特征渲染、C/scout、selected PR或dense全C PR、CPU FP64评分/排序、I/O；FIT动作标签生成与旧PR已见FIT单列。当前evalbody共用全C数值参考，明确不是单独部署计时，不能扣除分项宣称加速。未来若要主张机制或效率，需要对应的完整计时、校准/动作损害与背景风险解释、重复和外部同规则证据；这些是结论条件，不是本轮新增实验授权。

依据：[锁定设计](N16_FAIR_UTILITY_SELECTOR_LOCKED_TRAINING_DESIGN_20261014.md)、[已发布缓存状态](N16_FULLFIT_CACHE_EXECUTION_STATUS_20261014.md)、[普通helper](code/N16_SCOUT_ACTION_LABELS_AND_SELECTION_UNBOUND_20261014_01.py)、[普通evalbody](code/N16_ALL100_SCOUT_SELECTOR_EVAL_BODY_UNBOUND_DRAFT_20261014_01.py)。实际资格与发布由ROOT负责。
