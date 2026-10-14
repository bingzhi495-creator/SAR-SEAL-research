# UNBOUND_DRAFT — N15 pure count action oracle

源码 N15_COUNT_BASED_ACTION_ORACLE_UNBOUND_DRAFT_20261014_01.py，标准库合成测试 test_N15_COUNT_BASED_ACTION_ORACLE_SYNTHETIC_ONLY_20261014_01.py。本任务未读实际SCI/数据/RAW/SPEC/Q/cache/权重，未导入Torch或运行科学载荷；没有科研资格、SAR结果或发布行为。这是旧动作oracle的纯计数独立交叉核验实现，不声称数学创新。

公开接口 oracle_counts(nativeTP, FP, FN, classes, K, mode)。classes必须精确含a,b,c,d,z，均非bool非负整数：a=FP→TN，b=FN→TP，c=TP→FN，d=TN→FP，z=真实unchanged bit。固定逐点after A、无动作交互、候选点去重且五类互斥穷尽是caller前提；真实接入须另核sum(classes)==|pool|，比较Ω/C时同K==|Ω|；helper只检查a<=FP、b<=FN、c<=nativeTP、0<=K<=sum(classes)，无TN/图像大小故不能验证真实物理pool。mode仅exact/atmost。

atmost选择至多K，按b、a，无需选noop/harm；exact必须选K个真实支持点，按b、a、z、d、c。z不够必须消耗实际伤害动作，不能凭空补noop。selected_support是支持点数量；actual_flips=a_sel+b_sel+c_sel+d_sel，两者在真实z下不同。输出selected_counts、candidate_support、legal_counts及其count-only范围、selected_support、selected_noops、helpful_flips、harmful_flips、after_counts、intersection/union/empty_union、iou(float)、iou_fraction{numerator,denominator}等。没有逐点index/mask输出。输入不合法抛ValueError；输入dict不修改。IoU精确整数分式：
(T+b_sel-c_sel)/(T+FP+FN-a_sel+d_sel)。
空并集规定IoU=1，empty_union=true且union=0；iou_fraction返回1/1。这是明示本helper语义，ROOT接入已有oracle前须核对一致性，不能静默更改既有科研协议。浮点iou仅方便展示，比较/阈值可用精确Fraction。

交换证明：将其他已选动作固定，余下交集I、并集U。若a、b同时可用，尚有至少1FP和1FN，因此U-I>=2，选b与选a的分数差=(U-I-1)/(U(U-1))>0。a/b不劣noop；noop不劣harm。若c、d同时可用，I>=1且U>=I；选d与选c之差=(U-I+1)/(U(U+1))>0，故强制harm时d优先c。唯一分母0情形是GT为空且最后FP被a移除，按空并集1单独处理，仍不劣其他选择。交换把任何更优解转为该类优先序，得到最优IoU；tie可多解，atmost并不保证所有最优解中最少翻转数。

Ω是旧abs(native L)<=2物理域，C是去重且包含Ω的候选扩域。固定同一native P/GT G/逐点after A/K，分别调用oracle_counts(...,classes_Ω,K,'atmost')与classes_C；两域各有真实z。保持相同动作在共有点上的分类、验证Ω⊆C后，oracle_C_atmost>=oracle_Ω_atmost；报告两数及其差，不能把Ω当空动作。exactC必须同K且只用C真实z，作为不同预算约束的诊断；不与atmost最优性混称。

GT replacement oracle亦只在同一C、同一atmostK内将afterbit设为G，不是无预算correctall。原PR五类counts不能推出GT替换所需分类，因为z未拆native TP/FP/FN/TN。caller另验证C内native confusion counts，则GT动作类a=C_FP、b=C_FN、c=d=0、z=C_TP+C_TN，再调用同helper。完整PR候选生成与推理费用仍须计入，oracle不免除它。

验证记录：依次hand-check助益优先、真实noop精确预算、强制harm顺序，三次新增测试先预期失败后实现通过。完整8项标准库synthetic tests通过；独立枚举1944小整数实例、36000合法selected-count vectors、19440 exact/atmost比较无反例，含空并集、K0、不合法计数/超pool预算、强制harm、atmost未用预算和大整数精确分式。使用python -I -S隔离站点包；测试仅此普通helper与标准库。expected RED退出是TDD开发记录，不是科研执行失败或RAW祖先。ROOT委派的独立逐像素枚举亦纯stdlib通过：0–4pixels下69,905真实物理cases、414,770 helpercalls，以Fraction逐真实subset比较；另0–3pixels下14,425 Ω⊆C域pairs验证native≤atmostΩ≤atmostC≤GTreplace-atmostC及exactΩ==actualΩ≤exactC≤atmostC，无反例。独立测试文件test_N15_COUNT_ORACLE_INDEPENDENT_PIXEL_EXHAUSTIVE_SYNTHETIC_ONLY_20261014_01.py；其逐点枚举与本稿count枚举不同。

单实例IoU oracle不证明AP/模型效用/收益归因。ROOT未来374匹配pairs的预锁GO门为meanΔ>=.005且fraction(slotgain>=4)>=.90且至少20/374 pairs Δ>=.005；本helper不计算这项总体决策。6 unmatched不进入374mean，不补0IoU。本稿未接入候选数据、未算任何实际Δ、未宣称SAR收益或后续执行合格。
