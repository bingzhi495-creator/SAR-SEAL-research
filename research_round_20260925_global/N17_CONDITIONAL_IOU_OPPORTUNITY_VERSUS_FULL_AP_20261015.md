# 条件IoU机会不能保证全量AP：N17结果后的最小诊断

2026-10-15。状态：独立方法判断与待执行诊断定义；未读取私有科学载荷、未运行新增模型/统计、未改变N17主端点或停止规则。

N15在固定374个query–GT配对上的理想动作选择，相对原支持得到平均+1.461603 IoU点。这证明该条件群体存在固定PR可利用的行动机会，不能据此预期全187图COCO mask AP至少提高1点。AP还取决于多IoU阈值、固定分数排序、重复预测及匹配变化；逐query IoU oracle不是AP上界。不能将该机会门写成已完成的AP效用证据。依据：[N15合格诊断与适用范围](N15_FIXED_E24_ACTION_OPPORTUNITY_QUALIFIED_RESULT_AND_NEXT_STEP_20261014.md)、[N17预锁实验](N17_LOCAL_NATIVE_LOGIT_ERROR_SELECTOR_EXPERIMENT_PLAN_20261015.md)。

若N17全量AP主收益非正，仍按预锁停止当前配方的重复扩展。之后先做一次固定配对的路由遗漏与动作能力分解，不以诊断替换主评价，不根据诊断重选最佳端点。若收益为正，重复种子仍按原门推进，本文不改变规则。

对原374个固定配对，设P为动作前二值mask，G为该配对GT，C为冻结候选，K为原预算，A为同一固定PR在C中每点的实际二值输出，S为当前策略实际选点（S包含于C且大小不超过K）。定义错误集合E={v∈C:P(v)≠G(v)}；可修集合B={v∈E:A(v)=G(v)}；有害集合H={v∈C:P(v)=G(v),A(v)≠G(v)}。

下列三项仅在K>0且全C实际动作完整的原固定配对子集上聚合，同时报告其占原374个配对的数量；零预算及缺失动作记录分别列出，不纳入PR能力分母。不能借条件筛选掩盖失败或缺失。分别报告：

- 预算归一的路由遗漏率：1−Σ|S∩B|/Σmin(K,|B|)。这是未选到可修点的程度，不是AP损失。
- 固定PR不可修比例：Σ|E\B|/Σ|E|。只描述C内、该配对群体与此PR。
- 选中翻转损害率：Σ|S∩H|/Σ|{v∈S:A(v)≠P(v)}|。仅以实际发生翻转的选中点作分母。

同时保留各实例分布与有效分母；空分母记NA。路由遗漏高与PR不可修比例高可同时发生，不能二选一地归因，更不能把像素统计当AP的因果分解。未匹配query不赋空GT，不将训练代理背景当真实背景。诊断不覆盖全量自然query的匹配与假阳性问题。

是否需要新推理取决于现成证据：Root须在原任务结束并通过原RAW资格前门后，核N15全C动作与N17的原生L、PR端点、候选C、预算K及坐标绑定一致，并确认全C动作实际保存。成立时复用已有动作和N17实际选点，无需新增模型推理；若未保存或绑定不符，仅凭被选中点不能估计全C能力，须标缺证，不能以代理值补齐。现阶段尚未核对这一复用条件，也没有上述统计结果。

直接先例与更广泛局限见[GLDet迁移分析](GLDET_N17_INCREMENTAL_RESEARCH_NOTE_20261015.md)、[最少后续证据](N17_MINIMAL_SCIENTIFIC_EVIDENCE_NEXT_ACTIONS_20261015.md)。本文件仅细化结果后的解释路径，不宣称新创新或TGRS就绪。

## 全C动作输出的静态接口可行性

普通源码接口检查（不是实际载荷核验）确认：K>0时，eval body以full_C_PR角色对完整C输出固定PR；decoder的query(q)返回native_L_grid、C_flat_indices、K、common_PR_chunks，每块含flat_indices、GPU_raw_residual、GPU_combined以及同坐标F/e/native_L。因此可按同一strict-sign规则定义A_C=(GPU_combined>0)，按flat=yW+x回到原栅格。P=(L>0)，Omega=(abs(L)<=2)，C=Omega∪P∪Chebyshev8(P)，K=|Omega|；原升序C和2048块规则保持。

每个epoch/arm/mode的read_profile返回实际选中S、选中PR块、二值mask、原RLE。八策略为E8/E24×LOCAL/CENTER×atmost/exact；S来自actual_selected_flat_indices，不用重新计算scout排名。atmost只保留score严格>0。二值mask由little-endian packbits按H×W截取，禁止以PNG或重采样替代原mask。

K=0是明确例外：common/scout块为空，即使C非空也没有完整A；全部S为空，只能记录零预算，不能推定其全C PR不可修或完全可修。包不含GT，后续仍需原固定query–GT配对和合法GT绑定，不能给未匹配query伪造空mask。

静态接口说明可行后，是否可以免新增推理仍取决于Root在原任务结束、纯RAW前门通过后，对实际全部query/原shared与新capture/模型/坐标/全C成员的核验；没有做这些核验前，不称实际数据已齐全，也不报告上述统计。接口依据：[普通eval body](code/N17_ALL100_LOCAL_CENTER_ERROR_SELECTOR_EVAL_BODY_UNBOUND_DRAFT_20261015_01.py)、[普通decoder](code/N17_NEW_IMAGE_BUNDLE_CPU_CORE_ADAPTER_UNBOUND_DRAFT_20261015_01.py)及其[合同](code/N17_NEW_IMAGE_BUNDLE_CPU_CORE_ADAPTER_UNBOUND_DRAFT_20261015_01.md)。本增补不改当前实验、主AP或停止门。
