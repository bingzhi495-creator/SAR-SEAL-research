# N25 固定掩膜评分诊断：真实结果与下一步

日期：2026-10-17。状态：原CPU任务真实退出0并已Root接收。

**GT辅助乘积评分在强R上显示+5.297021 AP点的排序机会；这是离线oracle诊断，不是新模型涨点，不是数学AP上界。** 本轮没有训练或更改任何掩膜。下一步进入固定R的可学习质量头基线，检验模型能否从推理可得信息预测质量。

## 完整主结果（AP百分点）

| 固定掩膜系统 | 原S0 | 主oracle：S0×GT质量 | 次控制：仅GT质量 | 主oracle−原S0 |
| --- | ---: | ---: | ---: | ---: |
| native_official | 67.071057 | 72.436298 | 3.216918 | +5.365241 |
| R | 68.099385 | 73.396406 | 3.380706 | +5.297021 |
| A_COMMON | 67.788929 | 73.014746 | 3.241553 | +5.225817 |
| A_ACTIVE | 67.603337 | 73.053722 | 3.247492 | +5.450384 |
| B_C | 67.807324 | 73.079966 | 3.292885 | +5.272642 |

R的主比较为68.09938494745545→73.39640612271616。五组和两oracle全部保留，没有选最高组或最高oracle替换预锁主结果。仅GT质量在R上为3.38070612906212，说明移除完整S0的信息会在此评分规则下严重损失AP；不能仅凭这一结果将原因唯一归于类别置信、重复候选或某一种错误。

R在IoU0.90处AP由9.937651%升至24.837940%，但达标GT候选仍为原112/380；全部top100几何完全不变。AP变化来自固定mask的评分替换与相应排序/COCO关联过程，不是几何变好，也不是可部署推断。

## 固定条件与完整性

内部DEV187图/380GT/每图100自然query，五组各18700预测。所有原mask RLE、类别、query身份、原始ordinal与S0冻结；同图同类全部GT计算q=max mask IoU，无同类GT或空mask取0。S0×q用FP64乘积，无epsilon、筛选、NMS、去重或额外归一化；并列分数沿官方稳定排序保留原ordinal。

三评分×五组均实际执行完整官方COCO12和十个IoU阈值AP；五组原S0的全部12项指标精确复现N23/N24。两个oracle均重新计算官方IoU，按真实query和GT ID对齐后，全部top100矩阵、每GT最佳IoU及阈值覆盖逐位相同。top10组成允许因排序而变化。

Root另从已合格N24的官方矩阵独立核对五组共93500个query的质量最大值、全部argmax并列GT、FP64乘积与原身份，并核评分后top10稳定顺序和指标聚合。Root未另行重算COCO或解码所有实际mask，不能把同一官方实现的多次结果称为独立评价器验证。

原CPU任务21.203秒；CPU0/1、线程2、空CUDA，无Torch、模型、权重或F/L缓存读取。最低可用RAM33539567616字节；整卡观测1564MiB是环境基线。当前235个正常终态、28个保留科学失败，共263，legacy pins243。不存在新模型训练或自动重试。

Root接收器在科学检查通过并写出接收产物后，因状态保存变量重名导致最后一次比较交换失败。恢复只重新核对原产物并完成状态保存；原接收产物、失败暂存与记录均保留，未重算COCO或重跑科学任务。它作为运维接收失败另记，不冒充新的模型失败或新科学运行。

## 下一步：已知质量方法的可学习性基线

固定强R，训练query–mask条件IoU回归基线：输入冻结query embedding及预测mask，监督来自FIT原真实matcher的预测mask—匹配GT IoU，保留低质量/空mask匹配，不将未匹配query伪标质量0。主评分预锁为S0×预测质量；qhat=1须精确返回原R。DEV GT不能进入学习特征、训练或轮次选择。

已有FIT缓存包含原native L、query embedding与匹配信息，但只保存抽样F/PR动作，不能把抽样动作拼成完整R mask。训练前需要准确获取匹配query的完整R读出及其GT独立特征；数据捕获与训练尚未开始。

Mask Scoring R-CNN已有可学习mask-IoU评分；原模型S0本身也含平均mask概率因子。因此该基线是已有方法在冻结query载体上的适配，缺少作者原ROI图像特征并且不联合训练，不能称完整作者复现或核心创新。[CVPR2019原文](https://openaccess.thecvf.com/content_CVPR_2019/html/Huang_Mask_Scoring_R-CNN_CVPR_2019_paper.html)、[有限直接先例及源码核查](N25_MASK_QUALITY_SCORING_DIRECT_PRIOR_CHECK_20261017.md)。

oracle机会不保证目标可学习，也不保证回收全部5.297点。正收益后仍需重复种子、公平强普通替代、成本与跨集证据；负收益仅限制所用输入与训练配方。N20/N23 STOP、自定义UNet++尚未完成专门实例分割验证、无核心创新/TGRS就绪结论均保持。

[完整15组指标](results/N25_FIXEDMASK_SCORE_ORACLE_AGGREGATE_20261017.json)、[执行前计划](../refine-logs/N25_FIXEDMASK_GT_QUALITY_SCORE_20261017_01/EXPERIMENT_PLAN.md)、[执行源码及合成检查范围](code/N25_FIXEDMASK_GT_SCORE_SOURCE_ONLY_20261017_01/README.md)、[结果到主张](N25_RESULT_TO_CLAIM_20261017/findings.md)、[下一项N26计划](../refine-logs/N26_QUERY_MASK_QUALITY_BASELINE_20261017_01/EXPERIMENT_PLAN.md)。
