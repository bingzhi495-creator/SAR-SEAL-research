# N19：固定方向代价读出的一次公平探针

日期2026-10-15。协议在UTC03:22:17、任何N19科学执行前锁定；当前未执行。它完成N13已经提出而尚未执行的固定1/2对照，是DEV再利用的探索性机制检验，不是新数学或确认性留出实验。

## 单一问题与反例

N18显示LOCAL比Ω减少有益及有害翻转，条件漏选仍高；这些像素数不能解释AP因果。检验同一已训练Utility概率下，按native符号区分FN填补与FP删除的固定方向代价，能否提升完整自然query AP。反例是：收益仅来自一般偏置或把更多点纳入，而非方向语义。原N16/N17失败及STOP保留；不追加旧seed、不扫r或阈值。

## 四种固定读出、同一个概率头

FP64 stable softmax生成help/harm/noop概率。SYMMETRIC=ph−pm，仅用于原结果精确重建；主DIR_HALF：P0用ph−0.5pm、P1用0.5ph−pm。GLOBAL_HALF无方向、始终ph−0.5pm；REVERSED_HALF将主方法两方向规则互换。后两者是事前反证对照，不能挑其最好者替换主方法。没有质量估计头或额外监督，不声称质量回归的效果。

固定E24 Utility初始化202610140161、原24轮/17784更新、原teacher/PR。候选C、K、query/分类分数/顺序、strict score>0与ties不变。共享原完整C的实际FP32 GPU logits及动作，CPU只排序、写回和导出，不重新训练/前向。K0直接native，不伪造缺失的logits。

## 执行和最低证据

| 块 | 证据 | 状态 |
|---|---|---|
| B0 | 一组手算score/方向/tie/K0/空输入，源码静态审查 | body通过，outer准备中 |
| B1 | 完整已qualified N16捕获与原root/真实退出链；原sym选点及mask逐位重建 | 待绑定执行 |
| B2 | 三个新读出各全18700预测、全187/380原GT官方COCO stats12 | 未执行 |
| B3 | 预锁主Δ与全部控制、成本/范围 | 待结果 |

主Δ=DIR_HALF_AP−max(SYMMETRIC_AP,ERROR_E24_atmost_AP,强R_AP,GLOBAL_HALF_AP,REVERSED_HALF_AP)。全部百分制，R=68.09938494745545；没有局部AP平均。若非正，停止此固定方向读出，不靠质量头、r、epoch或seed救火；若正，仍仅探索支持，先另外锁新的同预算重复训练协议才允许后续科学执行。预留新seed候选191/192尚未准入，旧162/163和172/173不准续训。

## 成本、资格和限制

一次CPU137叶任务，原normal137/failed21/pins158；CPU0/1、线程2、CUDAempty、RAM可用至少16GiB、RSS最多16GiB、全GPU≤8192MiB、1800秒不自动重试。完成先核实际退出、纯RAW链再读结果。GPU模型前向小时0；主要成本为捕获读取/校验与COCO。不能把缓存重放时间当部署总时延。

新选点写回旧common-C动作是离线缓存动作模拟。即使AP正收益，新subset batch的实际GPU PR与独立CPU数值/完整导出资格仍须验证；本轮不宣称已验证真实部署数值或速度。三类概率校准、期望IoU最优、AP最优、域外适用、正式TEST、跨集和创新成立均未证。

质量头暂不实施：现有FIT max-native-IoU-to-any-GT不一定对应原matcher的assigned G，不能直接用于四方向推导中的同一GT质量；如后续确有必要，另建mapped-GT全图标签及公平监督对照。

[定向查新](../../research_round_20260925_global/N19_IOU_DIRECTIONAL_ACTION_RISK_TARGETED_PRIOR_20261015.md)、[N18结果](../../research_round_20260925_global/N18_QUALIFIED_ROUTING_CAPABILITY_RESULT_AND_NEXT_HYPOTHESIS_20261015.md)。
