# N25 固定mask GT质量评分：执行前源码

状态：源码冻结，4项纯合成检查通过，独立静态审阅未发现阻塞问题；真实DEV诊断尚未执行。执行前原计划保持历史原文。

本诊断固定五组原187图/380GT/18700query，每图全部100候选；R为主，不择最高组。原S0、主oracle S0×最大同图同类GT-mask IoU、次控制仅GT IoU分别完整运行官方COCO12及逐IoU AP。两oracle只改变score，GT用于离线诊断，不能视为部署收益或数学AP上界。

按真实query和原ordinal重排官方DT行，按GT ID对齐列，检查稳定并列顺序。两oracle重新评价后全部top100 IoU矩阵、每GT最佳IoU和覆盖计数须与原评分逐位相同；top10组成可以改变。五组原COCO12必须精确复现N23/N24。私有逐query评分/质量/argmax并列GT审计保留本地，不发布原RLE/标注。

3项核心合成检查覆盖全GT最大重叠、两评分函数、FP64乘积、无GT、非有限/越界与行错配；另1项接口检查使用187张手写8×8图、非平凡query顺序、反序GT、空GT图和并列score，验证双轴对齐与重排后几何不变。这不替代真实数据检查或独立COCO复算。

`python -B -m unittest discover -p 'test_*.py' -v`运行4项纯合成检查。真实运行需私有原预测/标注/绑定和终态链，不能承诺一键复现。resource_control.py是已有资源Control源码的发布副本，真实入口只调用Control，不执行其训练入口；CPU0/1、线程2、空CUDA、RSS/可用RAM阈值各16GiB，无Torch/模型/权重/F/L读取。

[预锁计划](../../../refine-logs/N25_FIXEDMASK_GT_QUALITY_SCORE_20261017_01/EXPERIMENT_PLAN.md)、[有限直接先例核查](../../N25_MASK_QUALITY_SCORING_DIRECT_PRIOR_CHECK_20261017.md)、[N24完整结果](../../N24_FULLDEV_ERROR_CENSUS_RESULT_20261017.md)。质量评分已有直接先例，不作为新创新。
