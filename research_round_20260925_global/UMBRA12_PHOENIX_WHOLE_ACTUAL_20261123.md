# Umbra Phoenix 作者整掩膜强对照已封存

在先行锁定的 12 个 Umbra 原生窗口和 1,166 个原检测候选上，完整执行了 Phoenix XL1 作者的 LVIS 非空掩膜分支。原本为空的 448 个候选按协议透传，保留原顺序和置信度。718 个非空提示完成五轮整掩膜推理，共 3,590 次候选轮次。模型返回的 13 个空掩膜原样保留。[强对照预锁](UMBRA12_PHOENIX_WHOLE_PRELOCK_20261123.md) · [Phoenix 作者评估代码](https://github.com/naver-ai/Phoenix/blob/13ee3996ef3d0292fab144316cc8e03b69c0ecfa/phoenix/eval/evaluators.py#L96-L137)

请求、DGX 推理与本机独立读回均实际退出 0。DGX 使用四个双线程 CPU 进程，墙钟推理测得约 185.5 秒。独立程序核对了逐候选位置、1024² 原生几何、RLE 以及全部二值增删计数，用时约 4.6 秒。无新检测器前向和优化步骤。真实判读文件在三个阶段均未读取。[实际执行与摘要核验](results/UMBRA12_PHOENIX_WHOLE_ACTUAL_SAFE_20261123.json) · [复现入口](code/UMBRA12_PHOENIX_WHOLE_20261123/README.md)

独立标注回传后，同一原始候选可同时评价原生、两种固定预算策略以及作者整掩膜方案。全掩膜分支的固定匹配连续 IoU、FP/FN 和独立重匹配覆盖为次要结果。原先按两份判读独立检验的等量端点主判据继续有效。当前栅格读回只证明模型按原生候选执行，精修的实用胜负需待原始轮廓评分。[双参考接收与评分准备](UMBRA12_DUAL_REVIEW_SCORE_READINESS_20261123.md)
