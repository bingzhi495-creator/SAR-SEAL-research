# 完整DiffSAR参考：完整状态恢复与原生数值波动分开报告

2026-11-07。这是运行前公开的第二版研究协议说明，不是训练完成或AP结果。形成时正式外部训练实际更新数为0，尚未启动；原健康32步、C/S验证12步和另行重复4步均作为独立开销，不转入正式轨迹。私有Root运维预锁不随本文发布。

第一版正式启动以旧1e-4数值恢复比较通过为条件。该比较实际退出1，条件未满足，第一版未启动且保留原样。后续同C4的新进程S/R重复也超出同一阈值，同时完整起点、输入轨迹和末记录RNG一致。新证据支持区分“完整训练状态实际加载”与“后续浮点路径逐位一致”，不把第一版改成成功，不更改原阈值，也不唯一归因CUDA原子运算。

## 冻结的正式训练配方

| 项目 | 前瞻规则 |
|---|---|
| 参考性质 | 作者公开代码构造初值下的完整系统适配参考；不是已认证的作者论文强复现或预训练匹配对照 |
| 模型与损失 | 固定作者R101/SCJE、500 proposals、6 heads和原损失；保留native优化器、全模型裁剪及原训练mapper，不修初始化 |
| 初值 | 新正式状态0，MODEL.WEIGHTS为空，不接入健康32步或C/S/R权重 |
| 数据与预算 | 原规范整数FIT741图/1661实例、730份原JPEG字节对象；保留741逻辑样本，batch1、50轮/37050更新 |
| 随机过程 | 全局与sampler seed40244023，进程启动前固定PYTHONHASHSEED，native TrainingSampler shuffle=true，workers=0显式适配 |
| 优化与调度 | 原native AdamW构造及clip；LR3.5e-5、wd1e-4；MAX_ITER37050、衰减点29640/35198、warmup20且factor1.0 |
| 分段 | 每段最多256次更新；保留model/buffers、param.grad、optimizer、scheduler、完整RNG及MapDataset/sampler状态 |
| 选择规则 | 只评价固定37050端点，不做中途DEV/TEST/AP或best checkpoint选择，不因短段无收益停止 |

每段仍须实际原工具/SSH/worker整数0、完整更新及样本轨迹、来源和可信checkpoint身份、续段起点全部状态实值核对、原损失与checkpoint有限性、资源及出生组回收通过。GPU allocator8GiB、组RSS16GiB、可用RAM下限16GiB、启动48GiB、每段期限1800秒；CPU0/1、线程2。原件和合格checkpoint保留，不自动删除；首次启动需按实际checkpoint大小和剩余段数核对空间。资源中断和数值异常均保留真实原因，不自动重试或改写旧失败。

此资格只支持实际状态加载和执行，不承诺与单进程不中断训练得到完全相同的浮点路径。已有C/S及S/R差异必须同时报告；一次短重复不能替代后续完整训练种子及AP稳定性。首段完成后接回优先的主PLAIN/FULL固定长训，切换仅发生在实际结束并核验后。

## 评价与公平性范围

固定端点使用已接触的内部DEV187图/380实例，不能称盲测官方TEST。共同主整数GT与既定本地候选浮点GT敏感性均报告；指标为COCO mask AP、AP50/AP75、AR及适用面积指标，maxDets100，native512测试预处理，预测seed40244024。没有GT提示或事后挑选有利标注、轮次和输出。

当前精修系统依赖官方COCO初始化后经过项目FIT24轮/17784次图像训练的冻结教师，以及缓存和部署时教师推理。教师上游完整训练数据祖谱、总GPU时及缓存成本仍有未知，须分层计账。相同37050次FIT图像曝光不等于相同上游信息、初始化、架构、增强、计算成本或收敛程度。随机初值DiffSAR参考的低AP本身不能证明超过作者论文/SOTA，也不能把系统差值唯一归因新结构。论文实际初始化来源仍未认证。

本协议不产生新AP、正式训练计数、方法成功或TGRS就绪结论。执行状态以之后经核验的实际记录为准。

[实际状态与数值比较](U57_DIFFSAR_STATE_AND_NUMERICAL_VARIATION_ACTUAL_20261107.md) · [初始化与预算范围](U57_DIFFSAR_FULL_BASELINE_BUDGET_INITIALIZATION_DESIGN_20261107.md) · [作者原配方核查](U57_DIFFSAR_NATIVE_RECIPE_SOURCE_REVIEW_20261107.md) · [绑定前完整训练源码快照](code/DIFFSAR_NATIVE_FULL_TRAIN_DRAFT_20261107_01/README.md)
