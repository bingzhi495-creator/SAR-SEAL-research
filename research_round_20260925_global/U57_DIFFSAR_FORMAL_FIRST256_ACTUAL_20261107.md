# 外部DiffSAR首256次正式更新：已真实完成，尚无AP

2026-11-07。新v2前向协议下，公开代码构造初值的完整DiffSAR R101/SCJE参考从新状态0完成256次原生优化器更新，固定终点37050次/50轮。原tool、SSH、worker均为严格整数0；Root结束后核验781项选定来源/输入、实际计数、同boot出生组消失、资源和CUDA回收。它是正式完整参考轨迹的第一段，不是已经充分训练的作者论文复现。

256个不同逻辑FIT样本按固定seed原生shuffle顺序读取，增强后实例访问共515；7个原mapper裁剪后空实例样本保持原位，没有为使梯度非零而挑换图像。全部24项原loss在这256步有限，完整末状态3186个tensor、551655708个元素均有限。检查点2207803747字节，保留模型/缓冲、每参数grad或None、原生optimizer/scheduler、Python/NumPy/TorchCPU/CUDA RNG、MapDataset RNG/fallback和sampler seed/cursor；原dtype缓冲保留，原子边界分组桶为空。权重和原始执行载荷不发布。

模型、原损失、原Trainer优化器与step内裁剪不变。batch1、worker0、seed/sampler40244023，固定FIT741/1661、730不同图像字节；学习率3.5e-5、wd1e-4、milestones29640/35198、warmup20/factor1和37050完整计划在运行前锁定。前健康32、C/S恢复验证12及R重复4共48次研究更新均另列，不拼入正式256；本段没有读取这些旧模型状态。

监督父进程约193.734秒、组RSS峰值5383974912字节、最低可用RAM60100493312字节，违规为空；CUDA allocator峰值6687235584字节，末allocated/reserved0/0。时间包含构造、输入验证、增强、训练、逐步摘要及完整状态落盘，不是部署FPS。按既有实际完整状态大小保留145份检查点加原子临时文件，启动时经验需322339655414字节、实测可用2269445562368字节；估计不是未来大小上界，后续每段按实际状态重估，不删检查点。

原C/S数值阈值检验仍失败，S/R同起点重复也有非逐位波动；本段不将它们改成成功。新v2以完整state加载和透明披露原生数值变异开展固定参考训练，不承诺与单进程未中断轨迹逐位相同。完整训练种子及最终AP稳定性仍需实际验证。[数值观察与范围](U57_DIFFSAR_STATE_AND_NUMERICAL_VARIATION_ACTUAL_20261107.md)、[新前向协议](U57_DIFFSAR_FORMAL50_STATE_CONTINUITY_FORWARD_PROTOCOL_20261107.md)。

同37050次图像曝光没有匹配当前精修系统的COCO初始化教师、项目教师FIT24轮17784次图像训练、缓存、输入及计算成本；作者论文正式预训练来源仍未知。不能把这个构造初值参考的低分自动解释为击败作者强方法，更不能把系统差值唯一归因新结构。[初始化与预算边界](U57_DIFFSAR_FULL_BASELINE_BUDGET_INITIALIZATION_DESIGN_20261107.md)。

尚未计算DEV/TEST/AP、选择best checkpoint或因短预算分数停止。正式轨迹目前256/37050（0.69%）；主PLAIN/FULL已各27392/37050（73.93%），本段关闭后已接回主27392→27648。训练进度不是论文完成度，FULL端点、完整强对照、容量/成本、完整训练重复和跨集仍缺，无TGRS就绪结论。

[本段安全实测摘要](results/U57_DIFFSAR_FORMAL256_ACTUAL_20261107.json)、[首段前冻结的科学源码快照](code/DIFFSAR_NATIVE_FULL_TRAIN_DRAFT_20261107_01/README.md)。源码README“未执行”保留形成时状态，当前实际以本页为准。
