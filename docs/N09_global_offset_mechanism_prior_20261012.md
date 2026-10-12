# N09 全局 offset 的机制先例与决策边界（2026-10-12）

只读文献核查；未执行训练、拟合或评估，未修改 SCI / RAW / Q / SPEC / 结果 / weights。以下 N07/N09 事实由主任务提供，文献事实与本项目推断分列。执行仍以原预锁为准。

| 直接先例（均链接原论文） | 原文支持的事实 | 对本项目的限定与推断 |
|---|---|---|
| Steyerberg et al., **Validation and updating of predictive logistic regression models: a study on sample size and shrinkage**, *Statistics in Medicine*, 2004，[原文摘要](https://onlinelibrary.wiley.com/doi/10.1002/sim.1844) | 将重新估计原预测线性项的截距、斜率与更全面模型修订区别讨论；截距更新已有长期先例。 | 固定原 logit、只学习加性 δ 是现有截距更新形式；N09 的作用是排查简单替代解释，不能据此申报新方法。此来源未提供本项目的像素概率校准证据。 |
| Kirillov, Wu, He & Girshick, **PointRend: Image Segmentation as Rendering**, CVPR 2020，[§3.1–3.2](https://arxiv.org/html/1912.08193v2#S3) | 共享点 MLP 接收图像细特征与 coarse prediction；后者提供实例/区域上下文。训练包含不确定点与 uniform 点（默认比例 75/25）。 | F / coarse logit / 实例信息的点预测与混合采样都有直接先例。N07 是冻结特征上的窄带残差，不是原版逐级 subdivision；构件相似不等于完整方法相同。 |
| Zhang et al., **RefineMask: Towards High-Quality Instance Segmentation with Fine-Grained Features**, CVPR 2021，[§3.3，Eq. 6](https://arxiv.org/html/2104.08569v1#S3.SS3) | 后续阶段边界区域采用新预测，其他区域继承上一级 mask；边界按到预测轮廓的距离定义。 | “局部更新、外部保持”已有直接先例。native `abs(L)<=2` 是不确定性集合，不能未经验证称为几何窄边界带；两者的差异需要保留。 |
| Cheng, Parkhi & Kirillov, **Pointly-Supervised Instance Segmentation**, CVPR 2022，[§4.1、Appendix E.1](https://arxiv.org/html/2104.06404v2#S4.SS1) | Implicit PointRend 为每个实例产生点头参数，输入相对坐标与图像点特征；论文分别删减坐标、图像特征作对照。 | 实例条件点头已有先例。N07 拼接 e 到共享 MLP，与动态生成整个点头参数不同；其有效性仍须分离 e、F 的作用，不能用总体 AP 代替机制对照。 |
| He et al., **UnfoldCRF: Structured Mask Refinement with Image-Conditioned Latent Regions**, arXiv:2609.33996v1，2026-09-27 **预印本**，[§4.1、Appendix B](https://arxiv.org/html/2609.33996v1#S4.SS1) | 普通 recurrent 网络对照使用相同输入、encoder、训练规则、stage 数及参数预算；结构项另做干预。 | 最新且会影响设计的先例：容量匹配与输入作用是不同问题。其结果与机制不能转移到本项目；作者尚称代码/支持材料将公开，本次未复现。 |

**N09 回答什么。** 锁定形式是 `L'=L+1[abs(L)<=2]*δ`：FIT 的 1661 个 GT 各取 1024 点（768 band + 256 uniform，replacement），仅学一个 δ，float32 dense 应用仍受 native band 限制。带内固定 0 判定等价于把原 logit 阈值移到 `-δ`；δ 超出带宽时受门控截断（本项目形式的数学推断）。因此它检验“单一阈值/截距调整是否足够”。偏向不确定区域、按 GT 等权的样本目标并非天然部署像素分布；BCE 拟合及 AP 收益不能自动证明概率校准。保持原预锁，不搜索 DEV 的 δ、带宽、判定余量或新子组。

**若一个 FIT offset 足以复现 C 的收益。** 按原解释规则收缩为：“在固定数据、冻结 E24、8 epoch 配方及固定 scores 的评估中，C 的总体收益可由 FIT 学得的单参数带内调整达到相近水平；目前没有空间/实例条件机制的必要性证据。”不能声称 C 内部一定学成常数、不能宣称该类方法总体失效，也不能推出已完成概率校准。AP 接近不等于 mask 或残差函数相同。R−C 均值 −0.012924 pt 的既有 patch NO_GO 不因文献先例而改变。

**若 offset 不足。** 唯一新增结论是“这个锁定的单参数替代解释不足”，还可能是普通非线性 logit 映射、额外容量或实例阈值差异。最小机制辨别设计应比较 `u(L)`、`u(F,L)`、`u(e,L)` 与既有 `u(F,e,L)`，形成 F / e 的 2×2 输入对照；前者排查普通非线性像素映射，后两者分别检验实例条件和点对齐 F 的增量作用。保持冻结 checkpoint、GT/点、loss、8 epoch、初始化规则、seeds、native band 与 scores 可比。普通 `u(F,L)` 还须有与 C **有效**参数预算可比的容量对照，并记录计算量；零填充输入虽保留名义参数，相关首层权重不参与有效学习，不能单凭总参数相同宣称容量已匹配。既有 C 本身就是普通共享条件像素 MLP，不能将 C 对 native 的收益命名成独立空间结构创新。现有权重的 e 交换 / F 位置打乱只能补充敏感性诊断，因组合可能偏离训练分布，不能替代上述独立训练对照。以上是证据条件，不启动新训练分支。
