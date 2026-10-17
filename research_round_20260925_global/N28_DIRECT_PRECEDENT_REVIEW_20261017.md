# N28：嵌套节点选择性门控的直接先例核查

访问日期：2026-10-17。状态：有界一手文献与作者源码查新；N28 尚未完成24轮训练，首段已受控超时退出、检查点恢复待核验。本报告不读取训练产物，不报告收益或确认创新。

结论：**“只门控特定分支、原特征继续编码”已有明确作者代码先例；嵌套跳接、UNet++ 加 CBAM、实例查询驱动掩膜预测也均已有先例。** N28 尚可检验的具体差异，是在嵌套图中按出边角色分配同一节点的两份表示，而不是首次引入注意力或实例条件化。未找到精确同构先例不等于证明首次性。

## 本轮核查对象

根据已锁设计，冻结教师提供 F、实例查询 e、粗掩膜 M，轻量 UNet++ 输出全图残差并加回原始 logit L。对允许路由的节点 D，仅算一次 T=G(D)：

- SELECTIVE：跨尺度边使用 U(T)，其他同尺度历史引用及继续编码的边使用 D。
- WHOLE_NODE：跨尺度边仍使用 U(T)，其他出边也使用 T。

两臂共同尺度共享 CBAM、模块容量、初始化、数据与训练预算、全掩膜监督；共同 stage0 已门控，最高分辨率解码节点不再门控。这里的 D 是“未经该次门控的节点张量”，不表示其全部上游特征从未经过注意力。

这个提炼继承用户自设计 model_new1 的 up_x 副本路由思想；它是修改输入、宽度及实例读出的适配版。用户原文件确有结构修改，与其修改是否形成可发表贡献是两个需要分别论证的问题。

## 五项最接近的一手先例

| 先例与证据 | 已覆盖的思想 | 与本轮设计的具体区别、证据边界 |
|---|---|---|
| **UNet++，2018 / TMI 2020**：[原论文](https://arxiv.org/abs/1912.05074)、[作者实现](https://github.com/MrGiovanni/UNetPlusPlus) | 同尺度历史节点的嵌套密集连接、多尺度解码融合；论文还包含 Mask R-CNN++ 实例分割扩展。 | 不能宣称首次将嵌套 U-Net 用于实例分割。N28 增加的是冻结教师下的查询条件残差读出及特定门控路由，嵌套骨架本身归属原先例。 |
| **Attention U-Net / Attention-Gated Networks，2018**：[原论文](https://arxiv.org/abs/1804.03999)、[作者 forward](https://github.com/ozan-oktay/Attention-Gated-Networks/blob/master/models/networks/unet_CT_multi_att_dsv_3D.py)、[作者融合模块](https://github.com/ozan-oktay/Attention-Gated-Networks/blob/master/models/networks/utils.py) | 作者实现先用 conv2/3/4 原张量池化、继续编码，另生成 g_conv2/3/4 给解码。因此原张量与门控副本服务不同分支已有直接代码证据。 | `UnetUp3_CT.forward` 只上采样第二输入：门控的是浅层横向 skip，深层输入另行上采样。N28 门控嵌套深层节点的上采样来源，并保留其同尺度历史引用及继续编码边；两者不完全同构，但分支选择概念重合。 |
| **ICUnet++，2023**：[原论文全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10208197/) | UNet++ 配合 CBAM/AG；表2的 C1–C5 已比较编码器、解码器与 skip 中的注意力位置。 | “UNet++ 加通道/空间注意力”及“位置影响表现”均不能作独立新颖点。论文所述位置实验不能直接替代 N28 的同容量出边对照；本轮未核验作者代码，不能断言其每个中间张量的出边一定不同于 N28。任务为 MR 脊柱语义分割。 |
| **QueryInst，ICCV 2021**：[原论文](https://arxiv.org/abs/2105.01928)、[作者仓库](https://github.com/hustvl/QueryInst)、[DynamicMaskHead](https://github.com/hustvl/QueryInst/blob/main/mmdet/models/roi_heads/mask_heads/dynamic_mask_head.py)、[QueryRoIHead](https://github.com/hustvl/QueryInst/blob/main/mmdet/models/roi_heads/query_roi_head.py) | 实例查询与对应 RoI 特征通过动态实例交互产生掩膜；多阶段掩膜头接受并行监督。实例身份条件化和多阶段掩膜预测已有直接先例。 | N28 广播拼接 e、共享全图 F 与粗 M，输出加回 L 的残差；作者代码是查询驱动的动态交互与 RoI 掩膜 logits，并非该残差接口。该区别是具体适配，不使简单拼 e 自动成为新机制。 |
| **ASGS-UNet，Scientific Reports，2026-04-29**：[出版社原文](https://www.nature.com/articles/s41598-026-46611-w) | 《Frequency-conditioned spatially adaptive decoding for bitemporal remote sensing segmentation》式6保留编码器递推，式7–9另生成受解码上下文与频率条件控制的 skip 门控副本；遥感中也存在分支门控。 | 是双时相遥感语义分割，正文明确不以显式变化检测为目标；不是 SAR 船舶实例分割。它使用频率条件 skip 与自适应退出，不是 UNet++ 多列节点的上采样/历史分工。本轮未核作者实现，不借其指标推断 N28 胜负。 |

代码证据补充：Attention-Gated-Networks 的 `unet_CT_multi_att_dsv_3D.forward` 分开保存 conv 与 g_conv；`UnetUp3_CT.forward(inputs1, inputs2)` 对 inputs2 上采样后再拼 inputs1。QueryInst 的 `DynamicMaskHead.forward(roi_feat, proposal_feat)` 将两者送入实例交互模块，最后输出 `conv_logits`；`QueryRoIHead._mask_forward` 取 RoI 特征并传入该阶段查询特征。以上仅静态网页阅读，未执行外部实现；分支名称相似不能代替张量流向核对。

## 当前能够与不能够主张的内容

1. **嵌套骨架、CBAM 与查询条件化：已知组成。** 不应把组合名称、新输入通道或迁移到 SAR 单独计为三个创新。可研究的整合贡献需要具体问题、直接先例差异和公平证据。
2. **具体角色路由：待检验候选。** N28 用严格两臂检查“跨尺度门控相同，其他边保留 D 或复用 T”的作用。即使正收益，也首先限定为该冻结教师、数据、预算与配方下的策略效用；不能单靠这组对照建立相对全部直接先例的创新性。
3. **信息保护或粗语义污染：尚未确认的机制解释。** 原张量仍可被某些边访问，不等于已证明其含有更正确的信息。改变继续编码边也改变后续表示、梯度路径及 BN 状态；A/B 的差异是完整路由策略效应，不能全部归为某个固定特征被保护。
4. **相对原精修器的改善：存在共同改动的替代解释。** 两臂都获得全掩膜改写、密集适配器及查询输入；相对旧精修器的提升不能独自证明 SELECTIVE。固定评分规则、实际掩膜指标和预锁终点必须保留；FIT loss、初始化探针或完成训练不构成 AP 收益。
5. **零或负结果同样有价值但须限域。** 若固定终点下 SELECTIVE 不稳定优于 WHOLE_NODE，应否定当前配方中已获支持的路由优势，不能宣称所有选择性注意力均失效。若后续获得重复种子及跨数据证据，才可讨论更广泛的整合贡献；本报告不判断已达 TGRS 发表水平。

## 检索与审阅范围

按本地 `novelty-check` skill 提炼嵌套融合、边/分支门控、查询条件掩膜三个主张，使用多组同义关键词及2024–2026、最近约六个月过滤思路，检索原论文、出版社/会议页面和作者仓库。近期检索也覆盖 adaptive skip、instance-conditioned mask refinement 及 ICLR/ICML/NeurIPS 关键词；这不是完整会刊或引用图穷举。主证据限定上表五项，未核实现或跨任务结果均明确标注。

当前工具未提供 skill 指定的 `mcp__codex__codex` 独立模型审稿入口；已另做一次网页源码的独立子审阅，但不冒称完成该跨模型审稿阶段。本报告只新增普通综述文档，不读取或修改运行中的 Source、载荷、权重、资格链及 Ops，不改变已锁实验。
