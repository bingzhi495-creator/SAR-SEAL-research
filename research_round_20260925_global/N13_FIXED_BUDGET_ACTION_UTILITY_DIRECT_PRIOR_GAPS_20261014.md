# 固定预算修补行动选择：两个直接先例缺口与最小否证检验

2026-10-14。**定向查新及设计分析；未执行新 SAR 实验，不修改当前 N13 三种子、固定 E48 配方或冻结源码。** 本文只读发布副本中的行动收益补充、固定 K 理论更正文档及 README 文献导航，另在线只读原论文和作者公开代码。没有读取本项目科学载荷、绑定源码、SOURCE、资格链、数据或权重，也未执行作者代码。由 ROOT 决定后续登记与同步。

## 结论及核验范围

本轮保留两个此前指定补充文档未列出的具体近例：**MagNet 已使用原预测与候选预测的联合置信度按点预算 top-k，再替换候选概率；所选图块分支可能进一步筛除点。PromptMoE 已采用学习的专家效用与交互评分、有限点提示及稀疏专家激活。** 因而“候选感知支持选择”“预测收益后再计算”“预算选点”和“将不确定性改成学习选择器”均不能单独作为首次机制。MagNet 的最终替换点数不能无条件记为 exact K，概率替换点数也不等于二值翻转数。

两个已读补充未列出这两篇，README 所见导航也没有显式列出；这只说明本轮可见文档中的查漏，**没有检查全部历史综述，不能声称整个项目从未阅读。** README 已有 Mask Transfiner、RefineMask 的专门监督目标核查入口；本轮不重复其分析。已有六篇 DVSNet/SegBlocks/BlockCopy/IteR-MRL/SAM-Veteran/RL-Axial 沿用此前补充的边界，不扩为第三轮泛搜。

证据分级：A 为已读原论文方法或作者实现；B 为官方论文索引、摘要或出版元数据；C 为本文提出的适配和未执行假设。A/B 只证明先例内容，不证明独立复现、在 SAR 上有效或本项目优于先例。第三方笔记只用于发现入口，不作为技术结论证据。

## 差异表

| 直接先例及证据 | 已核到的具体机制 | 与当前未来候选的差异和未证内容 |
|---|---|---|
| **MagNet / Progressive Semantic Segmentation，CVPR 2021，16755–16764 页**。官方索引 B，方法 §3.2 的 primary 搜索正文及作者实现 A。[CVF](https://openaccess.thecvf.com/content/CVPR2021/html/Huynh_Progressive_Semantic_Segmentation_CVPR_2021_paper.html)、[论文](https://openaccess.thecvf.com/content/CVPR2021/papers/Huynh_Progressive_Semantic_Segmentation_CVPR_2021_paper.pdf)、[作者仓库](https://github.com/VinAIResearch/MagNet) | 排序图结合旧预测不确定性与新候选确定性，平滑后按预算 top-k。作者 `demo.py` 先计算所选图块的完整细化候选，再构造分数、选点和 scatter 替换；点预算按图块覆盖缩放，`opt.n_patches > 0` 时 top-k 后还用 `sample_mask` 筛除点。替换的是候选概率，不保证同数二值翻转。 | 多尺度语义分割、图块候选及候选计算后选点；不是固定单实例 PR 的逐点 help/harm 学习，也不是先选支持后只计算 K 个 PR 点。候选区别须是可核验的行动预测与稀疏执行收益，不能仅是新增打分头。其置信度乘积不等于真实错误概率或期望 IoU 改善。 |
| **PromptMoE: A Segmentation Refinement Framework Leveraging Mixture of Experts for Improved Prompting，CVPR 2026，6325–6335 页**。正式 proceedings 状态为 B；原论文第 4–5 页方法及作者公开推理代码为 A。[CVF](https://openaccess.thecvf.com/content/CVPR2026/html/Price_PromptMoE_A_Segmentation_Refinement_Framework_Leveraging_Mixture_of_Experts_for_CVPR_2026_paper.html)、[原文](https://openaccess.thecvf.com/content/CVPR2026/papers/Price_PromptMoE_A_Segmentation_Refinement_Framework_Leveraging_Mixture_of_Experts_for_CVPR_2026_paper.pdf)、[作者仓库](https://github.com/sprice134/PromptMoE) | 以执行后的专家质量监督单专家遗憾和专家组合额外收益，再稀疏激活专家；点提示按分数最大值及空间抑制放置。代码会提前停止正点放置，另外加入负点，之后以提示和 mask/box 进行 SAM 全局重解码。 | 专家级策略与提示点，改变整张输出；不是原生分数固定的 exact K 像素替换。正提示预算、实际翻转数与 PR 点计算预算不等价。高收益稀疏路由已有先例，但不能据此断言与本项目逐点 help/harm 标签等价。论文选对规则与现有代码有具体区别，尚未执行一致性核验。 |

MagNet 作者公开实现的可引用位置：[固定候选、排序、选点、替换](https://github.com/VinAIResearch/MagNet/blob/7f5b8760cb471230402015d80117559de0e38600/demo.py#L160)、[预算缩放及 top-k 后筛除](https://github.com/VinAIResearch/MagNet/blob/7f5b8760cb471230402015d80117559de0e38600/demo.py#L201)、[certainty 与固定数量 top-k](https://github.com/VinAIResearch/MagNet/blob/7f5b8760cb471230402015d80117559de0e38600/magnet/utils/geometry.py#L90)。本轮核到的仓库分支头为 `7f5b8760cb471230402015d80117559de0e38600`，只读在线文本，没有复制或运行代码。

PromptMoE §3.2.3 给出单专家质量 `q_e=(IoU_e+BIoU_e)/2`、遗憾 `r_e=max_j(q_j)−q_e`，第一阶段拟合遗憾；第二阶段冻结单专家头，以 `q_ij−max(q_i,q_j)` 监督组合额外收益，最后联合微调。因此它已直接利用执行结果质量，不能概述成普通不确定性选择器。单专家遗憾如何映射到 §3.2.1 所称高分高效用，需要完整训练实现才能进一步确认。[原论文第 5 页、§3.2.3](https://openaccess.thecvf.com/content/CVPR2026/papers/Price_PromptMoE_A_Segmentation_Refinement_Framework_Leveraging_Mixture_of_Experts_for_CVPR_2026_paper.pdf#page=5)

作者公开实现的可引用位置：[utility/interaction 与路由](https://github.com/sprice134/PromptMoE/blob/66d61d66238b372a5e38859864a1590260628ed4/PromptMoE.py#L41)、[最多 k 正点、抑制及额外负点](https://github.com/sprice134/PromptMoE/blob/66d61d66238b372a5e38859864a1590260628ed4/PromptMoE.py#L659)、[重复全局解码及质量分数选 mask](https://github.com/sprice134/PromptMoE/blob/66d61d66238b372a5e38859864a1590260628ed4/PromptMoE.py#L964)。本轮分支头为 `66d61d66238b372a5e38859864a1590260628ed4`。论文 §3.2.2 选最高分无序专家对，再对该对单专家效用 softmax；公开推理路由则用 pair-score 的 logsumexp 边缘值再 top-k，二者不能记为已核实等价。README 仍写 submitted，发表状态采用 CVF 的正式索引。未确认独立 router 训练入口，不能宣称作者训练链已取得或已复现。

## 哪些原理不能作为创新

1. **错误监督改名为行动收益监督。** 已读项目文档给出固定二值候选关系 `u=δ(2e−1)`；标签不产生新的 GT 信息。若输入描述符压缩，预测难度可能不同，但这只是待检验的有限容量建模作用。
2. **预算选点和候选感知置信度。** MagNet 已是直接先例，但不是所有模式均最终 exact K 替换的证明。把旧不确定性换成一个学习得分或把输出叫“效用”，还没有证明技术差异的必要性。
3. **预测效用后稀疏激活。** 既有 DVSNet 等工作与本轮 PromptMoE 已覆盖此原则；候选必须核算具体执行路径，而非只报告选择数或 MAC。
4. **实例质量权重为 IoU 最优原理。** 已读固定 K 更正中，真实标签的五类排序在限定条件下是 oracle，固定内部权重 r=1/2 已足够；预测概率加权即使校准也无期望最终 IoU 最优保证。估计质量头的必要性要靠共享头对照，不能靠该理论命名。

本轮没有确认与当前候选完全同构的 SAR 组合，但定向未命中不支持领域空白、首次组合、核心创新或 TGRS 成熟结论。

## 最小决定性检验建议：固定修补器下检验“收益预测”的额外作用

**当前 N13 完整主评价结束后才考虑另行登记。** 最能否定“行动收益标签带来新机制”的强替代，仍是已有同容量 error-head 加实际候选翻转的分解，而不是另加大型 SAM、MoE 或第三个 selector：

\[
s_{factorized}=\delta_{actual}(2\hat p_{error}-1).
\]

它与候选用相同 PR、输入范围、训练划分、点监督曝光、训练预算、容量和预锁种子；在相同允许域及 native K 上排序，固定原生类别、分数、query 次序、空输出和评估规则。GT 只用于训练与离线诊断，不能决定部署支持。实际候选需要全域 PR，因而这是候选信息更充分的质量强替代；不将其与先选支持方案误写成部署信息和计算完全相同。

这次查漏增加的最低成本检验是：**在同一次固定候选评价中，额外保存一个无需再训练的 MagNet 式排序结果**。在二值概率表示下可用

\[
s_{MagNet}=(1-|2p_{native}-1|)|2p_{PR}-1|.
\]

这是 MagNet 排序原则的二类结构适配：先限定各臂共同允许域 Ω，再以该分数降序、固定 flat 下标升序处理平局，作 stable top-K；在 |Ω|≥K 时严格选 K 个不同位置，K=0 和空 query 沿用共同原生规则，不在 top-k 后另行筛点。使用同一完整 PR 候选，只在这些位置替换，其余像素继承原生；K 是支持位置数，仍不等于实际二值翻转数。该 exact K 约束来自本项目统一协议，不是作者原版所有模式的事实，也不是计算匹配的完整作者复现。若省略原文的中值平滑，要明确标作“无平滑结构适配”；如保留平滑，则核函数及边界规则须在新实验前锁定，不能通过 DEV 搜索最优核。该后处理不新增 selector 训练，也不能替代 error-head 容量及监督控制。

把它加入同一次评价，可排查“只是同时利用原生不确定性和候选确定性”的简单解释；主否证对象依然是 error＋actual flip。质量权重另沿用同一概率头、固定 r=1/2 的既有控制，不再训练第三个头。三者的参数、全域候选计算、排序/平滑和真实稀疏执行成本分别记账。

预登记需同时给出 AP/高阈值几何质量的效用门槛和质量等效区间，报告全部配对种子、净修复、FP/FN 修复与破坏、不变点及实际翻转数。若 matched error＋actual flip 在质量上达到预锁等效标准，则在本数据和配方下未支持“收益标签具有独立质量优势”；若简单 MagNet 适配足以达到同范围效用，则候选感知打分仍不足以成为核心创新。这些替换不是唯一因果分解。只有先选支持方案在锁定质量等效区间内降低包括 selector、排序、采点、PR、传输/导出的实际全流程成本，才支持有限范围的质量—成本贡献。单纯 ΔIoU 相关性或 K 较少不能代替该证据。

这是一个对现有未来设计的最小查漏，**没有改变现行 N13 训练、评价、停止规则或执行任何新候选。** 主效用、重复稳定性、机制、正式 TEST 与跨数据集仍须实际取得证据。

## 本轮来源与访问边界

在线检索集中于 mask refinement 的固定点数、错误预测与候选收益，不扩展一般主动标注、视频时序、3D 点云或同名异常检测 PromptMoE。MagNet 的方法由 primary 搜索正文和作者实现共同支持；PromptMoE 的出版信息由官方 CVF 支持，方法与训练标签由官方原链接 PDF 第 3–5 页支持，推理细节由作者代码支持。官方 PDF 的 web 直接访问受限，另从同一公开原链接在内存中读取方法页，未保存或分发论文全文；这不是完整逐页审阅或训练链复现。本文不附第三方全文或作者权重，不报告作者数值为本项目复现结果。

本文只新建为普通研究文档；没有编辑 README、发布副本、实验状态或任何科学文件，也未提交 Git。形成后交 ROOT 审阅与同步。
