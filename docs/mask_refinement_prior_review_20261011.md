# 当前掩膜修正方案的直接先例与对照边界

更新于 2026-10-12。本文记录公共论文和官方源码查新，不包含新的本地 AP、模型运行或投稿结论。当前固定 A/B 长训保持不变；以下对照只在主方案通过既定效用筛选后安排。

## 判断

小型点 MLP、粗掩膜修正、实例条件和原图点监督已有直接先例。当前方案的可检验差异是复用最终 mask feature 与对应实例的 final mask embedding，以 9441 个增量参数计算原图残差 `L + u`，同时输入局部概率差异。它尚不能被称为新的通用范式；是否构成有价值的 SAR 方法，仍取决于公平增益、先例对照、成本和适用范围。

## 已核对的公共机制

| 先例 | 已公开的机制 | 对当前主张的限制 |
| --- | --- | --- |
| [PointRend，CVPR 2020](https://arxiv.org/abs/1912.08193)；[官方 StandardPointHead](https://github.com/facebookresearch/detectron2/blob/main/projects/PointRend/point_rend/point_head.py) | 细粒度像素特征与粗预测输入点 MLP，点 BCE；官方头在隐藏层可重复拼接粗预测，直接输出点 logits | “特征＋粗 mask＋MLP 修正”不是新原语。当前残差、输入与原图配方不同，但差异不能自动推出优越性 |
| [Implicit PointRend，CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Cheng_Pointly-Supervised_Instance_Segmentation_CVPR_2022_paper.html)；上述官方源码 ImplicitPointHead | 按实例生成点预测函数的参数；官方实现支持像素特征、位置编码和实例动态 MLP | “实例条件点预测”已有先例；仅显式输入 final embedding 不能建立首创性。当前是共享小头的条件输入，不等同于该动态权重算子 |
| [Mask Transfiner，CVPR 2022](https://github.com/SysCV/transfiner)；[作者全文 v1](https://arxiv.org/html/2111.13673v1) | 从 GT 掩膜降采样重建损失定义 incoherent 区域；预测区域并构建四叉树，结合多尺度 ROI 特征、粗 mask、位置和 3×3 特征邻域，经 Transformer 直接预测节点标签 | 实例内上下文、粗 mask 与局部邻域的精修已有先例。其像素节点 query 不等于对象 embedding，特征残差不等于当前 L+u；这些区别仍须实验支持价值 |
| [SAMRefiner，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/90800f46f84381b7891e1378ee850013-Abstract-Conference.html)；[论文](https://arxiv.org/html/2502.06756v1) | 从粗 mask 提取抗噪视觉 prompts，利用 SAM 细化；另有候选 IoU 排名适配 | 通用粗 mask 修正和质量排名适配不能作为空白领域。SAM 强预训练及提示协议与当前端到端方案不同 |
| [Phoenix，2026 预印本](https://arxiv.org/html/2607.29059v1)；[作者代码](https://github.com/naver-ai/Phoenix) | 基于 SAM 的对抗 mask 扰动与三向特征对比，训练修正解码器；论文有 Mask2Former 等输入模型的修正评价 | 真实错误模式驱动的修正已有工作。它的 contrastive learning 不等于当前 3×3 posterior contrast 数值输入。作者标注 ECCV 2026，本轮未独立核实正式会议录 |
| [O²Former，JSTARS 2026](https://doi.org/10.1109/JSTARS.2026.3677088)；[2025 预印本 v1](https://arxiv.org/html/2506.11913v1) | v1 在 SAR Mask2Former 中以多尺度特征生成解码前 query，并增加方向感知特征 | SAR query 优化和方向特征已有先例。最终 e0 残差读出与解码前 query 生成不同，不能把泛化的 query 增强作为新贡献；正式版全文与代码未核验 |

这些论文的发表数值不是我们的同协议复现。不同标注、划分、预训练、输入模型及评价规则下的 AP 不可直接拼成 SOTA 排名。

## Mask Transfiner 补查与主张限制（2026-10-12）

这是直接相关先例补漏。实际方法阅读依据为作者 arXiv v1 和当前官方代码，未核验正式版全文。[官方仓库](https://github.com/SysCV/transfiner#results-on-coco-test-dev)注明实现属于重新实现，不能把当前 main 当作论文原始执行轨迹。[区域监督代码](https://github.com/SysCV/transfiner/blob/main/detectron2/modeling/roi_heads/mask_head.py#L87-L98)采用双线性重建残差，与 v1 式(1)的最近邻/异或定义不同。

当前设计相对于该先例的具体差异，是直接复用最终 F 与对应 final mask embedding、概率邻域输入、小型共享点头及原生 logit 上的 L+u；这些暂属于待验证的适配差异，不自动构成 SAR 机制创新。现有原图插值也不能替代该先例的多尺度细特征。

主效用得到支持后，仍优先完成下述同协议 PointRend 式点头控制及必要几何/成本诊断，不新增当前训练分支。该控制能检验相近点头替代解释，不能证明优于完整 Mask Transfiner。若之后保留“局部概率邻域具有必要作用”的主张，再事前锁定有针对性的删除/替换及其否定条件；本次补查不启动该实验。

## O²Former 发表记录与强对照资格补查（2026-10-12）

ROOT 独立读取 [IEEE 向 Crossref 登记的元数据](https://api.crossref.org/works/10.1109/JSTARS.2026.3677088)，确认 publisher 为 IEEE、type 为 journal-article、期刊为 JSTARS、published 年为 2026。登记题名与上述论文一致；正式作者登记为六位。这里只核验发表记录，未取得正式版全文，不能把 v1 表格写成正式版结果。

[v1 第 IV 节](https://arxiv.org/html/2506.11913v1#S4)报告 PSeg-SSDD 928/232、HRSID 3642/1962，近岸/离岸 mask AP 分列；这些是作者报告的划分数量，不是对实际 IDs 的独立核验。实现段写 ImageNet-R50→HRSID→SSDD 初始化、500 轮、batch 8、AdamW、初始 LR 1e-4、300/400 轮降 LR。数据段称官方划分，实现段又写随机划分；Table I 面积分档写 64² 边界。实际 split IDs 与评价实现须由代码澄清。

本次未定位可核验作者仓库、配置、split IDs 或权重，不等于它们不存在。O²Former 是同载体 SAR 机制直接先例和强对照候选，尚非已复现对照。后续若自行实现，应说明同协议实现的范围，计入预训练与预算，不能将其刊载数值与内部 DEV 主表混排。当前首个点头适配控制及 A/B 运行顺序不变。

## 成功后的第一优先直接先例对照

采用同一原生载体、E24 初始化、数据与 A/B 的 24+48 两阶段预算，准备一个明确标注为“PointRend 式点头适配”的控制：细粒度输入固定为相同 F，粗输入固定为 L；遵循 StandardPointHead 的多层 ReLU、粗预测拼接及直接点 logits 语义，使用相同点监督预算。主输出 `C_direct_official` 按与当前 `B_posterior_official` 相同的 official export 规则，对控制自身的直接 logits 重新计算前景 sigmoid 均值，并与自身原生分类概率相乘；全部 query 与空 mask 规则相同。固定原生分数数值的 `C_direct_native_score` 仅列为几何诊断，不得替代主比较。额外参数、训练成本和推理成本单独报告。

该适配改变了作者的载体、细粒度特征来源、采样和推理流程，不能叫完整 PointRend 作者复现。它首先检验当前改动是否优于相近的已知点头机制，不独自证明实例条件作用或 SAR 物理机制。

另一个“第二 F 投影替换 e0、保持 9441 参数”的原型，只能称为参数匹配的 pixel 容量对照。它仍由 L 包含实例信息，且 F 通道与梯度路径不同；保留我们自己的 posterior 输入、SiLU 和残差输出也使它不等于 PointRend。不得用这个控制冒充直接先例复现。现有容量原型尚未完成数值验证或正式训练。

## 需要收缩的主张

- 原图坐标上的插值没有引入新的高分辨率特征，不能写成高分辨率细节恢复支路。
- 当前 A/B 都有相同原图点 BCE，A/B 比较不缺少这个监督对照；收益仍可能涉及额外容量和输出参数化。
- `posterior contrast` 在这里是局部概率差异输入，不是 Phoenix 的特征对比学习，也不是已证明的概率校准。
- 9441 是新增头的参数数，不能把它当整个模型的参数或据此直接声称推理更快。
- AP 提高若没有固定分数的几何证据支持，应报告实际效用并收缩局部几何修复解释。
- 主方案只有通过固定开发筛选、重复、直接先例差异与正式跨数据集验证后，才讨论 TGRS 投稿依据；当前这些科学结论尚未成立。
