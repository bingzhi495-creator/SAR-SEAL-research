# TGRS 文风指南（基于真实接收稿逐页深读提炼）

**样本**: Gao, Zhong, Sun, Hussain, Zhou, "BBox-Free SAR Ship Instance Segmentation Method Based on Gaussian Heatmap", TGRS 2024 接收稿（修订版, 含审稿修改红/蓝标记）, 18 页双栏。与我们同领域（SAR 船舶实例分割）、同出版社（IEEE TGRS）、同数据集（PSeg-SSDD + HRSID）。
**佐证**: `E:\SAR\seal\docs\LIT_LEDGER.md` 中 O2Former 条目（同领域另一篇的协议记录）。
**所有英文摘录均逐字来自样本 PDF，括注为样本刊内页码（p.1–p.18）。**

---

## 1. 章节骨架与各节篇幅

| 节 | 位置 | 篇幅 | 内部结构 |
|---|---|---|---|
| Abstract | p.1 | 1 段, ~11 句, ~240 词 | 见 §2 |
| Index Terms | p.1 | 5 个词条, 按字母序 | `BBox-free, feature decoupling (FD), instance segmentation, ship detection, synthetic aperture radar (SAR)` |
| I. Introduction | p.1–3 | ~2.2 页 (≈4.5 栏), ~14 段 | 含 3 条贡献列表 + 组织段, 见 §3 |
| II. Related Work | p.3–4 | ~1.5 页 | 4 个小节 A–D, 每小节 2–5 段, 每小节末尾一句"gap 句"收束 |
| III. Methodology | p.4–10 | ~5.5 页 | 5 个小节 A–E; 29 个编号公式; Fig.2–6; Algorithm 1; 小节内用 *1) Feature Extractor:* 式斜体 run-in 三级标题 |
| IV. Experiments | p.10–16 | ~6 页 | 6 个小节 A–F: 数据集与设置 / 评价指标 / 对比 / 消融 / 损失权重选择 / 可视化; Table I–VI; Fig.7–13 |
| V. Conclusion | p.16 | 单段, ~180 词 | 见 §7 |
| References | p.16–18 | 76 条 | 大量引用 TGRS/JSTARS/GRSL/Remote Sensing 本社期刊 |
| 作者简介 | p.18 | 每人一段 | IEEE 标准格式 |

要点：**方法节与实验节篇幅几乎 1:1（各约 6 页），合计占正文 3/4**。Introduction 不惜篇幅（2+ 页），Related Work 反而紧凑。

---

## 2. Abstract 句式结构（逐句功能分解, 全部摘自 p.1）

| # | 功能 | 原句 |
|---|---|---|
| 1 | 领域大背景（1 句即止） | "Recently, deep learning methods have been widely adopted for ship detection in synthetic aperture radar (SAR) images." |
| 2 | 困难 1 | "However, many of the existing methods miss adjacent ship instances when detecting densely arranged ship targets in inshore scenes." |
| 3 | 困难 2+3（合并） | "Besides, they suffer from the lack of precision in the instance indication information and the confusion of multiple instances by a single mask head." |
| 4 | 提案 1（机制一句话） | "In this paper, we propose a novel center point prediction algorithm, which detects the center points by finding a long distance variation relationship between two points." |
| 5 | 提案 1 的性质 | "The whole prediction process is anchor-free and does not require additional bounding box (BBox) predictions for non-maximum suppression (NMS)." |
| 6 | 性质→收益（Therefore 句） | "Therefore, our algorithm is BBox-free and NMS-free, solving the problem of low recall rates when conducting NMS for densely arranged targets." |
| 7 | 提案 2（Furthermore + to tackle） | "Furthermore, to tackle the deficiency of position indication information in localization tasks, we introduce a feature fusion module with feature decoupling (FD)." |
| 8 | 提案 2 机制+收益 | "This module uses classification branch to provide guidance information for localization branch, while suppressing the influence of the gradient flow mixing, effectively improving the algorithm's segmentation performance of ship contours." |
| 9 | 提案 3（Finally + 命名） | "Finally, through principal component analysis (PCA) of the Gaussian distribution covariance matrix, we propose a loss function based on the distance between centroids and the difference of angle, called centroid and angle constraint (CAC)." |
| 10 | 提案 3 机制 | "CAC guides the network in learning the criterion that a single dynamic mask head is only valid for a single instance." |
| 11 | 实验一句收束 | "Experiments conducted on PSeg-SSDD and HRSID demonstrate the effectiveness and robustness of our method." |

模板化规律：**1 句背景 → 2 句困难 → 每个组件"提出+机制+收益"各 2–3 句（用 Furthermore/Finally 串联）→ 1 句实验**。注意：
- Abstract 内**零引用编号**；
- **所有缩写在 Abstract 内首次出现即展开**（SAR/BBox/NMS/FD/PCA/CAC 全部如此）；
- 末句不自称 SOTA，只说 effectiveness and robustness。

---

## 3. Introduction 展开模式（p.1–3）

段落流水线（每段一个功能，顺序固定）：

1. **领域背景**（1 段）：
   > "WITH the advantage of all-weather, all-day imaging, synthetic aperture radar (SAR) plays a key role in water traffic surveillance, fisheries monitoring, marine vessel management and intelligence acquisition [1]–[3]." (p.1)
2. **传统方法及其不足**（1 段, CFAR/GLRT）：
   > "Due to the complexity of the detection steps and the reliance on a priori statistical properties of the image, these methods are not robust enough when facing changes in imaging radar or the imaging scene." (p.1)
3. **过渡到深度学习**（真实过渡句）：
   > "In recent years, with the development of deep learning technology, algorithms based on deep convolutional neural network (DCNN) have achieved great success in the field of optical target detection." (p.1)
4. **任务分层定位**（BBox→RBox→InstSeg, 借 Fig.1 说明为何做实例分割, p.1）。
5. **方法分类学**（anchor-based vs anchor-free, p.1–2）。
6–8. **三个具体困难，每个困难独占一段**，段首用递进过渡句衔接（真实摘录）：
   > "The most important step in target detection is the distinction of multiple targets within a single image." (p.2, 困难 1: NMS)
   > "Apart from the challenges of densely arranged ship detection, existing instance segmentation algorithms face the problem of being unable to perceive the relative position relationship between pixels and targets during mask segmentation." (p.2, 困难 2)
   > "In addition to the two challenges mentioned above, Dice Loss [29], the most widely used mask segmentation loss function, cannot provide more refined guidance for network training." (p.2, 困难 3)
9. **本文观察/切割段**（已有工具用得不够，我们用得更全）：
   > "Their application of Gaussian heatmaps is limited to the work of center point proposals ... In contrast, we make a comprehensive utilization of Gaussian features." (p.2)
   段末总起一句: "Our contributions can effectively improve the performance of instance segmentation from the perspective of center point detection, mask segmentation, and training loss function." (p.2)
10–12. **三个 "To address..." 段，与三个困难一一对应**（镜像结构）：
   > "To address the challenge of detecting densely arranged ship targets, our method encodes each ship target as an elliptical Gaussian heatmap and regresses the heatmap by classification branch." (p.2)
   > "In order to address the problem that Dice Loss cannot perform a more refined evaluation of the predicted mask, we conduct principal component analysis (PCA) on the covariance matrix of the Gaussian distribution to obtain the orientation vector." (p.2)
13. **贡献列表引入句 + 编号列表**（每条 2–3 句: 提出了什么+性质+收益）：
   > "The main contributions of our research are summarized as follows." (p.2)
   > 例条目: "The center point detection algorithm based on the elliptical Gaussian distribution is proposed. ... The entire detection process is anchor-free, BBox-free, NMS-free, and exhibits a robust detection performance, particularly in scenes involving densely adjacent ship targets." (p.2–3)
14. **组织段**（固定句式）：
   > "The rest of this paper is divided into four parts. Section II describes the related work. Section III details the methodology of this paper. Section IV describes the experiments conducted on ... Section V concludes this paper." (p.3)

核心纪律：**困难数 = "To address"段数 = 贡献条数 = 消融模块数**（本样本均为 3），全文一根轴贯穿。

Related Work 补充：每小节末尾必有一句"批评+指向本文"的 gap 句，如：
> "None of the above loss functions can evaluate the orientation difference of the mask, which is the focus of the proposed CAC in this paper." (p.4)
> "However, the cumbersome setting of anchor's hyperparameters and time-consuming ROI extraction make it weaker than anchor-free algorithms in generality and real-time performance." (p.3)

---

## 4. 方法节：记号与公式的引入方式（p.4–10）

**节首路线图段**（固定套路）：
> "In this section, we first introduce the overall structure of our method. After this, three important parts of our method are detailed, namely, center point prediction method, FD and CAC. Finally, we detail the loss function of our method." (p.4)

**公式引入三件套**：导语句（以逗号或冒号结尾）→ 编号公式 → "where" 句逐一解释每个符号。真实样例：

- 导语: "Ship targets can be encoded as elliptical Gaussian distributions according to the ship BBox annotations. The empirical formula of Gaussian distribution is" → Eq.(1) → "where Σ is the two-dimensional covariance matrix, X is the coordinate vector (x, y) and u is the center point coordinate (x̄, ȳ)." (p.5)
- 目的驱动的公式修改: "To normalize the heat value of center point to 1, the Gaussian distribution formula is modified to" → Eq.(2) (p.5)
- 多式连发: "Maximum likelihood estimation of the ship mask is performed to obtain the mean and covariance matrix of the elliptical Gaussian distribution. The specific formulas are" → Eq.(3)–(5) → "where Mask_ship is the set of ship mask's coordinates and g_x,y denotes the value of the image at (x, y)." (p.5)
- 先文字步骤后公式化: 先列 1) 2) 3) 步骤, 再 "The above steps can be expressed by the following formula:" → Eq.(6)–(8) → "where G(x, y) is the heat value at (x,y) and CandPoints denotes the set of candidate points." (p.6)
- 算子名直接入式并在 where 句解释: "The specific process can be described as follows," → Eq.(9)–(11) → "where AdaptiveAvgPool2d and Mean denote the adaptive average pooling and channel-wise mean, respectively. F_in ∈ R^(C×H×W) denotes the input feature." (p.7)
- 损失公式: "In the mask segmentation training process, Dice Loss is used as the loss function, and the formula of Dice Loss is as follows:" → Eq.(17) → "where Mask_gt represents the ground truth of the mask and Mask_pred represents the predicted mask. ∩ denotes the intersection of two sets and |•| denotes the size of the set." (p.8)
- 假设式引入记号: "Assuming the predicted centroid position of the mask is (x_pred, y_pred) and the ground truth centroid position is (x_gt, y_gt), the relative distance between them is calculated as follows:" → Eq.(18) → "where H, W represent the height and width of the feature map." (p.9)
- 总损失+超参数值当场给出: "The loss functions of the three tasks are multiplied with the corresponding weights and summed to obtain the overall loss function:" → Eq.(29) → "where α1, α2 and α3 are the weights of each of the three tasks. In this paper, we set these three weights to 1000, 1 and 1 respectively." (p.10)

其他习惯：
- 张量维度用 ∈ R^(C×H×W) 记号在 where 句给出（p.7–8）；
- 超参数取值随公式当场交代并引来源: "we set the compression ratio r to 2 [26], [52] to prevent the loss of useful features during the compression process" (p.8)；
- 伪代码用 `Algorithm 1 Candidate points de-duplication.` 环境（p.6），配 input/output 行 + 行内注释；
- 设计动机句紧跟机制句: "It should be emphasized that the threshold M in the second criterion has a fundamental difference from the IoU threshold in NMS." (p.7)；
- 小节内部收束: "In summary, our approach has the following advantages." + 编号列表 (p.7)。

---

## 5. 实验节表述习惯（p.10–16）

**节首路线图**（Experiments 开门先摆全节结构）：
> "This section presents the detailed results of experiments conducted on PSeg-SSDD and HRSID. First, the datasets and evaluation metrics used in the experiments are introduced. Then, we compare our method with other mainstream segmentation methods to verify the effectiveness and robustness of the proposed method. Subsequently, ablation experiments are conducted to verify the effectiveness of each module. Finally, we conduct the visualization experiments of center point prediction, FD and CAC." (p.10)

**协议一段讲全**（init/优化器/LR 日程/epoch/框架/硬件, 一个不落）：
> "During training, we used parameters pre-trained on ImageNet to initialise DLA-34 and used random initialisation for each subsequent branch of the network. Adam was used as the training optimization algorithm, with the initial learning rate set to 0.0001 and the learning rate reduced by a factor of 10 at epoch 80 and 120, for a total of 140 training epochs. The experimental part of the code in this paper is implemented using the Pytorch framework [69], and the MMdetection [70] framework and the AdelaiDet framework [71] are used for comparison experiments. The experimental platform is configured with Ubuntu 20.04 system, 32G RAM, and 3080ti is used for training and inference." (p.10)

（旁证：O2Former 同样在一处集中给全 500ep/bs8/AdamW/LR1e-4×10@ep300,400 + init 来源——本领域审稿人期待协议可复现到这个粒度，见 LIT_LEDGER O2Former 条目。）

**数据集段给官方划分与场景计数**：
> "According to the official partition method, 928 images are used as the training samples and 232 images as the testing samples. Among the testing set, 186 images belong to the offshore scene and 46 images belong to the inshore scene, with an offshore-to-inshore image ratio of 8:2." (p.10)

**评价指标单列一小节**，连 precision/recall/AP 都给公式 Eq.(30)–(32)，COCO 指标做成 Table I；Flops/Params 也解释含义（p.10）。

**表格分析句式 = "编号结论列表"**。每张表后固定一句引入，再逐条结论：
> "Table II demonstrates the quantitative performance of different methods. An analysis of the experimental results is provided below." (p.11)
> "From this table, we can draw the following conclusions." (p.14)

结论条目三种典型句：
- **罗列增量**: "our method achieves improvements of 7.9%, 4.7%, 7.3%, 14.5%, 7.7%, 18.5%, 8.3%, 4.3%, 4.3% and 5.6% over Mask R-CNN, Cascade Mask R-CNN, Mask Scoring R-CNN, Yolact, CondInst, SparseInst, RTMDet, HQ-ISNet, AFSS-Inst and SRNet on AP, respectively." (p.11)
- **机制归因**: "This is mainly because our proposed Gaussian center point localization method can suppress scattering noise and locate neighboring ship instances from complex scenes based on Gaussian long-range trends." (p.11)
- **诚实认输+解释**: "Only the AP75 on HRSID is second to HQ-ISNet and SRNet, which use the complex network structure and has the highest computational consumption." (p.11)

**消融**：每个模块一张表 + "we can draw the following conclusions" + 编号条目，首条常报开销：
> "FD only brings a computational and parameter cost of 0.01 GFlops and 0.51 MParams, which is a lightweight network structure." (p.14)
> "The adoption of FD bring about an improvement in instance segmentation accuracy of 5.51% for inshore scenes and 0.60% for offshore scenes." (p.14)
消融还跨载体验证鲁棒性: "CAC is effective for both the two-stage algorithm Mask R-CNN and the single-stage algorithm RTMDet, proving the robustness of CAC." (p.14)

**超参选择实验单列小节**（审稿修改中新增, 蓝字）：
> "We vary the weights in steps of multiples of 10 and choose four different sets of weights to compare the performance of our method under different weights." (p.14)

**PR 曲线/可视化的描述句**（指着图逐区间讲）：
> "Fig. 7(a) reveals that when the recall is between 0.3 and 0.8, the precision of Yolact and SparseInst decreases rapidly." (p.11)
> "Fig. 11 shows the center point prediction process. The images from left to right are ground truth, Gaussian heatmap, center point proposal, and center point prediction result, respectively." (p.15)

---

## 6. 图表 caption 真实样例（3+2 条）

1. **多子图极简式** — "Fig. 1. Illustrative results. (a) BBox detection. (b) RBox detection. (c) Instance segmentation." (p.1)
2. **架构图自足式**（caption 内解释所有缩写与部件）— "Fig. 2. Overall architecture of our method. The network structure can be divided into three parts: the feature extractor, the classification branch and the localization branch. Cond Conv means conditional convolution. Map G, M, S are the Gaussian map, mask map and semantic map. The training loss supervises G, M, S to learn the ground truth in the training process. The center point extractor combines the characteristics of Gaussian distribution to predict the center point on Map G." (p.5)
3. **示意图图例式**（逐色解释）— "Fig. 6. The blue mask represents the ground truth of the ship instance, and the red mask represents errors made during the segmentation process. The blue dots represent the centroid of ground truth and the red dots represent centroid of predicted mask. The blue arrow represents the direction of the long axis of the ground truth, the red arrow represents the direction of the long axis of predicted mask." (p.8)
4. **对比结果网格式** — "Fig. 8. Instance segmentation results of different methods on PSeg-SSDD. (a) Ground truth. (b) Our method. (c) Mask R-CNN. (d) Cascade Mask R-CNN. (e) Mask Scoring R-CNN. (f) Yolact. (g) CondInst. (h) SparseInst. (i) RTMDet. (j) HQ-ISNet. (k) AFSS-Inst. (l) SRNet." (p.13)
5. **表标题全大写小体 + 脚注说明加粗/下划线约定** — "TABLE II INSTANCE SEGMENTATION PERFORMANCE OF DIFFERENT METHODS ON PSEG-SSDD AND HRSID"; 表下脚注: "Bold items denote the optimal offshore values in the columns, the underlined items represent the optimal inshore values in the columns." (p.12)

规律：caption 必须自足（不看正文能懂），架构图 caption 可长达 5 句；表内最优值 bold/下划线约定写在表脚注而非正文。

---

## 7. 结论写法（p.16, 单段）

结构（全部过去时）：总起 → 逐组件"困难+动作+效果"各 2 句 → 实验一句 → 消融一句 → 未来工作一句。真实骨架摘录：
> "In this article, we proposed a BBox-free SAR ship instance segmentation method based on Gaussian heatmap. To tackle the issue of BBox-based NMS often omitting ships in densely arranged scenes, we introduced a center point prediction approach grounded in the mathematical properties of Gaussian distribution. This approach significantly enhances the accuracy of center point detection. ... we devised a feature fusion module incorporating FD. FD effectively addresses the spatial location discrepancies in gradient flows during feature fusion. ... we introduced the CAC. CAC addresses the limitation of the Dice Loss, which cannot perceive where the mask error occurs during instance segmentation. Our experiments, conducted on PSeg-SSDD and HRSID, demonstrated that our method proficiently extracts ship center points and precisely segments the ships. Ablation experiments underlined the effectiveness of each component introduced in this paper. In the future, we plan to delve deeper into SAR ship instance segmentation algorithms to achieve higher accuracy while minimizing resource consumption." (p.16)

要点：动词轮换 proposed / introduced / devised 避免重复；**不引入任何新数字、新引用**；未来工作只给一句、方向模糊即可。

---

## 8. 高频学术搭配词表（20 条, 全部为样本中真实出现的原文短语）

| # | 短语 | 出处页 |
|---|---|---|
| 1 | plays a key role in | p.1 |
| 2 | has received extensive attention in recent years | p.1 |
| 3 | poses a significant challenge in the realm of | p.2 |
| 4 | To address the challenge of ... | p.2 |
| 5 | In addition to the two challenges mentioned above | p.2 |
| 6 | The main contributions of our research are summarized as follows. | p.2 |
| 7 | The rest of this paper is divided into four parts. | p.3 |
| 8 | It should be emphasized that | p.7 |
| 9 | In summary, our approach has the following advantages. | p.7 |
| 10 | The specific process can be described as follows | p.7 |
| 11 | To verify the effectiveness and robustness of our method | p.10 |
| 12 | According to the official partition method | p.10 |
| 13 | As shown in Table II / As evidenced by our experimental results | p.11 |
| 14 | achieves improvements of ... over ... on AP, respectively | p.11 |
| 15 | This is mainly because | p.11 |
| 16 | is second only to / is only second to | p.11 |
| 17 | It can be observed from Table II that | p.11 |
| 18 | we can draw the following conclusions | p.14 |
| 19 | proving the robustness of | p.14 |
| 20 | demonstrate the effectiveness and robustness of our method | p.1, p.10 |

高频功能词习惯：`respectively` 高密度使用（几乎每个多对象罗列句必带）；`In order to` 与 `To` 开头的目的状语句是段首标配；`namely,` 用于列举部件名（p.4, p.10）。

---

## 9. 该刊避免的写法（从样本反推的负面清单）

1. **不自称 SOTA**。最强的自评只到 "our method has a competitive performance" (p.11) 与 "which indicates the superiority of our method" (p.11)；程度副词上限是 significantly / effectively，"dramatically improves" 全文仅一次 (p.11)。
2. **不回避劣势**。落后的指标明写并给结构性解释（AP75 认输句, p.11; 大图实验自曝 "there are still some false alarms in complex inshore scene (pink mask in slice 2)", p.13）。
3. **Abstract 零引用、零未展开缩写**；正文缩写首次出现必展开，即使 NMS/BBox 这类常识缩写也不例外。
4. **无第一人称单数、无反问句、无口语缩写**（全文无 don't/can't）；一律 "we/our method"。
5. **无未被正文引用的图表**；Fig.1–13、Table I–VI 全部按序在正文点名（"Fig. 4 illustrates ..."、"Table IV shows ..."）。
6. **公式后不许出现无解释符号**——每个公式必配 where 句，符号逐个交代（见 §4）。
7. **数字不裸奔**：增量必须报到具体百分数并逐方法罗列（§5），且加粗/下划线规则在表脚注声明。
8. **时态纪律**：正文一律一般现在时，结论一律过去时；混用会被 copyedit。
9. **"paper" vs "article"**：样本混用（"in this paper" p.1 / "in this article" p.4, p.16）——IEEE 定稿风格倾向 "article"，我们应统一用 "article" 免得 copyeditor 改。
10. **拼写一致性**：样本残留英式拼写 "initialise" (p.10)、"utilise" (p.2)——IEEE 用美式拼写，投稿前统一为 initialize/utilize（accepted 稿都有此瑕疵，说明审稿不卡但 copyedit 会改；自查省事）。
11. **资助信息放首页脚注**，格式固定: "This work was supported in part by the National Natural Science Foundation of China under Grant 62371022." (p.1)。

**附: 修订标记（红/蓝字）透露的审稿人真实关注点**——样本接收稿中被审稿人逼出来的增补，恰是我们应预先写足的部分：(a) 大场景整图推理实验（ALOS-2 18001×14804 整图切片, p.12–13）；(b) 损失权重选择实验独立成节（p.14）；(c) 对比方法背景差异的机制解释句（p.11 大段蓝字归因分析）；(d) 每个 caption 的解释性扩写（Fig.3/Fig.12 caption 中的红字结论句, p.6, p.15）。
