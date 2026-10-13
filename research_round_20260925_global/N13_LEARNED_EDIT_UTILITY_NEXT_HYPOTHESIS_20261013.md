# N13 后续候选：预测修补收益的同配额支持分配

2026-10-13；**未来待检验假设，方案审阅完成，训练未执行。当前两臂、三种子与48epoch预算不改。**

先质疑：预测错误图早有先例，学边界分数 Top-K、质量预测也已存在；仅把 absL 换为“错误概率”不足以建立创新。错误概率高不代表当前修补头能修正，低错误概率也可能被修补头破坏。新增可检验问题是：**同 K 时预测实际修补的正负收益，是否胜过同容量错误概率选择器？**

## 直接先例

| 原始来源与状态 | 已知内容及本假设须区分之处 |
|---|---|
| [Zhang & Chung，Fine-Grain Error Map Prediction，MICCAI2019；预印本2019-07-29](https://arxiv.org/abs/1907.12244)；[作者单位发表记录](https://researchportal.hkust.edu.hk/en/publications/a-fine-grain-error-map-prediction-and-segmentation-quality-assess/) | 已将预测 mask 与 GT 不同的像素作误差图监督；因此 error-map head 不能称首次。其质量判断不等于固定修补器作用后的正负收益。 |
| [Mask Transfiner，CVPR2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Ke_Mask_Transfiner_for_High-Quality_Instance_Segmentation_CVPR_2022_paper.pdf) | 学习检测下/上采样信息损失定义的 incoherent 区域并稀疏精修；“学习哪里修”已有直接先例。 |
| [PointFlow，CVPR2021；预印本2021-03-11](https://arxiv.org/abs/2103.06564)；[作者原文 §3.2](https://zhouchenlin.github.io/Publications/2021-CVPR-PointFlow.pdf) | 遥感预测边界 confidence Top-K 与显著点传播已有实现；不能以学习排序本身区别。 |
| [Mask Scoring R-CNN，CVPR2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Huang_Mask_Scoring_R-CNN_CVPR_2019_paper.pdf) | 预测实例 mask IoU 并调分；这里预测逐像素修补收益，固定原生类别/分数/全部query，禁止把调分收益混入。 |

## 一个具体设计及否证条件

未来另登记 FIT 按图确定性80/20划分 A/B。冻结 E24，原生 `P=1[L>0]`；原 PR 在 A 的每 GT 1024全图均匀、有放回固定点上训练，每实例均值BCE，**去训练门控**，使域外点也有残差数据梯度。B 未参与该 PR 训练；冻结 PR 后，生成候选 `P'=1[L+r>0]`，监督标签 `u=1[P!=GT]-1[P'!=GT]∈{-1,0,+1}`，分别是破坏、不变、修复。

选择器输入逐点 LayerNorm F256（无affine、eps1e-5）与四量 `[clip(L,-8,8)/8,P,log1p(min(D,32))/log33,no_opposite]`；D仍是原域内异类距离，仅描述符截断。共享线性 F256→32（无bias），与逐query四量→32（有bias）的输出逐元素相加，再经ReLU、32→3（有bias）；总8451参数（256×32+4×32+32+32×3+3）。冻结选择器推理时F投影每图一次；训练不能跨每实例优化器更新复用旧投影，未加权三类 CE、每实例点均值。推理按 `p修复-p破坏` 降序、flat升序取原生区域 |L|≤2 的像素数 exact K，仅这些位置写 L+r，K0保持原输出；GT不参与选择。仅当条件概率估计可靠、候选残差与P'固定且不随所选支持集合改变时，排序近似最大化exact K下的期望像素错误减少；无需假设各像素误差统计独立；强制选满K仍可能包含负收益点，**不是 IoU/AP 最优收益**。

三种子各固定48轮、每 GT 一次step，AdamW lr3e-4/wd1e-4固定。N_A/N_B是A/B中实际参与训练的匹配实例数。每seed PR更新48N_A、单个selector更新48N_B；三seed单配置为144(N_A+N_B)，utility/error两配置共用三个PR时总更新144(N_A+2N_B)。两个selector均使用B每GT相同的1024全图均匀有放回固定点，监督点曝光为上述更新数乘1024。按图80/20不保证实例80/20，参数/曝光相同也不等于同计算。所有比较共享同一 A-trained PR。强对照为相同8451参数/输入/预算的三类错误预测（正确、FP、FN，分数pFP+pFN）；另保留 absL、D、随机同K支持。分列标签生成及全图selector/距离/排序成本；共享F投影后的激活与输出仍逐query计算。报告三类占比、改变像素上的判断能力、高置信错分、修复/破坏、全预测AP及峰值显存，避免不变类多数掩盖失败。

未来执行前登记配对AP与净修复判据，报告逐seed配对差及不确定性。若固定端点AP不超过错误预测对照，记为**本预算、划分及配方下未支持相对任务效用**；若绝对净修复≤0，未支持减少原生像素错误的机制，但不能据此单独否定相对错误选择器的分配优势。像素改善但AP不升也不称任务有效。B 标签含模型条件性、未匹配query与跨集偏移仍待验证，无query的FN不能由该头恢复。现有教师见过FIT，B又训练selector；A/B只隔离新PR的标签生成，不能用B证明泛化。最终使用独立评价集；同一split三seed只衡量该split下随机性。此方案同时改变训练覆盖与分配，机制对照须共享新PR；不能直接与全FIT、带内训练的N13归因比较。完成固定 N13 结果后才决定是否执行，不宣称组合首次或投稿成熟。
