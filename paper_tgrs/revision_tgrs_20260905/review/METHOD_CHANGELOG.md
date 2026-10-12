# 方法章节实质修订记录

文件：`sec_method.tex`。日期：2026-09-05。依据：当前 `PAPER_PLAN.md`、前次完整审阅 `PAPER_AUDIT_CN.md`、三个载体保存的训练配置/日志、`mask_tta.py`、`maskiou_head.py`、`bpr_apply.py`、`bpr_refiner.py`、`scic_rerank.py`，以及已修订图件的来源说明。

## 主线

章节从“未改网络的普遍保留规律”重写为“已训练载体上的候选融合、局部边界修正和质量重评分”。原有 4 个 subsection 与所有交叉引用 label 保留，减少方法部分重复报告 AP 和辩论式动机；数值结果仍由实验章节承担，不产生新实验或更改已有指标。

## 实质变化

1. **训练身份**：明确 COCO 初始化的 Mask2Former-R50 训练时采用 `L_M2F + β L_quality`，β=0.5；辅助质量分支未 detach，因此作用于共享表示。推理分数混合 λ=0 与训练监督分开解释。冻结发生在载体训练之后。删除 unmodified、never updated、全部架构控制失败即可推出保留规律等错误说法。
2. **权重与候选**：汤为离线参数处理，保留 3 个种子和各数据集真实 TTA 尺度；明确 3 次训练成本。说明掩码映射回原图坐标后才能合并。移除“共同初始化/BN 一致是有用权重平均的必要充分质量条件”这类过强推论。
3. **融合定义**：贪心聚类只要求与种子达到 mask-IoU 阈值，并不要求两两相似或传递闭包；候选支持数与 distinct-view count 不同。门控保留真实 2≤|C|≤6 条件，说明偶数票平局保留前景、|C|=2 为并集。
4. **输出与算法**：`eq:score` 改为部署事实 `s_out=s_seed`，一般 γ 形式保留在解释中；不再把 |C|/V 称为 [0,1] 概率或方差。算法显式输入 class/view ID、移除已消耗候选、保留统计和 top100 输出；不宣称最优聚类或普遍安全。
5. **BPR**：明确借鉴既有 BPR，训练补丁来自单视图；部署来自融合输出。保留 64×64→128×128→64×64、图像 bicubic、coarse mask bilinear、sigmoid 后面积缩采样、重叠概率平均的真实流程。保留原样本数量和 patch 验证方式；删除“独立图像验证”暗示。
6. **边界约束**：使用 3×3 方核迭代 4 次 dilation/erosion XOR 定义形态学带；输出需同时满足带内和有 patch 覆盖。保留空掩码回退。可保证带外像素、scores 和检测记录不变，不能保证拓扑或每实例 IoU 改善。
7. **重排**：保留 14 特征并说明 singleton 的 off-diagonal IoU 统计为零、几何来自 BPR 前 masks；训练行来自已保存融合候选且 score≥0.01，删除“所有被筛掉候选也参与”的过宽表述。target 为 best-GT IoU，不等于 COCO 一对一匹配标签。
8. **组合依赖**：明确 BPR 后记录按分数顺序、mask-IoU≥0.5 与原始候选做一对一贪心匹配，使用原始几何/统计评分；unmatched 保留分数。删除 disjoint writes 推出算子交换性、互不抵消、重排必然提高 AP 的说法。
9. **图件证据**：总图用 `figs/fig1_framework_ai.png`，caption 明示概念插图，不含实测 SAR/模型张量。融合图 caption 明示 top16 截断候选重放；在 full800 中两个示例支持度为 10、9，均保留 seed。BPR 图 caption 明示单视图素材，并非最终 TTA+fusion 的前后配对。可选 readout 留作补充，不增加主文浮动图。

## 保留的 labels

`sec:method`, `sec:overview`, `sec:fusion`, `sec:bpr`, `sec:rerank`; `fig:framework`, `fig:m-fusion`, `fig:m-bpr`; `alg:fusion`; `eq:soup`, `eq:views`, `eq:iou`, `eq:cluster`, `eq:fuse`, `eq:score`, `eq:guard`, `eq:rrfeat`, `eq:rrblend`。

**新增 label：无。删除 label：无。** 九个编号公式仍保持原顺序；新增质量训练公式仅内联、不占编号。

## 引文变化

- 保留：`mask2former2022`, `modelsoups2022`, `bpr2021`。
- 方法节新增使用：`msrcnn2019`（key 已存在于共享 bibliography，无需新增条目）。
- 方法节移除：`o2former2025`（跨论文初始化差异移出方法因果叙事；其他章节仍可引用）。
- 未添加任何新 bibliography key。

## 验证与集成

已用脚本比较原/新 method 的 label 与 cite 集合，确认无 label 丢失。源文件约 1500 个空白分词（包含 LaTeX 命令），相较原节压缩重复结果和修辞，以约 2.5–3 页为布局目标。实际页数取决于根代理生成总图的比例与全文浮动图排版，必须在集成 PDF 中最终确认。

本子任务未运行训练、推理、COCOeval 或 bootstrap；没有生成新科学指标。完整 LaTeX 编译、引用解析和逐页视觉检查由根代理在图件到位后完成。原 `E:/SAR/seal/paper_tgrs/sec_method.tex` 未改动。

## 交接后的只读核查补充

**训练身份必须按数据集区分。** 上述 β=0.5 的保存配置和损失日志证据属于 HRSID；PSeg-SSDD 的 `runs/cp_pseg/s0/s2/s3/cp_m2f_cocoinit_pseg.py:226` 均为标准 `Mask2FormerHead`，三个 resolved config 均无 mask-IoU 辅助分支。s0 的 `20260728_232720/20260728_232720.log:2307` 训练损失项也不含 `loss_maskiou`。因此根代理接管后的最终方法/概念总图应标注 HRSID auxiliary quality supervision，PSeg-SSDD standard head，不能将辅助项泛化到两个载体。

**重排 α 来源得到文本日志支持。** HRSID 的 `runs/_scic.out:16` 至 `:22` 包含训练图像 holdout 的 α sweep 并选择 α=0.5；对应 `_scic_r11_R11_{inshore,official,offshore}.log:5` 的实际 apply 也为 0.5。PSeg-SSDD 的 `runs/_ssddr11.out:1699`、`:1721`、`:1743`、`:1765` 记录原分数及三个 α 值的训练 holdout 评估，`:1766` 选择 α=1.0；最终 `_ssddr11fix_inshore.log:6`、`_ssddr11fix_offshore.log:6` 和 `_ssddr11_full.log:6` 实际使用 α=1.0。这里只读取已有日志，没有重算指标或加载 pickle 模型。

PSeg-SSDD 曾存在旧错误关联产物：`_ssddr11_inshore.log:6` 显示 matched=0，但最终应以 `ssdd_r11fix_*` 配对文件为准。最终 full 文件的匹配数 7140 等于修正后 inshore 2793 与 offshore 4347 之和。这是产物链路核查提示，不是新增性能实验。

根代理将把正文融合/BPR主图替换为 imagegen 生成的概念机制图，并把真实 top16 replay 与单视图 BPR 证据图放入补充材料；故上文图注路径变化由根代理最终集成，本文不再持有 `sec_method.tex` 编辑锁。
