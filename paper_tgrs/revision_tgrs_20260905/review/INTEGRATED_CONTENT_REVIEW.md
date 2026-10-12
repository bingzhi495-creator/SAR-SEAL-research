# 完整修订稿独立交叉审查

审查对象：`revision_tgrs_20260905` 中 main、sec_intro、sec_related、sec_method、sec_exp、sec_disc、ai_disclosure，以及正文引用的三张独立表。行号对应本报告写入前读取的版本。仅写本报告，没有修改正文。

结论：本轮没有发现需要推翻主要结果的新增数值错误；载体身份、两数据集的 α、候选数与视图数、已有选择偏差及条件 bootstrap 的表述已基本一致。下面列出仍可直接修复的 6 项，主要是部署规则/训练规格遗漏及一处比较对象错接。P2 表示应在本次正文定稿时修复；P3 表示准确性或可读性的小修。

## 1. P2：BPR 定义漏掉实际部署的分数门槛和 patch 上限

- 位置：[sec_method.tex:93](../sec_method.tex)、[sec_method.tex:97](../sec_method.tex)，以及 Eq. `eq:guard`。
- 证据：`carrier/bpr_apply.py:113` 默认 `score_thr=0.05`；154–159 对低分记录直接透传。`refine_instance` 第 55–60 行默认最多 64 patches。`carrier/bpr_patches.py:66,75,115` 训练收集默认最多 32 patches。`run_bpr_queue.sh:51–54,92–96` 使用这些默认值，没有覆盖。
- 影响：目前正文会使复现者对所有 fused instances 应用 BPR，并遍历所有边界格子；这与产生现有结果的实现不同，也直接影响被修改的像素及计算量。当前“one center per grid cell”还未说明确定性选择顺序。
- 最小修正（93 行补充）：

> Boundary pixels are scanned in row-major order; the first pixel in each grid cell is retained, with at most 32 patches harvested per instance.

- 最小修正（97 行首句替换）：

> At deployment, instances with score below 0.05 pass through unchanged. For the remaining fused masks, the same deterministic sampling rule extracts at most 64 patches per instance; Eq. (\ref{eq:guard}) applies to these processed instances.

## 2. P2：BPR “remaining fitting details”交叉引用没有真正给出拟合规格

- 位置：[sec_method.tex:95](../sec_method.tex) 和 [sec_exp.tex:14](../sec_exp.tex)。方法指向实验，但实验仅写 20 epochs 和约 19 minutes。
- 证据：`carrier/bpr_refiner.py:46–65` 定义 4-down/4-up UNet；128–136 为增强；153–158 为参数；176–181 为固定划分；188–190 为优化器/调度；232–237 按最小 holdout loss 保存。`run_bpr_queue.sh:63–64` 指定 batch 64/lr 1e-3。没有执行新训练。
- 最小修正（实验 BPR 句后补充）：

> The refiner uses AdamW with batch size 64, learning rate $10^{-3}$, weight decay $10^{-4}$, and cosine decay to $10^{-5}$ over 20 epochs. Training patches receive independent horizontal and vertical flips with probability 0.5 and a random multiple-of-90-degree rotation. The 5\% patch holdout uses a fixed permutation seed of 123; the checkpoint with minimum holdout BCE-plus-Dice loss is retained.

- 方法可用一句补足 UNet 的可辨识规格：

> The four-down/four-up UNet has encoder widths 32/64/128/192/192, double convolution--batch normalization--ReLU blocks, and bilinear decoder upsampling with skip concatenation.

## 3. P2：F1 比较的 0.882 属于 OAEM，而上下文只明确提到了 NDM

- 位置：[sec_exp.tex:209](../sec_exp.tex)。该句先写 BPR vs NDM，接着写“0.902 versus 0.882”，随后又写“these three methods”。
- 证据：[tab_fafn.tex:24](../tab_fafn.tex) 的 NDM 最优 F1 为 **0.881**；25 行 OAEM 为 **0.882**；28 行 BPR 为 **0.902**。Offshore NDM、OAEM、BPR 在 0.30 阈值均为 369/374，但当前第三个对象没有明确引入。
- 最小修正：保留全部原数字，明确命名比较对象。

> The per-method optimum $F_1$ is 0.902 for BPR and 0.882 for OAEM; these thresholds are selected on the evaluated split and are descriptive, not deployment thresholds. At score 0.30 offshore, BPR, NDM, and OAEM each detect 369 of 374 ships.

## 4. P2：重排公式应区分“匹配到的特征来源”与“当前记录的原分数”

- 位置：[sec_method.tex:131](../sec_method.tex)–136，Eq. `eq:rrblend`。
- 证据：`carrier/scic_rerank.py:294–321` 允许 BPR mask 与 pre-BPR pool 贪心匹配；`stats[ci]`、`scores[ci]`、`geom_feats(rles[ci])` 构造特征，混合项却明确使用当前 BPR 记录的 `d['score']`，不是 matched candidate 的 `scores[ci]`。二者通常对应同一原实例，但贪心几何匹配不保证这一点。
- 影响：当前同一个未加下标的 $s^{out}$ 既出现在特征中又出现在混合公式，容易误读为必须继承匹配候选的分数；这在匹配重分配时不等于实现。
- 最小修正：设 $b$ 为 BPR 记录，$c(b)$ 为匹配的 pre-BPR 候选。公式写成

```latex
s_b^{\mathrm{final}}=
\alpha\,\operatorname{clip}(g(\boldsymbol\phi_{c(b)}),0,1)
+(1-\alpha)s_b^{\mathrm{in}},
```

> Here, $s_b^{\mathrm{in}}$ is the score already attached to the BPR record, which BPR leaves unchanged; the feature vector comes from its matched pre-refinement candidate.

## 5. P2：α 的验证目标与相关系数种类仍未指明

- 位置：[sec_method.tex:129](../sec_method.tex)、[sec_exp.tex:14](../sec_exp.tex)。划分和候选 α 已正确，但“selected using”没有给出最优化的评价量；“residual correlations”也无法区分 Pearson/Spearman。
- 证据：`carrier/scic_rerank.py:175–190,225–239` 对 held-out **原始 fused masks（BPR 前）**运行 COCO AP，并以 AP 最大的候选 α 为最终值。代码也评估 α=0 的原分数参照，但它不在三个候选的选择循环中。`runs/_scic_step3b.log:2` Pearson=0.8764，**Spearman=0.6766**，所以正文 0.68 是 Spearman，不是一般意义下未指定的相关系数。
- 最小修正：

> The blend is selected by maximizing COCO mask AP on the held-out images using the pre-BPR fused masks, with the original scores evaluated as a reference; the regressor is then refitted on all rows.

- 同时把 `Held-out residual correlations` 改成 `Held-out Spearman correlations between predicted quality and target mask IoU`。不改变 α=0.5/1.0 或现有相关系数。

## 6. P3：“detection records are preserved”字面上与更新 mask/score 冲突

- 位置：[sec_method.tex:106](../sec_method.tex)、111、136。
- 原因：完整检测记录包含 segmentation 和 score 字段，BPR 和 re-ranking 恰好分别修改其中之一。想表达的应为记录数量和身份保持，而不是整个记录逐字不变。
- 最小修正：BPR 两处用 `scores, detection count and identities, and all pixels outside ...`；重排句用 `All masks, detection count, and image/class identities are preserved; only scores change.` 不必再新增一段局限性。

## 已核查而不作为问题上报

- 摘要为 **222 个空格分隔词**，满足计划要求的 150–250 词。
- 扫描所审正文及三张输入表：`\ref`/`\eqref` 没有缺失 label，`\cite` 没有缺失 bib key。
- HRSID β=0.5 auxiliary head 与 PSeg-SSDD standard head 在摘要、引言、相关工作、方法、实验和讨论中一致；α 两数据集分别 0.5/1.0 已正确。
- 条件 voting 采用候选数、foreground tie 产生二成员 union、seed-relative 非传递聚类的表述与实现一致。
- 本轮没有再次把 test-set selection、patch holdout、缺少新确认实验或没有公开仓库列为“新问题”；正文已披露这些事实。
- SARSAM 的当前“parameter-efficient adaptation”概括与[出版商原文](https://www.mdpi.com/2072-4292/17/7/1209)一致，不应因为其长标题未含 SARSAM 就误判为错引。未对全部参考文献的原始数字做新的独立复现实验。
- 新 Fig. 1–3 的 conceptual 图路径与其 AI-use 披露由根任务集成，本审查没有把正在替换的路径当作错误。S1/S2 的 stored-candidate / single-view evidence 范围已有正文说明。
- 上述建议均是文稿修正或补齐已存在配置；不要求新训练、评估或新增指标行。
