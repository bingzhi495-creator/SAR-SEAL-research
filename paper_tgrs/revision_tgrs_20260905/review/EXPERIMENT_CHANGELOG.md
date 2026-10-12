# 实验、讨论与结论修订记录

修订日期：2026-09-05。独占编辑 `sec_exp.tex`、`sec_disc.tex`、`tab_fafn.tex`、`tab_soupn.tex`、`tab_univ.tex`；父目录原稿未改。

## 结构与证据定位

- 顺序调整为数据/训练、选择/统计协议、同载体消融、文献及本地比较、迁移、负结果。核心固定候选池证据先于跨论文对比；所有旧 LaTeX labels 保留。
- HRSID c0 使用权重 0.5 的辅助质量损失；推理 λ=0 不能撤销训练影响。PSeg-SSDD 的保存配置使用标准 Mask2FormerHead，不带该目标，因此迁移的是融合规则和校正流程，不是相同训练 head/objective。
- HRSID fusion sweeps 明确标为探索性 test-set selection；保留 ceiling/训练集 sweep 数值，删除“窄幅 sweep 已约束所有乐观偏差”和“训练集重扫不可能改善选择”的辩解。
- 区分随机 patch holdout、image-ID holdout、out-of-fold residual；明确 BPR 的最佳 s0 选择及后续 re-ranking 继承的选择条件。
- paired bootstrap 仅覆盖固定预测/设置下的图像重采样；不涵盖训练、开发选择和父场景相关性。确定性、filename separation、annotation-free apply 均不再等同于全面无泄漏认证。
- PSeg-SSDD 的 46 张近岸图上，两项阶段区间跨零；不写所有阶段都显著、普遍适用或 SOTA。
- 文献 reported values 与 local reimplementation 分开，列明初始化、训练预算、checkpoint selection、soup 和推理计算差异；删除 one identical protocol 和“best-on-test 就是 upper bound”的解释。
- 同候选池结果区分 mask-space seed selection 与 gate 增量：37.1→38.2→38.8。Matrix-NMS 的极低值保留为 tested configuration 失败；相同概率均值/投票指标不再被称为 bitwise 等价。
- Swin-S、200 epoch、跨论文 pretraining、Gaussian–Wasserstein 以及负结果保留数字，改为探索性控制；删除 preservation law 和机制必然性。
- 跨载体表改为有限 transfer evidence。M2F λ=0.2 baseline 明确是另一 rescoring operating point，不是主表 λ=0 的额外严格匹配试验。
- 定性图 captions 披露偏向有利改善的选图规则，保留原独有计数和 IoU；不把精选图片当代表性样本。
- 讨论明确写字段约束不蕴含 commutativity 或 AP monotonicity；集中说明独立确认、OOF、场景划分和实测 compute 要求。
- 结论保留 HRSID 44.5/48.3 与 PSeg-SSDD 62.1，撤销已隔离预训练作用、SOTA、传感器信息上限和完成独立确认的主张。
- 删除虚构 `XXX/XXX` 发布链接和无来源致谢。Data/Code 如实写为私有本地记录，未建立 public archive。作者未改；AI-use disclosure 由主代理单独维护。

## 成本口径

- Soup 的 0 training 改为额外两次成员训练，明确每成员 55 epochs、合计 3×55。零只指该后处理操作无需拟合。
- 约 6×/8× cost 改为 6/8 view count，BPR/rank 另列；没有同步计时证据时不把视图数当 runtime multiplier。
- 既有 19 min、19 s 等保留为 reported corrector-fitting time，未新增计时。

## 验证

- 8 张表抽取的 433 个逐行结果数值与原稿完全一致，见 `EXPERIMENT_NUMERIC_CHECK.json`；soup 的 mean±s.d. 行也原样保留。
- 原 `sec_exp.tex` 与 `sec_disc.tex` 联合出现的独特小数值，在修订后的两文件中全部保留；所有旧 labels 保留。
- 实验源空白分词长度从 6328 压缩至约 4800，含表格。保留大表和原定性图，未通过缩小字体硬压五页。
- 独立 jobname `qa_experiments` 编译当前整稿成功，输出 `review/qa_experiments.pdf`。这是排版检查；参考文献及最终全稿交叉引用由主代理统一构建。
- 主表一度横向溢出 53.6 pt，已用缩短重复行名和多行列头解决。调整结果/浮动体顺序后，一次 7.7 pt 纵向溢出也消失；最后检查日志无 overfull。
- 已检查关键实验表和定性图渲染。主代理还需在新方法图/参考文献稳定后统一复核图表落位、原定性图最终字号和跨页留白。

没有训练、推理、COCOeval、bootstrap 重算或 checkpoint 读取。数值一致性检查保留已有证据，不是独立复现实验认证。
