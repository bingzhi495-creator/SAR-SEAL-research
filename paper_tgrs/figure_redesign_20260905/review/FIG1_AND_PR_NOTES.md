# 总览与 PR 图的交付说明

## Fig. 1：参数准备与图像推理分开

新总览保留原说明中的核心模块和视觉语义，但纠正原图将权重汤画在图像预测之后的问题。上路是三个训练完成的参数副本经离线平均；下路是 SAR 图像进入使用该固定汤模型的 Wide TTA，再进入候选融合、BPR 和重排。Wide TTA 框包含冻结网络的全部多视图前向调用，而不只是 resize/flip。

仅使用 `assets_fig1/chip.png`，它与 `E:/SAR/seal/paper_tgrs/figs/assets/chip.png` 完全一致。输出使用符号实例集合，因为现有 `pred_masks.png` 只是单视图输出展示，不能放在最终重排之后冒充该级真实预测。

“frozen”仅指训练完成后的推理状态。保存的训练配置及日志显示辅助 mask-quality loss 被启用，因此此图没有使用 unmodified、never trained 或从未改变损失等标签。主体标签保留 Mask2Former-R50 家族名，图注必须补充实际训练身份。

主标签/次标签物理字号为 8/7 PDF pt；方法框均两行；宽 516 pt，高 122 pt（4.304 cm）；44 个空白分词；线宽 0.5/0.8 pt；两类非实线图案；一个锁；10 种矢量颜色；唯一嵌入图片为 800×800，绘出 48×48 pt，1200 dpi。高饱和朱红实框保留给共识融合，绿色虚框表示经训练的纠错器。所有六项机械验收通过；最终色彩与灰度预览均经检查。没有组织独立真人一秒阅读测试。

### English caption

Overview of the inference cascade with parameter preparation separated from image processing. The three fitted Mask2Former-R50 carrier checkpoints are averaged offline, and the resulting weights remain fixed throughout inference. Wide TTA denotes both the view transforms and all forward calls to this frozen souped model. Consensus vote merges candidate instances; the separately trained boundary refiner updates mask pixels, and the reranker updates scores. The input is a real HRSID chip; the final instance set is symbolic. The carrier training records include an auxiliary mask-quality objective, although its inference-time rescoring is disabled. Frozen therefore describes deployment after fitting. The two correctors preserve different stored fields; their performance interaction is an empirical question.

## Fig. 4：保持已有曲线，仅修改显示

原图已经是矢量 PR 图，包含每条曲线全部 101 个 COCO recall-grid 顶点。本轮直接恢复并保留这些已有坐标，再以 matplotlib 重绘，未加载 detector、未运行 COCOeval、未重算 AP。每面板八条，两个面板共 16 条曲线。

`sources/fig4_original.pdf` 是原 `paper_tgrs/figs/fig_pr_ssdd.pdf` 的逐字节副本；`sources/make_pr_original.py` 保存原评估绘图代码供溯源。新 `make_pr.py` 仅依赖包内源 PDF，生成 `fig4_pr.pdf/.svg` 与 `review/PR_CURVE_PROVENANCE.json`。

五条对照仍按两档灰度表示；base / +stack / +BPR 使用蓝点划、朱红实线、绿虚线，并在下图直接标注。减少垂直网格使视线集中到曲线分离处。浅绿色面积是已有 stack/BPR 曲线间的几何差，不是新计算的 AP 结论。

独立复核比较了原 PDF 与新 PDF 的所有 16×101 个顶点，并确认归一化坐标范围与实际轴线一致。最大 recall 坐标差 4.82e-8，最大 precision 坐标差 2.12e-7，仅为 PDF 浮点坐标舍入。详见 `QA_PR_ROUNDTRIP.json`。此检查证明绘图忠实，不等于独立验证预测 dump 或论文数值。

画幅 252×324 pt（8.89×11.43 cm），字体 7/8 pt，线宽 0.5/0.75/1.4 pt，无图像栅格，字体作为 TrueType 嵌入，无 Type 3。六道验收全部通过，文字 38 个空白分词，未印 AP 数字。

### English caption

Precision-recall curves on the PSeg-SSDD inshore split at (a) mask IoU = 0.50 and (b) mask IoU = 0.75, before learned reranking. The three cascade variants are identified directly in (b); the five comparison methods use two gray tiers named in (a). The shaded region shows the gap between the stored stack and BPR curves. This redesign preserves the 101 sampled points of every curve in the existing source figure. The underlying prediction evaluation is described in the original experiment protocol and was not rerun for this graphical revision.

## 与绘图说明的明确差异

- Fig. 1 改为上下两路，避免权重汤被误读为 mask 后处理；最后用符号集合，避免跨阶段冒用真实图片。
- Fig. 2 用同一批存储成员重新导出矩阵，并在图注明确 top-16 重放与全池部署不同。
- Fig. 3 用 coarse masks 代替不真实的 fused masks，并增加 gray + mask 说明实际输入。
- 对图注的上述修正优先于说明文档的固定文案。图内不添加精度数字，不生成或伪造数据图。
- IEEE 当前图形指南建议 full-size 约 9–10 pt；本次按用户说明保留 8/7 pt 两档，机械符合用户规则，不宣称其是 IEEE 官方的普遍最低字号。若编辑部要求更大字号，应删减图中文字或改版面，不能靠缩放挤入。

官方依据：[图形尺寸与分辨率](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-graphics-for-your-article/resolution-and-size/)，[格式与字体嵌入](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-graphics-for-your-article/file-formatting/)，[TGRS 作者说明](https://www.grss-ieee.org/publications/author-resources/tgrs-information-for-authors/)。检查日期：2026-09-05。
