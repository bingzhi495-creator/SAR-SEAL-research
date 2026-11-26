# TGRS 完整稿与图件入口

- [25 页 IEEE 双栏完整稿](boundary_refinement_complete.pdf) · [LaTeX 源码](boundary_refinement_complete.tex)
- [现行精简稿](boundary_refinement_main.pdf) · [作者信息与投稿资料](submission_20261126/AUTHOR_HANDOFF_ZH.md)
- [实验和论文证据索引](../TGRS_MANUSCRIPT_EVIDENCE_20261123.md) · [可编辑图件目录](figures/README.md)

完整稿包括引言、相关工作、总体流程和模块图、原生网格与占据率公式、同预算三路决策及固定总量分配、平台和数据、量化结果、正负定性案例、消融、误差和开销讨论、结论及参考文献。定性分析包括 Instance-RSDD 保留集原生／距离优先／限距重配／独立选点的同目标对照、HRSID 重叠得失、高分三号匹配预算、PSeg-SSDD 的阈值与同量写回、SegRefiner 整区写回、SAR-SSP 跨实例空间汇总。图注逐一标明原始对照和参考。主定性对照每行分别固定四、八次增删交换，选例按预定符号中位数，真实正负例并列。

[主定性图 SVG](figures/rsdd74_common_cap_cases.svg) · [选例规则、输入与重绘](figures/RSDD74_COMMON_CAP_CASES.md) · [高分三号 SVG](figures/gf3_matched_cases_binned.svg) · [PSeg SVG](figures/pseg_budget_cases_binned.svg) · [SegRefiner SVG](figures/rsdd74_segrefiner_cases_binned.svg) · [消融 SVG](figures/independent_sides_ablation.svg)

公开 PDF 的遥感案例使用实测输出的汇总矢量图和精确尺度的分数标注。带原始 SAR 裁片、作者轮廓和具体实例叠图的作者审阅稿保存在本机，依项目发行范围处理。正式投稿还需作者资料、相关图件复用权限及最终作者确认。前瞻 Umbra 两批仍待各自的独立参考评分。稿件保留了未过门的实验及适用范围。

在当前目录运行两到三次 `pdflatex -interaction=nonstopmode -halt-on-error boundary_refinement_complete.tex` 可重建公开版。论文需要 IEEEtran 及同目录已发布的矢量图件。
