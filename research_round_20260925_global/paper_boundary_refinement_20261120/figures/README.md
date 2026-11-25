# Manuscript figures

## November 2026 visual revision

| Figure | Editable SVG | Vector PDF |
| --- | --- | --- |
| 1. Inference path | [SVG](matched_budget_pipeline.svg) | [PDF](matched_budget_pipeline.pdf) |
| 2. Local occupancy and supervision | [SVG](method_native_occupancy.svg) | [PDF](method_native_occupancy.pdf) |
| 3. Fixed versus independent selection at the same count (complete evidence draft) | [SVG](method_signed_selection.svg) | [PDF](method_signed_selection.pdf) |
| 4. Dual-reference allocation at a fixed total count | [SVG](method_signed_allocation.svg) | [PDF](method_signed_allocation.pdf) |
| 5. Controlled Phoenix writes | [SVG](method_phoenix_controls.svg) | [PDF](method_phoenix_controls.pdf) |
| 7. Same predictions under two author references | [SVG](reference_disagreement_effects.svg) | [PDF](reference_disagreement_effects.pdf) |
| 8. Variable signed allocation and measured area/intersection terms | [SVG](dual_reference_budget_effect.svg) | [PDF](dual_reference_budget_effect.pdf) |
| 12. Observed SegRefiner gain and loss cases | [SVG](rsdd74_segrefiner_cases_binned.svg) | [PDF](rsdd74_segrefiner_cases_binned.pdf) |

The 10-page review manuscript uses the corresponding three-writer figure with a recomputed bounded assignment between the distance-first and independent actions. Rebuild its editable [SVG](method_bounded_selection.svg) and [PDF](method_bounded_selection.pdf) with `python draw_selection_controls.py --output .`. The illustrative case verifies all three actions at two additions and two removals.

From this directory use `python draw_core_method_redesign.py --output .` for Figs. 1 and 3, and `python draw_supporting_stages_redesign.py --out .` for Figs. 2 and 5. For Fig. 4, run `python draw_signed_allocation.py --output .`. Its illustrative occupancies produce equal and variable allocations with four flips each. The first script uses the worked example in `draw_matched_budget_pipeline.py`. For Fig. 7, run `python plot_reference_disagreement.py --results ../../results --figures .` to rebuild the vector plot from the published aggregate scores. For Fig. 8, run `python plot_dual_budget.py --score ../../results/SPSD_PSEG_SCORE_VARIABLE_SIGNED_BUDGET_SEED5297_20261125_SAFE.json --figures .`. For Fig. 12, run `python plot_rsdd_cases_redesign.py --help` to see the required released summary and closed local inputs. [Case selection](../../results/RSDD74_SEGREFINER_STOCK_CASES_SAFE_20261123.json) · [Review and rationale](../../style_review/STYLE_REDESIGN_20261123.md)


## Other main figures

| Editable file | Content and data |
| --- | --- |
| [DEV705 overlap decomposition](dev705_overlap_cases.svg) | Two observed cases near the positive and negative medians. Each arm edits eight pixels per side. |
| [Cross-cohort effects](transfer_effects.svg) | Within-head writer contrasts and learned-versus-dynamic rank contrasts. AP dots represent fits. Horizontal IoU intervals represent paired group resampling. |
| [Gaofen-3 outcomes](gf3_matched_cases_binned.svg) | Four observed cases near the respective sign medians on 425 common eligible matches, with distinct images and eight swaps per arm. The released views pool local geometry to an eight-by-eight display grid. Values below each panel use the original continuous annotations. |
| [Phoenix support and edit count by dataset](phoenix_budget_by_dataset.svg) | The same four operating rules, shown for 74 Instance-RSDD images and 232 PSeg-SSDD images. Fixed geometry uses 47 and 293 original eligible matches, respectively. [Instance-RSDD scores](../../RSDD74_PHOENIX_SUPPORT_BUDGET_ACTUAL_20261123.md) · [PSeg-SSDD scores](../../PSEG232_PHOENIX_SUPPORT_BUDGET_ACTUAL_20261123.md) |
| [PSeg-SSDD qualitative outcomes](pseg_budget_cases_binned.svg) | Actual median-sign cases showing foreground removed by thresholding and a narrow tip partially recovered only at the larger budget. The SVG has pooled spatial geometry, with exact native-area scores outside the panels. [Case selection and measured outcomes](../../PSEG232_PHOENIX_SUPPORT_BUDGET_ACTUAL_20261123.md) |

The SVGs preserve text, geometry and connectors as editable vector elements. The corresponding PDFs use the same layouts for the manuscript. The method graphic was manually reconstructed as vectors from an illustrative layout concept. Its generator checks the assignment, selected slots and final areas before saving.

Run `python plot_transfer_effects.py` to rebuild the cross-cohort graphic from the published result summaries. The script checks each input hash and supports `--results` for a separate copy of the results directory. The Gaofen-3 script accepts the separately obtained author annotations and the closed detector 5203 replay produced by the [carrier reproduction](../../INSTANCE_RSDD_CARRIER5203_ACTUAL_20261123.md). Run `python plot_gf3_matched_cases.py --help` for its input directories. It places exact overlays in a separate private output directory. The published SVG and PDF contain pooled spatial measurements with no source imagery, full outlines or identifiers.

## Supplementary and previous figures

`native_grid_selection.svg` preserves the editable text, native-pixel grid, mask, endpoint glyphs, and connectors. Its three-panel layout follows a generated concept preview. The vector redraw separates fixed pairing from independent selection by choosing different sites at the same edit count. All mask shapes in this method schematic are illustrative.

`fit109_query_contrasts.svg` is a supplementary plot of the independently reproduced FIT109 comparison. The paired source-group 95% intervals apply to continuous IoU. [Verified FIT109 result](../../RFDETR_QUERY_CHANNEL_FIT109_ACTUAL_20261121.md)

`dev705_overlap_cases.svg` shows two observed RF-DETR matches reconstructed from the closed DEV705 predictions, sparse edits and source polygons. For seed 5217, the record contains 1,012 improvements, 252 deteriorations and 197 differences within $10^{-12}$ in continuous IoU. The two displayed pairs are observed matches nearest the median of their respective sign, with the first image excluded when choosing the second case. Both methods make eight swaps in each case. The JSON contains the scalar overlap and count inputs to the plot. [Closed matched comparison](../../RFDETR_CENTER5202_DEV705_MATCHED_RESULT_20261120.md)

`hrsid_official_seed_effects.svg` and `hrsid_official_seed_effects.pdf` show the three point-head differences and arithmetic means for mask AP and AP75 on the historical HRSID official allocation. The segments span observed seed effects. [Verified official score](../../HRSID_OFFICIAL_FROZEN_REFINEMENT_ACTUAL_20261122.md)

Rebuild each figure with its adjacent Python script. Figure legibility, editable labels and embedded fonts were checked using the [scientific visualization checklist](https://github.com/K-Dense-AI/scientific-agent-skills/blob/ec59a29a03d5c40f73ab40878df04860bc00e876/skills/scientific-visualization/SKILL.md).

Run `plot_cross_dataset_phoenix_budget.py` with the two released score summaries to generate the cross-dataset plot. The PSeg case generator uses the separately authorized author images, polygons and closed prediction streams. It saves the binned publication panels and precise internal overlays in separate directories.
# Additional method and ablation figures



`independent_sides_ablation.svg` and its companion PDF use the observed COCO mask AP from the [same-count side intervention](../../INDEPENDENT_SIDE_FACTORIAL_ACTUAL_20261123.md). The script `plot_independent_sides_ablation.py` reads the released aggregate at its verified digest. The date-separated view is nested. AP markers are point estimates.

## Supplementary matched-distance ablation

[Editable SVG](phoenix_bounded_matching.svg) · [Vector PDF](phoenix_bounded_matching.pdf) · [Source aggregates and selection rules](../../PHOENIX_BOUNDED_MATCH_ACTUAL_20261123.md). The left panel compares observed mask AP points. The dashed line marks the 0.30-point advancement threshold. The right panel gives fixed-match continuous IoU with paired cluster intervals. The two nested views are sensitivity analyses.

## Stock SegRefiner-LR cases on Instance-RSDD

[Editable SVG](rsdd74_segrefiner_cases_binned.svg) · [Vector PDF](rsdd74_segrefiner_cases_binned.pdf) · [Case selection and caption data](../../results/RSDD74_SEGREFINER_STOCK_CASES_SAFE_20261123.json). For 39 originally matched, gate-exposed hulls, the two separate images are closest to the respective medians of gains and losses from stock SegRefiner-LR. The public graphic pools exact mask and reference areas on an 8 × 8 grid. The local review PDF overlays all three masks and the author outline on native SAR crops.

## SAR-SSP 冻结编辑与分母

[矢量 PDF](sarssp_public_summary.pdf) · [可编辑 SVG](sarssp_public_summary.svg) · [聚合输入](sarssp_public_aggregates.json)。图中 52 个变更配对来自 92 个原生固定匹配，来自 19 个母景。零差组属于输出掩膜改变但 IoU 保持的匹配。空间格汇总每艘船的候选政策分歧像素中，独立相对距离配对净判对比例。置信区间保持全部 139 个船体和 23 个含船母景。公开文件只有聚合统计。

从本目录运行 `python make_public_sarssp_summary.py sarssp_public_aggregates.json --output .` 重绘图件。 [评分协议与矩阵](../../SAR_SSP_FROZEN_TRANSFER_ACTUAL_20261125.md) · [覆盖和等量编辑上限](../../SAR_SSP_COVERAGE_AND_EDIT_CEILING_ACTUAL_20261125.md)
