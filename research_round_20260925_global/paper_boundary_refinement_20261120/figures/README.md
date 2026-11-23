# Manuscript figures

## Current main figures

| Editable file | Content and data |
| --- | --- |
| [Matched-budget pipeline](matched_budget_pipeline.svg) | Native mask and signed support, source-only point supervision, and fixed versus independent writers with a worked example. The geometry and scores are illustrative. |
| [DEV705 overlap decomposition](dev705_overlap_cases.svg) | Two observed cases near the positive and negative medians. Each arm edits eight pixels per side. |
| [Cross-cohort effects](transfer_effects.svg) | Within-head writer contrasts and learned-versus-dynamic rank contrasts. AP dots represent fits. Horizontal IoU intervals represent paired group resampling. |
| [Gaofen-3 outcomes](gf3_matched_cases_binned.svg) | Four observed cases near the respective sign medians on 425 common eligible matches, with distinct images and eight swaps per arm. The released views pool local geometry to an eight-by-eight display grid. Values below each panel use the original continuous annotations. |
| [Phoenix support and edit count](phoenix_support_budget.svg) | Observed mask AP on 74 images and fixed continuous IoU on 47 original matches, crossing narrow and whole support with threshold and exact-count rules. [Frozen scoring result](../../RSDD74_PHOENIX_SUPPORT_BUDGET_ACTUAL_20261123.md) |

The SVGs preserve text, geometry and connectors as editable vector elements. The corresponding PDFs use the same layouts for the manuscript. The method graphic was manually reconstructed as vectors from an illustrative layout concept. Its generator checks the assignment, selected slots and final areas before saving.

Run `python draw_matched_budget_pipeline.py` to rebuild the method graphic. Run `python plot_transfer_effects.py` to rebuild the cross-cohort graphic from the published result summaries. The script checks each input hash and supports `--results` for a separate copy of the results directory. The Gaofen-3 script accepts the separately obtained author annotations and the closed detector 5203 replay produced by the [carrier reproduction](../../INSTANCE_RSDD_CARRIER5203_ACTUAL_20261123.md). Run `python plot_gf3_matched_cases.py --help` for its input directories. It places exact overlays in a separate private output directory. The published SVG and PDF contain pooled spatial measurements with no source imagery, full outlines or identifiers.

## Supplementary and previous figures

`native_grid_selection.svg` preserves the editable text, native-pixel grid, mask, endpoint glyphs, and connectors. Its three-panel layout follows a generated concept preview. The vector redraw separates fixed pairing from independent selection by choosing different sites at the same edit count. All mask shapes in this method schematic are illustrative.

`fit109_query_contrasts.svg` is a supplementary plot of the independently reproduced FIT109 comparison. The paired source-group 95% intervals apply to continuous IoU. [Verified FIT109 result](../../RFDETR_QUERY_CHANNEL_FIT109_ACTUAL_20261121.md)

`dev705_overlap_cases.svg` shows two observed RF-DETR matches reconstructed from the closed DEV705 predictions, sparse edits and source polygons. For seed 5217, the record contains 1,012 improvements, 252 deteriorations and 197 differences within $10^{-12}$ in continuous IoU. The two displayed pairs are observed matches nearest the median of their respective sign, with the first image excluded when choosing the second case. Both methods make eight swaps in each case. The JSON contains the scalar overlap and count inputs to the plot. [Closed matched comparison](../../RFDETR_CENTER5202_DEV705_MATCHED_RESULT_20261120.md)

`hrsid_official_seed_effects.svg` and `hrsid_official_seed_effects.pdf` show the three point-head differences and arithmetic means for mask AP and AP75 on the historical HRSID official allocation. The segments span observed seed effects. [Verified official score](../../HRSID_OFFICIAL_FROZEN_REFINEMENT_ACTUAL_20261122.md)

Rebuild each figure with its adjacent Python script. Figure legibility, editable labels and embedded fonts were checked using the [scientific visualization checklist](https://github.com/K-Dense-AI/scientific-agent-skills/blob/ec59a29a03d5c40f73ab40878df04860bc00e876/skills/scientific-visualization/SKILL.md).
