# Manuscript figures

`native_grid_selection.svg` preserves the editable text, native-pixel grid, mask, endpoint glyphs, and connectors. Its three-panel layout follows a generated concept preview. The vector redraw separates fixed pairing from independent selection by choosing different sites at the same edit count. All mask shapes in this method schematic are illustrative.

`fit109_query_contrasts.svg` is generated from the independently reproduced FIT109 comparison. The paired source-group 95% intervals apply to continuous IoU. [Verified FIT109 result](../../RFDETR_QUERY_CHANNEL_FIT109_ACTUAL_20261121.md)

`dev705_overlap_cases.svg` shows two observed RF-DETR matches reconstructed from the closed DEV705 predictions, sparse edits and source polygons. For seed 5217, the record contains 1,012 improvements, 252 deteriorations and 197 differences within $10^{-12}$ in continuous IoU. The two displayed pairs are observed matches nearest the median of their respective sign, with the first image excluded when choosing the second case. Both methods make eight swaps in each case. The JSON contains the scalar overlap and count inputs to the plot. [Closed matched comparison](../../RFDETR_CENTER5202_DEV705_MATCHED_RESULT_20261120.md)

The SVGs retain editable text and the PDFs contain vector copies for LaTeX. Rebuild each figure with its adjacent Python script. Figure legibility, editable labels and embedded fonts were checked using the [scientific visualization checklist](https://github.com/K-Dense-AI/scientific-agent-skills/blob/ec59a29a03d5c40f73ab40878df04860bc00e876/skills/scientific-visualization/SKILL.md).
