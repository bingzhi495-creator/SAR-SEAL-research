# Equal-count D-FINE examples on the Instance-RSDD reserve

The two examples use the identical score-ranked native assignments from the accepted historical 74-chip evaluation. The 44 eligible hulls reproduce the reported group-averaged continuous IoU difference for independent versus bounded selection to within `1e-12` points. Using the same tolerance, 25 hulls improved, 11 deteriorated and eight tied. For each sign, choose the hull nearest its median difference, excluding the first image from the second selection. Image, annotation and frozen candidate order break exact ties. The cases illustrate spatial outcomes at a matched budget.

| Case | Added / removed per writer | N | F | L | I | I minus L |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| (a) Median gain | 4 / 4 | 51.71 | 51.52 | 51.73 | 51.92 | +0.18 |
| (b) Median loss | 8 / 8 | 67.67 | 67.73 | 68.00 | 67.85 | −0.15 |

The unrounded geometry uses the exact intersection of the author polygon and native foreground cells. N is the unedited mask, F sorts distance-first pairs, L re-pairs under the inherited per-pair distance cap, and I ranks both sides independently. F, L and I reuse each candidate's realized addition/removal count. The native private view repeats each original crop across the six columns, with a common 2nd–98th percentile grayscale stretch, teal foreground and a dashed author outline. The public view uses an 8 × 8 display grid. Reference shades show polygon-covered area and each mask shade shows the fraction of foreground cells in that display bin. The last column gives I's foreground fraction minus L's at each display bin, with color saturation normalized to the maximal absolute difference per row. The underlying IoUs and the edit counts retain the original native geometry.

To reconstruct the editable public PDF and SVG in a fresh directory, use:

~~~sh
python plot_rsdd74_common_cap_cases.py --safe-input rsdd74_common_cap_cases_safe.json --out pooled_redraw
~~~

The rendering source can reconstruct the private view from the original closed inputs using its `--closed`, `--capture`, `--intake`, `--archive`, and `--code` options. It verifies the observed score closure, reads back the original capture and independent edit receipts via the accepted worker, uses the original COCO matching function, checks all eligible cases against the accepted grouped mean, and verifies exact decoded image and edit counts for both displayed samples. Python 3.10.13, NumPy 1.26.4, Matplotlib 3.10.9, Shapely 2.0.2 / GEOS 3.11.2 and qualified pycocotools 2.0.11 were used. No additional inference, training or target fitting was run.

The public JSON includes pooled display values and scalar summaries. The public SVG has editable cells and labels, and its PDF embeds no images. The original crops, contour overlays, native masks and case identifiers are retained locally. [Closed evaluation](https://github.com/bingzhi495-creator/SAR-SEAL-research/blob/main/research_round_20260925_global/code/DFINE_RSDD74_COMMON_CAP_20261126/ACTUAL.md)
