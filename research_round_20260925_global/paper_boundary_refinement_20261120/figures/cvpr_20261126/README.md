# CVPR-inspired figure revision

The refreshed full manuscript uses all 16 vector panels in this directory. The local author-view manuscript adds the native-SAR SAR-SSP panel and uses native case figures in the four qualifying comparisons. The public build uses the spatially pooled variants. The complete manuscript source still points directly to the original remote-data protocols and evidence.

| Figures | Panels | Generator / baseline |
| --- | --- | --- |
| 1–5 | Overview, fractional occupancy, equal-count selection, signed allocation, Phoenix controls | [Draw method figures](draw_method_figures_cvpr.py) · [Fixtures and design](METHOD_REDESIGN.md) |
| 6 and SAR-SSP summary | Three-policy transfer, scene-balanced SAR-SSP and spatial effects | [Draw evidence figures](plot_cvpr_evidence.py) |
| 7, 8 and ablation charts | Reference disagreement, signed allocation, independent sides, Phoenix support and count | [Draw quantitative figures](plot_quantitative_redesign.py) |
| 9–13 | Matched-cap runs, HRSID examples, Gaofen-3, PSeg-SSDD and SegRefiner | [Case redraw source](style_case_panels.py) · [Case checks](CASE_STYLE_README.md) |

Run from this directory with the same release tree and the existing scientific plotting dependencies. Use a fresh output directory.

~~~sh
python draw_method_figures_cvpr.py --fixtures-dir .. --output redraw
python plot_quantitative_redesign.py --results ../../../results --output-dir redraw
python plot_cvpr_evidence.py --code ../../../code --sarssp ../sarssp_public_aggregates.json --output redraw
python style_case_panels.py --public .. --out case_redraw
~~~

The last command restyles the original, completed SVG files in the parent directory once. To apply the current ROH column labels to its outputs, run `python rename_roh_case_labels.py --public case_redraw/public --out roh_case_redraw`. Use the resulting two public SVG and PDF pairs alongside the other case figures. It preserves the original panel geometry and metric labels. The source uses a Chromium-based browser to print SVG to PDF, preserving dashed reference contours. Native case figures have their own local entry point and ownership. The new transfer plot reads five SHA-256-pinned research summaries. Its original independent-versus-distance and attraction overview remains available in the parent figure directory. The SAR-SSP chart reads the unchanged released aggregate. No model fits or evaluations were run during the figure revision.

[Research design and figures](../../design/DESIGN_RATIONALE_ZH.md) · [Complete manuscript](../../COMPLETE_MANUSCRIPT.md)
