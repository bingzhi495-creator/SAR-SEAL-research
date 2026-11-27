# Case selection and reconstruction

## Historical 74-chip Instance-RSDD reserve

The two D-FINE examples compare independent ranking with bounded re-pairing at the count supplied by the distance-first rule. Both use the same scores, native candidate sites and signed edit counts. Among 44 eligible fixed matches, 25 gained continuous IoU, 11 lost, and eight tied. The displayed cases are the admitted hulls nearest the sign-specific medians. They give the only plated within-case comparison of this specific distance constraint in the reviewed collection. At four additions and four removals, changed removals account for the positive result despite slightly weaker additions. At eight per side, the observed loss comes from weaker additions. The original selection and public mask fractions are recorded in the project repository. [RSDD case record](https://github.com/bingzhi495-creator/SAR-SEAL-research/blob/e467ab80ca9bbd2a2c80b27ff38bd2f803b29c0c/research_round_20260925_global/paper_boundary_refinement_20261120/figures/RSDD74_COMMON_CAP_CASES.md)

## Gaofen-3 ranking

The four RF-DETR fit B examples compare ROH ranking and dynamic spatial attraction at the dynamic-attraction count. Each method adds eight cells and removes eight in all four cases. The original near-median set has two gains and two losses on four distinct images. Of the 425 eligible matches, 268 gained IoU, 154 lost and three tied. The main pair maximizes the absolute change in GT intersection within each sign of these four admitted examples. This selection reflects the capacity to inspect the changed hull area on each side. The supplementary figure retains all four, in order of sign and then absolute change in intersection. [Original registered case-figure generator and selection rule](https://github.com/bingzhi495-creator/SAR-SEAL-research/blob/e467ab80ca9bbd2a2c80b27ff38bd2f803b29c0c/research_round_20260925_global/paper_boundary_refinement_20261120/figures/plot_gf3_matched_cases.py)

## Coverage, intersection and detail windows

The first view shows binned polygon coverage. The second and third show the compared foreground assignments. The last uses the marked 4-by-4 window of an 8-by-8 spatial grid. Its location maximizes the GT intersection area contained in the symmetric difference of the two masks, independent of which rule is favored. Ties in area to four decimal places favor the window nearest the field center, then the smallest row and column. For RSDD the detail encodes the mean signed foreground difference. For Gaofen-3 it sums signed intersection changes over the native cells in each bin. Both rows and columns use a common orientation within a case. RSDD uses the previously released eight-bin partitions. Gaofen-3 uses eight equal bins per dimension and mask coverage rounded to quarter levels. The dashed contour in the public view is the 50% contour of the pooled GT coverage. Colors in a given contrast share one scale across its cases.

The whole-case numbers use polygons and native unit cells. For a native cell with GT fraction q, its contribution to the intersection difference is q times the difference in final binary foreground. Summing this over original background cells gives the addition term. Summing it over original foreground cells gives the removal term. A positive removal term means the first-named rule removed less reference area. Net change is computed before display rounding. The mechanism checks used the registered paths and masks of the authorized review plates and reproduce the previously reported IoUs, equal edit counts and all four Gaofen-3 intersection changes at their original precision. Published files contain only aggregated values and coordinates at the eight-bin display scale.

## Public reproduction

Run `python draw_observed_cases.py --data case_aggregates.json --output rebuilt` with NumPy and Matplotlib. This writes the three paired-case figures in PDF and SVG, with editable vector paths and text. The public routine consumes only the aggregated JSON by default. Source-to-figure derivation of the native case plates is an author-review operation and remains local. No detector, head or policy was rerun for this display.

## Whole-crop comparison

The two gate-exposed Instance-RSDD hulls are nearest the median loss and median gain of SegRefiner-LR relative to its native input. The complete masks in these two examples use different supports and edit counts. The first row favors local ROH by 4.52 continuous-IoU points and the second favors SegRefiner-LR by 2.07 points. Both lower-end views use one shared window per hull. The source public plate comprises pooled coverage on an 8-by-8 grid with the reference contour at 25 percent. Rebuild the public, editable PDF and SVG from this directory with PyMuPDF.

~~~sh
python retained_hulls.py --source ../journal_20261126/rsdd74_segrefiner_cases_binned.pdf --out retained_hulls
~~~

## Development-set overlap

Supplementary Fig. S14 uses the recorded addition and removal intersections for two HRSID DEV705 hulls under RF-DETR fit 5217. The original JSON supplies the per-arm geometry totals, signed-count checks, median selection, eligibility counts and tie tolerance. The two cases have no image identifiers in the released record. Their numerical comparison can be reproduced from that record. [HRSID aggregate input](../dev705_overlap_cases.json)
