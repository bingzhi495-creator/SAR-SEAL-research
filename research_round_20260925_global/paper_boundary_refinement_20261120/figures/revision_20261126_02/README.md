# NORA core figure revision 02

The five SVG/PDF pairs cover the overall interface, ROH matrices, F/L/I placement, fixed-total allocation and the frozen-Phoenix factorial control. `NORA_core_revision.drawio` contains five pages of editable shapes, paths and labels, rather than image-only wrappers. The manuscript uses the corresponding assets in `method_atlas_20261126` and `journal_20261126`.

```powershell
python draw_core_revision.py --output rebuilt
```

Requires local Python, NumPy, Matplotlib and PyMuPDF. `schematic_selection.json` records the illustrative endpoint scores and the enumerated F/L/I choices; these are explanatory values, not experimental measurements. The first-fit dual-reference result plot remains separate from the three-fit table. Quantitative figures retain their verified published inputs and plotting lineage.

Review the SVG/PDF at final manuscript size. The Draw.io XML structure was checked for five populated pages; desktop export of this companion was not executed. The SVG/PDF pairs are the visually verified rendering authority.

Native SAR figures remain in the local author-review workspace. Public assets contain schematic or aggregate vectors only. See [revision audit](../../design/CONTINUATION_AUDIT_20261126_02.md).

The overall network and ROH pages distinguish the scalar polygon–cell area fraction from the two binary cell labels used to fit the dual head. The first binary mask is the COCO rasterization of the matched PSeg polygon. The second is the matched object in the native SPSD panoptic mask. The dual sigmoid outputs estimate these labels. Use `--figure overview` or `--figure roh` to rebuild either diagram alone.
