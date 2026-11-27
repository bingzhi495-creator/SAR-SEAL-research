# NORA core figure revision 02

The five SVG/PDF pairs cover the overall interface, ROH matrices, F/L/I placement, fixed-total allocation and the frozen-Phoenix factorial control. `NORA_core_revision.drawio` contains five pages of editable shapes, paths and labels, rather than image-only wrappers. The manuscript uses the corresponding assets in `method_atlas_20261126` and `journal_20261126`.

```powershell
python draw_core_revision.py --output rebuilt
```

Requires local Python, NumPy, Matplotlib and PyMuPDF. `schematic_selection.json` records the illustrative endpoint scores and the enumerated F/L/I choices; these are explanatory values, not experimental measurements. The first-fit dual-reference result plot remains separate from the three-fit table. Quantitative figures retain their verified published inputs and plotting lineage.

Review the SVG/PDF at final manuscript size. The Draw.io XML structure was checked for five populated pages; desktop export of this companion was not executed. The SVG/PDF pairs are the visually verified rendering authority.

Native SAR figures remain in the local author-review workspace. Public assets contain schematic or aggregate vectors only. See [revision audit](../../design/CONTINUATION_AUDIT_20261126_02.md).

The overall network and ROH pages distinguish the scalar polygon–cell area fraction from the two binary cell labels used to fit the dual head. The first binary mask is the COCO rasterization of the matched PSeg polygon. The second is the matched object in the native SPSD panoptic mask. The dual sigmoid outputs estimate these labels. Use `--figure overview` or `--figure roh` to rebuild either diagram alone.

## Sentinel-1A input in the public overview

The left panel is a historical VV SAR observation of the Suez Canal on 25 March 2021. The remaining network stages illustrate the method. The public vector figure displays 32 rows by 28 columns of local mean brightness. Its [aggregate grid](suez_s1a_vv_20210325_grid.json) contains the acquisition identifier, crop and display definition. Contains modified Copernicus Sentinel data (2021). RTC processing by Catalyst, via Microsoft Planetary Computer (CC BY 4.0). [Source scene](https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-1-rtc/items/S1A_IW_GRDH_1SDV_20210325T154823_20210325T154848_037157_046001_rtc) · [Sentinel data terms](https://ewds.climate.copernicus.eu/licences/ec-sentinel) · [RTC collection license](https://creativecommons.org/licenses/by/4.0/).

The display uses γ⁰ in VV polarization. It converts intensity to dB, clips at −8 and +10 dB, and maps the interval to 0–255. Each drawn cell is the rounded mean of a disjoint 2×2 block of display brightness. Rows run north to south. The spatial aspect ratio uses local WGS84 ground distances at the crop center. Every grayscale cell in the public overview is an editable vector rectangle.

Run `python draw_core_revision.py --figure overview --output .` to regenerate the public PDF and SVG from the included grid. Run `python refresh_overview_drawio.py` to update the overview page in both workbooks. The [window-to-grid script](rebuild_suez_vv_grid.py) regenerates the aggregate from the published source using NumPy and tifffile. A separately admitted author-review image can be used through the atlas wrapper with its local output directory.
