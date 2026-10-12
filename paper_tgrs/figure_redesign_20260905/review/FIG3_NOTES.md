# Fig. 3: boundary-patch refinement

## Deliverables

- `../fig3_bpr.tex`: standalone TikZ source, portable relative references to `../assets_fig3/` from this notes directory.
- `../fig3_bpr.pdf`: print-ready figure, 516 × 148 PDF pt (18.2033 × 5.2211 cm).
- `../previews/fig3_bpr_150dpi.png` and `../previews/fig3_bpr_gray_150dpi.png`: color and grayscale proofs.
- `fig3_checks.json`: repository six-gate validator output.
- `fig3_measurements.json`: additional geometry, font, stroke, asset, and SHA-256 measurements.

Compile from the deliverable directory using `pdflatex -interaction=nonstopmode -halt-on-error fig3_bpr.tex`. No external shared style file is required. The visual settings were checked against the source project's `tikz_style.tex` and `tikz_style_v5.tex`, but implemented locally to preserve exact physical sizes and only two stroke weights.

## Corrected caption, ready for adaptation

Boundary-patch refinement illustrated using a real HRSID inshore chip and coarse masks from a single-view frozen soup prediction. (a) A native 64 × 64 boundary window at 20× the overview scale. The stored coarse contour (red) and refined contour (green) are overlaid on the same grayscale SAR patch; these overlays are visualizations, while the UNet receives two separate channels: grayscale intensity and the binary coarse mask. The native pair is resized to 128 × 128 for the separately trained refiner, whose probabilities are reduced to the native grid and averaged where windows overlap. (b) The ±4-pixel morphological band, shown at the same 20× scale. Amber marks the only pixels eligible for writing; uncovered band pixels and all pixels outside it retain the original mask. An empty-mask fallback preserves the original instance, and scores and ranks are unchanged. The example window was selected by the asset generator for its number of changed pixels, so it illustrates the mechanism rather than typical accuracy.

## Provenance corrections to the supplied brief

The brief labels the overview “fused masks” and describes the example as the preceding consensus-vote rung. The actual manifest `_source` and `carrier/dump_figure_assets.py` use `carrier/configs/c0_maskiou_v2_m2f_hrsid.py`, checkpoint `runs/c0_maskiou_v2/soup_s023/iter_100872.pth`, and the ordinary single-view resize pipeline. The phrase “fusion head” in the asset tensor description refers to the detector's instance-output head; it does not establish cross-view consensus fusion. Accordingly, this version uses **coarse masks**, and the caption explicitly identifies a single-view example. No inference was run to make a substitute.

The original Fig. 3 painted the UNet gray. It is a separately trained corrector, so the revised module uses the prescribed green fill and short dashed border. “gray + mask” was added to prevent the display overlay from being mistaken for the actual neural-network input. These two wording changes are intentional departures from the brief's exact-string list, justified by the supplied implementation.

`carrier/bpr_apply.py:44–49` defines the band using four iterations of dilation and erosion with the square 3 × 3 kernel from `carrier/bpr_patches.py`. This is a morphological band; it is not an exact Euclidean signed-distance threshold. The caption therefore says morphological band. Native windows are 64 × 64, but the implementation passes 128 × 128 arrays through UNet (`bpr_apply.py:64–77`; `bpr_refiner.py:1–8`). “64 × 64 patch” means the window on the original image, not the network input tensor size.

`dump_figure_assets.py:348–358` ranks windows by the number of changed pixels and selects the maximum-change window by default. Its selected real example is not a random or typical sample. The figure makes no accuracy claim and contains no AP or other performance numbers.

## Exact assets and geometry

All four images are byte-for-byte copies of existing files under `E:/SAR/seal/paper_tgrs/figs/assets/`. The original files were not edited. Source and copied SHA-256 hashes match, as recorded in `fig3_measurements.json`.

| Copied asset | Meaning | Embedded pixels | Drawn size | Embedded dpi |
|---|---|---:|---:|---:|
| `assets_fig3/pred_masks.png` | Single-view coarse instance masks over SAR | 800 × 800 | 20 × 20 mm | 1016.0 |
| `assets_fig3/patch_before_x2.png` | Coarse contour over the native SAR patch | 1024 × 1024 | 32 × 32 mm | 812.8 |
| `assets_fig3/patch_after_x2.png` | Refined contour, same SAR window | 1024 × 1024 | 32 × 32 mm | 812.8 |
| `assets_fig3/band_annulus_x3.png` | Real writable band around instance 24 | 1026 × 1314 | 28.5 × 36.5 mm | 914.4 |

The insets are existing integer nearest-neighbor embedding copies. Their high file dpi does not add native spatial information: the patch remains 64 × 64 measurements and the band crop remains 57 × 73 measurements. Neither smoothing nor generated imagery was used. Their baked contour and amber colors are preserved exactly, including the original generator's amber alpha of 0.55.

The chip is `P0101_1200_2000_3000_3800.jpg`, HRSID image 1528. The coarse/refined window is `(y,x)=(536,0)` with native side 64. The band crop is `(y0,x0,y1,x1)=(524,0,597,57)`, using exclusive ending coordinates. Both nearly coincident locator rectangles are derived directly from those values. A solid vermillion locator denotes the patch; a dashed amber locator denotes the band.

The overview scale is 20 mm / 800 = 0.025 mm per native pixel. Both enlarged panels use 0.5 mm per native pixel, so **×20 is geometrically exact**, not a nominal display zoom. In page coordinates, the overview top left is `(12,62.8070866142)` PDF pt; adding column or row × 56.6929133858 / 800 yields each locator corner.

## Visual changes and validation

The module now has enough width to state its actual two-channel input, the before/after tiles share a baseline and equal native scale, and the gap before the band panel is reduced. Neutral magnification leaders distinguish a crop correspondence from the black inference arrows. The only filled colored module is the green trained refiner; the real amber support remains the most distinctive region, including in the grayscale proof.

| Gate | Result |
|---|---|
| Canvas | Exactly 516 × 148 PDF pt |
| Height / aspect ratio | 5.2211 cm ≤ 5.4 cm; 3.4865 ≥ 3.3 |
| Text | 23 whitespace tokens; no performance numbers |
| Font sizes | Exactly 7.000 and 8.000 physical PDF pt; two tiers |
| Fonts | Two embedded Type 1 Times-family fonts; no Type 3 |
| Strokes | 0.500 and 0.800 PDF pt; two tiers |
| Dashes | One short-dash pattern |
| Colors | Eight chrome/text colors including white; raster intensity/color values are real data and are not quantized |
| Lock / nested containers | One lock; no enclosing nested containers |
| Raster floor | 812.8 dpi, above 600 dpi |
| Repository validator | WIDTH, FONT, EMBED, RASTER, STROKE, OVERLAP all pass |
| Ink overlap | `_overlap.py`: TT = 0, TR = 0, TI = 0 |
| Visual inspection | Color and grayscale 150 dpi proofs inspected |

The source explicitly sets the Stealth head outline to 0.5 PDF pt while keeping arrow shafts at 0.8 pt. PGF otherwise silently caps the outline of the requested 1.6 × 1.2 mm Stealth geometry to approximately 0.765 pt, creating a third stroke tier. The head remains the specified Stealth family and dimensions. Fonts are stated in `bp` to meet the physical 7 pt floor; canvas dimensions are explicit lengths to avoid a 515.994 pt multiplication-rounding artifact.

The validator checks embedded raster dpi and vector stroke widths, not the recoverability of raw masks from colored overlays. These assets are adequate for an honest mechanism illustration but do not constitute an independent reproduction, a random-case evaluation, or proof of consensus-rung performance.
