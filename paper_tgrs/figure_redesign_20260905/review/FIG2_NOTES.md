# Fig. 2 — consensus-gated fusion

## Deliverables

- `fig2_consensus.tex` — editable standalone TikZ source; compile in the output folder.
- `fig2_consensus.pdf` — final, exactly 516 × 196 PDF points.
- `previews/fig2_consensus_150dpi.png` — 150 dpi color preview.
- `previews/fig2_consensus_grayscale.png` — 150 dpi grayscale proof.
- `previews/fig2_consensus_blur.png` — Gaussian-blur salience proof.
- `derive_fig2_assets.py` — deterministic derivation from existing native binary masks; imports no model or detector code.
- `assets_fig2/provenance.json` — mask identities, source SHA-256 hashes, exact pairwise intersections/unions, viewports, consistency assertions, and packaged-source manifest.
- `review/fig2_verification.json` and `review/fig2_metrics.json` — measured checks.

The source inputs needed to regenerate the figure are packaged under `assets_fig2/source_fusion/` and `assets_fig2/source_iou_matrix_meta.json`. The `.tex` file references only local packaged assets. Original project files were preserved.

## Caption suitable for insertion

Consensus-gated vote fusion illustrated by a deterministic replay on the 16 highest-scoring stored candidates from one HRSID inshore chip. Top: sampled TTA overlays and the pairwise mask IoU of the same 16 masks, reordered by score-seeded greedy clustering (white: zero; blue: one). Solid vermilion blocks mark supports in the voting band; dotted gray blocks mark the other supports. (a) The six-member subset cluster is pixel-voted. (b) The seven-member subset cluster keeps its seed; overlaying the seed contour on the rejected majority mask reveals the hull pixels that voting would remove. Both panels use 110 × 86 native-pixel viewports at 22.58 mm width, preserving the same magnification. The left edge in (a) is the actual image boundary. This top-16 replay illustrates the operator; it is not the full 800-candidate deployment, where these two seeds collect 10 and 9 members, respectively, and both keep the seed.

The last sentence is required for scientific accuracy with these supplied assets. It should not be deleted merely to shorten the caption.

## Scientific issues found and resolved

1. **The supplied matrix and fusion crops were not from identical stored candidates.** `figs/assets/fusion/manifest.json` explicitly reports `matrix_percent_equal=false`, `candidates_equal=false`, and `matched_iou_matrix_clusters=false`. A figure caption claiming one identical candidate pool for those two supplied asset groups would therefore be unsupported.
2. **The new matrix repairs that inconsistency without inference.** Each of the 16 native member masks was restored to its global 800 × 800 canvas using `window_yx`. Every restored area and bounding box equals its manifest record. Exact intersection/union yields a matrix whose greedy seed order and partition reproduce the four stored groups, of sizes 6, 7, 1, and 2. The source IDs equal the 16 IDs recorded in the fusion manifest. Its integer percentage values differ from the old matrix in two symmetric entries. All four stored vote-count arrays and gated fused masks were checked to be pixel-identical to arithmetic on these restored masks.
3. **The examples are a restricted-candidate replay, not deployment outcomes.** The source manifest says all 800 candidates are used in deployment; the same four seeds then gather 10, 9, 11, and 10 candidates. In particular, the displayed six-member example is a seed-preserving cluster in the complete pool. We have retained the required figure text and made the restricted replay explicit in the caption and source comments. This figure is evidence of operator mechanics; it cannot by itself establish where the deployed gains come from.
4. **Cluster cardinality does not generally equal distinct-view support.** Full-pool metadata contains same-view duplicates: some in-band clusters have two or more candidates from only one view. The gate uses candidate cardinality. The manuscript should not call `|C|` an exact count of agreeing views or imply that seven/eight candidates means near-unanimous agreement across eight views unless a one-candidate-per-view constraint is actually implemented and verified.
5. **The original in-band crop is 266 × 266, not 118 × 118.** A remote, single-vote component creates the long union bounding box. The redesigned illustration shows an exact 110 × 86 viewport, excluding that remote speck from the close-up; all matrix and voting arithmetic retain every original pixel. The vessel touches the source chip's left boundary, so that truncation remains visible and is disclosed.

## Visual changes

The candidate pool is a flat montage of real view overlays, avoiding the old tensor-stack metaphor for a set of instances. Both lower branches now use equal-sized rectangular viewports at exactly the same native scale, making the seed-versus-rejected-vote comparison larger and easier to inspect. A wider evidence chain uses the full upper row, while aligned panels and restrained line styles keep the decision structure legible in grayscale.

The seed contour in the counterfactual panel is an exact 114-vertex pixel-center boundary extracted from the provided native seed mask with `CHAIN_APPROX_SIMPLE`, without smoothing or hand adjustment. The solid-versus-dotted border and crossed dashed path retain the intended-versus-rejected distinction when color is removed.

## Measured acceptance checks

| Check | Measured value | Result |
|---|---:|---|
| PDF page | 516.000 × 196.000 pt | Pass |
| Physical page | 18.2033 × 6.9144 cm | Pass |
| Aspect ratio | 2.63265:1 | Pass, requested 2.5–2.7 |
| Words | 35 alphabetic tokens; 45 whitespace tokens including numbers/symbols | Pass, ≤60 |
| Accuracy numbers in figure | 0 | Pass |
| Font sizes at final size | 7.000 and 8.000 PDF pt only | Pass |
| Embedded fonts | Seven Type 1 subsets; no Type 3 | Pass |
| Vector stroke widths | 0.500 and 0.800 PDF pt only | Pass |
| Dash patterns | Two non-solid patterns | Pass |
| Vector drawing colors | Four; raster colors excluded | Pass |
| Locks | 0 | Pass |
| Nested containers | 0 | Pass |
| Embedded raster instances | 11 | Measured |
| Minimum embedded sampling density | 768 dpi, matrix | Pass, ≥600 |
| Lower-panel sampling density | 990 dpi | Pass |
| Flow arrows | Orthogonal; one Stealth design | Pass |
| Ink-level text/text, text/rule, text/raster collisions | 0 / 0 / 0 | Pass |
| Visual inspection | Color, grayscale, and blur previews inspected | Completed |
| Independent human one-second reading test | Not performed | Unverified subjective gate |

Raster dpi describes embedded output sampling, not new source information. The matrix is still 16 × 16 measured values, nearest-neighbor enlarged; lower panels are 110 × 86 measured pixels, nearest-neighbor enlarged. This duplication preserves original pixels and does not claim additional spatial resolution. Salience remains constrained by the naturally high contrast of real SAR and binary masks; no invented color blocks were added to increase color occupancy.

Commands run successfully:

```powershell
& 'C:/Users/houbingzhi/anaconda3/python.exe' 'derive_fig2_assets.py'
& 'E:/miktex/miktex/bin/x64/pdflatex.exe' -interaction=nonstopmode -halt-on-error 'fig2_consensus.tex'
& 'C:/Users/houbingzhi/anaconda3/python.exe' '../figs/verify_fig.py' --target 516 --min-pt 7 --min-dpi 600 --json 'review/fig2_verification.json' 'fig2_consensus.pdf'
& 'C:/Users/houbingzhi/anaconda3/python.exe' '../figs/_overlap.py' 'fig2_consensus.pdf'
```

MiKTeX prints an unsupported-Windows notice after successful compilation; exit status is zero and the PDF passed both font-embedding checks.

## Embedded-asset inventory

All original sources below are under `E:/SAR/seal/paper_tgrs/figs/assets/`.

| Embedded local file | Original source / derivation |
|---|---|
| `assets_fig2/views_0.png` | `fusion/views_0.png` |
| `assets_fig2/views_2.png` | `fusion/views_2.png` |
| `assets_fig2/views_4.png` | `fusion/views_4.png` |
| `assets_fig2/views_6.png` | `fusion/views_6.png` |
| `assets_fig2/iou_matrix_x48.png` | Recomputed from `fusion/inband6/members_0.png` … `members_5.png`; `fusion/outband7/members_0.png` … `members_6.png`; `fusion/single1/members_0.png`; `fusion/pair2/members_0.png`, `members_1.png`; 16 × 16 native matrix enlarged ×48 by nearest-neighbor |
| `assets_fig2/inband_members.png` | `fusion/inband6/members_contours_x4.png`, native viewport `(x0,y0,x1,y1)=(0,180,110,266)`, exact ×2 duplication of the crop |
| `assets_fig2/inband_votes.png` | `fusion/inband6/vote_count_color_x4.png`, same viewport and duplication |
| `assets_fig2/inband_fused.png` | `fusion/inband6/fused_x4.png`, same viewport and duplication |
| `assets_fig2/outband_members.png` | `fusion/outband7/members_contours_x8.png`, native viewport `(0,12,110,98)`, no further resampling |
| `assets_fig2/outband_seed.png` | `fusion/outband7/seed_x8.png`, same viewport |
| `assets_fig2/outband_counterfactual.png` | `fusion/outband7/vote_would_be_x8.png`, same viewport |
| `assets_fig2/seed_contour.tex` (vector) | Exact pixel-center boundary of `fusion/outband7/seed.png`, mapped into the same counterfactual viewport |

Additional native `vote_count.png` and `fused.png` files for all four clusters are packaged solely to reproduce the consistency assertions; they are not extra images in the PDF. Source manifests and the old matrix metadata are retained as audit evidence. SHA-256 values for all packaged originals appear in `assets_fig2/provenance.json`.
