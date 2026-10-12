# Built-in imagegen prompt and selection record

All artworks were produced with OpenAI's built-in image-generation tool, not the CLI. User requested CVPR/TPAMI-style scientific visualization during revision. Earlier block-flow drafts are superseded. No generated pixels were inserted into empirical SAR/mask/PR evidence.

Selected for manuscript: cvpr_overview_compact_v2, cvpr_fusion_v2, cvpr_bpr. Expanded conceptual plate: cvpr_overview_v2 (not used at journal print size). Native raster dimensions are preserved without upsampling; main figure placement is bounded to maintain at least 300 effective DPI.

## overview

Create a polished publication-quality conceptual system diagram for an IEEE TGRS research paper. This is a scientific schematic, NOT an empirical visualization. Wide landscape canvas approximately 2.7:1, white background, crisp flat vector-like illustration, restrained palette (blue #0072B2 for data, dark gray #555555 for carrier, vermilion #D55E00 for deterministic processing, green #009E73 for learned correction), clean sans-serif large legible English text, thin dark outlines, generous spacing, NO gradients, NO 3D, NO shadows, NO decorative ocean/ships, NO numerical performance claims, NO invented SAR images or heatmaps. No external overall title.
The diagram must have exactly two horizontal bands.
Top narrow band headed "TRAINING". From left to right three rounded boxes: "COCO initialization"; arrow to "SAR fine-tuning" with smaller line "Segmentation + mask-quality losses"; arrow to "Three seed checkpoints"; arrow to "Weight averaging". A single downward arrow from Weight averaging reaches a gray box labeled "Frozen carrier" near the left of the lower band. A small closed-padlock icon by Frozen carrier indicates weights frozen AFTER training.
Bottom larger band headed "INFERENCE". Leftmost blue card "SAR image" points into "Frozen carrier" (the carrier is the exact box reached by the vertical Weight averaging arrow). Then flow strictly left to right: orange box "Multi-scale + flip views" with smaller line "4 scales × 2 flips"; orange box "Cluster-support-gated fusion" with smaller line "Vote or keep seed"; green box "Boundary refinement" with smaller line "Update mask pixels"; green box "Score recalibration" with smaller line "Update confidence"; blue output card "Scored instances".
Draw only this topology, no bypasses, no arrows back into training. Put a small note below the diagram in readable dark gray: "Conceptual illustration; empirical results are reported separately."
Make all labels exact, every word spelled correctly. Green means separately trained correction modules; add small green legend text "Learned correction" and small orange legend text "Deterministic processing" at bottom left above the note. Do not include neural-network tensor dimensions, measured pixel masks, curves, graphs, probabilities, equations, or benchmark scores. Use typography and structured layout suitable for the main architecture figure in a serious remote sensing journal.

Generated images are saved to C:\Users\houbingzhi\.codex\generated_images\01a06fb0-8d09-7551-a0c9-fe43804c0ff9 as C:\Users\houbingzhi\.codex\generated_images\01a06fb0-8d09-7551-a0c9-fe43804c0ff9\exec-72aeeace-d6ef-4633-a262-a8f0e1796632.png by default.
If you need to use a generated image at another path, copy it and leave the original in place unless the user explicitly asks you to delete it.
The generated image is already displayed to the user. There is no need to render it in the final response as a Markdown image or file link.

## overview_v2

Revise the attached scientific diagram and fix its architecture, keeping the crisp white background, flat outlined boxes and blue/orange/green/gray palette. Render at the highest available native resolution, preferably at least 2400 pixels wide, landscape 2.7:1. Large readable type. Remove tiny legends and all footnotes; the journal caption will explain them.
Use TWO rows, with exactly the following connections and labels:
TOP row labeled "TRAINING": three horizontally arranged boxes occupying the LEFT TWO THIRDS of the canvas: "COCO initialization" → "SAR fine-tuning" (subtitle "Segmentation + mask-quality losses") → "Weight averaging" (subtitle "Three seed checkpoints"). Leave top right clear. The ONLY parameter connection between rows starts at the BOTTOM CENTER OF THE WEIGHT AVERAGING BOX, goes down through the gap between rows, then LEFT horizontally if needed, then DOWN into the TOP CENTER of the multi-view inference box in the lower row. It must visibly originate from Weight averaging, never from initialization or the row border. Use a dashed line and arrowhead for this parameter connection.
BOTTOM row labeled "INFERENCE": five main nodes in a left-to-right data flow: blue "SAR image" → gray "Multi-view inference" (subtitle line 1 "Frozen Mask2Former", subtitle line 2 "4 scales × 2 flips", small padlock) → orange "Gated mask fusion" (subtitle "Vote or keep seed") → green "Boundary refinement" (subtitle "Update mask pixels") → green "Score recalibration" (subtitle "Update confidence") → blue "Scored instances".
The gray multi-view inference box represents augmentation AND repeated forward passes together. There must be NO separate Frozen carrier box and NO separate Multi-scale + flip views box. No numeric AP results, no data thumbnails, no graph plots. All arrows in lower row are solid left-to-right. All lowercase/uppercase and spellings exactly as requested. No connection from any training row border, no connection from COCO initialization directly into inference.
At bottom, only a compact legend "Dashed arrow: averaged parameters" may appear. Keep diagram balanced and compact for journal reproduction, generous white space, all text large enough to read. This is conceptual artwork only, not measured data.

Generated images are saved to C:\Users\houbingzhi\.codex\generated_images\01a06fb0-8d09-7551-a0c9-fe43804c0ff9 as C:\Users\houbingzhi\.codex\generated_images\01a06fb0-8d09-7551-a0c9-fe43804c0ff9\exec-e53163c4-d978-4808-af30-13c7bd8a8bef.png by default.
If you need to use a generated image at another path, copy it and leave the original in place unless the user explicitly asks you to delete it.
The generated image is already displayed to the user. There is no need to render it in the final response as a Markdown image or file link.

## fusion

Create a scientific algorithm schematic for an IEEE Transactions on Geoscience and Remote Sensing paper, precise English labels, pure white background, clean flat vector-like shapes, high native resolution at least 2400px wide if supported, wide landscape 2.8:1. Large type, thin crisp outlines, no gradients, no shadows, no artistic scene. This is a CONCEPTUAL flowchart: no SAR photographs, no fabricated masks, no fake heatmap, no graphs, no AP numbers.
Main flow left to right: blue outlined box "Aligned candidates" → dark gray outlined box "Greedy mask-IoU clustering" (subtitle "Highest-score seed; threshold 0.5") → orange outlined decision diamond with text "2 ≤ cluster size ≤ 6?".
From decision diamond two clearly labeled outgoing branches: upper branch "Yes" leads to an orange box "Pixel vote" (subtitle "Foreground on ties"); lower branch "No" leads to a gray box "Keep seed mask". Both branch boxes feed one blue outlined output box at the far right "One mask per cluster" (subtitle "Inherit seed score").
At the bottom, a single very legible note: "Cluster size counts candidates, not distinct views."
Show arrowheads consistently pointing forward. Do not display any example measured candidate pool or IoU matrix; association and threshold are algorithm settings. Make both branches fit without crossing, with equal visual weight and minimal text. Exactly one decision diamond, exactly the described five rectangular boxes (including the two branch alternatives) plus output. Black/dark-gray text with blue/orange accents. Publication-ready conceptual artwork.

Generated images are saved to C:\Users\houbingzhi\.codex\generated_images\01a06fb0-8d09-7551-a0c9-fe43804c0ff9 as C:\Users\houbingzhi\.codex\generated_images\01a06fb0-8d09-7551-a0c9-fe43804c0ff9\exec-3edd3734-2699-4442-b332-a644895e5a45.png by default.
If you need to use a generated image at another path, copy it and leave the original in place unless the user explicitly asks you to delete it.
The generated image is already displayed to the user. There is no need to render it in the final response as a Markdown image or file link.

## bpr

Create a precise conceptual scientific schematic for an IEEE TGRS paper showing boundary-patch refinement. White background, flat clean vector-like artwork, thin gray outlines, green #009E73 for trained UNet, orange #D55E00 for write constraint, blue #0072B2 for inputs and output. Highest native resolution supported, prefer 2400px wide, landscape 3:1. Very legible English labels, no shadows, no gradients. No empirical SAR imagery, no fabricated pixel mask examples, no invented performance measurements.
Draw a horizontal flow of six labeled boxes, from left to right:
1 blue "Image + coarse mask" with subtitle "Native 64 × 64 patches"
2 gray "Resize to 128 × 128" with subtitle "Two input channels"
3 green "UNet" with subtitle "Separately trained"
4 gray "Sigmoid + resize" with subtitle "Back to 64 × 64"
5 gray "Average probabilities" with subtitle "Across overlapping patches"
6 orange "Band-constrained update" with subtitle "Threshold at 0.5"
then a small arrow to a blue label "Refined mask".
Under this chain, draw a single thin orange bypass arrow from the bottom of the FIRST box, running below the other boxes and entering the bottom of the LAST orange box. Label this bypass "Original mask outside the covered boundary band". It preserves original pixels.
Add ONE short bottom note in large type: "Scores and detection records are retained."
Use a tiny closed padlock icon next to the bypass note to communicate the write constraint. The band is a discrete morphological band defined by four iterations of 3×3 square dilation and erosion, but do not write that long definition in the image; it will be in the caption. Do not create an image-like measured ring or sample hull. Six pipeline boxes plus output label only. Clear left-to-right arrowheads between adjacent boxes, no feedback to UNet. All text spelled exactly and make clear probabilities (NOT logits) are averaged. Use two text lines in long boxes and ample padding to prevent overlap.

Generated images are saved to C:\Users\houbingzhi\.codex\generated_images\01a06fb0-8d09-7551-a0c9-fe43804c0ff9 as C:\Users\houbingzhi\.codex\generated_images\01a06fb0-8d09-7551-a0c9-fe43804c0ff9\exec-d85586ed-a513-4d92-a4af-1606c094b7bf.png by default.
If you need to use a generated image at another path, copy it and leave the original in place unless the user explicitly asks you to delete it.
The generated image is already displayed to the user. There is no need to render it in the final response as a Markdown image or file link.

## cvpr_overview

Design a sophisticated, visually rich computer-vision research architecture figure at the visual standard of a strong CVPR / ICCV / TPAMI paper. This must look like a carefully authored scientific neural-network diagram, not a generic corporate flowchart. High native resolution, wide landscape approximately 2.3:1. Pure white background, subtle pale lavender / ice blue / mint / apricot panel backgrounds, deep slate text, thin meticulous connectors, restrained orange highlights, coherent professional typography. Flat orthographic or very shallow axonometric tensor stacks are encouraged for meaningful scientific structure. NO glossy 3D, no shadows, no gradients, no clip-art ship, no decorative ocean, no fake performance charts. All mask tiles are SIMPLE ABSTRACT SCHEMATIC GLYPHS (elongated ship silhouettes on plain grid tiles), never pretend to be recorded SAR observations. Include small text "Schematic" near these tiles.

Overall structure: an elegant main horizontal pipeline across the upper 60% with five visually distinct modules. Use narrow panel headers and varied meaningful internal illustrations, not just empty rectangular boxes.
(a) left panel header "Multi-view inference": small grayscale input tile labeled "SAR input"; fan out into eight tiny tiles in a 2×4 grid labeled "4 scales × 2 flips"; arrows into a stylized neural carrier drawn as a 4-level feature pyramid feeding a small mask decoder, labeled "Frozen Mask2Former". Put a small padlock by the carrier. Its output is a layered stack of tiny binary-mask glyph cards labeled "Candidates".
Above the carrier, a small detached training inset: three checkpoint cubes merging to one cube labeled "Weight soup", with dashed parameter arrow into the frozen carrier; small note "After SAR fine-tuning". Do not claim the carrier was never trained.
(b) next panel header "Gated mask fusion": show a small cluster of overlapping schematic mask cards converging at an orange gate symbol, then one selected mask card. A short arrow links Candidates to this panel.
(c) next panel header "Boundary refinement": show a coarse outlined elongated mask glyph, a small green UNet icon, then a smoother mask glyph; emphasize only a narrow orange boundary ring on the coarse glyph. Small label "Local mask update".
(d) next slim panel header "Quality re-ranking": a vertical set of three score bars or numbered candidate cards reordered (no numeric performance axis), label "Score update".
(e) final small output panel: three non-overlapping colored elongated instance glyphs with small confidence tags represented by short bars, label "Scored instances".

Lower 40%: two aligned magnification insets, connected by thin dashed guides to panels b and c.
Left inset title "(b) Cluster-support gate": split into two clean mini branches with small abstract mask stacks. Label top "2 ≤ |C| ≤ 6" → "Pixel vote"; bottom "|C| = 1 or |C| > 6" → "Keep seed". One note at bottom "Candidate count ≠ distinct-view count". Voting glyphs may show orange contour agreement; do not display measured IoU heatmap.
Right inset title "(c) Boundary write constraint": an enlarged simple elongated outline with an orange morphological band around its edge; inside and outside the band shaded faint gray, small padlock; green arrows indicate only the band can update. Alongside draw tiny two-channel patch stack → UNet → probability map glyph, label "64² → 128² → 64²". Small note "Average probabilities; preserve other pixels".

Use large, crisp, limited English text exactly as specified; avoid adding prose paragraphs or needless technical text. Scientific visual texture should come from meaningful tensor stacks, branching views, mask glyphs, zoom connectors, and constrained boundary illustration. Preserve clear left-to-right flow and no crossing arrows. The two learned modules are green, the deterministic fuser orange, frozen carrier lavender-gray, data blue. Do not add AP scores, quantitative results, datasets measurements, model layers not requested, or a big title. The image is an AI-assisted conceptual architecture schematic.

Generated images are saved to C:\Users\houbingzhi\.codex\generated_images\01a06fb0-8d09-7551-a0c9-fe43804c0ff9 as C:\Users\houbingzhi\.codex\generated_images\01a06fb0-8d09-7551-a0c9-fe43804c0ff9\exec-83d47816-46ca-463a-9271-8ed24b656cd4.png by default.
If you need to use a generated image at another path, copy it and leave the original in place unless the user explicitly asks you to delete it.
The generated image is already displayed to the user. There is no need to render it in the final response as a Markdown image or file link.

## cvpr_overview_v2

Refine the attached CVPR-style architecture figure. Keep its rich scientific composition, five top panels, two bottom zoom panels, meaningful tensor stacks and restrained pastel palette. Fix scientific annotation errors precisely; this revision must be suitable as a journal conceptual figure. Highest native resolution, crisp type.
1. Add a small clearly visible top-center label "Conceptual schematic". All input and mask drawings are conceptual, not empirical SAR measurements.
2. In the "Frozen Mask2Former" box, change the four feature labels from P3, P4, P5, P6 to C5, C4, C3, C2 respectively from top (smallest) to bottom (largest). Do not add other pyramid labels.
3. Replace the faucet/tap-like icon in the orange "Gated mask fusion" module by a clean mathematical decision diamond with only the symbol "|C|". A water-tap icon is scientifically inappropriate. Preserve the incoming mask-card arrows and outgoing mask.
4. In the lower right zoom, the "Two-channel patch stack" must show EXACTLY TWO separated abstract tiles: the upper tile is a grayscale intensity schematic labeled "Image"; the lower tile is a simple white binary elongated mask on black labeled "Mask". The upper tile must not be another binary mask. Place a small "Schematic" label directly below both tiles. Avoid adding fake noisy real SAR texture.
5. Remove the redundant extra "64²" under the patch stack; keep only the single dimension sequence "64² → 128² → 64²" under the network.
6. Change the top header for (d) Quality re-ranking from yellow/orange to the same green family as boundary refinement, because both are separately trained correction modules.
7. Replace first input label "SAR input" by "SAR input (schematic)".
8. Keep the exact inequalities in the lower left: voting when 2 ≤ |C| ≤ 6, keep seed when |C| = 1 or |C| > 6. Do not alter them.
Use crisp dark type, pure white outside panels. No performance numbers, no extra invented neural layers, no title banner. Keep all scientific arrows clean.

Generated images are saved to C:\Users\houbingzhi\.codex\generated_images\01a06fb0-8d09-7551-a0c9-fe43804c0ff9 as C:\Users\houbingzhi\.codex\generated_images\01a06fb0-8d09-7551-a0c9-fe43804c0ff9\exec-a1e223ba-3614-4097-ba0c-a96f7f764037.png by default.
If you need to use a generated image at another path, copy it and leave the original in place unless the user explicitly asks you to delete it.
The generated image is already displayed to the user. There is no need to render it in the final response as a Markdown image or file link.

## cvpr_overview_compact

Create a publication-ready neural-network architecture illustration with the refined visual language of CVPR / TPAMI, as a SINGLE WIDE ROW. Wide landscape canvas around 3.2:1, high native resolution. White background, thin slate outlines, very pale lavender for frozen network, pale orange for mask fusion, pale mint for learned correction, blue for data. Large readable text only: do not add any label not explicitly listed. No overall title. All miniature masks are abstract elongated ship-shaped glyphs, and all are conceptual illustrations, not empirical observations.

Exactly four main panels in one horizontal row, with clear arrows joining them:
PANEL 1, about 38% width, header "(a) Candidate generation". Meaningful visual content: at far left an input glyph, then a 2×4 grid of eight mini view tiles, then a frozen network illustrated by four stacked feature maps feeding a decoder block, then a stack of candidate masks at the panel right edge. ONLY interior labels: "SAR image", "8 views", "Frozen carrier", "Candidates". Small padlock by the carrier. A small inset ABOVE the frozen network shows three small checkpoint cubes converging into one cube, text "Weight soup", with a dashed arrow from merged cube INTO carrier. No feature-level indices, no tensor dimensions, no tiny extra notation. This is a diagram of view transforms followed by repeated carrier inference, not output augmentation.
PANEL 2, about 24% width, header "(b) Gated fusion". Two mini mask stacks merge through a clean orange diamond labeled "|C|"; tiny visual branches show a composite vote and an unchanged seed card, recombining to one output mask. ONLY additional labels: "Vote" and "Keep seed". No faucet icon.
PANEL 3, about 24% width, header "(c) Boundary refinement". A coarse mask outline with narrow orange ring, a small green UNet hourglass icon, then a refined mask outline. Draw a small square crop callout from coarse boundary into a two-tile patch stack before UNet. ONLY labels "UNet" and "Boundary band". Do not imply the diagram is measured before/after data.
PANEL 4, about 14% width, header "(d) Re-ranking". Two columns of THREE small colored mask cards connected by crossing permutation arrows; mask colors blue/green/orange, order changes but silhouettes remain identical. Tiny horizontal bars beside the cards represent scores with no numbers. ONLY labels "Update scores" and at far right of panel "Output". This module is green because it is separately trained.

ONE centered caption-like label below all four panels: "Conceptual illustration".
Everything should be crisp and scholarly, with visual richness from tensor stacks, masked glyphs, crop links and score permutations, not ornament. No lower inset row, no large whitespace at top/bottom, no detailed equations, no AP numbers, no realistic SAR photo. Text should remain legible when the entire figure is printed 180 mm wide; prioritize large labels and minimum 30px lettering over micro details. No gradients or glossy effects.

Generated images are saved to C:\Users\houbingzhi\.codex\generated_images\01a06fb0-8d09-7551-a0c9-fe43804c0ff9 as C:\Users\houbingzhi\.codex\generated_images\01a06fb0-8d09-7551-a0c9-fe43804c0ff9\exec-be2d8bfb-608e-4af6-8c01-9edc9bd9663b.png by default.
If you need to use a generated image at another path, copy it and leave the original in place unless the user explicitly asks you to delete it.
The generated image is already displayed to the user. There is no need to render it in the final response as a Markdown image or file link.

## cvpr_overview_compact_v2

Edit this CVPR-style compact architecture figure with ONLY the following important scientific repairs. Preserve the four-panel layout, typography, pastel palette, and scientific visual richness.
1. Under "8 views" there are currently six tiles: replace this with EXACTLY EIGHT tiles in TWO ROWS and FOUR COLUMNS. Four top-row tiles show the same upright diagonal elongated silhouette at four different sizes. Four bottom-row tiles are horizontal flips of the corresponding top-row tile. Do NOT include arbitrary rotations. Keep the heading "8 views".
2. In panel (b), the |C| diamond currently incorrectly connects directly to the bottom output. Remove that direct diamond-to-output connector. Draw EXACTLY TWO outgoing connections from the diamond: left connection into the "Vote" card, right connection into the "Keep seed" card. Only these two cards then merge into the bottom output mask. Add a small but readable "2–6" by the vote branch and "otherwise" by the keep-seed branch. Thus the topology is diamond -> alternative card -> output, never diamond -> output directly.
3. In panel (c), the top patch tile immediately before UNet should be a binary coarse-mask crop, clearly labeled "Mask"; the lower patch tile is grayscale intensity, labeled "Image". Keep the crop guide. The ring is a conceptual write band, not a measured gain.
4. Remove internal ship deck holes/details from all glyphs; use clean solid elongated silhouettes to make these abstract instance masks rather than ship clip art.
5. Preserve the bottom label "Conceptual illustration". No new AP numbers or claims. Keep main text large, white background and high-resolution clean outlines. Do not change any other module or arrow.

Generated images are saved to C:\Users\houbingzhi\.codex\generated_images\01a06fb0-8d09-7551-a0c9-fe43804c0ff9 as C:\Users\houbingzhi\.codex\generated_images\01a06fb0-8d09-7551-a0c9-fe43804c0ff9\exec-2eff27d6-85ea-4399-a5c1-02e917e223b2.png by default.
If you need to use a generated image at another path, copy it and leave the original in place unless the user explicitly asks you to delete it.
The generated image is already displayed to the user. There is no need to render it in the final response as a Markdown image or file link.

## cvpr_fusion

Create a refined computer-vision methods figure for a CVPR/TPAMI-style paper: richly visual, scientifically precise, pure white canvas, wide landscape 2.7:1, very pale blue/orange/gray panel backgrounds, thin contours, clean typography and large labels. Show three labeled panels in one row. This is a conceptual diagram, and the bottom must say "Schematic masks; no measured IoU or accuracy".
(a) left third header "Mask-space association". Show three elongated clean abstract binary masks as translucent blue/orange contour layers on a faint grid. Two masks overlap one elongated target, a third nearby separate elongated target. Depict their bounding rectangles overlapping, but the third target has disjoint foreground pixels. To the right of these masks within panel show two small stacks of candidate mask cards separating into two groups. Only small text "Greedy seed clustering". No numeric IoU matrix and no empirical heatmap. Don't imply a learned clustering network.
(b) middle third header "Vote within the support band". Draw three mask cards as a visual ellipsis-based depiction of a cluster, converging into an orange overlay contour glyph and then a single output card. Put the exact large formula "2 ≤ |C| ≤ 6" above this operation. Below place "Foreground on ties". The three input masks are slightly different contours of one elongated shape; the resulting conceptual overlay is not claimed to be more accurate.
(c) right third header "Keep seed outside the band". Draw a stack of pale gray mask cards and highlight one top card with a dark blue border and tiny crown-free simple dot marker. A clear arrow copies that exact top-card shape to one output card. Put exact condition "|C| = 1 or |C| > 6" above it. Below place "Seed mask and score".
Along the bottom above the schematic disclaimer a compact orange-to-blue line reads "Cluster size counts candidates, not distinct views." Use meaningful geometry and mask layers, not generic box flowcharts; no 3D heavy blocks, no deck or boat details, no realistic SAR photos, no performance plots, no fabricated numeric examples. Make panel headers large and crisp. Use diagram typography suitable for 180mm print width.

Generated images are saved to C:\Users\houbingzhi\.codex\generated_images\01a06fb0-8d09-7551-a0c9-fe43804c0ff9 as C:\Users\houbingzhi\.codex\generated_images\01a06fb0-8d09-7551-a0c9-fe43804c0ff9\exec-828d54ac-925f-4bee-a82e-96db70d526db.png by default.
If you need to use a generated image at another path, copy it and leave the original in place unless the user explicitly asks you to delete it.
The generated image is already displayed to the user. There is no need to render it in the final response as a Markdown image or file link.

## cvpr_fusion_v2

Edit this scientific fusion diagram while keeping its three-panel CVPR-style layout and palette. Precisely fix these visual semantics:
- In center panel (b), every blue input silhouette currently slopes differently from the orange voted mask. Redraw all THREE blue input silhouettes so their long axes have EXACTLY the same orientation as the orange central overlay and orange output: top left to bottom right, steeply diagonal about 60 degrees relative to horizontal. Their native-grid locations and scale must match. Only tiny boundary differences vary; pixel voting must NEVER rotate a mask. Preserve the exact mathematical condition 2 ≤ |C| ≤ 6 and the text Foreground on ties.
- In right panel (c), the highlighted seed silhouette and the output silhouette must be identical in shape, orientation and scale. Keep seed-copy arrow and blue dot marker.
- Break header (b) into two lines "(b) Vote within" / "the support band"; break header (c) into two lines "(c) Keep seed" / "outside the band" so neither touches the panel edge.
- Keep all contents explicitly schematic and all inequalities unchanged.
- No new performance values or empirical heatmap. Do not change left panel or layout beyond adding proper padding for wrapped headings.

Generated images are saved to C:\Users\houbingzhi\.codex\generated_images\01a06fb0-8d09-7551-a0c9-fe43804c0ff9 as C:\Users\houbingzhi\.codex\generated_images\01a06fb0-8d09-7551-a0c9-fe43804c0ff9\exec-a9c673e6-dd99-4fb9-b89b-9e391b8a3098.png by default.
If you need to use a generated image at another path, copy it and leave the original in place unless the user explicitly asks you to delete it.
The generated image is already displayed to the user. There is no need to render it in the final response as a Markdown image or file link.

## cvpr_bpr

Create a visually refined boundary-refinement architecture illustration for a strong CVPR / TPAMI paper. Scientific schematic, wide landscape 2.7:1, white background, very pale blue/green/orange panels, thin exact-looking contours, clean professional large typography, high native resolution. Visual interest comes from meaningful crop callouts, two-channel tensors, UNet skip connections, and a boundary write mask. No corporate flowchart, no glossy 3D, no realistic generated SAR texture, no performance plots.

Three panels left to right:
(a) header "Boundary patches". Show one large abstract elongated ship-instance silhouette with a jagged gray coarse contour. Highlight a NARROW ORANGE BAND that straddles this contour on both its inner and outer sides. Mark a small square on the boundary, and draw two dashed zoom guides down to a pair of stacked crop tiles. Upper tile is a smooth grayscale intensity schematic labeled "Image". Lower tile is a sharp binary black/white coarse-mask schematic labeled "Mask". Large label "64 × 64". These are conceptual glyphs.
(b) header "Local refinement". Left pair of patch tiles → resize symbol → GREEN UNet depicted as a U-shaped encoder-decoder with THREE shallow levels of decreasing and then increasing feature-map stacks and thin same-level horizontal skip-connection arrows. Main label "UNet". Under input label "128 × 128". Output is a soft grayscale probability-map glyph labeled "Sigmoid". Then a downward arrow into a set of overlapping probability tiles, label "Resize to 64 × 64". Small bottom label "Average probabilities". Never say average logits.
(c) header "Constrained write". Show three aligned miniature panels within this section: gray input contour plus orange thin boundary band; a green local update glyph; final outlined mask with orange only at the boundary. A small closed padlock in the interior and a second padlock in the exterior show pixels outside the orange writable band are retained. Use an orange arrow from averaged probabilities in panel (b) to this boundary band. ONLY labels "Covered band", "Other pixels fixed", and at bottom "Scores unchanged". Do not draw any learned correction outside the band. Do not show a guarantee of better accuracy.

One compact footer centered across the complete figure: "Conceptual illustration; recorded examples are provided separately."
Arrow flow must be left-to-right for all data. Render crisp, tasteful and balanced, large labels readable in a two-column journal. Avoid extra equations, legends, abbreviations or quantitative result numbers.

Generated images are saved to C:\Users\houbingzhi\.codex\generated_images\01a06fb0-8d09-7551-a0c9-fe43804c0ff9 as C:\Users\houbingzhi\.codex\generated_images\01a06fb0-8d09-7551-a0c9-fe43804c0ff9\exec-2367924f-1498-4220-888c-20c94d514219.png by default.
If you need to use a generated image at another path, copy it and leave the original in place unless the user explicitly asks you to delete it.
The generated image is already displayed to the user. There is no need to render it in the final response as a Markdown image or file link.

