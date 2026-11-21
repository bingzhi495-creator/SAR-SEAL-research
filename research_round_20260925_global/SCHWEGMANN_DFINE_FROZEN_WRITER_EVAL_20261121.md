# Frozen D-FINE boundary writer on Schwegmann center scatter masks

The fixed author accession contains 1,596 annotated center-ship crops. Use the earlier group's log-intensity mapping, HH or VV eligibility, finite nonnegative pixels and content deduplication. Evaluate all qualifying distinct RGB crops after excluding the 32 images already scored in the project and any RGB duplicates of them. Keep all included groups in the original audited acquisition mapping. The previously used acquisition groups are historically exposed. This holdout excludes images, with acquisition groups shared with the older comparison. Product provenance distinguishes this dataset's captures from HRSID.

Freeze the original HRSID-trained D-FINE E24 detector, 1,321-parameter FIT109 point head, standardizer, support, score and swap policy. Use the qualified author 640-square inference and restore masks to 101 by 101 native pixels. Choose the highest native-ranked top-100 box whose center lies in [40,61) on both axes, as in the earlier center-mask work. Freeze selection before edits and target mask access. Assign zero overlap to missing or empty selections. Use the admitted existing size band, point support and maximum swap counts. Compare the raw mask, same-head fixed pairs, independent selection at the fixed count, independent selection at the natural count and native ranking at the natural count.

Report acquisition-equal IoU, Dice, 0.50 and 0.75 recovery, exact candidate and edit counts, preservation of area and per-group paired contrasts. Use 4,096 paired acquisition-group bootstrap draws at seed 254101. The center reference is the original author scatter-support mask, with limited physical interpretation. The panel remains below the earlier 20-acquisition confirmation threshold. This frozen auxiliary experiment provides a descriptive effect on center masks. Dataset-wide instance AP, full-scene false alarms and accuracy on physical hull boundaries require other data.

[Original source and protocol](HRSID_TO_SCHWEGMANN_CENTRAL_SHIP_PROTOCOL_20261110.md)


The exact image intake retained 750 unique decoded 101-pixel crops after excluding the 32 previously scored inputs. A second readback reconstructed every PNG and verified the acquisition counts and original image digests. The source and split were fixed before detector inference. [Image-only aggregate](results/SCHWEGMANN_DFINE_IMAGESET_SAFE_20261121.json)


## Frozen panel outcome

The frozen D-FINE selects a ranked center candidate on 572 of 750 crops. Among these, 465 masks are empty and 107 contain 1–31 native pixels. The other 178 have no selected candidate. The minimum edit area is 32 pixels. No candidate meets it. All five arms produce identical masks, with zero added or removed pixels.

| Quantity | Result |
| --- | ---: |
| Images / conservative acquisition groups | 750 / 16 |
| Nonempty selected masks / area-eligible masks | 107 / 0 |
| Acquisition-equal IoU, scaled by 100 | 0.9222 |
| Acquisition-equal Dice, scaled by 100 | 1.3835 |
| Original IoU of at least 0.50 / 0.75 | 3 / 1 |

The capture, writer and original scoring process each exited 0. A separate reader reproduced all image and capture counts, the original reference masks, mask selection, the 16 group summaries and all paired intervals. The missing, empty and too-small masks localize the failure to upstream coverage. The zero paired effect describes the frozen recipe on this evaluation panel. The effect on eligible center candidates remains open.

The 32 previous scoring images are excluded. The 16 acquisition groups were previously sampled in another detector comparison. The center-only annotations measure the selected scatter support inside each crop. These results characterize center-mask overlap.

- [Exact frozen protocol](code/SCHWEGMANN_DFINE_FROZEN_WRITER_20261121/PROTOCOL.md)
- [Original score](results/SCHWEGMANN_DFINE_FROZEN_CENTER_SCORE_SAFE_20261121.json)
- [Independent actual recomputation](results/SCHWEGMANN_DFINE_FROZEN_CENTER_INDEPENDENT_20261121.json)
- [Author accession](https://sarmaps.gitlab.io/analysis_data.html)
