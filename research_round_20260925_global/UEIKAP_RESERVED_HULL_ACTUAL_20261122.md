# Frozen Sentinel-1 transfer on the reserved UEIKAP hulls

All 108 author PNGs, 139 published hull polygons and 18 reserved acquisition families were evaluated. At native COCO mask IoU 0.50, the frozen HRSID D-FINE matched **2 of 139 labeled hulls (1.44%)** in two families. Both original predicted masks had zero positive fixed-pair exchanges. The matched-count difference in continuous polygon IoU was **0.000 points**, with a paired family interval of [0.000, 0.000]. The prespecified requirement was 20 matched hulls with positive fixed-pair budgets across eight families, plus a positive lower bound. [Protocol frozen before evaluation](UEIKAP_RESERVED_HULL_PRELOCK_20261122.md) · [Aggregate score](results/UEIKAP_RESERVED_HULL_SCORE_ACTUAL_20261122.json)

| Frozen contrast | All labeled hulls, 18 equal-weight families | Two original native matches |
| --- | ---: | ---: |
| Independent selection minus fixed pairing, same count | 0.0000 | 0.000 |
| Natural independent selection minus native masks | +0.0025 | +0.853 |
| Learned ranking minus dynamic spatial attraction, same count | −0.0056 | −0.914 |

Values are continuous IoU points. The all-hull analysis assigns zero overlap and zero paired difference to 137 unmatched hulls. The original matched masks covered 136 and 146 pixels, sufficient for the native predicted-area guard. Neither matched hull crossed 0.75 or 0.90 under any arm. Both original matches favored the dynamic spatial-attraction adaptation over learned ranking at its realized count, with two matches providing descriptive evidence only. Acquisition-level transport of the refinement effect remains unconfirmed.

The actual forward pass used the accepted HRSID D-FINE E24 EMA, a frozen FIT109 BCE point head with 1,321 parameters, and the original scaler, thresholds and support. All 108 native chip outputs retained 100 ranks each. Of the 10,800 retained masks, 9,035 were empty and kept in the capture. Equal-count fixed and independent strategies each applied 699 exchanges across the same 258 candidate masks. These edited candidates provided zero positive budgets among the two original hull matches. Frozen capture, independent capture readback, label-free edits, deterministic replay and geometry scoring each exited with code 0. The deterministic replay reproduced the endpoint choices for each policy and verified exact XOR counts and foreground area. [Source and stage inputs](code/UEIKAP_RESERVED_HULL_FROZEN_20261122/README.md)

The author source is wake-centric, and completeness of whole-scene hull annotation remains unknown. This is the first scored model evaluation on the reserved families. Their labels and partial TIFF prefixes had previously been used for structural and acquisition metadata checks. The earlier complete image previews and human review sample were drawn from other families. The chip populations and outputs are preserved without filtering according to the observed metrics. [Author archive](https://zenodo.org/records/18391027) · [Author hull and wake protocol](https://www.iris.unina.it/retrieve/8417698a-520f-4668-a2e4-4d045eb9bb9a/From_Ship_Wakes_to_Pseudo-AIS_Dark_Vessel_Detection_with_Sentinel-1_and_Sentinel-2.pdf)

The next decisive transfer experiment requires an independent acquisition set with enough correct stock matches and verified completeness of instance masks. Instance-RSDD remains a candidate for whole-image AP once the author-hosted instance labels and source identities can be verified. Published DACF-Net results remain contextual because its official repository was still a source placeholder on 2026-11-22. [Instance-RSDD source](https://github.com/XIDIAN212Lab/SRNet#the-instance-rsdd-dataset) · [DACF-Net author repository](https://github.com/yang199809/DACF-Net)

## Candidate coverage and a geometric upper bound

An independently checked post-hoc audit measures whether each labeled hull has any qualifying candidate. At IoU 0.50, the top 100 contain an adequate native box for 21 hulls and an adequate native mask for two. All 30,910 post-NMS candidates contain adequate boxes for 39 and adequate masks for five. The three additional matching masks have ranks 115, 133 and 226. Among the 21 hulls with an adequate top-100 box, the candidate with best box IoU has median mask area 38.5% of its label area and median mask IoU 0.378. These are candidate-coverage measurements on the released label population.

| Candidate scope | Box / mask ≥0.25 | Box / mask ≥0.50 | Box / mask ≥0.75 | Oracle with up to eight swaps ≥0.50 |
| --- | ---: | ---: | ---: | ---: |
| First 100 native ranks per chip | 92 / 49 | 21 / 2 | 0 / 0 | 4 |
| All archived post-NMS ranks | 116 / 61 | 39 / 5 | 0 / 0 | 7 |

The last column preserves candidate area and permits up to eight ideal swaps anywhere on the image with true raster labels. For native area \(P\), truth area \(G\) and original intersection \(I\), the most favorable intersection is \(\min(P,G,I+8)\). Every candidate is evaluated separately against each hull. Even this relaxed upper bound covers zero hulls at IoU 0.75. Allowing arbitrary rearrangement within the original crop at the same area raises the 0.50 ceiling to 8 top-ranked or 40 full-list hulls. Allowing arbitrary area inside the crop raises it to 73 or 128. The gaps identify localization, area and local-edit budget as distinct constraints on these archived candidates. The source of the cross-domain error remains open. [Candidate aggregates](results/UEIKAP_CANDIDATE_OPPORTUNITY_SAFE_20261122.json) · [Independent bound checks](results/UEIKAP_CANDIDATE_BOUNDS_SAFE_20261122.json) · [Code](code/UEIKAP_RESERVED_HULL_FROZEN_20261122)

The diagnostics had two early stops before emitting results, one from a terminal-checksum transcription error and one from a strict interval check on reversed native box coordinates. The final audit retains every candidate, treats nonpositive original box extents as empty, and confirms that each such candidate has an empty native mask. The main and independent checks ran to completion.

On the same released hull cohort, a later frozen RF-DETR carrier yielded 17 native matches with positive fixed-pair budgets across four acquisitions. Its paired effect was positive within this coverage, which remained below the original requirement. [Second-carrier experiment](UEIKAP_RFDETR_CARRIER_ACTUAL_20261122.md)
