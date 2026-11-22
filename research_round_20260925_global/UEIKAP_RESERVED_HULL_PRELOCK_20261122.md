# Frozen evaluation on the UEIKAP reserved acquisition families

## Reference population

Use all 108 author PNG chips and the 139 released ship polygon annotations in the 18 Sentinel-1 acquisition families reserved on 2026-09-24. The original reservation has SHA-256 `1f3b4d3bf0ff8f0ed1767cfdd796b541e60f2a4c7d76b03c316f0b1e81dbded1`. Acquisition groups are mission, absolute orbit and datatake. Use the original reservation and its already audited mapping. Preserve every family and chip, including zero predictions and zero editable matches.

Use the [author's UEIKAP SAR ZIP](https://zenodo.org/records/18391027), byte length 1,571,288,093 and archive directory SHA-256 `e7a969a39b4f6723d9293acdc9297e917d2830fcba54eeac3dd22b26b4f56610`. Verify each requested PNG and YOLO file by the exact original ZIP member and CRC32, then record the actual member SHA-256. Use the publisher PNG in its original raster coordinates. Convert normalized YOLO polygon coordinates by native width and height. In this archive, class 1 matches the separately published wake polygons, and class 0 corresponds to the separately described ship annotations. The project paper reports separate ship and wake results and describes ship hull annotation for SAR. [Author paper, Sec. II-A and Table II](https://www.iris.unina.it/retrieve/8417698a-520f-4668-a2e4-4d045eb9bb9a/From_Ship_Wakes_to_Pseudo-AIS_Dark_Vessel_Detection_with_Sentinel-1_and_Sentinel-2.pdf)

The reserved March-to-April 2024 source window comes after the HRSID and SSDD reference releases used for model development. Verify the reservation against the archived acquisition mapping at intake. These families were kept outside the five initial image previews and QA50 image selection. Their public labels and partial TIFF headers were inspected in the repository's earlier data audit. This is the first model evaluation proposed for the reserved families. The dataset is wake-centric and many labels are attached to targeted vessels. The evaluation therefore measures agreement with released ship polygons. It does not estimate complete-scene AP, unannotated-ship false positives or open-water prevalence.

## Frozen model and comparisons

Use the previously accepted HRSID D-FINE E24 EMA and the single frozen FIT109 BCE point head and scaler as carried to PSeg-SSDD. Use the stock author preprocessing and NMS, native masks, original scores and up to 100 native slots; any lower native counts require their actual count to be recorded. No target refit, domain-selected normalization, threshold change, sample choice or proposal substitution. Use the original 32 to below 1,024 native pixel editing-area guard; retain all other masks as native. Verify native author-mask parity and exact probability, rank and identity linkage before closing the initial capture.

- N is the native prediction.
- SF uses the same learned endpoint predictions and fixed distance pairing.
- SU=F removes the pairing restriction at exactly the realized number of SF exchanges per mask.
- SU uses independent selection with its original positive-margin count.
- The spatial-attraction adaptation replays the previously frozen radius-5, decay-length-5 rule on the same endpoints. Compare it with static attraction and learned independent ranking at its own realized count. This is the matched-count adaptation of a published rule, with no target adjustment. [Atkinson original method](https://doi.org/10.14358/PERS.71.7.839)

Generate all arms before opening target geometry for scoring. A separate reader verifies original identity and ordering, rank and score equality, area preservation, and per-mask realized XOR and count equality for the specified paired arms.

## Primary estimate and evidence limits

Fix each target match once, using the native score order and COCO mask IoU 0.50 on the labeled-vessel population. Every valid labeled vessel remains in the primary denominator. A vessel without an original match has zero overlap and zero paired effect; an unchanged match also contributes zero effect. Compare continuous polygon IoU at those fixed identities with equal weight to the 18 families, averaging vessels within each family. The primary contrast is SU=F minus SF. Also report SU minus N and learned versus spatial-attraction ranking at the spatial-attraction count. Report conditional fixed-match results, exact match and edit coverage, counts and five-number or quantile summaries for native area, and gains/losses at continuous 0.75 and 0.90. No contrast changes native assignment or labels.

Use paired family bootstrap percentile intervals with 4,096 draws and PCG64 seed 5217. A transport claim requires at least 20 available original matched vessels with a positive SF edit budget across at least eight families, plus a positive lower 95% bound for the primary all-vessel difference. Retain the estimate even if native transfer, edit support or the interval fails. With coverage below the gate, interpret the numbers descriptively. Keep unmatched, noneditable and unchanged counts. The study concerns released hull annotations in these reserved acquisitions under the frozen detector/head.
