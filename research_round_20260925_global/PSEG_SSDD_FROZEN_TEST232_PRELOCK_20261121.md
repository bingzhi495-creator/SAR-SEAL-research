# Frozen HRSID-to-PSeg-SSDD test evaluation

Use the original PSeg-SSDD combined COCO test split with the 232 native JPEGs and 546 nonempty floating-point instance polygons. The local files were decoded and audited independently. File lengths and Git blob hashes match a pinned third-party mirror for all 233 test files. Author-hosted byte authentication remains unresolved. The 232 split has historical project exposure.

Run the frozen HRSID-trained D-FINE-seg-S EMA at E24 plus the unchanged HRSID FIT109 1,321-parameter head, scaler and editing policy from DEV187. One label-free inference over the full 232 supplies the original candidates to all five arms: native mask, fixed pairs, independent selection at the same per-instance count, natural positive-gain independent selection and original decoder ranking at the natural count. Keep original scores, ranks, candidate identities, boxes, foreground counts and native raster dimensions. Use zero target updates and no target parameter selection.

After closing and independently verifying the prediction and mask-edit streams, score class 0 original floating polygons with the verified class mapping. The primary result uses all 232 images. A sensitivity excludes the ten images previously confirmed to have byte-exact copies in historical project PSeg-SSDD FIT741. It reuses the same predictions and keeps 222 images. The deduplication addresses that prior target-side exposure. It does not verify HRSID-to-SSDD acquisition-scene independence. Always report the raw and matched-count contrasts and compare natural-policy gains with the prior +0.50 AP over raw and +0.30 AP over fixed pairing thresholds as descriptive references. Retain all arms and both populations.

- [Official PSeg-SSDD release](https://github.com/TianwenZhang0825/Official-SSDD)
- [Original paper](https://doi.org/10.3390/rs13183690)
- [Pinned third-party file mirror](https://github.com/DonnieRaymond3/cs231n_ship_detection/tree/30cdc0e7e7fa28b67c8a2f200f25ec7bc08dd97b/SSDD/BBox_RBox_PSeg_SSDD/coco_style)
