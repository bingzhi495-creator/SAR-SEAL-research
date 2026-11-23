# Whole-mask Phoenix transfer on the archived Instance-RSDD 74 cohort

## Question and fixed population

How does Phoenix change whole-instance mask AP, valid hull coverage and native-matched geometry when its public refiner writes the entire mask? Compare the full writer with the already measured matched-count occupancy and Phoenix endpoint rankings. This is a new same-cohort comparator on the previously scored 74 chips and 13 added parents. The 42-image date-separated view is nested in this cohort. No result from either view qualifies as a previously unexamined geographic replication. No target information enters the new inference.

Pre-fix HRSID RF-DETR center5202 E24 as the detector. Use the exact observed B5 stock prediction bytes and image identities associated with the closed [matched endpoint run](INSTANCE_RSDD_PHOENIX_EQUAL_COUNT_ACTUAL_20261123.md). Retain all 7,346 stock candidates in all arms, with original order, scores and ship class, including 1,267 empty masks. Use exact original RGB. No detector, head or Phoenix training or target-based selection.

## Author writer and experimental arms

Use [pinned Phoenix 13ee3996](https://github.com/naver-ai/Phoenix/blob/13ee3996ef3d0292fab144316cc8e03b69c0ecfa/phoenix/eval/evaluators.py#L96-L167) and the official EfficientViT-XL1 instance weight with SHA-256 9b6ba211e0d6d434085d3d5e379eb527d8b56ef413628f42cb587153cc51af7d. Apply the public final binary mask operation with five iterations, smooth kernel 1, binary prompt and multimask selection. Reuse the admitted DGX CPU FastGeodis 1.0.5 path with four workers, two threads per worker and at most two candidate masks per batch. Initialize seed 5202 for each native image. The batch cap is an execution choice. The experiment carries public gate and writer semantics to SAR with a frozen detector and CPU runtime.

1. Primary whole-mask transfer uses the authored LVIS area rule. Refine every nonempty mask on the original full image, including masks outside the 32–1023 range. Total 6,079 nonempty inputs. Keep any newly empty mask and retain the stock RLE on skipped entries.
2. Secondary whole-mask transfer uses the authored COCO gate, identical to Cityscapes, at strictly greater than 1% of native image area. The 172 candidates that pass this gate run separately through the same writer. Keep the stock RLE on all skipped entries. A 512-square image puts the gate above the entire 32–1023 target range.
3. Archive references are stock RF-DETR, the three matched-count occupancy heads and Phoenix ranked on the exact endpoints and dynamic counts shared by those heads. Use the archived frequencies and unchanged score order. The main comparison of whole-mask with matched Phoenix changes support, signed quotas and eligibility jointly. It measures realized performance, and the controls cannot attribute the difference to one of these changes.

Observe actual successful process termination and verify worker, request, ordered ledger, prediction masks and complete footers. Validate image digests and all original mask positions before scoring. Return no image or mask to the public repository. Report eligible, unchanged, changed, newly empty, addition, deletion and signed area totals plus CPU stage timing. This is CPU stage timing on loaded frozen detector outputs.

## Evaluation

Score the complete annotated 74-chip population with the preserved COCO protocol for mask AP, AP75, AP90, small AP and AR100. Also report positive detections and original matched hulls that retain IoU 0.50, matched raster recall at 0.50, and continuous IoU on frozen original eligible matches. Give paired intervals on fixed-match IoU using the existing equally weighted parent and 4,096 sensor/date block protocol. Where computed, also give normalized FP and FN and group coverage. Retain the 42-date-separated view, both sensor views and predeclared decoded duplicate view. Report counts of unsupported cohorts.

Whole LVIS-policy versus archived matched Phoenix is the primary contrast, with COCO-policy whole write and comparison with stock and each occupancy head as secondary. A supported whole-writer advantage would require an increase of at least 0.30 mask AP points and a positive lower paired IoU bound with at least 20 valid native eligible matches in eight sensor/date blocks. A supported loss requires the mirror conditions. Report conflicting signs, any coverage loss and both AP/IoU point differences regardless of outcome. No arm or gate is changed after any target scores open.
