# RF-DETR E1 EMA export candidate

The adapter starts from one already admitted eager eval forward. Its synthetic checks use only the standard library and fixture objects. Real E1 weights, images, masks, codec, and evaluation were outside this candidate check.

`export_e1_image(raw_outputs, *, image_id, original_hw, torch_api, postprocess_type, canonical_export, encode_mask)` accepts the pinned Torch module, RF-DETR `PostProcess` class, project `export_coco_mask_predictions` function, and fixed compressed COCO encoder wrapper. Pass the canonical image ID and actual original height and width.

The selected Seg Nano recipe has 100 queries and two classifier slots. In eval the model uses one query group, while training uses 13. The adapter requires one image, 100 actual dense eval masks and two logits per query. It calls stock PostProcess once with all slots, original sizes, no score threshold, and 100 selected rows. Stock ranks 200 query/class pairs by sigmoid score, with stable order for ties. A query may therefore appear twice. The adapter keeps the 100 native rows and does not assign its row rank as a query ID. Stock converts each selected mask to original size by bilinear logit interpolation, then thresholds at logit > 0.

The returned `native` dictionary contains the 100 rank-aligned score, label, box, and bool mask tensors. It is private and is not a JSON payload. `coco` contains the mapped prediction records. It invokes the existing canonical exporter one row at a time to bound Python list memory. Slot 0 maps to COCO category 1. Slot 1 remains in the native result and is excluded from COCO. Some images may have fewer than 100 COCO rows or no mapped rows. Continuous scores, native order, and legal zero masks survive for retained rows. The adapter has no rescore, refill, query padding, ground truth, or metrics path.

For an actual 2–4 image qualification, independently bind the E1 archive at 186 updates and the native EMA average state. Resolve EMA model keys and verify the eager model actually holds that state before calling `eval()` and `inference_mode()`. Use the admitted validation image transform and IDs without calling the validation metrics callback. Record actual input and output dtype, dimensions, raw query count, stock row count, and unmapped count per image. Compare the fixed codec's RLE round-trip after JSON serialization with the original native bool mask. The small sample qualifies interfaces. Whole DEV187 canonical evaluation requires a separate locked request.

- [RF-DETR pinned eval query selection and final mask output](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/models/lwdetr.py#L483-L602)
- [RF-DETR pinned postprocessing](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/models/postprocess.py)
- [RF-DETR pinned EMA callback](https://github.com/roboflow/rf-detr/blob/639eac4555ceb89ef399ccda3fde716bbcfe34cb/src/rfdetr/training/callbacks/ema.py#L459-L485)

The existing canonical exporter used for the interface review has SHA-256 `b2e99e581acb0236f6cee5542321bb3e01b2c620f2c727062b28b71eedfd1820`. The synthetic check injects a fixture, while the eventual real entry must bind this independent exporter and the admitted codec.
