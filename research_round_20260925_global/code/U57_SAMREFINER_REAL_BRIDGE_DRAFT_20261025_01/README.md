# Author API bridge draft: unbound and model-unexecuted

This is a small call-site bridge to a **caller-supplied, source-verified** base SAMRefiner function. It does not import the author package, load/check a model, read an image, or establish source/checkpoint authenticity. Tests use a fake author callable and small NumPy CPU masks only.

The author function returns arrays without query IDs. This bridge tags returned rows with the submitted IDs; these labels **cannot detect a permutation inside SAM if the returned rows are internally wrong**. The existing indexed adapter can validate the declared order, not independently certify the author's internal correspondence. That remains a real-execution qualification requirement.

## Interface and scope

`backend = make_samrefiner_backend(image_path, loaded_sam, verified_author_refiner)`

`backend(query_ids=ids, masks=masks, image_size=(height, width), iters=5)`

- `image_path` is the fixed string passed unchanged to the author, which reads the image itself. There is no captured RGB-array parameter. The actual file, uint8 RGB interpretation, real dimensions and path-to-mask correspondence must be read and verified by the future caller/qualification layer. This draft cannot infer them from cached masks or the declared size.
- The bridge accepts 1 to 100 unique submitted IDs, supporting the nonempty subset from the existing adapter's explicitly chosen passthrough strategy. It creates one independent, contiguous NumPy `uint8` batch, with exactly the caller-declared `(N,H,W)` and pre-existing boolean/integral 0/1 values. No threshold, resize, clipping or GT is used.
- It calls the author once, with the captured model object unchanged and the source defaults fixed: five iterations; point, box, mask and negative prompts enabled; margin 0, gamma 4, strength 30; `resize_transform=None`, `use_samhq=False`, `ddp=False`, `is_train=False`. The bridge never calls `.to()`, `.eval()` or any other model method.
- The expected author result is its three-element tuple. Only the first element is consumed: an actual NumPy `uint8` array of the same `(N,H,W)` containing 0/1. SAM quality/logit auxiliaries are not converted or used as scores. Returned masks become immutable Python tuples for the indexed adapter.
- Empty-mask policy remains upstream: default rejection or explicitly disclosed passthrough in the frozen adapter. This bridge introduces no silent skipping, fallback, retry, NMS, score replacement or empty-mask repair. Author exceptions propagate.

## Finite validation and remaining work

Seven test methods cover the exact author call, copied array boundary, invalid input/output/declarations, no retry, and one read-only integration with the frozen 73-line adapter. The integration preserves all 100 IDs/categories and bit-exact float S0, including negative zero. It uses only synthetic masks and a fake callable; the previous 13-test candidate is not modified.

Run `python -B -S -X utf8 -m unittest -v test_samrefiner_bridge`. The test harness exposes the already-installed NumPy path without site startup and scopes TEMP/TMP to its own D-directory fixture folder. No installation or Torch import occurs. No real image, SAR data, weights, GPU, SSH, author inference, actual five-iteration execution, latency/memory measurement or AP has been performed. See `VALIDATION.json` for observed RED/GREEN records and hashes.

Static source contract, commit `4bb7f95738c7a2e805f189210fb09e02a2197557`: [outer signature, image-path reading and NumPy return](https://github.com/linyq2117/SAMRefiner/blob/4bb7f95738c7a2e805f189210fb09e02a2197557/sam_refiner.py), [prompt/mask preparation](https://github.com/linyq2117/SAMRefiner/blob/4bb7f95738c7a2e805f189210fb09e02a2197557/utils.py), [author SAM entry](https://github.com/linyq2117/SAMRefiner/blob/4bb7f95738c7a2e805f189210fb09e02a2197557/segment-anything/segment_anything/modeling/sam.py#L133). The latter exposes `forward_with_image_embeddings` and leaves postprocessed masks as logits; the outer function performs the final `>0`/NumPy `uint8` conversion. These are source facts, not real model execution evidence. No third-party source was copied into this candidate.
