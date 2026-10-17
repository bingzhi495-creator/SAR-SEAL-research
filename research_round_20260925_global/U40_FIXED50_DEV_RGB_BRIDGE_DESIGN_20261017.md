# U40 fixed-E50 DEV RGB bridge — static candidate

Status: design only, not implemented or executed. No Torch/NumPy import, image/cache/weight read, teacher capture, qualification or evaluation was performed for this note. Formal evaluation remains gated by a Root-accepted paired endpoint of 37,050 updates per arm. No additional qualification leaf is proposed.

## Minimal composition

Reuse the published N28 DEV reader unchanged. Add a shallow RGB wrapper around its `load(index)` result. Reuse the existing `image_geometry.make_geometry(..., 800, 400)` and its `prepare_rgb`/`render_delta` methods unchanged. The JPEG/EXIF/RGB loading checks can follow U37 lines 103–138 directly; do not call its FIT-specific `prepare` or `_RGBReader`.

The U37 reader requires GT-matched `record.identity`, `G` and `query_ids`, and its factory binds a specific FIT reader. DEV metadata instead has `qids` and may lack `identity.image`. No synthetic `G=100`, GT facade, matcher, or semantic reinterpretation of `G` is needed or permitted.

Proposed future API (not current implementation):

```text
prepare(torch, np, c, context, original_image_refs,
        *, dev_reader, image_geometry, source_refs) -> reader
reader.records: original DEV metadata, unmodified
reader.input_references: original DEV references plus the 187 admitted JPEG references
reader.load(index): original DEV item plus rgb400, geometry, rgb_receipt
```

`original_image_refs` is a length-187 strict-ref list in the original dataset ordinal order. The Root binder must supply these references from original admitted image metadata, register them before import/execution, and retain provenance. The bridge must not guess filenames, scan directories, or read GT to derive image identity. The currently inspected public DEV row grammar alone does not identify all JPEG refs; this is the remaining binding input, not a reason to recapture teacher tensors.

At load time, first call the unchanged DEV `load(index)`. Require its actual `identity.image` strict ref to equal `original_image_refs[index]` before opening the JPEG with `c.closed`. Require exact resolved path, JPEG format, original dimensions, and EXIF orientation absent or integer 1. Convert to RGB uint8 without auto-orientation. Build the existing geometry from the unchanged `identity.original_shape`, `actual_img_shape` and `[800,800]` canvas; call its existing `prepare_rgb` to obtain finite detached CPU FP32 `[1,3,400,400]` in `[0,1]`.

Return a shallow copy of the DEV item, retaining the same `F`, `e`, `M`, `L`, `identity` and `query_ids` objects, with only three new keys. Reject collisions. `rgb_receipt` records the actual image ref, original/content shapes, 800/400 sizes, original/decoded modes, EXIF and no-auto-orientation, and the new-branch RGB/255 convention. It must describe an all-100 natural-query baseline, not U37's GT-matched baseline; RGB/255 is not a reconstruction of teacher normalization. The wrapper never supplies GT, scores, categories or prediction dictionaries.

## Query-axis contract

The DEV item has `F=[1,256,200,200]`, `e=[100,256]`, `M=[100,200,200]`, `L=[100,H,W]`. These tensor axes use natural query ids 0–99. `query_ids` is the original prediction-export permutation and must stay unchanged. The evaluation caller must index `e`, `M` and `L` by the actual query ids of each export chunk, not slice by export rank. No M reconstruction from resized L is allowed.

The evaluator, not this bridge, owns fixed S0/class/order, each arm's one image encode, query chunks, residual rendering and exact zero-residual native RLE calibration. Keep all 100 queries and the original full-population COCO merge. The bridge does not grant AP or training eligibility.

## Source closure and static evidence

Runtime dependencies for a future bridge are the injected DEV-reader and geometry modules, their explicitly bound source refs, plus the bridge's own source. U37 is implementation provenance for the short JPEG-loading section; it need not become a runtime module dependency. All original reader dependencies and the 187 actual JPEG refs remain explicit admission inputs. Do not hash or copy live bound science for this design.

Paths below are ordinary published files under `E:/SAR/seal/github_release_20261011/research_round_20260925_global/code/`:

| File | Exact static basis |
| --- | --- |
| `N28_FIXED24_DEV_EVAL_DRAFT_20261017_01/dev_reader.py` | `prepare_records` lines 26–51; original packet/bundle identity and natural/export joins 95–119; F/e/M/class shapes 120–129; return and factory 130–143. 7,780 bytes; SHA256 `daabec8388e23d9ff646ae78eba69ba2168099a72971578a8d34cef04f1268c0`. |
| `U37_FULL_RGB_MATCHED_READER_DRAFT_20261017_01/rgb_matched_reader.py` | FIT-only row checks 96–102; image/JPEG/EXIF/RGB path 103–138; shallow return and FIT-specific receipt 142–152. 8,884 bytes; SHA256 `678ce3781872e21e961b24a7b0a4d0ecc69f0f8a08b8feb7eb0b2e8624d57025`. |
| `U33_REAL_IMAGE_400_PROBE_DRAFT_20261017_01/image_geometry.py` | `make_geometry` line 8; integer keep-ratio rounding 25–28; `prepare_rgb` 31–42; `render_delta` 44–53. 3,024 bytes; SHA256 `65db107de80c5f1d4dade7cb2ffd3f241fd6bf7f9b233cb1aa9c90cbfd00abf7`. |

The inspected sources establish API feasibility only. They do not establish that all required current DEV artifacts exist or pass actual hashes, decoding, zero-render calibration, model restoration or COCO evaluation. Those checks belong to the future Root-owned fixed-endpoint evaluation.
