# U59 all100-query DEV portable package — UNBOUND / UNEXECUTED

This is an ordinary source-only helper for a future Root-owned export and read.
It is not a launcher, binder, resource monitor, endpoint restorer, evaluator, or
qualification decision. No genuine Torch/NumPy execution, real DEV export,
serialization equivalence, transfer, zero-residual RLE calibration, or AP has
been performed for this draft. Passing the tests below does not change that.

## Reviewed interfaces

The static reference is the ordinary published source of
`N28_FIXED24_DEV_EVAL_DRAFT_20261017_01/dev_reader.py`,
`U40_FIXED50_DEV_RGB_READER_IMPLEMENTATION_DRAFT_20261017_01/dev_rgb_reader.py`,
`U31_IMAGE_GEOMETRY_DRAFT_20261017_01/image_geometry.py`, and the ordinary
`U40_FIXED50_DEV_PREDICTION_DRAFT_20261017_01/prediction_core.py`.
The first three source SHA256 pins are explicit in `SOURCE_PINS`; the ordinary
published files were checked against them during static implementation.
No bound SOURCE, actual input/index, weight, Ops file, or qualification chain
was accessed to implement or test this helper.

The unchanged `_calibration_and_prediction` consumes natural-query-axis
`e/M/L` by the original arbitrary `query_ids` export permutation. It also needs
`rgb400`, the original geometry object, row `dataset_index/image_id`, and the
100 original native prediction dictionaries. It does not need `F` or GT.
The transported reader therefore returns precisely those inputs.

## Package whitelist

The package has only `schema`, `metadata`, `tensors`, and `proof`.

| Tensor | Required genuine runtime representation |
| --- | --- |
| `rgb400` | detached finite CPU FP32 `[1,3,400,400]`, values in `[0,1]` |
| `e` | detached finite CPU FP32 `[100,256]`, natural query ids `0..99` |
| `M` | detached finite CPU FP32 `[100,200,200]`, natural query ids `0..99` |
| `L` | detached finite CPU FP32 `[100,original_H,original_W]`, natural query ids `0..99` |

Export clones only these four fields. There is no F, target/matched tensor,
point scout, shared bundle, class-logit tensor, or arbitrary saved object.
`proof` contains only a schema, canonical metadata SHA256, and each field's
shape, FP32 dtype name, and CPU byte digest. A digest is an integrity check;
it does not prove that arbitrary caller-supplied data came from a qualified run.

Metadata is recursively plain finite JSON and strictly whitelisted:

- Original ordinal `0..186`, image id, natural query ids `0..99`, and the full
  original 100-query export permutation; no filtering, sorting, or GT matching.
- Identity contains only `image_id`, the original image `path/bytes/sha256`
  reference, `original_shape`, `actual_img_shape`, and `batch_input_shape`.
- Geometry contains only original/content shapes and canvas800/branch400.
  Content dimensions must equal the original integer round-half-up keep-ratio
  rule. A geometry object is not serialized.
- Native predictions retain the complete original 100 dictionaries in their
  exact export order: required `image_id/query_id/category_id/score/segmentation`,
  and optional original `bbox/area`. Score, numeric type, signed zero, category,
  bbox, area, order, and opaque string-count RLE dictionaries are preserved.
  No score threshold, empty-mask deletion, RLE decoding/re-encoding, bbox/area
  regeneration, or score replacement occurs. Empty masks remain candidates.
  An unknown dictionary/RLE ABI is rejected, not silently truncated or guessed.
- RGB convention is the unchanged reviewed RGB uint8 → FP32/255 path with no
  automatic orientation or teacher photometric reconstruction.
- Provenance contains only the three `source_refs` and five `input_refs`:
  `dev_context/native_predictions/original_capture_manifest/bundle/dense_packet`.
  Every ref is exactly `path/bytes/sha256`; no whole row, context, manifest,
  source text, association metadata, or target payload is copied inside it.

The original reader loads historical bundle containers that can also hold GT
association fields. This helper excludes their contents from its output; it
does **not** establish that the complete upstream export process never touched
any GT-bearing container. A future scientific report must retain that caveat.

## Future Root-owned use

All runtime interfaces require caller-injected **genuine Torch**, an already
admitted Control implementing `closed(ref)` and `check()`, and the original
source-loaded `image_geometry` module with the three explicit local source refs.
`closed(ref)` must authenticate the admitted bytes; a function that merely
returns a path is not an adequate production control. Source pins and the
geometry module path are checked, but these helpers do not construct or qualify
the control or authenticate a caller's Python object provenance.

1. `export_item(torch, c, dataset_index, item, originals, *, image_geometry,
   source_refs, input_refs)` projects the already admitted original RGB-reader
   item and the corresponding full100 native dictionaries. The original
   reader's checks and lineage remain required; a caller-provided ref alone
   cannot establish origin.
2. `save_export(torch, c, value, output_path, *, image_geometry, source_refs)`
   writes a new one-image file exclusively, explicitly reloads with
   `map_location='cpu', weights_only=True`, checks metadata/proofs and all four
   tensors, then returns an entry with only
   `dataset_index/image_id/query_ids/package_ref`. Existing output is never
   replaced. Failed write/load/check leaves its partial output and raises;
   there is no unsafe load fallback or automatic retry.
3. Root assembles the 187 entries in original ordinal order, verifies their
   provenance and actual completed export, and handles transport/admission.
   `validate_entries` checks full187 unique-image metadata and 18700 candidates;
   it does not open or validate the tensor population. When moving hosts, Root
   supplies the actual destination package refs and local original-source refs.
   Historical source/input/image refs inside package metadata stay unchanged
   and are not opened on the destination host.
4. `reader = prepare(torch, c, entries, *, image_geometry, source_refs)` validates
   the full manifest. `item, row, originals = reader.load(index)` closes and
   weights-only loads only the selected package, checks every field proof,
   joins ordinal/image/query mapping, and calls the original
   `make_geometry(torch, original_shape, actual_img_shape, 800, 400)`.
   `item` contains only `rgb400/e/M/L/identity/query_ids/geometry`;
   `row` contains only `dataset_index/image_id/qids`. The caller can pass these
   three outputs to the original per-image calibration. This tuple interface
   is **not** a drop-in replacement for the old `_arm` reader interface.

Before using any predictions, Root must execute and verify the original
zero-residual calibration of **every** original native RLE and all non-mask
dictionary fields, including empty masks, under the admitted endpoint and
resource policy. That numerical calibration and the complete187/18700 result
are outside this helper. No matched-query subset, interim model selection,
target annotation, AP computation, or new scientific conclusion is introduced.

## Tests and limits

Run `python -B test_metadata.py` in this directory. The tests import only
stdlib plus this helper. They use tiny opaque dependency-injection tokens,
synthetic metadata, and temporary tiny token files. No raster, dense tensor,
genuine Torch serialization, renderer arithmetic, real source control, cache,
DEV image, prediction population, network, or training is executed. Stub tests
only verify orchestration, whitelist rejection, axis/order metadata, explicit
weights-only arguments, proof mismatch refusal, output exclusivity, geometry
constructor arguments/ABI refusal, and unsupported-load refusal without retry.
Runtime byte equality, source/packet origin, whole-population calibration,
resource limits, and cross-environment behavior remain **UNEXECUTED**.
