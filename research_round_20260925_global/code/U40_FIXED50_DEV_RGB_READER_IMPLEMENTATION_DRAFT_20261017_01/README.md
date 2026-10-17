# U40 fixed-E50 DEV RGB bridge — unbound implementation draft

Status: only three synthetic standard-library metadata tests have run. The numerical image/tensor path has not run; no real image, cache, weight, Torch/NumPy/PIL import, neural network, COCO evaluation or new qualification leaf was used. This is a reader bridge, not a runner or evaluation qualification. Root acceptance of the paired fixed E50 endpoint (37,050 updates per arm) remains a future evaluation prerequisite.

`dev_rgb_reader.prepare(torch, np, c, context, original_image_refs, *, dev_reader, image_geometry, source_refs)` composes the unchanged source-pinned DEV reader with the unchanged source-pinned image geometry. `source_refs` has exactly `dev_reader` and `image_geometry`; the caller must also admit this bridge's own source and the original reader's full dependencies. Torch and NumPy are injected. PIL is imported only during `load`.

`original_image_refs` is the original ordinal-ordered list of 187 strict `path/bytes/sha256` JPEG refs, pre-registered by the Root binder. No filename inference, directory scan or GT read supplies missing identity. Each `load(index)` requires the actual packet's `identity.image` to equal that ordinal's ref, then closes the JPEG through the same admitted control. Metadata enumeration does not read all JPEG bytes.

The bridge follows the original U37 JPEG format, dimensions, absent/identity EXIF, no auto-orientation and RGB uint8 checks. It calls the existing geometry for RGB/255, integer keep-ratio content sizing, right/bottom padding to 800 and bilinear resize to 400. It does not claim to reconstruct teacher photometric normalization. U37 is provenance only; its FIT-specific reader, GT-matched receipt and `G` contract are not imported or adapted.

The returned item is a shallow copy retaining the same original `F`, `e`, `M`, `L`, `identity` and `query_ids` objects, plus only `rgb400`, `geometry` and `rgb_receipt`. It supplies no GT or synthetic count field. `reader.records` remains the original DEV reader's metadata. Its `image_references` is the ordinal mapping; `input_references` deduplicates original inputs, JPEG refs and the two module refs while rejecting conflicting path identities.

Native tensors are `F=[1,256,200,200]`, `e=[100,256]`, `M=[100,200,200]`, `L=[100,H,W]` with **natural query axes 0–99**. `query_ids` is the separate **original export permutation**. The evaluator must select e/M/L by actual query ids, not export ranks. This bridge never sorts queries, reconstructs M from L, or changes scores, categories, native tensors or prediction dictionaries.

Source pins:

- Published `N28_FIXED24_DEV_EVAL_DRAFT_20261017_01/dev_reader.py`: SHA256 `daabec8388e23d9ff646ae78eba69ba2168099a72971578a8d34cef04f1268c0`.
- Published `U33_REAL_IMAGE_400_PROBE_DRAFT_20261017_01/image_geometry.py`: SHA256 `65db107de80c5f1d4dade7cb2ffd3f241fd6bf7f9b233cb1aa9c90cbfd00abf7`.
- JPEG-loading provenance: published `U37_FULL_RGB_MATCHED_READER_DRAFT_20261017_01/rgb_matched_reader.py` lines 103–138, SHA256 `678ce3781872e21e961b24a7b0a4d0ecc69f0f8a08b8feb7eb0b2e8624d57025`.

The three tests exercise metadata mapping, image/added-field conflict rejection, and shallow preservation of opaque metadata objects plus arbitrary query export order. Opaque JSON carriers are not mock tensors or neural-network evidence. Actual decoding, tensor properties, full-187 population, zero-render native calibration, restored models and COCO must be checked within a future authorized Root-owned evaluation. No efficacy, AP or innovation conclusion follows from these tests.
