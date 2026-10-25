# SAMRefiner first original image and native100 input check

Status: one real CPU input check completed and independently closed by Root on 2026-10-25. This is preparation for a future external method comparison; it is not SAMRefiner inference or evidence of method improvement.

## Entry and private inputs

The worker CLI is:

```text
python first_input_check.py --spec <absolute-private-spec-path> --spec-sha <exact-spec-SHA256>
```

Root must supply a prospectively pinned private specification, authenticated source/input references, a fresh output directory, and the external bounded execution/cleanup controller. The CLI must not be treated as an unsupervised standalone execution recipe. CUDA is hidden before scientific library imports.

The accepted attempt closed 16 explicitly admitted source/input/library references. The closure covers the selected worker/control/reader/geometry/decoder components and the specific private manifest, stage metadata and original input required by the pinned specification. The exact 16-entry inventory, paths, bytes and SHA256 values remain in the private specification and Root closure evidence. This is a finite selected-source footprint, not a complete closure of Python, C extensions, shared libraries, image codecs or the operating system.

The U59 portable reader is reused through its real prepare/load interface, and U65 validate_loaded_metadata checks the consumed single-image ABI. Only the original first DEV ordinal is loaded. No GT-based selection, sorting of candidates, score replacement, mask deletion or SAM model loading is part of this worker.

## Actual scope and observations

Root reported the following accepted observations:

| Item | Actual observation |
| --- | --- |
| Selected input | ordinal 0, image_id 4 |
| Original image geometry | height 317, width 420 |
| Native candidates | all original 100, all nonempty in this image |
| Pixel round trip | current PIL RGB pixels, saved lossless PNG and OpenCV-decoded RGB pixels matched exactly |
| Candidate/mask outputs | original ordering and nonmask metadata retained; saved masks and candidate metadata reloaded and checked |
| Worker elapsed time | 1.798 seconds |
| Worker peak self RSS | 782,811,136 bytes |
| Parent resource observations | 9 samples |
| Maximum sampled group RSS | 733,487,104 bytes |
| Minimum sampled available RAM | 78,076,952,576 bytes |
| Resource violations | none observed |
| SAM / GT / AP | no SAM inference, GT matching or AP computation |

Worker maximum self RSS and the parent's maximum sampled group RSS are different measurements. Their difference does not establish that memory use remained below a limit between parent samples.

Original execution evidence: session 16439; tool, SSH and child exit codes were all 0, with terminal observation 0e9094. Independent Root closure 9e0183 exited 0 and verified the 16 admitted references, output bytes, the same boot/birth identity, and a strictly empty owned process group. The first post-observation attempt 7017721 failed because its metadata observer lacked sys; that observer failure was preserved and corrected without replaying the numerical work.

## Test and revision history

Three stdlib tests were run test-first with genuine failing and passing observations. They cover only:

1. Relocation of an identical original JPEG reference without rewriting historical provenance.
2. Rejection of a changed image digest.
3. Rejection of an image identity that tries to reuse another image's reference.

These tests do not exercise PIL, OpenCV, pycocotools, Torch, the portable package, real image pixels or the complete worker. Root observed the final three-test pass after the thread checks were added as a6d806, exit 0. An earlier whole-module Windows import failed at a5f4ea1 because resource was imported at module scope; resource was moved inside main.

The earlier unexecuted draft also lacked part of the declared two-thread setup/checks. Static review identified the missing Torch interop setting and the need to set/check the OMP, MKL and OpenBLAS thread environment. Root reported that these were corrected before the real attempt. The executed revised worker SHA256 is 5db5ac9b741d99b32c8791024ef8fe0572721459600dd676f9f4dc49db482315, verified against the closed private execution source. The earlier draft must not be confused with the executed revision.

## Resource supervision and termination scope

The outer Root controller used GNU timeout with a 90-second KILL bound. The child execution was also monitored against an 85-second deadline. Execution retained an independent session/process-group identity and external birth/PGID cleanup verification. Root alone accepted the actual terminal and strictly empty owned group.

CPU/RAM supervision consists of actual control checks and parent samples, including sampled group RSS and available RAM. It is not a continuous RAM guarantee, a proof about unobserved intervals, or complete whole-system memory certification. No automatic retry or silent increase of limits is established by this result.

## Interpretation and remaining checks

The original JPEG was joined to its historical image reference by exact bytes/digest while allowing a separately declared relocated path. The PIL-to-PNG-to-OpenCV check verifies lossless preservation of the RGB array decoded by the current installed PIL stack. It does not prove that the current JPEG decoder reconstructs exactly the same pixels as the historical Windows decoder, or that cached preprocessing tensors match a reconstruction from the current decoder.

A future separately registered and bounded check should reconstruct rgb400 from the current PIL-decoded original using the exact reviewed historical RGB/geometry preprocessing, then compare that result with the stored historical rgb400 field. Decoder behavior, resizing, geometry, dtype and normalization must be distinguished; any mismatch must be measured and reported before deciding whether the original pixel/preprocessing contract is reproduced. That comparison has not been performed by this single-image check.

This result does not validate all 187 DEV images, certify a SAM checkpoint or model, test the author's internal mask-return order, compute AP, establish an accuracy gain, or establish publication readiness. It also does not establish complete binary dependency closure or historical Windows/current decoder full-pixel equality.

## Publication boundary

This directory's reviewable worker, stdlib tests and explanatory README may be handled according to the project's publication rules. Operational specifications and manifests, credentials or host details, original JPEGs, PNGs, masks, native candidate outputs, weights and other private payloads must not be published with this source candidate. This README records Root-reported execution evidence and scope without rereading or redistributing those payloads.

