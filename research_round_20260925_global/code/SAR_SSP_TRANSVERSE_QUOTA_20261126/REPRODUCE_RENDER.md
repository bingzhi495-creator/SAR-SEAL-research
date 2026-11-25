# Label-free transverse quota replay

The renderer applies the published prelock to the original closed SAR-SSP transfer. It uses native mask pixel centers in row/column coordinates. An eligible mask has one eight-connected component, positive major-axis variance, and a covariance eigenvalue ratio of at least four. The axes are fixed from the original mask. Projected variance for each candidate mask is about that mask's own projected mean.

The original maximin objective and its original first-tie order determine one set of selected sites per feasible addition count. The transverse arm allows quota winners whose short-axis variance remains within the locked numerical tolerance of the balanced winner. The two controls admit the same number of quota choices, including the balanced winner. Their remaining options are ordered by absolute area change and by longitudinal variance, respectively. All three use the original objective for final selection and preserve \(2k_s\) actual flips per positive-budget candidate. The original variable writer is used for any geometrically ineligible candidate. No radar look direction is inferred from the raster axes.

## Replay

Use the original [closed SAR-SSP variable transfer](../SAR_SSP_DFINE_BCE5297_VARIABLE_SIGN_TRANSFER_20261126/README.md) and its Python 3.11.15, Torch 2.13.0 CPU, NumPy 1.26.4 and pycocotools environment. Set CUDA_VISIBLE_DEVICES to -1 and numerical libraries to one thread. The archive-dependent renderer binds the original worker and original render and score receipts. The standalone quota rule only imports NumPy and uses the previously frozen writer's probability helpers.

Run the archived capture observer around the renderer, with arguments shown by the renderer's help. Invoke the Python child with interpreter flags -I -B -X utf8. Pass render as the phase, the closed original variable-transfer directory, the new core and SHA-256, the published prelock and SHA-256, the verified commit, a fresh private work root and its absent render child. Use capture as the observer phase, with a separate observed_render directory and render/RESULT_SAFE.json as the result path.

The result footer binds the edit ledger, original reference receipts, locked core and original writer. The actual per-image ledger remains private. Only closed summaries, protocols and sources belong in the release.

For synthetic tests, set SAR_SEAL_VARIABLE_WRITER to the released original variable_writer.py and SAR_SEAL_ARCHIVED_TRANSFER_WORKER to the closed original transfer.py. Run test_transverse_quota.py. Compilation of the archived worker checks the three code seams without running the archived program or accessing scientific payloads.

