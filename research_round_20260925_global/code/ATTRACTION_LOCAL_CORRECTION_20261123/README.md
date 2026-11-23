# Corrections fitted at disputed native boundary locations

The [locked source-only experiment](../../ATTRACTION_LOCAL_CORRECTION_PRELOCK_20261123.md) keeps the original D-FINE candidates, mask scores, support, area and signed exchange counts. C fits the two score differences. Q adds the 14 original endpoint feature differences. CM uses the C ordering and exactly the number of Q replacements in each candidate and sign.

Run the workers from this release in the pinned CPU runtime used by [the original D-FINE signed-side replay](../ATTRACTION_SIGNED_SIDES_20261123/README.md). Use isolated Python with `-I -B -X utf8`, one native compute thread and no CUDA visibility. Use fresh output directories and an observer recording the actual exit, worker SHA-256, result SHA-256 and footer. The [observer example](../RFDETR_FRACTIONAL_TIE_20261123/README.md) describes this format. Keep the original images, labels, native predictions and frozen head bundles local.

| Phase and observer label | Worker |
| --- | --- |
| `source` | `source_pairs.py --root DIR --code DIR --protocol FILE --protocol-sha SHA --gate-sha SHA --output DIR/source`. Root is the parent of the accepted original FIT109 and capture runs. The worker reconstructs the legacy five folds and binds the fractional polygon and fixed-match identities. |
| `fit` | `gate_model.py --source DIR/source --source-terminal FILE --protocol FILE --protocol-sha256 SHA --output DIR/model`. Close both ridge fits before applying them to target pairs. |
| `prepare-pairs` | `target_pairs.py --cohort NAME --root DIR --code DIR --protocol FILE --protocol-sha256 SHA --gate-model-sha256 SHA --output DIR/features`. Use `dev705` with the accepted HRSID PNG ZIP via `--archive`, or `pseg232` with the accepted image-only owner directory via `--owner`. |
| `replay` | `apply_correction.py --root DIR --code DIR --cohort NAME --manifest FILE --protocol FILE --protocol-sha256 SHA --features DIR/features --feature-terminal FILE --fit DIR/model --fit-terminal FILE --output DIR/replay`. The manifest is the original accepted image-only roster for the indicated cohort. |
| `score` | `score_correction.py --root DIR --code DIR --cohort NAME --manifest FILE --phase-root DIR --protocol FILE --protocol-sha256 SHA --fit-result FILE --fit-terminal FILE --feature-result FILE --feature-terminal FILE --shapely-site DIR --output DIR/score`. PSeg scoring also takes the exact historical float annotation JSON using `--gt`. All actual mask inputs close before opening target geometry. |

The exact legacy resources and archived hashes are identified by each worker and the linked [HRSID comparison](../../HRSID_DEV705_CLASSICAL_ATTRACTION_ACTUAL_20261122.md) and [PSeg comparison](../../PSEG_TEST232_CLASSICAL_ATTRACTION_ACTUAL_20261122.md). Inference phases keep the legacy base head frozen. Report detector forwards, base point-head readout, outside-fold optimizer steps, ridge solves, and elapsed observation times separately.

