# Extra UEIKAP acquisition families

The [locked comparison](../../UEIKAP_ADDITIONAL_FAMILIES_PRELOCK_20261123.md) uses 342 author PNGs and 528 released hull polygons from 13 additional Sentinel-1 acquisitions. They come from the same Danish-region release as the earlier 108 images. The population excludes the earlier scored families and the individually logged preview and QA chips.

The [author ZIP](https://zenodo.org/records/18391027) remains with the publisher. Reconstruct the input roster from the exact historical preflight and recovery inputs named by `build_roster.py`. Its IDs are assigned in lexicographic order of the original TIFF ZIP member name. The original files and the previous closed intake provenance remain local to the operator. After the roster exits successfully, `fetch_author_subset.py` verifies the observed result, reads the pinned ZIP ranges and saves only the selected original PNGs and TXT files. It stores separate image and label manifests and checks decoded RGB uniqueness across the two cohorts.

Use the [B5 Windows CPython 3.10.13 environment](../INSTANCE_RSDD_B5_REPRO_20261123/README.md) and the original center5202 checkpoint, the same seed-5217 frozen HRSID point bundle and its accepted fit receipts. On the Conda-based Windows host use the documented stand-alone interpreter if the venv launcher fails to load the PyTorch libraries. Use a process observer that records an actual exit, worker SHA-256, output footer verification and result SHA-256. The [observer example](../RFDETR_FRACTIONAL_TIE_20261123/README.md) shows the receipt format. Each worker refuses preexisting output directories.

| Step | Entry point and inputs |
| --- | --- |
| Roster | `build_roster.py --preflight DIR --recovery DIR --output DIR` |
| Intake | `fetch_author_subset.py --roster FILE --roster-result FILE --roster-terminal FILE --protocol FILE --prior-provenance FILE --output DIR` |
| Forward | `capture_additional.py capture --code DIR --intake DIR --intake-terminal FILE --protocol FILE --protocol-sha256 SHA --checkpoint FILE --output DIR` |
| Read back | `capture_additional.py verify-capture` with the shared options, `--capture DIR --capture-terminal FILE` in place of `--checkpoint`, and a separate output |
| Edit | `edit_additional.py prepare-edits --authority FILE --authority-sha SHA --output DIR`, followed by `verify-edits` with the observed edit inputs |
| Score | `score_additional.py --code DIR --protocol FILE --roster FILE --intake DIR --intake-terminal FILE --intake-terminal-sha SHA --intake-result-sha SHA --capture DIR --prepared DIR --readback FILE --readback-sha SHA --readback-terminal FILE --readback-terminal-sha SHA --output DIR` |

Run Python with `-I -B -X utf8`. Use `CUDA_VISIBLE_DEVICES=0` for forward and `-1` for readback, editing and scoring. The edit authority contains the absolute paths and digests in `edit_additional.py` and uses the capture readback and its observer. Include the prepared result and observer only for edit verification. The scorer uses that verified result and observer. It opens the author masks only after validating the frozen predictions and edit stream. Results concern agreement with the released hulls and retain the no-match cases in the acquisition-weighted denominator.

[Verified results and execution receipts](../../UEIKAP_ADDITIONAL_FAMILIES_ACTUAL_20261123.md).
