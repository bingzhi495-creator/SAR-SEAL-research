# Frozen SAR-SSP transfer with variable signed boundary editing

[Results and interpretation](ACTUAL.md) · [Prespecified protocol](PRELOCK.md) · [Label-free render summary](render/RESULT_SAFE.json) · [Scoring summary](score/RESULT_SAFE.json) · [Verified execution summary](EXECUTION_SAFE.json)

This archive documents a retrospective frozen transfer. It reports publication-scope summaries and a portable path adapter for the executed worker. Reproduction requires the previously frozen candidate capture and source weights. Obtain the SAR-SSP author data through the established dataset channels. The render used Python 3.11.15, NumPy 1.26.4, PyTorch 2.13.0+cu130 on CPU, and pycocotools 2.0.11. The earlier SAR-SSP run supplied the qualified NumPy and RLE reader. The worker checks the previously established input digests before replay.

Provide a local JSON file whose keys map to the paths listed below. Root paths are directories, and explicit source paths are files. Relative paths are resolved against the JSON file.

| Key | Required input |
| --- | --- |
| prior_sar_ssp_run | Closed D-FINE E24 SAR-SSP run with capture, prepare, verify, score and the original replay worker |
| source_variable_score_run | Closed source-development score for the variable signed-budget experiment |
| source_variable_render_run | Closed source-development render and its exact variable writer |
| released_code_root | Research code directory with the established D-FINE point features and SAR-SSP evaluator |
| sar_ssp_intake | Original image-only intake with the verified manifest and JPEG files |
| first_signed_action_gate_run | Closed first matched signed-action source gate, with the checkpoint loader |
| earlier_gate_render_source | Exact earlier source gate render Python file that provides the natural-exit validator |
| first_matched_fit_run | Closed BCE seed 5297 matched training run with its accepted checkpoint and scaler |
| occupancy_core_source | Exact matched occupancy training core Python file |
| fixed_count_selector_source | Exact fixed-count selector Python file |
| author_zip | Original SLGSea author dataset archive |

Set the process to CPU with CUDA_VISIBLE_DEVICES=-1. Use the repository's observe_capture.py actual-exit wrapper, stored under the PSEG_RFDETR_FROZEN_SOURCE_HEAD_20261122 code directory. Run the path adapter with -I -B -X utf8 and phases render and score. Give the commands the same existing empty private work root and a private inputs JSON. The output directories must be work-root/render and work-root/score. The capture observer writes work-root/observed_render/TERMINAL.json. After it exits successfully and verifies the result footer, start the verify observer for the score. The score opens author annotations only after independently reconstructing the new edited masks from the closed render.

The two RESULT_SAFE files are byte-identical to the original aggregate receipts. EXECUTION_SAFE records their verified natural exits and the path-adapted source hash. The portable version preserves the experiment logic after replacing host-specific input paths, so its worker and timing fields differ from the original run. Full replay requires the bound artifacts and the former environment.
