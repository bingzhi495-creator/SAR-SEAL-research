# Float-primary SSDD DEV187 owner integration

The adapter is a worker with stages `EXPORT_NATIVE`, `EXPORT_CENTER` and `SCORE`. It uses the pinned public CUDA-init stock stage for configuration, FP32 export, image-only scope, candidate artifacts and original COCO scoring. It reads new float-specific E24 arm/pair receipts from the independent verifier. The scorer never constructs historical integer acceptance records.

Invoke with the pinned Windows interpreter using `-I -B -X utf8 float_dev187.py --code-root <ordinary release code> --protocol <frozen PROTOCOL.md> --stage <stage> --request <fresh original request> --request-sha256 <exact request SHA>`. Keep `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, CUDA visible only for the export stage. Use one CPU thread for scoring. Run each export under a separate original owner and verify its natural terminal before the CPU scorer. An original failure yields no independent export acceptance.

Each file reference is the exact canonical `{path, bytes, sha256}` object from the original immutable file. Common request fields are `status=SSDD_FLOAT_PRIMARY_DEV187_STAGE_REQUEST_V1`, `stage`, `training_protocol`, `evaluation_protocol`, `training_pair_ref`, `native_arm_ref`, `center_arm_ref`, `image_manifest_ref` and `output_directory`. The export adds `archive_ref` and the original image-only `dev_root`. The score adds `float_gt_ref`, `integer_gt_ref`, `historical_ref`, `native` and `center`. Each score arm has `closure`, `report_ref`, `predictions_ref`, `export_manifest_ref`, `checkpoint_ref` and `acceptance_ref`. Its `closure` fields retain the original admission format with `original_os_wait=true`, `original_os_exit=0`, `resource_stop=false`, `worker_closed=true`, `terminal_sha256` and `report_sha256`. The score does not open either checkpoint.

The independent float export acceptance document has the following exact fields:

- `status=SSDD_FLOAT_COMMON_DEV187_EXPORT_INDEPENDENTLY_ACCEPTED_V1`
- `mode`, `evaluation_protocol`, `training_pair_sha256`, `export_report_sha256`, `request_sha256`
- `original_child_terminal_ref` pointing to the original `OWNER_TERMINAL_PRIVATE.json`
- `original_outer_wait_ref` pointing to the retained original external tool wait

The original terminal must independently show natural exit 0, assigned JobObject, no remaining processes and no stop reason. Its original `OWNER.stdout` ends with the exact worker marker `SSDD_FLOAT_COMMON_DEV187_STAGE_REPORT_WRITTEN_V1`, stage, request SHA and report SHA. The retained outer tool-wait document uses status `SSDD_FLOAT_COMMON_DEV187_EXPORT_ORIGINAL_TOOL_WAIT_V1`, `original_session_id`, `exact_tool_wait` and the shared keys `stage`, `request_sha256`, `export_report_sha256` and `original_child_terminal_sha256`. The original outer tool must have naturally exited 0 and its own final line must be JSON with `status=SSDD_FLOAT_COMMON_DEV187_EXPORT_ORIGINAL_OWNER_CLOSED_V1` and the four shared keys. Preserve the full original tool response in `exact_tool_wait`. The worker does not mint export or scoring acceptance.

The sealed `SCORE_SAFE.json` retains the full scores and first-seed gate, original per-arm candidate counters, FP32 export timings and peak allocation. Accept it only after the score owner's original wait and matching footer. Training cost can be drawn from the separately accepted training receipts/log aggregates when available.

Synthetic CPU tests use the real byte-pinned COCO backend with generated 187/380 fixtures and the already closed public aggregate reference. They do not read real DEV labels, JPEGs, predictions, FIT archives or TEST.
