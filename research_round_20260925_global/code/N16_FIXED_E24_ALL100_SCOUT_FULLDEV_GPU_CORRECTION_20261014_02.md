# N16 FullDEV GPU entry 02 — local variable shadowing correction

UNBOUND, not scientifically executed. This revision changes only the local CPU spec variable in `training_gate` from `cpu` to `cpu_spec`; the Boolean `cpu` mode is preserved. All evaluation, capture, budget and decoder behavior follows the unchanged [01 contract](N16_FIXED_E24_ALL100_SCOUT_FULLDEV_GPU_ROOT_UNBOUND_DRAFT_20261014_01.md). Use the 02 Python source for future ROOT binding; 01 contains a confirmed TypeError and must not execute.

Stdlib-only owner-prefix regression executes ordinary AST statements with synthetic read/lineage callbacks: 01 reproduces the exact `int(dict)` failure for both modes; 02 GPUFalse/109 and CPUTrue/110 both bind training owners107/108, and shifted CPU108/109 is rejected. Four tests passed. No science file, model, CLI, CUDA, Source hash or actual RAW was accessed. Other actual ABI/qualification/resource obligations remain ROOT responsibilities.
