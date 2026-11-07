# UNBOUND DiffSAR R101 CPU model-load probe

2026-11-07. This is a new, unbound candidate. Six standard-library contract tests passed; **the worker has not imported a scientific runtime, read weights, constructed a real model, or executed an actual load**. Root alone may freeze the inputs, grant one attempt after the main scientific task has actually ended, supervise the original process group, and accept the resulting observations. Neither a lease flag nor this worker's report authenticates those external facts by itself.

The future operation constructs a separate complete author DiffSAR model on CPU and checks an explicit ImageNet-R101 backbone transfer. It does not resume or modify the current random-initialization baseline, train, create an optimizer, run a model or normalizer forward, load FIT/DEV/TEST images or GT, evaluate COCO, or save a checkpoint. A successful CPU load does not prove GPU or forward compatibility, efficacy, paper initialization, or a fair full-system comparison.

## Interface

`python -B worker.py --spec /explicit/SPEC.json --spec-sha SHA256`

Exact top-level spec fields:

`schema, operation, run_id, source_refs, library_sources, config_ref, checkpoint_ref, root_lease_ref, output_directory, prelock`.

- Schema: `Root-R101-DiffSAR-CPU-load-probe-spec-v1`.
- Operation: `CPU_DIFFSAR_R101_EXPLICIT_BACKBONE_LOAD_PROBE_V01`.
- Run identity: `DiffSAR-R101-CPU-load-v01-` followed by 1–80 ASCII letters/digits/underscores.
- `source_refs` has exactly `worker`, `config_adapter`, `actual_load`, `coverage`.
- `library_sources` has exactly `LIBRARY_NAMES` from `worker.py`: the selected Torch/NumPy/author/D2 modules and binaries, including the newly explicit `frozenbn` module and `tv_extension` binary. These are a selected-source guarantee, not a complete transitive-runtime or environment-isolation proof.
- All refs are exact canonical POSIX `path/bytes/sha256` triples. Output must be fresh and outside all input/source paths. JSON duplicate keys and nonfinite constants are rejected before ordinary dictionaries hide them.
- `prelock` must match the exported `PRELOCK` with JSON type distinctions intact; it is not a set of caller-tunable resource values.

Worker/config/adapter/coverage/helper/checkpoint bytes must be independently bound by Root. The candidate additionally fixes the existing SSDD YAML, reviewed inactive-BIFPN schema adapter, strict 520-key metadata checker, and actual tensor-comparison helper identities. The V1 candidate is 178793939 bytes with observed full SHA256 `63fe2227b86e8f1f2063f43a75c84d195911b6a0eace650907dd3dc62dd49a0a`. That observed pin does not retrospectively create an independent official full-SHA or cache-provenance certificate.

## Exact Root lease

Lease schema is `Root-R101-DiffSAR-CPU-load-probe-lease-v1`, status `ROOT_GRANTED`. Exact fields are:

`schema, status, operation, run_id, prelock, source_refs, library_sources, config_ref, checkpoint_ref, output_directory, boot_id, single_attempt, main_scientific_processes_reaped, exclusive_project_scientific_leaf, whole_device_exclusive, main_accepted_prefix, worker_ref, prior_scientific_groups, main_acceptance_ref`.

The operation/run/prelock/source/library/config/checkpoint/output bindings must equal the spec. `worker_ref` must equal `source_refs.worker`, as required by the existing generic external parent. `prior_scientific_groups` must contain explicit positive-PID identity dictionaries; actual birth and whole-group absence are checked externally by that parent/Root, not inferred by the worker's metadata check. `main_acceptance_ref` is a syntactically valid explicit ref; Root must join it to the true accepted main boundary and observations. The worker neither reads Ops nor grants itself authority.

Main boundary is an integer from 31744 through 37050 on the existing 256 grid, with the sole final37050 exception. Flags require single attempt, all prior main scientific processes reaped, project-scoped exclusive scientific leaf, and no whole-device exclusivity. Boot must match the actual runtime boot. Adding arbitrary lease fields is rejected rather than silently ignored.

## Intended actual execution

The worker verifies its own/spec/selected source bytes and Root lease, requires the existing Linux/aarch64 Python3.12.14, Torch2.10.0+cu130, NumPy1.26.4 and Torchvision0.25.0+cu130 runtime, verifies actual imported module paths/bytes and the loaded TV binary, and keeps the author/config packages in one explicitly selected compatible root. These version gates are carried from the ordinary native full-training worker's existing runtime guard, not guessed from a framework compatibility table and not independently verified by this candidate-development task; the future real call must still pass them. It does not rewrite source files. `CUDA_VISIBLE_DEVICES` must be empty; CPU affinity is0/1, thread variables and actual Torch counts are2, startup free RAM48GiB, running floor16GiB, worker RSS cap16GiB, and deadline300seconds. The external parent remains responsible for whole-group RSS, all child identities, enforced deadline and cleanup; the worker's finite checkpoints are not a substitute.

Configuration comes from the unchanged pinned SSDD YAML and the reviewed inactive schema registration. Only `MODEL.DEVICE=cpu`, `SEED=40244023`, and `DATALOADER.NUM_WORKERS=0` are changed. The original `Trainer.build_model(cfg)` constructs the complete model; no Trainer instance is constructed, avoiding automatic training/evaluation/checkpoint hooks. `MODEL.WEIGHTS` remains empty, with explicit `torch.load(..., map_location='cpu', weights_only=True)` performed only after construction. No solver schedule is shortened or repurposed.

Actual checks include the author model class,500 proposals/six heads/one class, all-CPU FP32 parameters, all12 original1000-element FP64 diffusion buffers, complete3/4/23/3 R101,104 FrozenBN modules,33 modulated-DCN blocks and66 zero-initialized offset parameters. Convolution strides and groups are checked on the real objects. The author's modulated DCN stores scalar stride/dilation whereas ordinary Conv2d stores pairs, so the strict `spatial_pair` normalization handles these representations without changing them.

An important author-interface detail is that pixel_mean/std are not model attributes or named buffers: they are the two tensors captured by the `normalizer` lambda. The worker checks the real closure names, CPU/dtype/shape and values, clones those tensors independently before transfer, and verifies closure identity and values after transfer. It does not call the normalizer or see any image; **actual image values and absence of double normalization in the data path remain unverified**.

The worker calls only `actual_load.load_backbone_verified(torch, model, source_state, coverage)`, from the exact reviewed helper. That helper performs real tensor extraction/comparison if executed; the standard-library checker alone cannot produce a successful actual-load receipt. The full helper observation is kept. Success still leaves `qualified=False`, `forward=False`, `AP=False`, and training credit0. The source/config/checkpoint refs are reverified after the operation; every failure is preserved with nonzero exit and traceback, without automatic retry.

Output is a fresh worker directory with atomic `REPORT.json` updates and final worker status. A failure before spec/output admission may only appear on the original stderr, which Root must retain. Parent and original tool/SSH/process closure must be independently accepted before reporting a successful actual observation.

## Limited verification and source facts

Six standard-library tests use explicit SYNTHETIC refs/config objects only. Seven RED→GREEN changes covered the initial spec, closed-main lease, raw duplicate JSON and unsupported routes, surgical CPU config overrides, generic-parent lease binding, scalar DCN attributes, and exact helper-byte pin. Final suite:6 tests, actual exit0,0.004seconds. These tests never import Torch/NumPy, open a checkpoint, construct a model, run a real conversion/load/forward, or validate the claimed resource/runtime facts.

Static source review confirmed CPU device selection and closure normalization in the [author detector](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/diffsarshipinst/detector.py#L220), the scalar modulated-DCN attributes in [deform_conv.py](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/detectron2/layers/deform_conv.py#L388), and native model creation in [build.py](https://github.com/XiaowoXuSAR/DiffSARShipInst/blob/bdc3362372213bc8c3a1f6d5b7b74dc224d1b871/detectron2/modeling/meta_arch/build.py#L14). This is static support, not actual CPU/ABI execution. The original modulated DCN forward explicitly rejects CPU input; therefore this load-only worker must not be described as a CPU full-model forward test.
