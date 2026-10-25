"""Unexecuted single-image SAMRefiner driver; stdlib-only at import."""
from pathlib import Path, PurePosixPath
import math

AUTHOR_COMMIT = '4bb7f95738c7a2e805f189210fb09e02a2197557'
CHECKPOINT_URL = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
CHECKPOINT_BYTES = 2564550879
OPERATION = 'GPU_SAMREFINER_ORIGINAL_ORDINAL0_SHARED_FUNCTION_PROBE_V02'
CPU_LIMITS = dict(allowed_cpus=[0, 1], threads=2,
              maximum_RSS_bytes=16*1024**3, minimum_available_RAM_bytes=16*1024**3,
              deadline_seconds=1800)
PARAMS = dict(iters=5, margin=0.0, gamma=4.0, strength=30, use_samhq=False)
SOURCE_PINS = {
    'fixed_adapter': 'eb0609708bd189ebfda381c9ef5700c2bdba53fbeba0837febeff24fdbb9cac2',
    'bridge': '3da3bbd6d64ba5c4f9a28b243528f9bfe095c7b2348297885e3b6105ae22d855',
    'author_refiner': 'b467d8e3e3c2fc395466a293635774cb821c5ffebdb39c4e5d64ac93c49dd3ad',
    'author_utils': '0d0c8e4b257935968c39f4223993b5b7de1868a4a7d886092c6a0074131299ef',
    'sam_init': 'c8d8eb649610b2d51f138481a89617a3dcb61106465121dcb1c2f23affc84c30',
    'sam_registry': '77bc4faf728118625ddd184dc4e6375bd1b5c636fc66fb429f9fc5276f331cc5',
    'sam_model': '588d294a48bbd4ce5c5f4f560fa862aee0a8325a8dadda4e7d848a9f23d27946'}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def strict_ref(value):
    need(type(value) is dict and set(value) == {'path', 'bytes', 'sha256'},
         'Exact path/bytes/SHA256 reference required')
    path = value['path']
    need(type(path) is str and PurePosixPath(path).is_absolute()
         and str(PurePosixPath(path)) == path and '..' not in PurePosixPath(path).parts,
         'Canonical absolute POSIX reference required')
    need(type(value['bytes']) is int and value['bytes'] >= 0
         and type(value['sha256']) is str and len(value['sha256']) == 64
         and set(value['sha256']) <= set('0123456789abcdef'),
         'Exact byte count and full lowercase SHA256 required')
    return value


def validate_spec(spec):
    need(type(spec) is dict and spec.get('exclusive_project_scientific_leaf') is True
         and spec.get('whole_device_exclusive') is False and 'exclusive_GPU' not in spec,
         'Only explicit shared v02 project-leaf scope is accepted')
    need(type(spec) is dict and spec.get('schema') == 'ROOT-SAMRefiner-one-image-shared-probe-spec-v02'
         and spec.get('operation') == OPERATION
         and type(spec.get('run_id')) is str and spec['run_id'].startswith('SAMRefiner-shared-v02-') and '/' not in spec['run_id'],
         'Root-pinned single-image trial specification required')
    pre = spec.get('prelock')
    need(type(pre) is dict and type(pre.get('limits')) is dict, 'Explicit resource fields required')
    limits = pre['limits']
    cap = limits.get('allocator_bytes')
    startup = pre.get('startup_min_available_RAM_bytes')
    need(type(cap) is int and cap > 0 and type(startup) is int
         and startup >= cap + CPU_LIMITS['maximum_RSS_bytes'] + CPU_LIMITS['minimum_available_RAM_bytes']
         and limits == dict(CPU_LIMITS, allocator_bytes=cap),
         'Explicit allocator cap and conservative startup RAM budget required; no automatic upgrade')
    need(pre == dict(author_commit=AUTHOR_COMMIT, model_type='vit_h',
         ordinal=0, image_id=4, precision='float32', amp=False, allow_cpu_fallback=False,
         params=PARAMS, limits=limits, startup_min_available_RAM_bytes=startup,
         resource_plan='INDEPENDENT_SHARED_GPU_FUNCTION_PROBE_V02'),
         'Fixed original image, author defaults, FP32 and separate Root resource plan required')
    refs = spec.get('source_refs')
    need(type(refs) is dict and set(refs) == set(SOURCE_PINS) | {'worker', 'resource_control'},
         'Nine explicit reviewed source roles required')
    for role, pin in SOURCE_PINS.items():
        need(strict_ref(refs[role])['sha256'] == pin, 'Fixed source changed: ' + role)
    libraries = spec.get('library_sources')
    inputs = spec.get('input_refs')
    need(type(libraries) is dict and set(libraries) == {'torch', 'numpy', 'psutil', 'cv2', 'PIL'}
         and type(inputs) is dict and set(inputs) == {'png', 'masks', 'candidates'},
         'Explicit selected library and already checked input references required')
    values = list(refs.values()) + list(libraries.values()) + list(inputs.values())
    need('gpu_background_observation_ref' in spec, 'Shared background observation reference required')
    values += [spec[k] for k in ('checkpoint_ref', 'checkpoint_stage_ref', 'input_report_ref', 'root_lease_ref', 'gpu_background_observation_ref')]
    unique = {}
    for value in values:
        strict_ref(value)
        need(value['path'] not in unique or unique[value['path']] == value,
             'Repeated reference paths must retain identical bytes/SHA256')
        unique[value['path']] = value
    need(spec.get('sources') == [unique[path] for path in sorted(unique)],
         'Exact explicit source/input closure required')
    out = spec.get('output_directory')
    need(type(out) is str and PurePosixPath(out).is_absolute()
         and str(PurePosixPath(out)) == out and '..' not in PurePosixPath(out).parts and out != '/',
         'Fresh canonical absolute output directory required')
    need(all(PurePosixPath(out) != PurePosixPath(path)
             and PurePosixPath(out) not in PurePosixPath(path).parents for path in unique),
         'Outputs cannot contain admitted inputs or sources')


def validate_checkpoint_stage(spec, stage):
    checkpoint = strict_ref(spec['checkpoint_ref'])
    need(checkpoint['bytes'] == CHECKPOINT_BYTES
         and spec.get('verified_checkpoint_sha256') == checkpoint['sha256'],
         'Root must pin the complete official ViT-H bytes and verified full SHA256')
    need(type(stage) is dict and stage.get('schema') == 'SAMRefiner-CHECKPOINT_STAGE-v1'
         and stage.get('status') == 'ROOT_VERIFIED_COMPLETE',
         'A closed Root-verified complete CHECKPOINT_STAGE is required')
    need(stage.get('checkpoint_ref') == checkpoint
         and stage.get('verified_sha256') == checkpoint['sha256']
         and stage.get('model_type') == 'vit_h'
         and stage.get('author_commit') == AUTHOR_COMMIT
         and stage.get('official_url') == CHECKPOINT_URL,
         'CHECKPOINT_STAGE must bind this exact official checkpoint and author variant')


def candidate_identities(rows, shape):
    need(type(rows) is list and len(rows) == 100, 'Exactly the original 100 candidates required')
    identities = []
    for row in rows:
        need(type(row) is dict and {'image_id', 'query_id', 'category_id', 'score', 'segmentation'} <= set(row),
             'Original native prediction fields required')
        need(type(row['image_id']) is int and row['image_id'] == 4
             and type(row['query_id']) is int and type(row['category_id']) is int
             and row['category_id'] == 1 and type(row['score']) in (int, float)
             and math.isfinite(row['score']) and 0 <= row['score'] <= 1,
             'Original image/query/class/raw finite S0 required')
        need(type(row['segmentation']) is dict and row['segmentation'].get('size') == list(shape)
             and type(row['segmentation'].get('counts')) is str, 'Original-grid opaque native RLE required')
        identities.append({key: row[key] for key in ('image_id', 'query_id', 'category_id', 'score')})
    need({row['query_id'] for row in identities} == set(range(100)),
         'Distinct original 100 IDs required; preserve their export permutation')
    return identities


def _verify(ref):
    """Bootstrap only the Root-pinned stdlib control before importing psutil."""
    import hashlib
    strict_ref(ref)
    path = Path(ref['path'])
    need(str(path.resolve(strict=True)) == ref['path'] and not path.is_symlink(),
         'Exact canonical admitted bootstrap path required')
    before = path.stat()
    digest = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            size += len(block)
            need(size <= ref['bytes'], 'Bootstrap file exceeded its admitted size')
            digest.update(block)
    after = path.stat()
    identity = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    need(identity(before) == identity(after) and size == ref['bytes']
         and digest.hexdigest() == ref['sha256'], 'Stable admitted bootstrap bytes required')
    return path


def _load(path, alias, originals):
    import sys
    import types
    originals.setdefault(alias, (alias in sys.modules, sys.modules.get(alias)))
    module = types.ModuleType(alias)
    module.__file__ = str(path)
    sys.modules[alias] = module
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def validate_background_observation(value, boot_id):
    import datetime
    import uuid
    need(type(value) is dict and set(value) == {'schema', 'boot_id', 'observed_UTC',
         'original_nvidia_smi_returncode', 'gpu_index', 'whole_device_exclusive', 'processes'}
         and value['schema'] == 'SAMRefiner-shared-GPU-background-observation-v02',
         'Exact v02 shared background observation schema required')
    need(type(value['boot_id']) is str and value['boot_id'] == boot_id
         and str(uuid.UUID(value['boot_id'])) == boot_id, 'Same canonical boot UUID required')
    need(type(value['observed_UTC']) is str, 'Original UTC observation time required')
    try:
        stamp = datetime.datetime.fromisoformat(value['observed_UTC'].replace('Z', '+00:00'))
    except ValueError as error:
        raise ValueError('Valid UTC observation time required') from error
    need(stamp.tzinfo is not None and stamp.utcoffset() == datetime.timedelta(0)
         and type(value['original_nvidia_smi_returncode']) is int
         and value['original_nvidia_smi_returncode'] == 0
         and type(value['gpu_index']) is int and value['gpu_index'] == 0
         and value['whole_device_exclusive'] is False,
         'Original successful nvidia observation, GPU0 and explicit nonexclusivity required')
    processes = value['processes']
    need(type(processes) is list and len(processes) > 0, 'Explicit observed background process list required')
    pids = []
    for process in processes:
        need(type(process) is dict and set(process) == {'pid', 'start_ticks', 'name', 'used_memory_MiB'}
             and type(process['pid']) is int and process['pid'] > 0
             and type(process['start_ticks']) is int and process['start_ticks'] > 0
             and type(process['used_memory_MiB']) is int and process['used_memory_MiB'] >= 0
             and type(process['name']) is str and process['name'].strip() == process['name']
             and process['name'] and not any(char in process['name'] for char in ('/', '\\', '\n', '\x00')),
             'Only PID/birth/kernel comm name/observed MiB permitted; no cmdline or weights')
        pids.append(process['pid'])
    need(len(set(pids)) == len(pids), 'Unique observed background PIDs required')


def validate_shared_lease(spec, lease, boot_id):
    need(spec.get('exclusive_project_scientific_leaf') is True
         and spec.get('whole_device_exclusive') is False and 'exclusive_GPU' not in spec,
         'Worker SPEC must explicitly use the shared v02 scope')
    strict_ref(spec['gpu_background_observation_ref'])
    need(type(lease) is dict and lease.get('schema') == 'Root-SAMRefiner-one-image-shared-lease-v02'
         and lease.get('status') == 'ROOT_GRANTED' and lease.get('run_id') == spec['run_id']
         and spec['run_id'].startswith('SAMRefiner-shared-v02-')
         and lease.get('operation') == OPERATION
         and lease.get('worker_ref') == spec['source_refs']['worker']
         and lease.get('prelock') == spec['prelock'] and lease.get('single_attempt') is True
         and lease.get('exclusive_project_scientific_leaf') is True
         and lease.get('whole_device_exclusive') is False and 'exclusive_GPU' not in lease
         and lease.get('main_scientific_processes_reaped') is True
         and lease.get('scope') == 'one_image_shared_GPU_resource_and_function_probe_only'
         and lease.get('gpu_background_observation_ref') == spec['gpu_background_observation_ref']
         and lease.get('boot_id') == boot_id,
         'Independent v02 project-leaf lease with explicit shared GPU background required')


def main(argv=None):
    import argparse
    import gc
    import hashlib
    import importlib
    import inspect
    import json
    import os
    import sys
    import time

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--spec-sha', required=True)
    args = parser.parse_args(argv)
    spec_path = Path(args.spec)
    need(spec_path.is_absolute() and spec_path.stat().st_size <= 2*1024**2, 'Small absolute private SPEC required')
    raw = spec_path.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == args.spec_sha, 'Exact Root-pinned SPEC digest required')
    spec = json.loads(raw)
    validate_spec(spec)
    need(sys.platform == 'linux' and os.environ.get('CUDA_VISIBLE_DEVICES') == '0',
         'External Root GPU lease must expose only device zero')
    need(not any(name.split('.')[0] in ('torch', 'numpy') for name in sys.modules),
         'Fresh child with no prior scientific imports required')
    os.sched_setaffinity(0, {0, 1})
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '2'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['MPLBACKEND'] = 'Agg'
    current_nice = os.getpriority(os.PRIO_PROCESS, 0)
    if current_nice < 19:
        os.nice(19-current_nice)
    out = Path(spec['output_directory'])
    need(str(out.resolve()) == str(out), 'Output parents must not relocate through symlinks')
    out.mkdir(mode=0o700, exist_ok=False)
    started = time.monotonic()
    aliases, handles = {}, []
    torch = c = sam = state = backend = refined = None
    report = dict(schema='SAMRefiner-one-image-shared-probe-report-v02', status='STARTED',
        run_id=spec['run_id'], operation=OPERATION, stage='STARTED',
        exclusive_project_scientific_leaf=True, whole_device_exclusive=False, speed_benchmark=False,
        gpu_background_observation_ref=spec['gpu_background_observation_ref'],
        background_load_locked=False,
        limits=spec['prelock']['limits'], startup_min_available_RAM_bytes=spec['prelock']['startup_min_available_RAM_bytes'],
        source_footprint='Selected admitted files; not complete Python/native/shared-library closure',
        checkpoint_stage_verified=False, checkpoint_bytes_verified=False,
        input_stage_verified=False, external_lease_record_verified=False,
        external_live_supervision_self_certified=False, model_constructed=False,
        checkpoint_deserialized=False, weights_loaded=False, model_to_CUDA_returned=False,
        author_refiner_entered=False, author_refiner_returned=False, CUDA_synchronized_after_refiner=False,
        inference_completed=False, configured_iterations=5, completed_rounds_observed=None,
        model_hooks={name: dict(pre=0, post=0) for name in ('image_encoder', 'prompt_encoder', 'mask_decoder')},
        AP=False, GT_loaded=False, training_updates=0, full187=False,
        author_internal_return_order_verified=False, historical_rgb400_vs_current_PNG_compared=False,
        historical_input_equivalence_closed=False, precision='float32', amp=False,
        cpu_fallback=False, automatic_retry=False, whole_GPU_memory_contract_certified=False)
    proc = Path('/proc/self/stat').read_text()
    report.update(pid=os.getpid(), pgid=os.getpgrp(),
                  start_ticks=int(proc[proc.rfind(')')+2:].split()[19]),
                  boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    events = (out/'STAGES.jsonl').open('x', encoding='utf-8')
    exit_code = 1

    def persist():
        temporary = out/'REPORT.next.json'
        temporary.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+'\n')
        os.replace(temporary, out/'REPORT.json')

    def event(stage, **details):
        report['stage'] = stage
        record = dict(stage=stage, elapsed_seconds=time.monotonic()-started,
                      model_hooks={key: dict(value) for key, value in report['model_hooks'].items()}, **details)
        events.write(json.dumps(record, sort_keys=True, allow_nan=False)+'\n')
        events.flush()
        os.fsync(events.fileno())
        persist()

    try:
        event('AUTHENTICATING_CONTROL')
        control = _load(_verify(spec['source_refs']['resource_control']), '_SAM_TRIAL_CPU_CONTROL', aliases)
        need(control.LIMITS == CPU_LIMITS, 'Reviewed unchanged CPU/RSS/file control component required')
        _verify(spec['library_sources']['psutil'])
        psutil = importlib.import_module('psutil')
        need(str(Path(psutil.__file__).resolve()) == spec['library_sources']['psutil']['path'],
             'Actual admitted psutil module required')
        c = control.make_control(dict(prelock=dict(limits=dict(CPU_LIMITS)), sources=spec['sources']), psutil)
        c.check()
        need(int(psutil.virtual_memory().available) >= spec['prelock']['startup_min_available_RAM_bytes'],
             'Resource failure: independent startup available-RAM gate not met')
        report['startup_available_RAM_bytes'] = int(psutil.virtual_memory().available)
        stage = json.loads(c.closed(spec['checkpoint_stage_ref']).read_bytes())
        validate_checkpoint_stage(spec, stage)
        report['checkpoint_stage_verified'] = True
        lease = json.loads(c.closed(spec['root_lease_ref']).read_bytes())
        validate_shared_lease(spec, lease, report['boot_id'])
        background = json.loads(c.closed(spec['gpu_background_observation_ref']).read_bytes())
        validate_background_observation(background, report['boot_id'])
        report['background_observed_UTC'] = background['observed_UTC']
        report['external_lease_record_verified'] = True
        event('ROOT_STAGE_AND_LEASE_RECORDS_VERIFIED')
        for ref in spec['sources']:
            if ref != spec['checkpoint_ref']:
                c.closed(ref)
        need(Path(__file__).resolve() == Path(spec['source_refs']['worker']['path']),
             'Actual admitted trial worker required')
        checkpoint = c.closed(spec['checkpoint_ref'])
        report['checkpoint_bytes_verified'] = True
        input_report = json.loads(c.closed(spec['input_report_ref']).read_bytes())
        need(input_report.get('status') == 'FIRST_ORIGINAL_RGB_AND_NATIVE100_INPUT_CHECK_PASSED'
             and input_report.get('ordinal') == 0 and input_report.get('image_id') == 4
             and input_report.get('queries') == 100 and input_report.get('nonempty_masks') == 100
             and input_report.get('PNG_ref') == spec['input_refs']['png']
             and input_report.get('masks_ref') == spec['input_refs']['masks']
             and input_report.get('candidates_ref') == spec['input_refs']['candidates'],
             'Closed Root-accepted first-image output references must join this trial')
        shape = input_report['original_shape']
        need(shape == [317, 420], 'Fixed real first-image original geometry required')
        report.update(input_stage_verified=True, ordinal=0, image_id=4, original_shape=shape, queries=100)
        event('CHECKPOINT_BYTES_AND_INPUT_STAGE_VERIFIED')
        modules = {}
        for name in ('torch', 'numpy', 'cv2', 'PIL'):
            module = importlib.import_module(name)
            need(str(Path(module.__file__).resolve()) == spec['library_sources'][name]['path'],
                 'Actual admitted selected library required: ' + name)
            modules[name] = module
            c.check()
        torch, np, cv2 = modules['torch'], modules['numpy'], modules['cv2']
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        need(torch.get_num_threads() == torch.get_num_interop_threads() == 2
             and all(os.environ[name] == '2' for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')),
             'Actual two-thread policy required')
        torch.set_default_dtype(torch.float32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        native = json.loads(c.closed(spec['input_refs']['candidates']).read_bytes())
        identities = candidate_identities(native, shape)
        coarse = np.load(c.closed(spec['input_refs']['masks']), allow_pickle=False)
        need(type(coarse) is np.ndarray and coarse.dtype == np.uint8
             and coarse.shape == (100, *shape) and np.logical_or(coarse == 0, coarse == 1).all()
             and coarse.reshape(100, -1).any(axis=1).all(), 'Original checked 100 nonempty uint8 masks required')
        need(hashlib.sha256(coarse.tobytes()).hexdigest() == input_report['decoded_native_masks_SHA256'],
             'Original decoded mask byte identity required')
        image_path = c.closed(spec['input_refs']['png'])
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        need(bgr is not None, 'Checked original PNG must decode')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        need(rgb.dtype == np.uint8 and rgb.shape == (*shape, 3)
             and hashlib.sha256(rgb.tobytes()).hexdigest() == input_report['original_RGB_SHA256'],
             'Current checked RGB pixel identity required')
        del bgr, rgb
        report.update(torch_version=str(torch.__version__), numpy_version=np.__version__,
                      selected_library_paths={name: str(Path(module.__file__).resolve()) for name, module in modules.items()})
        adapter = _load(c.closed(spec['source_refs']['fixed_adapter']), '_SAM_TRIAL_FIXED_ADAPTER', aliases)
        bridge = _load(c.closed(spec['source_refs']['bridge']), '_SAM_TRIAL_BRIDGE', aliases)
        _load(c.closed(spec['source_refs']['author_utils']), 'utils', aliases)
        author = _load(c.closed(spec['source_refs']['author_refiner']), '_SAM_TRIAL_AUTHOR', aliases)
        need(Path(sys.modules['utils'].__file__).resolve() == Path(spec['source_refs']['author_utils']['path']),
             'Author absolute utils import must belong to the fixed source')
        package = importlib.import_module('segment_anything')
        registry_module = importlib.import_module('segment_anything.build_sam')
        model_module = importlib.import_module('segment_anything.modeling.sam')
        for role, module in (('sam_init', package), ('sam_registry', registry_module), ('sam_model', model_module)):
            need(str(Path(module.__file__).resolve()) == spec['source_refs'][role]['path'],
                 'Actual author-vendored installed package required: ' + role)
        method = model_module.Sam.forward_with_image_embeddings
        parameters = list(inspect.signature(method).parameters.values())
        need([p.name for p in parameters] == ['self', 'image_embeddings', 'batched_input', 'multimask_output']
             and all(p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD for p in parameters),
             'Actual custom Sam method parameter contract required; annotations are preserved')
        need(package.sam_model_registry['vit_h'] is registry_module.build_sam_vit_h
             and package.sam_model_registry['default'] is package.sam_model_registry['vit_h'],
             'Actual ViT-H author registry required')
        report['custom_Sam_signature'] = str(inspect.signature(method))
        need(torch.cuda.is_available(), 'CUDA required; CPU fallback forbidden')
        props = torch.cuda.get_device_properties(0)
        cap = spec['prelock']['limits']['allocator_bytes']
        need(cap <= props.total_memory, 'Requested allocator cap exceeds actual CUDA memory')
        fraction = cap / props.total_memory
        torch.cuda.set_per_process_memory_fraction(fraction, device=0)
        torch.cuda.reset_peak_memory_stats(0)
        report.update(GPU_allocator_cap_bytes=cap, GPU_allocator_fraction_requested=fraction,
                      allocator_cap_set_call_returned=True, CUDA_device_name=props.name)
        c.check()
        event('CONSTRUCTING_FP32_VIT_H_CPU')
        sam = package.sam_model_registry['vit_h'](checkpoint=None)
        report['model_constructed'] = True
        event('MODEL_CONSTRUCTION_RETURNED')
        state = torch.load(str(checkpoint), map_location='cpu', weights_only=True)
        report['checkpoint_deserialized'] = True
        event('CHECKPOINT_DESERIALIZATION_RETURNED')
        need(isinstance(state, dict) and all(isinstance(value, torch.Tensor) and value.device.type == 'cpu'
             and (not value.is_floating_point() or value.dtype == torch.float32) for value in state.values()),
             'FP32 CPU tensor state dictionary required')
        sam.load_state_dict(state, strict=True)
        report['weights_loaded'] = True
        state = None
        gc.collect()
        c.check()
        event('WEIGHT_LOADING_RETURNED')
        sam.eval()
        event('MOVING_MODEL_TO_CUDA')
        sam.to(device=torch.device('cuda:0'), dtype=torch.float32)
        report['model_to_CUDA_returned'] = True
        event('MODEL_TO_CUDA_RETURNED')
        need(all(p.device == torch.device('cuda:0') and (not p.is_floating_point() or p.dtype == torch.float32)
                 for p in list(sam.parameters()) + list(sam.buffers()))
             and all(not module.training for module in sam.modules()), 'FP32 CUDA eval model required')
        report['model_CUDA_FP32_verified'] = True

        def hook(role, when):
            def observe(module, *unused):
                report['model_hooks'][role][when] += 1
                event(role.upper() + '_' + when.upper())
                c.check()
                return None
            return observe

        for role in report['model_hooks']:
            module = getattr(sam, role)
            handles.append(module.register_forward_pre_hook(hook(role, 'pre')))
            handles.append(module.register_forward_hook(hook(role, 'post')))
        module = None

        def tracked_author(*values, **kwargs):
            report['author_refiner_entered'] = True
            event('AUTHOR_REFINER_ENTERED')
            result = author.sam_refiner(*values, **kwargs)
            report['author_refiner_returned'] = True
            event('AUTHOR_REFINER_RETURNED')
            return result

        candidates = tuple(adapter.Candidate(row['query_id'], row['category_id'], row['score'], mask.tolist())
                           for row, mask in zip(native, coarse))
        backend = bridge.make_samrefiner_backend(str(image_path), sam, tracked_author)
        event('STARTING_FULL100_AUTHOR_DEFAULT_REFINEMENT')
        with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
            need(not torch.is_autocast_enabled('cuda'), 'AMP must remain disabled')
            refined = adapter.refine_candidates(candidates, shape, backend, empty_policy='reject')
        torch.cuda.synchronize(0)
        report['CUDA_synchronized_after_refiner'] = True
        event('AUTHOR_RETURN_AND_CUDA_SYNCHRONIZATION_COMPLETED')
        need(refined.backend_calls == 1 and refined.configured_iterations == 5 and not refined.is_adaptation,
             'One full100 author call with original defaults required')
        after = [dict(image_id=4, query_id=row.query_id, category_id=row.category_id, score=row.score)
                 for row in refined.candidates]
        need(json.dumps(after, allow_nan=False) == json.dumps(identities, allow_nan=False),
             'Original query order, category and raw S0 must remain unchanged')
        output_masks = np.asarray([row.mask for row in refined.candidates], dtype=np.uint8)
        need(output_masks.shape == (100, *shape) and np.logical_or(output_masks == 0, output_masks == 1).all(),
             'Full original-grid binary masks required')
        mask_path = out/'refined100_masks.npy'
        np.save(mask_path, output_masks, allow_pickle=False)
        need(np.array_equal(np.load(mask_path, allow_pickle=False), output_masks), 'Saved mask reload must match')
        identity_path = out/'candidate_identity.json'
        identity_path.write_text(json.dumps(after, indent=2, allow_nan=False)+'\n')
        report.update(inference_completed=True, backend_calls=1,
                      output_masks_ref=dict(path=str(mask_path), bytes=mask_path.stat().st_size,
                                           sha256=hashlib.sha256(mask_path.read_bytes()).hexdigest()),
                      output_identity_ref=dict(path=str(identity_path), bytes=identity_path.stat().st_size,
                                              sha256=hashlib.sha256(identity_path.read_bytes()).hexdigest()),
                      refined_empty_positions=np.flatnonzero(~output_masks.reshape(100, -1).any(axis=1)).tolist())
        c.check()
        report['status'] = 'ONE_IMAGE_AUTHOR_REFINER_RETURNED_PENDING_ROOT_REVIEW'
        event('OUTPUTS_WRITTEN_PENDING_CLEANUP')
        exit_code = 0
    except BaseException as error:
        is_oom = isinstance(error, MemoryError) or (torch is not None and isinstance(error, torch.cuda.OutOfMemoryError))
        report.update(status='RESOURCE_FAILURE_OOM_NO_RETRY' if is_oom else 'FAILED_PRESERVED_NO_RETRY',
                      error_type=type(error).__name__, error=str(error))
        event('FAILED', error_type=type(error).__name__, error=str(error))
        exit_code = 1
    finally:
        for handle in handles:
            handle.remove()
        backend = sam = state = refined = module = None
        gc.collect()
        if torch is not None and torch.cuda.is_initialized():
            try:
                torch.cuda.synchronize(0)
                report['backend_workspace_environment'] = {name: os.environ.get(name) for name in
                    ('CUBLAS_WORKSPACE_CONFIG', 'CUBLASLT_WORKSPACE_SIZE', 'TORCH_CUBLASLT_UNIFIED_WORKSPACE')}
                report['cleanup_before_backend_workspace_clear'] = dict(
                    allocated_bytes=torch.cuda.memory_allocated(0), reserved_bytes=torch.cuda.memory_reserved(0))
                event('CLEANUP_BEFORE_BACKEND_WORKSPACE_CLEAR')
                clear_workspaces = getattr(torch._C, '_cuda_clearCublasWorkspaces', None)
                need(callable(clear_workspaces), 'PyTorch2.10 private cuBLAS workspace cleanup interface unavailable')
                clear_workspaces()
                report['backend_workspace_clear_call_returned'] = True
                event('BACKEND_WORKSPACES_CLEAR_RETURNED')
                torch.cuda.empty_cache()
                torch.cuda.synchronize(0)
                report['cleanup_after_backend_workspace_clear'] = dict(
                    allocated_bytes=torch.cuda.memory_allocated(0), reserved_bytes=torch.cuda.memory_reserved(0))
                event('CLEANUP_AFTER_BACKEND_WORKSPACE_CLEAR')
                report.update(CUDA_allocated_after_cleanup=torch.cuda.memory_allocated(0),
                              CUDA_reserved_after_cleanup=torch.cuda.memory_reserved(0),
                              CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(0))
                need(report['CUDA_allocated_after_cleanup'] == report['CUDA_reserved_after_cleanup'] == 0
                     and report['CUDA_peak_allocated_bytes'] <= spec['prelock']['limits']['allocator_bytes'],
                     'Resource cleanup/allocation budget failed')
            except BaseException as cleanup_error:
                report.update(cleanup_error_type=type(cleanup_error).__name__, cleanup_error=str(cleanup_error))
                if exit_code == 0:
                    report['status'] = 'RESOURCE_CLEANUP_FAILURE_PENDING_ROOT_REVIEW'
                exit_code = 1
        if c is not None:
            try:
                c.check()
            except BaseException as resource_error:
                report['final_resource_error'] = str(resource_error)
                if exit_code == 0:
                    report['status'] = 'FINAL_RESOURCE_FAILURE_PENDING_ROOT_REVIEW'
                exit_code = 1
            report['sampled_control'] = c.summary()
        for alias, (existed, previous) in aliases.items():
            if existed:
                sys.modules[alias] = previous
            else:
                sys.modules.pop(alias, None)
        report.update(elapsed_seconds=time.monotonic()-started, worker_exit_code=exit_code,
                      outer_terminal_and_owned_group_acceptance_required=True)
        event('WORKER_RETURNING')
        events.close()
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())


