"""Unbound full-DEV SAMRefiner worker; scientific imports occur only in main."""
from pathlib import Path, PurePosixPath
import math

AUTHOR_COMMIT = '4bb7f95738c7a2e805f189210fb09e02a2197557'
OPERATION = 'GPU_SAMREFINER_FULL187_NATIVE100_SHARED_V01'
SCHEMA = 'ROOT-SAMRefiner-full187-shared-worker-spec-v01'
CPU_LIMITS = dict(allowed_cpus=[0, 1], threads=2,
                  maximum_RSS_bytes=16 * 1024 ** 3,
                  minimum_available_RAM_bytes=16 * 1024 ** 3,
                  deadline_seconds=7200)
PARAMS = dict(iters=5, margin=0.0, gamma=4.0, strength=30, use_samhq=False)
SOURCE_PINS = dict(
    author_refiner='b467d8e3e3c2fc395466a293635774cb821c5ffebdb39c4e5d64ac93c49dd3ad',
    author_utils='0d0c8e4b257935968c39f4223993b5b7de1868a4a7d886092c6a0074131299ef',
    sam_init='c8d8eb649610b2d51f138481a89617a3dcb61106465121dcb1c2f23affc84c30',
    sam_registry='77bc4faf728118625ddd184dc4e6375bd1b5c636fc66fb429f9fc5276f331cc5',
    sam_model='588d294a48bbd4ce5c5f4f560fa862aee0a8325a8dadda4e7d848a9f23d27946')
CHECKPOINT_BYTES = 2564550879
CHECKPOINT_SHA256 = 'a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e'
CHECKPOINT_URL = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'


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
         and set(value['sha256']) <= set('0123456789abcdef'), 'Full lowercase SHA256 required')
    return value


def validate_images(images):
    need(type(images) is list and len(images) == 187, 'Exactly 187 image records required')
    ids = set()
    for ordinal, image in enumerate(images):
        need(type(image) is dict and set(image) == {'ordinal', 'image_id', 'shape', 'jpeg_ref'},
             'Exact ordinal/image_id/shape/jpeg_ref image record required')
        need(type(image['ordinal']) is int and image['ordinal'] == ordinal
             and type(image['image_id']) is int and image['image_id'] not in ids,
             'Original ordinal order and distinct integer image IDs required')
        shape = image['shape']
        need(type(shape) is list and len(shape) == 2
             and all(type(v) is int and v > 0 for v in shape), 'Positive original H,W required')
        strict_ref(image['jpeg_ref'])
        ids.add(image['image_id'])
    return images


def validate_population(predictions, images):
    """Return per-roster-image original list indices without sorting candidates."""
    validate_images(images)
    need(type(predictions) is list and len(predictions) == 18700,
         'Exactly the original 18700 native predictions required')
    ordinal_by_id = {image['image_id']: image['ordinal'] for image in images}
    groups = [[] for image in images]
    for index, row in enumerate(predictions):
        need(type(row) is dict and {'image_id', 'query_id', 'category_id', 'score', 'segmentation'} <= set(row),
             'Original prediction fields required')
        need(type(row['image_id']) is int and row['image_id'] in ordinal_by_id
             and type(row['query_id']) is int and 0 <= row['query_id'] < 100
             and type(row['category_id']) is int and row['category_id'] == 1
             and type(row['score']) in (int, float) and math.isfinite(row['score'])
             and 0 <= row['score'] <= 1, 'Original identity, category and finite raw S0 required')
        ordinal = ordinal_by_id[row['image_id']]
        rle = row['segmentation']
        need(type(rle) is dict and set(rle) == {'size', 'counts'}
             and rle['size'] == images[ordinal]['shape']
             and type(rle['counts']) is str and rle['counts'], 'Original-grid compressed RLE required')
        groups[ordinal].append(index)
    for group in groups:
        need(len(group) == 100 and {predictions[i]['query_id'] for i in group} == set(range(100)),
             'Every image must retain all 100 distinct native queries')
    return groups


def replace_segmentations(predictions, encoded_by_index):
    """Preserve the complete original global list; update only each mask RLE."""
    from copy import deepcopy
    need(type(encoded_by_index) is dict and set(encoded_by_index) == set(range(len(predictions))),
         'Exactly one refined RLE for every original list position required')
    output = deepcopy(predictions)
    for index, row in enumerate(output):
        rle = encoded_by_index[index]
        need(type(rle) is dict and set(rle) == {'size', 'counts'}
             and rle['size'] == row['segmentation']['size']
             and type(rle['counts']) is str and rle['counts'], 'Encoded original-grid RLE required')
        row['segmentation'] = deepcopy(rle)
    return output


def invoke_author(image_path, masks, sam, author_refiner):
    """Use the unchanged author defaults, including its own empty-mask path."""
    need(len(masks) == 100, 'All original 100 masks must enter the author function')
    output = author_refiner(image_path, masks, sam, resize_transform=None,
        use_point=True, use_box=True, use_mask=True, add_neg=True, iters=5,
        margin=0.0, gamma=4.0, strength=30, use_samhq=False, ddp=False, is_train=False)
    need(type(output) is tuple and len(output) == 3, 'Original three-element author return required')
    return output


def validate_spec(spec):
    need(type(spec) is dict and spec.get('schema') == SCHEMA
         and spec.get('operation') == OPERATION
         and type(spec.get('run_id')) is str
         and spec['run_id'].startswith('SAMRefiner-full187-v01-')
         and '/' not in spec['run_id'], 'Root-pinned full187 worker specification required')
    need(spec.get('exclusive_project_scientific_leaf') is True
         and spec.get('whole_device_exclusive') is False and 'exclusive_GPU' not in spec,
         'Exclusive project scientific leaf on explicitly shared physical GPU required')
    pre = spec.get('prelock')
    need(type(pre) is dict and type(pre.get('seed')) is int and 0 <= pre['seed'] < 2**32,
         'Explicit prospective seed required')
    need(pre == dict(author_commit=AUTHOR_COMMIT, model_type='vit_h', precision='float32',
        amp=False, allow_cpu_fallback=False, params=PARAMS,
        limits=dict(CPU_LIMITS, allocator_bytes=16*1024**3),
        startup_min_available_RAM_bytes=48*1024**3,
        resource_plan='FULL187_FIXED_NATIVE100_SHARED_V01',
        empty_policy='AUTHOR_ALL100_UNFILTERED', expected_images=187, queries_per_image=100,
        seed=pre['seed']), 'Exact full187 prospective recipe/resource contract required')
    refs = spec.get('source_refs')
    need(type(refs) is dict and set(refs) == set(SOURCE_PINS) | {'worker'},
         'Six explicit selected source roles required')
    for role, sha in SOURCE_PINS.items():
        need(strict_ref(refs[role])['sha256'] == sha, 'Fixed author source changed: ' + role)
    libraries, inputs = spec.get('library_sources'), spec.get('input_refs')
    need(type(libraries) is dict and set(libraries) ==
         {'torch', 'numpy', 'psutil', 'cv2', 'PIL', 'pycocotools_mask'}
         and type(inputs) is dict and set(inputs) == {'native_predictions', 'image_manifest'},
         'Explicit selected libraries and two full population metadata inputs required')
    values = list(refs.values()) + list(libraries.values()) + list(inputs.values())
    values += [spec[k] for k in ('checkpoint_ref', 'checkpoint_stage_ref', 'root_lease_ref',
                                'gpu_background_observation_ref')]
    for value in values:
        strict_ref(value)
    checkpoint = spec['checkpoint_ref']
    need(checkpoint['bytes'] == CHECKPOINT_BYTES and checkpoint['sha256'] == CHECKPOINT_SHA256
         and spec.get('verified_checkpoint_sha256') == CHECKPOINT_SHA256,
         'Previously full-byte verified official ViT-H asset required')
    need(type(spec.get('sources')) is list, 'Explicit full selected source/input closure required')
    for ref in spec['sources']:
        strict_ref(ref)
    out = spec.get('output_directory')
    need(type(out) is str and PurePosixPath(out).is_absolute() and out != '/'
         and str(PurePosixPath(out)) == out and '..' not in PurePosixPath(out).parts,
         'Canonical absolute fresh output directory required')
    need(all(PurePosixPath(out) != PurePosixPath(ref['path'])
         and PurePosixPath(out) not in PurePosixPath(ref['path']).parents for ref in spec['sources']),
         'Output cannot contain an admitted input or source')


def validate_source_closure(spec, images):
    values = list(spec['source_refs'].values()) + list(spec['library_sources'].values())
    values += list(spec['input_refs'].values()) + [image['jpeg_ref'] for image in images]
    values += [spec[k] for k in ('checkpoint_ref', 'checkpoint_stage_ref', 'root_lease_ref',
                                'gpu_background_observation_ref')]
    unique = {}
    for ref in values:
        strict_ref(ref)
        need(ref['path'] not in unique or unique[ref['path']] == ref,
             'Repeated path must retain one byte/SHA256 identity')
        unique[ref['path']] = ref
    need(spec['sources'] == [unique[path] for path in sorted(unique)],
         'Sources must exactly close selected modules, metadata, asset and every JPEG')


def verify_file(ref, check=lambda: None):
    """Hash stable file bytes without importing scientific packages."""
    import hashlib
    strict_ref(ref)
    path = Path(ref['path'])
    need(str(path.resolve(strict=True)) == ref['path'] and not path.is_symlink(),
         'Exact canonical non-symlink file required')
    before = path.stat()
    digest, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            check()
            size += len(block)
            need(size <= ref['bytes'], 'File exceeds admitted byte count')
            digest.update(block)
    after = path.stat()
    identity = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    need(identity(before) == identity(after) and size == ref['bytes']
         and digest.hexdigest() == ref['sha256'], 'Stable exact admitted file bytes required')
    return path


def load_source(path, alias, originals):
    import sys
    import types
    originals.setdefault(alias, (alias in sys.modules, sys.modules.get(alias)))
    module = types.ModuleType(alias)
    module.__file__ = str(path)
    sys.modules[alias] = module
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def validate_stage_and_lease(spec, stage, lease, background, boot_id):
    need(stage.get('schema') == 'SAMRefiner-CHECKPOINT_STAGE-v1'
         and stage.get('status') == 'ROOT_VERIFIED_COMPLETE'
         and stage.get('checkpoint_ref') == spec['checkpoint_ref']
         and stage.get('verified_sha256') == CHECKPOINT_SHA256
         and stage.get('model_type') == 'vit_h' and stage.get('author_commit') == AUTHOR_COMMIT
         and stage.get('official_url') == CHECKPOINT_URL, 'Closed official checkpoint stage required')
    need(lease.get('schema') == 'Root-SAMRefiner-full187-shared-lease-v01'
         and lease.get('status') == 'ROOT_GRANTED' and lease.get('run_id') == spec['run_id']
         and lease.get('operation') == OPERATION and lease.get('boot_id') == boot_id
         and lease.get('worker_ref') == spec['source_refs']['worker']
         and lease.get('prelock') == spec['prelock'] and lease.get('single_attempt') is True
         and lease.get('exclusive_project_scientific_leaf') is True
         and lease.get('whole_device_exclusive') is False and 'exclusive_GPU' not in lease
         and lease.get('main_scientific_processes_reaped') is True
         and lease.get('scope') == 'full187_frozen_native100_no_GT_no_AP'
         and lease.get('output_directory') == spec['output_directory']
         and lease.get('gpu_background_observation_ref') == spec['gpu_background_observation_ref'],
         'Independent Root lease must bind this full187 recipe, output, birth and shared scope')
    need(background.get('schema') == 'SAMRefiner-shared-GPU-background-observation-v02'
         and background.get('boot_id') == boot_id
         and type(background.get('original_nvidia_smi_returncode')) is int
         and background['original_nvidia_smi_returncode'] == 0
         and type(background.get('gpu_index')) is int and background['gpu_index'] == 0
         and background.get('whole_device_exclusive') is False
         and type(background.get('processes')) is list, 'Original same-boot shared-GPU observation required')
    import datetime
    stamp = datetime.datetime.fromisoformat(background['observed_UTC'].replace('Z', '+00:00'))
    need(stamp.tzinfo is not None and stamp.utcoffset() == datetime.timedelta(0),
         'Explicit UTC background observation timestamp required')
    pids = set()
    for process in background['processes']:
        need(set(process) == {'pid', 'start_ticks', 'name', 'used_memory_MiB'}
             and type(process['pid']) is int and process['pid'] > 0 and process['pid'] not in pids
             and type(process['start_ticks']) is int and process['start_ticks'] > 0
             and type(process['used_memory_MiB']) is int and process['used_memory_MiB'] >= 0
             and type(process['name']) is str and process['name']
             and not any(c in process['name'] for c in ('/', '\\', '\n', '\x00')),
             'Only process birth/kernel name/observed GPU MiB background metadata permitted')
        pids.add(process['pid'])


def main(argv=None):
    import argparse
    import gc
    import hashlib
    import importlib
    import inspect
    import json
    import os
    import random
    import sys
    import time

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--spec-sha', required=True)
    args = parser.parse_args(argv)
    spec_path = Path(args.spec)
    need(spec_path.is_absolute() and spec_path.stat().st_size <= 2*1024**2,
         'Small absolute Root SPEC required')
    raw = spec_path.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == args.spec_sha, 'Exact Root-pinned SPEC digest required')
    spec = json.loads(raw)
    validate_spec(spec)
    need(sys.platform == 'linux' and os.environ.get('CUDA_VISIBLE_DEVICES') == '0',
         'Fresh leased Linux GPU0 worker required')
    need(not any(name.split('.')[0] in ('torch', 'numpy') for name in sys.modules),
         'No scientific import before fresh worker admission')
    os.sched_setaffinity(0, {0, 1})
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '2'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['MPLBACKEND'] = 'Agg'
    current_nice = os.getpriority(os.PRIO_PROCESS, 0)
    if current_nice < 19:
        os.nice(19-current_nice)
    out = Path(spec['output_directory'])
    need(str(out.resolve()) == str(out), 'Output parents cannot relocate through symlinks')
    out.mkdir(mode=0o700, exist_ok=False)
    (out/'png').mkdir(mode=0o700)
    started = time.monotonic()
    aliases, handles = {}, []
    torch = sam = state = author_packet = module = None
    psutil = None
    report = dict(schema='SAMRefiner-full187-shared-worker-report-v01', status='STARTED',
        run_id=spec['run_id'], operation=OPERATION, stage='STARTED',
        exclusive_project_scientific_leaf=True, whole_device_exclusive=False,
        limits=spec['prelock']['limits'], source_footprint='Selected explicit Python files, JPEGs and asset; not complete native/shared-library closure',
        AP=False, GT_loaded=False, training_updates=0, automatic_retry=False,
        precision='float32', amp=False, cpu_fallback=False,
        expected_images=187, expected_candidates=18700, images_completed=0, candidates_completed=0,
        empty_policy='AUTHOR_ALL100_UNFILTERED', configured_iterations=5,
        author_internal_return_order_verified=False, historical_rgb400_equivalence_proven=False,
        speed_benchmark=False, background_load_locked=False,
        external_live_supervision_self_certified=False,
        model_hooks={name: dict(pre=0, post=0) for name in ('image_encoder', 'prompt_encoder', 'mask_decoder')})
    stat = Path('/proc/self/stat').read_text()
    boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    report.update(pid=os.getpid(), pgid=os.getpgrp(), boot_id=boot_id,
                  start_ticks=int(stat[stat.rfind(')')+2:].split()[19]))
    events = (out/'STAGES.jsonl').open('x', encoding='utf-8')
    exit_code = 1
    peak_rss, min_available = 0, None

    def check():
        nonlocal peak_rss, min_available
        need(time.monotonic()-started < CPU_LIMITS['deadline_seconds'], 'Worker finite deadline exceeded')
        need(set(os.sched_getaffinity(0)) <= {0, 1}, 'Worker CPU affinity escaped CPU0/1')
        if psutil is not None:
            rss = int(psutil.Process().memory_info().rss)
            available = int(psutil.virtual_memory().available)
            peak_rss = max(peak_rss, rss)
            min_available = available if min_available is None else min(min_available, available)
            need(rss <= CPU_LIMITS['maximum_RSS_bytes'], 'Worker sampled RSS cap exceeded')
            need(available >= CPU_LIMITS['minimum_available_RAM_bytes'], 'Worker available RAM floor violated')

    def reference(path):
        b = path.read_bytes()
        return dict(path=str(path), bytes=len(b), sha256=hashlib.sha256(b).hexdigest())

    def write_json(path, value):
        with path.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())

    def event(stage, **details):
        report['stage'] = stage
        events.write(json.dumps(dict(stage=stage, elapsed_seconds=time.monotonic()-started,
                                    **details), sort_keys=True, allow_nan=False)+'\n')
        events.flush()
        os.fsync(events.fileno())
        temp = out/'REPORT.next.json'
        with temp.open('w', encoding='utf-8') as stream:
            json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(temp, out/'REPORT.json')

    try:
        event('AUTHENTICATING_FULL187_INPUTS')
        verify_file(spec['library_sources']['psutil'])
        psutil = importlib.import_module('psutil')
        need(str(Path(psutil.__file__).resolve()) == spec['library_sources']['psutil']['path'],
             'Actual admitted psutil source required')
        check()
        need(int(psutil.virtual_memory().available) >= 48*1024**3, 'Startup available RAM gate not met')
        report['startup_available_RAM_bytes'] = int(psutil.virtual_memory().available)
        images = json.loads(verify_file(spec['input_refs']['image_manifest'], check).read_bytes())
        validate_images(images)
        validate_source_closure(spec, images)
        for ref in spec['sources']:
            verify_file(ref, check)
        need(Path(__file__).resolve() == Path(spec['source_refs']['worker']['path']), 'Actual admitted worker required')
        stage = json.loads(Path(spec['checkpoint_stage_ref']['path']).read_bytes())
        lease = json.loads(Path(spec['root_lease_ref']['path']).read_bytes())
        background = json.loads(Path(spec['gpu_background_observation_ref']['path']).read_bytes())
        validate_stage_and_lease(spec, stage, lease, background, boot_id)
        report.update(source_count_verified=len(spec['sources']), checkpoint_stage_verified=True,
                      checkpoint_bytes_verified=True, external_lease_record_verified=True,
                      background_observed_UTC=background['observed_UTC'])
        native = json.loads(Path(spec['input_refs']['native_predictions']['path']).read_bytes())
        groups = validate_population(native, images)
        report['population_metadata_verified'] = True
        modules = {}
        for name, role in (('torch', 'torch'), ('numpy', 'numpy'), ('cv2', 'cv2'),
                           ('PIL', 'PIL'), ('pycocotools.mask', 'pycocotools_mask')):
            imported = importlib.import_module(name)
            need(str(Path(imported.__file__).resolve()) == spec['library_sources'][role]['path'],
                 'Actual admitted selected library required: '+name)
            modules[role] = imported
            check()
        torch, np, cv2, mask_api = modules['torch'], modules['numpy'], modules['cv2'], modules['pycocotools_mask']
        from PIL import Image
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        need(torch.get_num_threads() == torch.get_num_interop_threads() == 2, 'Actual torch thread limit required')
        torch.set_default_dtype(torch.float32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        load_source(Path(spec['source_refs']['author_utils']['path']), 'utils', aliases)
        author = load_source(Path(spec['source_refs']['author_refiner']['path']), '_SAM_FULL187_AUTHOR', aliases)
        package = importlib.import_module('segment_anything')
        registry_module = importlib.import_module('segment_anything.build_sam')
        model_module = importlib.import_module('segment_anything.modeling.sam')
        for role, imported in (('sam_init', package), ('sam_registry', registry_module), ('sam_model', model_module)):
            need(str(Path(imported.__file__).resolve()) == spec['source_refs'][role]['path'],
                 'Actual fixed author SAM package required: '+role)
        method = model_module.Sam.forward_with_image_embeddings
        need([p.name for p in inspect.signature(method).parameters.values()] ==
             ['self', 'image_embeddings', 'batched_input', 'multimask_output'], 'Custom author SAM method required')
        need(package.sam_model_registry['vit_h'] is registry_module.build_sam_vit_h,
             'Actual ViT-H registry required')
        need(torch.cuda.is_available(), 'CUDA required; CPU fallback forbidden')
        props = torch.cuda.get_device_properties(0)
        cap = spec['prelock']['limits']['allocator_bytes']
        need(cap <= props.total_memory, 'Allocator cap exceeds physical CUDA memory')
        torch.cuda.set_per_process_memory_fraction(cap/props.total_memory, device=0)
        torch.cuda.reset_peak_memory_stats(0)
        seed = spec['prelock']['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        report.update(seed=seed, torch_version=str(torch.__version__), numpy_version=np.__version__,
                      selected_library_paths={k: str(Path(v.__file__).resolve()) for k, v in modules.items()},
                      CUDA_device_name=props.name, GPU_allocator_cap_bytes=cap,
                      deterministic_bitwise_reproducibility_claimed=False)
        event('CONSTRUCTING_FP32_VIT_H_ONCE')
        sam = package.sam_model_registry['vit_h'](checkpoint=None)
        state = torch.load(spec['checkpoint_ref']['path'], map_location='cpu', weights_only=True)
        need(isinstance(state, dict) and all(isinstance(value, torch.Tensor) and value.device.type == 'cpu'
             and (not value.is_floating_point() or value.dtype == torch.float32) for value in state.values()),
             'FP32 CPU checkpoint state dictionary required')
        sam.load_state_dict(state, strict=True)
        state = None
        gc.collect()
        check()
        sam.eval()
        sam.to(device=torch.device('cuda:0'), dtype=torch.float32)
        need(all(p.device == torch.device('cuda:0') and (not p.is_floating_point() or p.dtype == torch.float32)
             for p in list(sam.parameters())+list(sam.buffers()))
             and all(not m.training for m in sam.modules()), 'Actual CUDA FP32 eval model required')
        report.update(model_load_count=1, model_CUDA_FP32_verified=True)

        def hook(role, when):
            def observe(unused_module, *unused):
                report['model_hooks'][role][when] += 1
                check()
            return observe

        for role in report['model_hooks']:
            module = getattr(sam, role)
            handles.append(module.register_forward_pre_hook(hook(role, 'pre')))
            handles.append(module.register_forward_hook(hook(role, 'post')))
        module = None
        encoded, image_reports = {}, []
        for image, indices in zip(images, groups):
            check()
            image_started = time.monotonic()
            ordinal, image_id, shape = image['ordinal'], image['image_id'], image['shape']
            event('IMAGE_STARTED', ordinal=ordinal, image_id=image_id)
            # Recheck the source JPEG immediately before current decode.
            jpeg_path = verify_file(image['jpeg_ref'], check)
            with Image.open(jpeg_path) as decoded:
                rgb = np.array(decoded.convert('RGB'), dtype=np.uint8, copy=True)
            need(rgb.shape == (*shape, 3), 'Original JPEG shape must join declared original grid')
            rgb_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
            png_path = out/'png'/('%06d_%08d.png' % (ordinal, image_id))
            Image.fromarray(rgb, mode='RGB').save(png_path, format='PNG')
            bgr = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
            need(bgr is not None, 'Derived lossless PNG must decode with author OpenCV')
            author_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            need(author_rgb.dtype == np.uint8 and np.array_equal(author_rgb, rgb),
                 'Current PIL JPEG to PNG to author OpenCV pixels must agree exactly')
            del rgb, bgr, author_rgb
            coarse = np.stack([mask_api.decode(native[index]['segmentation']) for index in indices], axis=0)
            need(coarse.dtype == np.uint8 and coarse.shape == (100, *shape)
                 and np.logical_or(coarse == 0, coarse == 1).all(), 'Original 100 decoded binary native masks required')
            empty_in = np.flatnonzero(~coarse.reshape(100, -1).any(axis=1)).tolist()
            before_hooks = {k: dict(v) for k, v in report['model_hooks'].items()}
            with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
                need(not torch.is_autocast_enabled('cuda'), 'AMP must remain disabled')
                author_packet = invoke_author(str(png_path), coarse, sam, author.sam_refiner)
            torch.cuda.synchronize(0)
            refined = author_packet[0]
            need(type(refined) is np.ndarray and refined.dtype == np.uint8
                 and refined.shape == (100, *shape) and np.logical_or(refined == 0, refined == 1).all(),
                 'All 100 original-grid binary author outputs required')
            delta = {k: {w: report['model_hooks'][k][w]-before_hooks[k][w] for w in ('pre', 'post')}
                     for k in report['model_hooks']}
            need(delta == dict(image_encoder=dict(pre=1, post=1),
                 prompt_encoder=dict(pre=5, post=5), mask_decoder=dict(pre=5, post=5)),
                 'One encoder pass and five complete prompt/decoder rounds required per image')
            for local, index in enumerate(indices):
                rle = mask_api.encode(np.asfortranarray(refined[local], dtype=np.uint8))
                encoded[index] = dict(size=[int(v) for v in rle['size']],
                                      counts=rle['counts'].decode('ascii'))
            empty_out = np.flatnonzero(~refined.reshape(100, -1).any(axis=1)).tolist()
            image_reports.append(dict(ordinal=ordinal, image_id=image_id, shape=shape,
                candidate_global_indices=indices, query_ids=[native[i]['query_id'] for i in indices],
                current_PIL_to_PNG_to_OpenCV_pixel_identity=True, current_RGB_SHA256=rgb_sha,
                original_JPEG_ref=image['jpeg_ref'], PNG_ref=reference(png_path),
                input_empty_positions=empty_in, output_empty_positions=empty_out,
                author_calls=1, configured_iterations=5, model_hook_deltas=delta,
                elapsed_shared_diagnostic_seconds=time.monotonic()-image_started,
                speed_benchmark=False, internal_row_correspondence_independently_verified=False))
            author_packet = refined = coarse = None
            gc.collect()
            check()
            report['images_completed'] += 1
            report['candidates_completed'] += 100
            event('IMAGE_COMPLETED', ordinal=ordinal, image_id=image_id)
        output = replace_segmentations(native, encoded)
        validate_population(output, images)
        predictions_path, image_reports_path = out/'refined_predictions.json', out/'image_reports.json'
        write_json(predictions_path, output)
        write_json(image_reports_path, image_reports)
        need(json.loads(predictions_path.read_bytes()) == output, 'Saved prediction JSON reload must match')
        report.update(output_predictions_ref=reference(predictions_path),
                      output_image_reports_ref=reference(image_reports_path),
                      all187_current_PIL_PNG_OpenCV_pixel_identity=True,
                      all100_candidates_submitted_each_image=True,
                      global_order_and_nonsegmentation_metadata_preserved=True,
                      status='FULL187_AUTHOR_RETURNED_PENDING_ROOT_REVIEW', inference_completed=True)
        event('FULL187_OUTPUTS_WRITTEN_PENDING_CLEANUP')
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
        author_packet = refined = coarse = sam = state = module = None
        gc.collect()
        if torch is not None and torch.cuda.is_initialized():
            try:
                torch.cuda.synchronize(0)
                report['cleanup_before_backend_workspace_clear'] = dict(
                    allocated_bytes=torch.cuda.memory_allocated(0), reserved_bytes=torch.cuda.memory_reserved(0))
                clear = getattr(torch._C, '_cuda_clearCublasWorkspaces', None)
                need(callable(clear), 'Qualified PyTorch private cuBLAS cleanup API required')
                clear()
                report['backend_workspace_clear_call_returned'] = True
                torch.cuda.empty_cache()
                torch.cuda.synchronize(0)
                report['cleanup_after_backend_workspace_clear'] = dict(
                    allocated_bytes=torch.cuda.memory_allocated(0), reserved_bytes=torch.cuda.memory_reserved(0))
                report.update(CUDA_allocated_after_cleanup=torch.cuda.memory_allocated(0),
                    CUDA_reserved_after_cleanup=torch.cuda.memory_reserved(0),
                    CUDA_peak_allocated_bytes=torch.cuda.max_memory_allocated(0))
                need(report['CUDA_allocated_after_cleanup'] == report['CUDA_reserved_after_cleanup'] == 0
                     and report['CUDA_peak_allocated_bytes'] <= spec['prelock']['limits']['allocator_bytes'],
                     'Strict CUDA cleanup/allocation budget failed')
            except BaseException as cleanup_error:
                report.update(cleanup_error_type=type(cleanup_error).__name__, cleanup_error=str(cleanup_error))
                if exit_code == 0:
                    report['status'] = 'RESOURCE_CLEANUP_FAILURE_PENDING_ROOT_REVIEW'
                exit_code = 1
        try:
            check()
        except BaseException as error:
            report['final_resource_error'] = str(error)
            if exit_code == 0:
                report['status'] = 'FINAL_RESOURCE_FAILURE_PENDING_ROOT_REVIEW'
            exit_code = 1
        for alias, (existed, previous) in aliases.items():
            if existed:
                sys.modules[alias] = previous
            else:
                sys.modules.pop(alias, None)
        report.update(elapsed_seconds=time.monotonic()-started, worker_exit_code=exit_code,
            sampled_worker_peak_RSS_bytes=peak_rss, sampled_worker_minimum_available_RAM_bytes=min_available,
            outer_original_terminal_source_postcheck_and_owned_group_acceptance_required=True)
        event('WORKER_RETURNING')
        events.close()
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
