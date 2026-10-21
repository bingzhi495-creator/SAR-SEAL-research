"""UNBOUND real CPU I/O wiring for U67; no import-time scientific work or launch."""
from copy import deepcopy
from contextlib import contextmanager
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import random
import sys
import time
import traceback
import types

RANGES = [[0,47], [47,94], [94,141], [141,187]]
COCO_KEYS = {'stats', 'metrics', 'iouType', 'maxDets', 'original_GT_preserved',
             'score_filter', 'official_COCO_crowd_ignore_semantics'}
MODULE_ROLES = ('population_contract', 'eval_body', 'official', 'cocoeval', 'mask_api', 'geometry_diagnostic', 'population_metrics')
LIBRARY_ROLES = ('numpy', 'psutil', 'pycocotools', 'coco_loader', 'mask_extension')
INPUT_ROLES = ('context', 'legacy_context', 'legacy_result', 'native_predictions', 'R_predictions', 'DEV')
CORE_PIN = dict(bytes=12084, sha256='ba69cf25fea20e1014124204a2c3e1bef41d0a080f4f81397e43696644cc4fe6')
SOURCE_PINS = {
    'population_contract': dict(bytes=4401, sha256='733a208c5df49113e9dff3338b2b88ad2d0d4b26b2bbdd3676ce10417e05c9a5'),
    'eval_body': dict(bytes=24972, sha256='0e4ec8c999b0e954ce6d588b3ffae97510be1051c4465f9d4510c3c2ec3b34bf'),
    'official': dict(bytes=49923, sha256='2357929fcd38ffc3020c5b8d204ee371da8da3972ff2e2e03398c8225c55e916'),
    'mask_api': dict(bytes=4693, sha256='1281ce0b5436ffe924c96521b2ae1d589f6f0e2a7b7281b6bb284f7b2245545c'),
    'geometry_diagnostic': dict(bytes=13722, sha256='6f6669eb91d0842f7c23eb09b992624642d16c9abe8d145c258ec9aae5ff36ac'),
    'population_metrics': dict(bytes=23923, sha256='f513d59321ffb571f6d823a74f31e3b6fd190086dc73bd77b400fe0500eb489a'),
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def strict_ref(value, *, original=False, allow_empty=False):
    need(type(value) is dict and set(value) == {'path', 'bytes', 'sha256'}
         and type(value['path']) is str and value['path']
         and type(value['bytes']) is int and value['bytes'] >= (0 if allow_empty else 1)
         and type(value['sha256']) is str and len(value['sha256']) == 64
         and set(value['sha256']) <= set('0123456789abcdef'), 'Exact file reference required')
    if not original:
        path = PurePosixPath(value['path'])
        need(path.anchor == '/' and str(path) == value['path'] and '..' not in path.parts
             and '\\' not in value['path'], 'Canonical actual POSIX reference required')
    return value


def relocate(role, original, actual):
    strict_ref(original, original=True)
    strict_ref(actual)
    need((original['bytes'], original['sha256']) == (actual['bytes'], actual['sha256']),
         'Original and actual content must be identical: '+role)
    return dict(role=role, original=deepcopy(original), actual=deepcopy(actual))


def signature(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _references(request):
    return [request['worker_source'], request['core_source']] + list(request['module_sources'].values()) + \
        list(request['library_sources'].values()) + list(request['input_refs'].values()) + request['shard_refs']


def validate_request(request):
    need(type(request) is dict and set(request) == {'worker_source', 'core_source', 'module_sources',
         'library_sources', 'input_refs', 'shard_refs', 'output_directory'}, 'Exact seven-key U70 request')
    for key, roles in (('module_sources', MODULE_ROLES), ('library_sources', LIBRARY_ROLES), ('input_refs', INPUT_ROLES)):
        need(type(request[key]) is dict and set(request[key]) == set(roles), 'Exact explicit roles: '+key)
    need(type(request['shard_refs']) is list and len(request['shard_refs']) == 4
         and len({signature(v) for v in request['shard_refs']}) == 4, 'Four distinct ordered full shard refs')
    for ref in [request['worker_source'], request['core_source']] + list(request['module_sources'].values()) + \
            list(request['input_refs'].values()) + request['shard_refs']:
        strict_ref(ref)
    for role, ref in request['library_sources'].items():
        strict_ref(ref, allow_empty=role == 'pycocotools')
    need(all(request['core_source'][k] == v for k, v in CORE_PIN.items()), 'Frozen U67 core identity required')
    for role, pin in SOURCE_PINS.items():
        need(all(request['module_sources'][role][k] == v for k, v in pin.items()), 'Original source identity: '+role)
    out = request['output_directory']
    need(type(out) is str and '\\' not in out and PurePosixPath(out).anchor == '/'
         and str(PurePosixPath(out)) == out and '..' not in PurePosixPath(out).parts and out != '/',
         'Canonical explicit output directory required')
    return request


def direct_references(request):
    """Pure metadata finite actual union; never follow nested historical refs."""
    validate_request(request)
    union = {}
    output = PurePosixPath(request['output_directory'])
    for ref in _references(request):
        path = ref['path']
        need(path not in union or union[path] == ref, 'Conflicting actual refs at one path')
        need(output != PurePosixPath(path) and output not in PurePosixPath(path).parents,
             'The output must not contain any input')
        union[path] = ref
    return [deepcopy(union[k]) for k in sorted(union)]


def closed_bytes(c, reference):
    c.check()
    path = c.closed(reference)
    need(str(path) == reference['path'], 'Exact admitted actual path required')
    raw = path.read_bytes()
    need(len(raw) == reference['bytes'] and hashlib.sha256(raw).hexdigest() == reference['sha256'],
         'Exact actual source/input bytes required')
    c.check()
    return raw


def no_torch():
    return not any(name.split('.')[0] == 'torch' for name in sys.modules)


def import_runtime(c, request):
    """Real future canonical imports; tests intercept this importlib boundary only."""
    need(platform.system() == 'Linux' and platform.machine() == 'aarch64'
         and platform.python_version() == '3.12.14', 'Original Linux/aarch64 Python3.12.14 runtime')
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and all(os.environ.get(k) == '2' for k in
         ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')), 'CPU-only GPU-hidden two-thread environment')
    need(no_torch() and not any(name.split('.')[0] in ('numpy', 'pycocotools') for name in sys.modules),
         'Fresh child without Torch or previous scientific imports')
    roles = (('psutil', 'psutil', request['library_sources']['psutil']),
        ('numpy', 'numpy', request['library_sources']['numpy']),
        ('pycocotools', 'pycocotools', request['library_sources']['pycocotools']),
        ('mask_extension', 'pycocotools._mask', request['library_sources']['mask_extension']),
        ('mask_api', 'pycocotools.mask', request['module_sources']['mask_api']),
        ('coco_loader', 'pycocotools.coco', request['library_sources']['coco_loader']),
        ('cocoeval', 'pycocotools.cocoeval', request['module_sources']['cocoeval']))
    for role, name, reference in roles:
        closed_bytes(c, reference)
    loaded = {}
    for role, name, reference in roles:
        value = importlib.import_module(name)
        need(type(value) is types.ModuleType and value.__file__ == reference['path'],
             'Actual canonical admitted runtime module: '+role)
        closed_bytes(c, reference)
        need(no_torch(), 'No Torch import during CPU runtime loading')
        loaded[role] = value
    need(str(loaded['numpy'].__version__) == '1.26.4' and str(loaded['psutil'].__version__) == '6.1.1'
         and c.psutil is loaded['psutil'] and loaded['psutil'].Process().cpu_affinity() == [0, 1],
         'Genuine original NumPy1.26.4/psutil6.1.1/control and CPU01')
    need(callable(getattr(loaded['coco_loader'], 'COCO', None))
         and callable(getattr(loaded['cocoeval'], 'COCOeval', None))
         and callable(getattr(loaded['mask_api'], 'decode', None)), 'Real COCO loader/evaluator/mask APIs required')
    return loaded


@contextmanager
def project_modules(c, request, runtime, proof):
    """Execute whole admitted stdlib definitions; restore every temporary alias."""
    previous, modules = {}, {}
    core = None

    def load(reference, alias):
        raw = closed_bytes(c, reference)
        previous[alias] = (alias in sys.modules, sys.modules.get(alias))
        value = types.ModuleType(alias)
        value.__file__ = reference['path']
        sys.modules[alias] = value
        exec(compile(raw, reference['path'], 'exec'), value.__dict__)
        need(value.__file__ == reference['path'] and no_torch(), 'Unchanged CPU source-backed module identity')
        return value

    try:
        for role in ('population_contract', 'eval_body', 'official', 'geometry_diagnostic', 'population_metrics'):
            modules[role] = load(request['module_sources'][role], '_U70_'+role)
        modules.update(mask_api=runtime['mask_api'], cocoeval=runtime['cocoeval'])
        for role, name in (('population_contract','merge_predictions'), ('eval_body','official_metrics'),
                           ('official','coco_metrics'), ('geometry_diagnostic','diagnose'),
                           ('population_metrics','relabel_geometry'), ('population_metrics','_delta_stats')):
            need(callable(getattr(modules[role], name, None)), 'Original project API: '+role+'.'+name)
        core = load(request['core_source'], '_U70_U67_CORE')
        need(callable(getattr(core, 'evaluate_u66_population', None)), 'Original U67 public API required')
        yield core, modules
    finally:
        modules.clear()
        core = None
        for alias, (existed, old) in reversed(list(previous.items())):
            if existed:
                sys.modules[alias] = old
            else:
                sys.modules.pop(alias, None)
        proof.update(aliases_restored=True, temporary_aliases=len(previous))


def write_json(path, value):
    raw = (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)+'\n').encode('utf-8')
    with path.open('xb') as stream:
        need(stream.write(raw) == len(raw), 'Complete exclusive JSON write')
        stream.flush()
        os.fsync(stream.fileno())
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def run_worker(c, request, *, report):
    """Actual future I/O path; genuine imports and U67, no owner/launch/Root gate."""
    need(type(report) is dict and not report, 'Fresh caller-owned report required')
    started, python_rng = time.monotonic(), random.getstate()
    out = None
    created = False
    runtime = np = numpy_rng = coco = core = modules = joined = result = None
    documents, shards, references = {}, [], []
    primary = None
    aliases = {}
    report.update(schema='U70-population-metrics-worker-report-v1', status='STARTED',
        qualified=False, ROOT_actual_terminal_self_certified=False, Root_external_admission_required=True,
        AP=False, DEV=True, DEV_technical_input=True, TEST=False, new_training=False, formal_training_budget_credit=0,
        AP_computation_status='NOT_STARTED',
        CPU_only=True, Torch_imported=False, CUDA_visible_devices='', automatic_retry=False,
        full_binary_closure_claimed=False, evaluation_started=False, annotation_JSON_parsed=False,
        module_loading=aliases, cleanup_errors=[], standalone_FPS_claimed=False)
    try:
        references = direct_references(request)
        need(request['worker_source']['path'] == str(__file__), 'Execute this exact admitted worker source')
        closed_bytes(c, request['worker_source'])
        report.update(request=deepcopy(request), direct_input_refs=references)
        out = Path(request['output_directory'])
        need(out.is_absolute() and out.resolve() == out and not out.exists() and not out.is_symlink()
             and out.parent.is_dir(), 'Fresh canonical output with existing parent; no retry or overwrite')
        out.mkdir()
        created = True
        for reference in [request['core_source']]+list(request['module_sources'].values())+list(request['library_sources'].values()):
            closed_bytes(c, reference)
        report['code_and_selected_library_bytes_closed_before_import'] = True
        for role in ('context', 'legacy_context', 'legacy_result', 'native_predictions', 'R_predictions'):
            documents[role] = json.loads(closed_bytes(c, request['input_refs'][role]))
        shards = [json.loads(closed_bytes(c, reference)) for reference in request['shard_refs']]
        joined = join_inputs(request, documents, shards)
        report['relocations'] = joined['relocations']
        report['original_context_and_population_metadata_joined'] = True
        runtime = import_runtime(c, request)
        np = runtime['numpy']
        numpy_rng = deepcopy(np.random.get_state())
        report['genuine_selected_runtime_modules_closed'] = True
        with project_modules(c, request, runtime, aliases) as (core, modules):
            closed_bytes(c, request['input_refs']['DEV'])
            report['annotation_bytes_closed'] = True
            coco = runtime['coco_loader'].COCO(str(c.closed(request['input_refs']['DEV'])))
            report['annotation_JSON_parsed'] = True
            need(list(coco.imgs) == joined['ids'] and len(coco.anns) == 380 and set(coco.cats) == {1},
                 'Actual original ordered DEV187/380/categoryID1 required')
            c.check()
            report['evaluation_started'] = True
            report['AP'] = None  # Original helpers may compute partially before an exception.
            report['AP_computation_status'] = 'STARTED_COMPLETION_UNKNOWN'
            result = core.evaluate_u66_population(np, c, shards=shards, native=joined['native'], strong=joined['strong'],
                reference_COCO=joined['reference_COCO'], coco=coco, ids=joined['ids'], modules=modules)
            need(type(result) is dict and set(result) == {'summary','private_pair_records'}
                 and result['summary']['status'] == 'METRICS_COMPLETE_PENDING_ROOT_ACCEPTANCE'
                 and result['summary']['qualified'] is result['summary']['ROOT_actual_terminal_self_certified'] is False
                 and result['summary']['AP'] is result['summary']['DEV'] is True
                 and result['summary']['TEST'] is result['summary']['new_training'] is False
                 and type(result['summary']['formal_training_budget_credit']) is int
                 and result['summary']['formal_training_budget_credit'] == 0
                 and type(result['private_pair_records']) is list, 'Exact pending U67 result ABI, no self qualification')
            report['AP'] = True
            report['AP_computation_status'] = 'COMPLETE_PENDING_ROOT_ACCEPTANCE'
            report['summary_ref'] = write_json(out/'SUMMARY.json', result['summary'])
            report['private_pair_records_ref'] = write_json(out/'FIXED_NATIVE_PAIRS_PRIVATE.json', result['private_pair_records'])
        for reference in references:
            closed_bytes(c, reference)
        report['all_direct_input_bytes_rechecked_after'] = True
        report['status'] = 'COMPLETE_PENDING_CLEANUP'
    except BaseException as error:
        primary = error
        report.update(status='FAILED', error=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
        traceback.clear_frames(error.__traceback__)
        error.__traceback__ = None
    finally:
        report['Torch_imported'] = not no_torch()
        report['CUDA_visible_devices'] = os.environ.get('CUDA_VISIBLE_DEVICES')
        coco = core = modules = joined = result = None
        documents.clear()
        shards.clear()
        gc.collect()
        for label, restore in [('Python_RNG', lambda: random.setstate(python_rng))] + (
                [('NumPy_RNG', lambda: np.random.set_state(numpy_rng))] if numpy_rng is not None else []):
            try:
                restore()
            except BaseException as error:
                report['cleanup_errors'].append(dict(operation=label, type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
                report['status'] = 'FAILED'
                if primary is None:
                    primary = error
        try:
            need(random.getstate() == python_rng and no_torch() and os.environ.get('CUDA_VISIBLE_DEVICES') == '',
                 'Original Python RNG and CPU-only/noTorch runtime preserved')
            if numpy_rng is not None:
                current = np.random.get_state()
                need(current[0] == numpy_rng[0] and bool(np.array_equal(current[1], numpy_rng[1]))
                     and current[2:] == numpy_rng[2:], 'Original caller NumPy RNG restored')
            report['source_aliases_restored'] = not aliases or aliases.get('aliases_restored') is True
            need(report['source_aliases_restored'], 'Temporary original module aliases restored')
            c.check()
            report['outer_runtime_states_restored'] = not report['cleanup_errors']
        except BaseException as error:
            report['cleanup_errors'].append(dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
            report['status'] = 'FAILED'
            if primary is None:
                primary = error
        try:
            report['resource_observations'] = c.summary()
        except BaseException as error:
            report['cleanup_errors'].append(dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
            report['status'] = 'FAILED'
            if primary is None:
                primary = error
        report['elapsed_seconds'] = time.monotonic()-started
        if report['elapsed_seconds'] > 1800:
            report['status'] = 'FAILED'
            if primary is None:
                primary = TimeoutError('Original CPU1800s worker deadline exceeded')
        if report['status'] == 'COMPLETE_PENDING_CLEANUP':
            report['status'] = 'COMPLETE_PENDING_ROOT_ACCEPTANCE'
        if created:
            try:
                write_json(out/'REPORT.json', report)
            except BaseException as error:
                report.update(status='FAILED', report_write_error=dict(type=type(error).__name__, message=str(error)))
                if primary is None:
                    primary = error
        runtime = np = numpy_rng = None
    if primary is not None:
        raise primary
    need(report['status'] == 'COMPLETE_PENDING_ROOT_ACCEPTANCE', 'Only complete I/O plus real U67 and cleanup can return')
    return report


def join_inputs(request, documents, shards):
    """Pure original-reference joins; never follow historical paths or qualify a run."""
    inputs = request['input_refs']
    context, legacy, result = (documents[k] for k in ('context', 'legacy_context', 'legacy_result'))
    relocations = [relocate('context.legacy_context', context['legacy_context'], inputs['legacy_context'])]
    for name, container in (('context', context), ('legacy_context', legacy)):
        for original_key, actual_key in (('native_predictions', 'native_predictions'),
                ('R_predictions', 'R_predictions'), ('legacy_ROOT_result', 'legacy_result')):
            relocations.append(relocate(name+'.'+original_key, container[original_key], inputs[actual_key]))
    relocations.append(relocate('legacy_context.DEV', legacy['DEV'], inputs['DEV']))
    relocations.append(relocate('legacy_context.COCO_source', legacy['COCO_source'], request['module_sources']['official']))
    ids = context['image_ids']
    need(type(ids) is list and len(ids) == len(set(ids)) == 187 and all(type(i) is int for i in ids)
         and signature(ids) == signature(legacy['image_ids']), 'Both original image_ids orders must be exact')
    native, strong = documents['native_predictions'], documents['R_predictions']
    need(type(native) is list and type(strong) is list and len(native) == len(strong) == 18700,
         'Original native/R are direct full18700 JSON lists')
    need([native[i*100]['image_id'] for i in range(187)] == ids, 'Native order must equal original image_ids')
    need(type(shards) is list and len(shards) == 4, 'Four complete original shards required')
    endpoint = shards[0]['endpoint']
    need(endpoint['schema'] == 'U58-U57-DGX-fixed-endpoint-metadata-v1' and endpoint['mode'] == 'FORMAL'
         and endpoint['formal_endpoint'] is True and signature(endpoint['cursor']) == signature(dict(
             global_updates=37050, completed_epochs=50, within_epoch=0)), 'Original FORMAL E50 endpoint required')
    for i, (shard, span) in enumerate(zip(shards, RANGES)):
        need(shard['schema'] == 'U66-DGX-fixed50-DEV-shard-pending-v1'
             and shard['status'] == 'PREDICTIONS_COMPLETE_PENDING_ROOT_ACCEPTANCE'
             and type(shard['shard_index']) is int and shard['shard_index'] == i
             and signature(shard['dataset_range']) == signature(span)
             and signature(shard['endpoint']) == signature(endpoint)
             and signature(shard['source_refs']) == signature(shards[0]['source_refs']),
             'Four fixed ranges with identical whole endpoint and prediction sources required')
        for role in ('population_contract', 'mask_api'):
            relocations.append(relocate('shard'+str(i)+'.'+role, shard['source_refs'][role], request['module_sources'][role]))
    reference = {name: deepcopy(result['control_results'][key]['COCO']) for name, key in
                 (('native_official', 'native_official'), ('R', 'strong_fixed_E24_Omega'))}
    need(all(type(v) is dict and set(v) == COCO_KEYS and type(v['stats']) is list and len(v['stats']) == 12
             for v in reference.values()), 'Both whole original seven-key COCO12 dictionaries required')
    signature(reference)
    return dict(ids=ids, native=native, strong=strong, reference_COCO=reference, relocations=relocations)
