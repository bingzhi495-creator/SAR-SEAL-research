"""UNBOUND source-closed wrapper for the published U39 probe; not executed.

The original probe owns all training math and first CUDA initialization. This
wrapper only loads closed modules, writes JSON/logs, and rejects FAILED output.
"""
import contextlib
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import types

PROBE_SHA256 = '378c955877674a65f6b2180d4c269e01355245881229ed0d2ec555b184ee7e5d'
LOAD_ORDER = (
    ('full_dsconv', 'dsconv'), ('full_head', 'head'), ('full_legacy', 'model_full'),
    ('full_feature', 'feature_backbone'), ('plain_legacy', 'plain_model1'),
    ('plain_feature', 'plain_feature'), ('instance_adapter', 'full_instance_adapter'),
    ('matched_reader', 'matched_reader'), ('image_geometry', 'image_geometry'),
    ('shared_initialization', 'shared_initialization'), ('rgb_matched_reader', 'rgb_matched_reader'),
    ('u36_helpers', 'U39_U36_HELPERS'),
)


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def _load(c, reference, name):
    path = Path(c.closed(reference))
    raw = path.read_bytes()
    need(len(raw) == reference['bytes'] and hashlib.sha256(raw).hexdigest() == reference['sha256'],
         'Exact source-closed component: '+name)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


class _Tee:
    def __init__(self, original, log):
        self.original, self.log = original, log

    def write(self, value):
        self.original.write(value)
        self.log.write(value)
        self.log.flush()
        return len(value)

    def flush(self):
        self.original.flush()
        self.log.flush()


def _recorded_peaks(value):
    peaks = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == 'CUDA_peak_allocated_bytes':
                need(type(item) is int and item >= 0, 'Actual nonnegative allocator peak')
                peaks.append(item)
            else:
                peaks.extend(_recorded_peaks(item))
    elif isinstance(value, list):
        for item in value:
            peaks.extend(_recorded_peaks(item))
    return peaks


def run(torch, np, spec, c, pre):
    started = time.monotonic()
    need(not torch.cuda.is_initialized()
         and torch.get_num_threads() == torch.get_num_interop_threads() == 2
         and torch.get_default_dtype() == torch.float32,
         'Probe wrapper must enter cold CUDA with pinned CPU threads and FP32')
    need(torch.backends.cudnn.deterministic is True and torch.backends.cudnn.benchmark is False
         and torch.backends.cudnn.allow_tf32 is False and torch.backends.cuda.matmul.allow_tf32 is False,
         'Original deterministic/FP32 settings before first CUDA initialization')
    need(pre['image_order'] == [54, 710, 54] and pre['updates_per_arm'] == 3
         and pre['total_pilot_updates'] == 6 and spec['admitted_tensor_indices'] == [54, 710],
         'Exact original two-image, three-update resource scope')
    output = Path(spec['output_directory'])
    output.mkdir()
    report_path, log_path = output/'REPORT.json', output/'probe.log'
    aliases = tuple(name for _, name in LOAD_ORDER)+('U39_CROSS_IMAGE_RESOURCE_PROBE',)
    absent = object()
    previous = {name: sys.modules.get(name, absent) for name in aliases}
    cpu_rng = torch.get_rng_state().clone()
    modules, imports, cold_checks = {}, [], []
    probe_report = {}
    report = dict(schema='U39-full-method-cross-image-resource-wrapper-report-pending-v1', status='STARTED',
        source=spec['body_source'], probe_source=spec['probe_source'], module_sources=spec['module_sources'],
        spec=c.execution_spec_ref, GPU_admission=c.execution_admission_ref, phase=spec['phase'],
        context=spec['context'], prelock=spec['prelock'], plan=spec['plan'],
        U38_qualified=spec['U38_qualified'], U38_report=spec['U38_report'],
        actual_import_records=imports, cold_CUDA_checkpoints=cold_checks, resource_probe=probe_report,
        full_metadata_population_images=741, full_metadata_population_GT=1661,
        admitted_tensor_indices=[54, 710], unique_tensor_images=2, image_order=[54, 710, 54],
        FIT=True, AP=False, DEV=False, TEST=False, qualified=False,
        formal_main_training_started=False, formal_training_budget_credit=0,
        full_method_training_eligible=False, pilot_only=True,
        automatic_retry=False, weights_loaded=False, weights_written=False)
    failure = None

    def cold(label):
        initialized = bool(torch.cuda.is_initialized())
        cold_checks.append(dict(label=label, CUDA_initialized=initialized))
        need(not initialized, 'Wrapper must defer first CUDA initialization to original probe: '+label)
        c.check()

    with log_path.open('x', encoding='utf-8', newline='\n') as log:
        with contextlib.redirect_stdout(_Tee(sys.stdout, log)), contextlib.redirect_stderr(_Tee(sys.stderr, log)):
            try:
                cold('before_component_imports')
                for role, name in LOAD_ORDER:
                    modules[role] = _load(c, spec['module_sources'][role], name)
                    imports.append(dict(role=role, module_name=name, source=spec['module_sources'][role]))
                    cold('after_import:'+role)
                need(spec['probe_source']['sha256'] == PROBE_SHA256, 'Exact Root-reviewed original resource probe')
                probe_module = _load(c, spec['probe_source'], 'U39_CROSS_IMAGE_RESOURCE_PROBE')
                imports.append(dict(role='probe_source', module_name='U39_CROSS_IMAGE_RESOURCE_PROBE',
                                    source=spec['probe_source']))
                cold('after_probe_import')
                need(modules['full_legacy'].DSConv is modules['full_dsconv'].DSConv
                     and modules['full_legacy'].CoordConv is modules['full_head'].CoordConv,
                     'Original dependencies resolved to exact admitted modules')
                context = json.loads(Path(c.closed(spec['context'])).read_bytes())
                print('Starting U39 original resource probe: [54,710,54], three updates per arm.', flush=True)
                # This single call owns the first CUDA init and 4GiB cap, all
                # model math, original renderer/loss, optimizer and arm cleanup.
                result = probe_module.probe(torch, np, c, context, modules, spec['module_sources'], pre,
                                            report=probe_report, image_control=None)
                need(result is probe_report, 'Original probe returns its caller-owned partial JSON')
                need(probe_report['status'] == 'COMPLETED_PENDING_ROOT_REVIEW',
                     'Original probe returned FAILED; preserve partial output and reject success')
                need(probe_report['image_order'] == [54, 710, 54]
                     and probe_report['actual_reader_metadata_images'] == 741
                     and probe_report['actual_reader_metadata_GT'] == 1661
                     and probe_report['distinct_probe_images'] == 2
                     and probe_report['successful_optimizer_updates'] == probe_report['optimizer_steps_entered'] == 6,
                     'Six returned actual optimizer updates; full population metadata only')
                need(probe_report['new_training'] is True and probe_report['pilot_only'] is True
                     and probe_report['formal_main_training_started'] is False
                     and probe_report['formal_training_budget_credit'] == 0
                     and all(probe_report[k] is False for k in ('AP', 'DEV', 'TEST', 'qualified',
                         'full_method_training_eligible', 'innovation_established', 'resource_speedup_established')),
                     'Disposable updates must never become formal budget or efficacy evidence')
                for arm in ('PLAIN', 'FULL'):
                    record = probe_report['arms'][arm]
                    need(record['status'] == 'PASS' and record['updates_completed'] == 3
                         and len(record['updates']) == 3
                         and [stage['dataset_index'] for stage in record['updates']] == [54, 710, 54]
                         and all(stage['status'] == 'PASS' and stage['optimizer_step_returned'] is True
                                 for stage in record['updates'])
                         and record['after_arm_release'] == dict(CUDA_allocated_bytes=0, CUDA_reserved_bytes=0),
                         'Both arms complete original sequence and hard-zero GPU release')
                need(len(probe_report['selected_metadata']) == 3, 'Three original metadata visits')
                for index, row in zip((54, 710, 54), probe_report['selected_metadata']):
                    need(row['dataset_index'] == index and row['identity']['image'] == spec['image_references'][index],
                         'Selected tensor/JPEG identities remain original ordinal-bound')
                need(probe_report['final_CUDA_cleanup'] == dict(CUDA_allocated_bytes=0, CUDA_reserved_bytes=0),
                     'Final cleanup comes from original resource probe')
                for value in list(spec['module_sources'].values())+[
                        spec['entry_source'], spec['body_source'], spec['runtime_source'], spec['probe_source']]:
                    need(c.ref(value['path']) == value, 'Executed source unchanged')
                c.check()
                report['status'] = 'COMPLETED_PENDING_ORIGINAL_ACTUAL0_AND_ROOT_ACCEPTANCE'
                print('U39 six disposable updates completed with both arm CUDA releases zero; pending Root intake.', flush=True)
            except BaseException as error:
                failure = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
                report['status'] = 'FAILED'
                report['error'] = failure
                traceback.print_exc()
            finally:
                torch.set_rng_state(cpu_rng)
                for name, value in previous.items():
                    if value is absent:
                        sys.modules.pop(name, None)
                    else:
                        sys.modules[name] = value
                modules.clear()
                gc.collect()
                report['source_aliases_restored_exact'] = all(
                    name not in sys.modules if value is absent else sys.modules.get(name) is value
                    for name, value in previous.items())
                report['CPU_RNG_restored_exact'] = bool(torch.equal(torch.get_rng_state(), cpu_rng))
                report['CUDA_initialized'] = bool(torch.cuda.is_initialized())
                intervals = _recorded_peaks(probe_report)
                current_peak = int(torch.cuda.max_memory_allocated(0)) if report['CUDA_initialized'] else 0
                report['CUDA_peak_allocated_bytes'] = max(intervals+[current_peak])
                report['CUDA_peak_scope'] = 'maximum original recursive stage peaks plus last actual allocator interval; peak resets retained'
                report['new_training'] = probe_report.get('new_training', False)
                report['successful_optimizer_updates'] = probe_report.get('successful_optimizer_updates', 0)
                report['actual_CPU_threads'] = torch.get_num_threads()
                report['actual_CPU_interop_threads'] = torch.get_num_interop_threads()
                report['actual_CPU_affinity'] = c.psutil.Process().cpu_affinity()
                report['peak_RSS_bytes'] = c.peak_rss
                report['minimum_available_RAM_bytes'] = c.minimum_available
                report['elapsed_seconds'] = time.monotonic()-started
                if (not report['source_aliases_restored_exact'] or not report['CPU_RNG_restored_exact']) and failure is None:
                    failure = dict(type='WrapperCleanupError', message='Source aliases or CPU RNG were not restored')
                    report['status'] = 'FAILED'
                    report['error'] = failure
                # A real error is preserved even when the resource controller
                # has already refused further successful work. Never qualify it.
                with report_path.open('x', encoding='utf-8', newline='\n') as stream:
                    json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
                    stream.write('\n')
    del cpu_rng
    gc.collect()
    if failure is not None:
        raise RuntimeError('U39 resource wrapper failed; preserved '+str(report_path)+' and '+str(log_path))
    c.check()
    return dict(report_ref=c.ref(report_path), log_ref=c.ref(log_path),
        pilot_optimizer_updates={'PLAIN': 3, 'FULL': 3}, successful_optimizer_updates=6, unique_images=2,
        new_training=True, pilot_only=True, formal_main_training_started=False, formal_training_budget_credit=0,
        AP=False, DEV=False, TEST=False, qualified=False, full_method_training_eligible=False)
