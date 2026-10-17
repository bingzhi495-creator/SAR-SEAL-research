"""UNBOUND CPU-only wrapper for the exact U38 qualifier; NOT EXECUTED.

Only a future original Root-admitted leaf may call run. No binder, guessed
image path, model forward, optimizer, automatic sharding/retry or CUDA work.
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

MODULE_LOAD_ORDER = (
    ('full_dsconv', 'dsconv'), ('full_head', 'head'), ('full_legacy', 'model_full'),
    ('full_feature', 'feature_backbone'), ('plain_legacy', 'plain_model1'),
    ('plain_feature', 'plain_feature'), ('instance_adapter', 'full_instance_adapter'),
    ('matched_reader', 'matched_reader'), ('image_geometry', 'image_geometry'),
    ('shared_initialization', 'shared_initialization'), ('rgb_matched_reader', 'rgb_matched_reader'),
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


def run(torch, np, spec, c, pre):
    started = time.monotonic()
    need(not torch.cuda.is_initialized()
         and torch.get_num_threads() == torch.get_num_interop_threads() == 2
         and torch.get_default_dtype() == torch.float32, 'Cold CPU FP32 qualifier')
    need(pre['images'] == 741 and pre['GT'] == 1661 and pre['dataset_indices'] == list(range(741)),
         'Only the entire original FIT population, no shard mode')
    output = Path(spec['output_directory'])
    output.mkdir()
    report_path, log_path = output/'REPORT.json', output/'probe.log'
    aliases = tuple(name for _, name in MODULE_LOAD_ORDER)+('U38_CPU_QUALIFIER',)
    absent = object()
    previous = {name: sys.modules.get(name, absent) for name in aliases}
    cpu_rng = torch.get_rng_state().clone()
    lazy_init = torch.cuda._lazy_init
    modules, imports, checks, attempts = {}, [], [], []
    qualification = {}
    report = dict(schema='U38-full-method-CPU-reader-init-report-pending-v1', status='STARTED',
        source=spec['body_source'], qualifier_source=spec['qualifier_source'],
        module_sources=spec['module_sources'], spec=c.execution_spec_ref,
        CPU_admission=c.execution_admission_ref, phase=spec['phase'], prelock=spec['prelock'],
        plan=spec['plan'], context=spec['context'], qualification=qualification,
        actual_import_records=imports, CUDA_checkpoints=checks, CUDA_initialization_attempts=attempts,
        CUDA_initialized=False, model_forward_executed=False, backward_executed=False,
        optimizer_created=False, optimizer_steps=0, new_training=False, AP=False, DEV=False, TEST=False,
        FIT=True, weights_loaded=False, weights_written=False,
        GPU_resource_gate_passed=False, full_method_training_eligible=False,
        qualified=False, original_actual0_self_certified=False, innovation_established=False,
        automatic_retry=False, automatic_sharding=False)
    failure = None

    def cold(label):
        value = bool(torch.cuda.is_initialized())
        checks.append(dict(label=label, CUDA_initialized=value))
        need(not value, 'CUDA must remain uninitialized: '+label)
        c.check()

    def forbidden_cuda_init(*args, **kwargs):
        attempts.append('torch.cuda._lazy_init attempted')
        raise RuntimeError('CUDA initialization forbidden in CPU741/1661 qualification')

    with log_path.open('x', encoding='utf-8', newline='\n') as log:
        with contextlib.redirect_stdout(_Tee(sys.stdout, log)), contextlib.redirect_stderr(_Tee(sys.stderr, log)):
            try:
                torch.cuda._lazy_init = forbidden_cuda_init
                cold('before_component_imports')
                for role, name in MODULE_LOAD_ORDER:
                    modules[role] = _load(c, spec['module_sources'][role], name)
                    imports.append(dict(role=role, module_name=name, source=spec['module_sources'][role]))
                    cold('after_import:'+role)
                qualifier = _load(c, spec['qualifier_source'], 'U38_CPU_QUALIFIER')
                imports.append(dict(role='qualifier_source', module_name='U38_CPU_QUALIFIER',
                                    source=spec['qualifier_source']))
                cold('after_qualifier_import')
                need(modules['full_legacy'].DSConv is modules['full_dsconv'].DSConv
                     and modules['full_legacy'].CoordConv is modules['full_head'].CoordConv,
                     'Original full dependency imports use closed objects')
                context = json.loads(Path(c.closed(spec['context'])).read_bytes())
                print('Starting complete FIT741/1661 CPU reader and common104 initialization checks.', flush=True)
                result = qualifier.qualify(torch, np, c, context, modules, spec['module_sources'],
                                           image_control=None, report=qualification)
                need(result is qualification and qualification['status'] ==
                     'CPU_READER_AND_COMMON_INITIALIZATION_CHECKS_COMPLETED'
                     and qualification['images_completed'] == 741 and qualification['GT_completed'] == 1661
                     and len(qualification['rows']) == 741
                     and qualification['image_reference_count'] == 741
                     and qualification['constructed_model_objects_released'] is True,
                     'Original qualifier completes entire population and releases fresh CPU pair')
                expected_images = qualifier.metadata_sha256(spec['image_references'])
                need(qualification['image_references_metadata_sha256'] == expected_images,
                     'Actual JPEG reference list equals Root pre-admitted list in original order')
                receipt = qualification['initialization']
                need(len(receipt['common_state']) == 104 and receipt['common_parameter_tensors'] == 68
                     and receipt['common_BN_buffers'] == 36
                     and receipt['seed'] == receipt['adapter_seed'] == 202610173101,
                     'Actual entire common state plus adapter initialization, no partial matching')
                cold('after_complete_reader_and_initialization')
                for value in list(spec['module_sources'].values())+[
                        spec['entry_source'], spec['body_source'], spec['runtime_source'], spec['qualifier_source']]:
                    need(c.ref(value['path']) == value, 'Executed source unchanged')
                need(not attempts, 'No attempted CUDA initialization')
                report['status'] = 'COMPLETED_PENDING_ORIGINAL_ACTUAL0_AND_ROOT_ACCEPTANCE'
                print('CPU741/1661 and common104 checks completed; pending original actual0 and Root acceptance.', flush=True)
            except BaseException as error:
                failure = error
                report['status'] = 'FAILED'
                report['error'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
                traceback.print_exc()
            finally:
                torch.cuda._lazy_init = lazy_init
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
                report['actual_CPU_threads'] = torch.get_num_threads()
                report['actual_CPU_interop_threads'] = torch.get_num_interop_threads()
                report['actual_CPU_affinity'] = c.psutil.Process().cpu_affinity()
                report['peak_RSS_bytes'] = c.peak_rss
                report['minimum_available_RAM_bytes'] = c.minimum_available
                report['elapsed_seconds'] = time.monotonic()-started
                if (not report['source_aliases_restored_exact'] or not report['CPU_RNG_restored_exact']
                        or report['CUDA_initialized'] or attempts) and failure is None:
                    failure = RuntimeError('CPU-only cleanup/source/RNG contract failed')
                    report['status'] = 'FAILED'
                    report['error'] = dict(type=type(failure).__name__, message=str(failure))
                # Preserve partial progress even after the controller has rejected
                # a deadline/resource check. This write does not qualify success.
                with report_path.open('x', encoding='utf-8', newline='\n') as stream:
                    json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
                    stream.write('\n')
    del cpu_rng
    gc.collect()
    if failure is not None:
        raise RuntimeError('U38 CPU qualification failed; preserved '+str(report_path)+' and '+str(log_path)) from failure
    c.check()
    return dict(report_ref=c.ref(report_path), log_ref=c.ref(log_path), images=741, GT=1661,
        CPU_reader_verified=True, common_initialization_verified=True,
        CUDA_initialized=False, qualified=False, new_training=False, AP=False, DEV=False, TEST=False,
        GPU_resource_gate_passed=False, full_method_training_eligible=False)
