"""UNBOUND U40 formal continuous-training wrapper; no scientific work on import.

Only the admitted core owns training, checkpoint serialization and first CUDA
initialization. This wrapper preserves partial JSON/logs and rejects FAILED.
"""
import contextlib
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import traceback
import types

CORE_SHA256 = '8f9b549a8383877aa6f0ae3b1d2a9786c44c0f486e45b447b06dea2e25f5a0c5'
LOAD_ORDER = (
    ('full_dsconv', 'dsconv'), ('full_head', 'head'), ('full_legacy', 'model_full'),
    ('full_feature', 'feature_backbone'), ('plain_legacy', 'plain_model1'),
    ('plain_feature', 'plain_feature'), ('instance_adapter', 'full_instance_adapter'),
    ('matched_reader', 'matched_reader'), ('image_geometry', 'image_geometry'),
    ('shared_initialization', 'shared_initialization'), ('rgb_matched_reader', 'rgb_matched_reader'),
    ('u36_helpers', 'U40_U36_HELPERS'), ('u39_training_ops', 'U40_U39_TRAINING_OPS'),
)


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def _load(c, reference, name):
    path = Path(c.closed(reference))
    raw = path.read_bytes()
    need(len(raw) == reference['bytes'] and hashlib.sha256(raw).hexdigest() == reference['sha256'],
         'Exact source-closed module: '+name)
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


def summarize_trace(c, path, start, stop, complete_reference=None):
    """Own-output JSONL scalars only; no tensor, GT mask or optimizer replay."""
    if complete_reference is not None:
        need(c.ref(path) == complete_reference, 'Exact core-produced trace bytes')
    rows, indices, gt_visits = 0, set(), 0
    peak = peak_reserved = rss = 0
    minimum_available = None
    events = {}
    with Path(path).open('r', encoding='utf-8') as stream:
        for line in stream:
            if rows % 32 == 0:
                c.check()
            value = json.loads(line)
            actual = value['actual_input']
            ordinal = actual['global_ordinal']
            need(type(ordinal) is int and ordinal == start+rows and ordinal < stop,
                 'Actual trace is a contiguous original global span')
            indices.add(actual['dataset_index']); gt_visits += actual['G']
            need(value['optimizer_step_returned'] is True, 'Trace row records an actual returned optimizer update')
            resource = value['resources']
            peak = max(peak, resource['CUDA_peak_allocated_bytes'])
            peak_reserved = max(peak_reserved, resource['CUDA_peak_reserved_bytes'])
            rss = max(rss, resource['RSS_bytes'])
            available = resource['available_RAM_bytes']
            minimum_available = available if minimum_available is None else min(minimum_available, available)
            for key, number in value['CPU_saved_tensor_events'].items():
                if type(number) is int:
                    events[key] = events.get(key, 0)+number
            rows += 1
    return dict(rows=rows, tensor_indices=sorted(indices), GT_visits=gt_visits,
        CUDA_peak_allocated_bytes=peak, CUDA_peak_reserved_bytes=peak_reserved,
        max_boundary_RSS_bytes=rss, min_boundary_available_RAM_bytes=minimum_available,
        CPU_saved_event_totals=events,
        scope='Original own-output JSONL metadata; cumulative event bytes are not unique/peak storage')


def run(torch, np, spec, c, pre):
    started = time.monotonic()
    need(not torch.cuda.is_initialized() and torch.get_default_dtype() == torch.float32
         and torch.get_num_threads() == torch.get_num_interop_threads() == 2,
         'Cold CUDA, FP32 and two threads before core import')
    need(torch.backends.cudnn.deterministic is True and torch.backends.cudnn.benchmark is False
         and torch.backends.cudnn.allow_tf32 is False and torch.backends.cuda.matmul.allow_tf32 is False,
         'Fixed deterministic/FP32 settings')
    output = Path(spec['output_directory']); output.mkdir()
    state_dir = Path(spec['segment_output_directory'])
    need(state_dir.resolve() == (output/'SEGMENT').resolve() and not state_dir.exists(),
         'Core creates its own fresh segment directory; wrapper must not pre-create it')
    report_path, log_path = output/'REPORT.json', output/'probe.log'
    aliases = tuple(name for _, name in LOAD_ORDER)+('U40_CONTINUOUS_TRAIN_CORE',)
    absent = object(); previous = {name: sys.modules.get(name, absent) for name in aliases}
    cpu_rng, python_rng, numpy_rng = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
    modules, imports, cold_checks, summaries = {}, [], [], {}
    core_report = {}
    report = dict(schema='U40-continuous-training-wrapper-report-pending-v1', status='STARTED',
        source=spec['body_source'], core_source=spec['core_source'], module_sources=spec['module_sources'],
        spec=c.execution_spec_ref, GPU_admission=c.execution_admission_ref, phase=spec['phase'],
        context=spec['context'], prelock=spec['prelock'], plan=spec['plan'], U39_qualified=spec['U39_qualified'],
        resume_qualified=spec['resume_qualified'], segment=spec['segment'], parent_checkpoint_refs=spec['parent_checkpoint_refs'],
        admitted_tensor_indices=spec['admitted_tensor_indices'], actual_import_records=imports,
        cold_CUDA_checkpoints=cold_checks, core_report=core_report, trace_summaries=summaries,
        pilot_only=False, epochs_target=50, total_updates_per_arm_target=37050,
        new_training=False, formal_training_attempted=False, formal_main_training_started=False,
        paired_prefix_advanced=False, qualified=False, AP=False, DEV=False, TEST=False,
        whole_trajectory_bitexact_claimed=False, automatic_retry=False)
    failure = None

    def cold(label):
        initialized = bool(torch.cuda.is_initialized())
        cold_checks.append(dict(label=label, CUDA_initialized=initialized))
        need(not initialized, 'Core must own first CUDA initialization: '+label)
        c.check()

    with log_path.open('x', encoding='utf-8', newline='\n') as log:
        with contextlib.redirect_stdout(_Tee(sys.stdout, log)), contextlib.redirect_stderr(_Tee(sys.stderr, log)):
            try:
                cold('before_component_imports')
                for role, name in LOAD_ORDER:
                    modules[role] = _load(c, spec['module_sources'][role], name)
                    imports.append(dict(role=role, module_name=name, source=spec['module_sources'][role]))
                    cold('after_import:'+role)
                need(spec['core_source']['sha256'] == CORE_SHA256, 'Exact Root-reviewed updated core')
                core = _load(c, spec['core_source'], 'U40_CONTINUOUS_TRAIN_CORE')
                imports.append(dict(role='core_source', module_name='U40_CONTINUOUS_TRAIN_CORE', source=spec['core_source']))
                cold('after_core_import')
                segment = spec['segment']; start, stop = segment['start'], segment['stop']
                need(segment['recipe'] == core.RECIPE and segment['limits'] == core.LIMITS,
                     'Unchanged full training recipe and limits')
                # Pure metadata check, not another numerical qualification leaf.
                # The core creates/reads the durable order itself during training.
                plan = core.generate_order() if start == 0 else json.loads(Path(c.closed(segment['order_ref'])).read_bytes())
                core.validate_order(plan); core.validate_segment(segment, plan)
                need(sorted(set(plan['order'][start:stop])) == spec['admitted_tensor_indices'],
                     'Source tensor membership exactly matches this prelocked segment')
                plan = None
                context = json.loads(Path(c.closed(spec['context'])).read_bytes())
                free_bytes = int(c.psutil.disk_usage(str(output.parent)).free)
                report['disk_budget'] = spec['disk_budget']
                report['output_free_bytes_before_core'] = free_bytes
                need(free_bytes >= spec['minimum_output_free_bytes'], 'Root-locked checkpoint retention/storage margin')
                print('Starting original U40 formal segment [%d,%d), same frozen order on both arms.' % (start, stop), flush=True)
                result = core.train_segment(torch, np, c, context, modules, spec['module_sources'], segment,
                    spec['parent_checkpoint_refs'], str(state_dir), report=core_report, image_control=None)
                need(result is core_report, 'Core returns original caller-owned partial report')
                need(core_report['status'] == 'SEGMENT_COMPLETED_PENDING_ROOT_ACCEPTANCE',
                     'FAILED/partial/one-sided core output cannot advance paired prefix')
                length = stop-start
                need(core_report['successful_optimizer_updates'] == core_report['optimizer_steps_entered'] == 2*length
                     and core_report['span'] == dict(start=start, stop=stop)
                     and core_report['new_training'] is True and core_report['formal_main_training_started'] is True
                     and core_report['formal_training_attempted'] is True
                     and core_report['paired_prefix_advanced'] is False
                     and core_report['paired_checkpoint_prefix'] == core.cursor(stop),
                     'Exact actual formal segment budget; paired frontier still awaits Root')
                for arm in ('PLAIN', 'FULL'):
                    record = core_report['arms'][arm]
                    need(record['status'] == 'CHECKPOINT_WRITTEN' and record['updates_completed'] == length
                         and record['end_cursor'] == core.cursor(stop)
                         and record['checkpoint']['serialization_roundtrip_verified'] is True
                         and record['GPU_cleanup'] == dict(allocated=0, reserved=0),
                         'Both complete CPU checkpoints and hard-zero arm cleanup')
                    need(record['restored_all_state_exact'] is (True if start else None),
                         'Resume must restore model/BN/optimizer/modes/all arm RNG exactly')
                    summaries[arm] = summarize_trace(c, state_dir/(arm+'.trace.jsonl'), start, stop, record['trace_ref'])
                    need(summaries[arm]['rows'] == length
                         and summaries[arm]['tensor_indices'] == spec['admitted_tensor_indices'],
                         'Every actual input in the span is traced; no skipped member')
                need(core_report['arms']['PLAIN']['actual_trace_sha256'] == core_report['arms']['FULL']['actual_trace_sha256']
                     and core_report['final_GPU_cleanup'] == dict(allocated=0, reserved=0),
                     'Same actual paired input prefix and final CUDA zero')
                pair = json.loads(Path(core_report['pair_commit_ref']['path']).read_bytes())
                need(c.ref(core_report['pair_commit_ref']['path']) == core_report['pair_commit_ref']
                     and pair['schema'] == 'U40-paired-checkpoint-commit-v1'
                     and pair['status'] == 'PAIRED_CHECKPOINTS_WRITTEN_PENDING_ROOT_ACCEPTANCE'
                     and pair['cursor'] == core.cursor(stop) and pair['identity'] == core_report['identity']
                     and pair['order_ref'] == core_report['order_ref'],
                     'Original core paired commit exists but never self-certifies Root acceptance')
                for value in list(spec['module_sources'].values())+[
                        spec['entry_source'], spec['body_source'], spec['runtime_source'], spec['core_source']]:
                    need(c.ref(value['path']) == value, 'Executed source unchanged')
                c.check()
                report['status'] = 'COMPLETED_PENDING_ORIGINAL_ACTUAL0_AND_ROOT_ACCEPTANCE'
                print('U40 paired formal segment written; original terminal and Root acceptance pending.', flush=True)
            except BaseException as error:
                failure = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
                report['status'] = 'FAILED'; report['error'] = failure
                traceback.print_exc()
            finally:
                # No failed arm or already returned optimizer update is erased.
                # Preserve any own-output trace that completed before failure.
                if failure is not None:
                    for arm in ('PLAIN', 'FULL'):
                        path = state_dir/(arm+'.trace.jsonl')
                        if arm not in summaries and path.is_file():
                            try:
                                summaries[arm] = summarize_trace(c, path, spec['segment']['start'], spec['segment']['stop'])
                            except BaseException as error:
                                report.setdefault('partial_trace_summary_errors', {})[arm] = dict(
                                    type=type(error).__name__, message=str(error))
                torch.set_rng_state(cpu_rng); random.setstate(python_rng); np.random.set_state(numpy_rng)
                for name, value in previous.items():
                    if value is absent:
                        sys.modules.pop(name, None)
                    else:
                        sys.modules[name] = value
                modules.clear(); gc.collect()
                report['source_aliases_restored_exact'] = all(
                    name not in sys.modules if value is absent else sys.modules.get(name) is value
                    for name, value in previous.items())
                n = np.random.get_state()
                numpy_equal = n[0] == numpy_rng[0] and np.array_equal(n[1], numpy_rng[1]) and n[2:] == numpy_rng[2:]
                report['external_RNG_restored_exact'] = bool(torch.equal(torch.get_rng_state(), cpu_rng)
                    and random.getstate() == python_rng and numpy_equal)
                report['external_RNG_scope'] = 'Python/NumPy/Torch CPU only; saved per-arm CUDA RNG belongs to core checkpoints'
                report['CUDA_initialized'] = bool(torch.cuda.is_initialized())
                current = int(torch.cuda.max_memory_allocated(0)) if report['CUDA_initialized'] else 0
                report['CUDA_peak_allocated_bytes'] = max([current]+[
                    value['CUDA_peak_allocated_bytes'] for value in summaries.values()])
                report['CUDA_peak_scope'] = 'maximum original per-update JSONL trace peaks plus last actual allocator interval'
                for key in ('new_training', 'formal_training_attempted', 'formal_main_training_started'):
                    report[key] = core_report.get(key, False)
                report['successful_optimizer_updates'] = core_report.get('successful_optimizer_updates', 0)
                report['optimizer_steps_entered'] = core_report.get('optimizer_steps_entered', 0)
                report['peak_RSS_bytes'] = c.peak_rss; report['minimum_available_RAM_bytes'] = c.minimum_available
                report['actual_CPU_threads'] = torch.get_num_threads()
                report['actual_CPU_interop_threads'] = torch.get_num_interop_threads()
                report['actual_CPU_affinity'] = c.psutil.Process().cpu_affinity()
                report['elapsed_seconds'] = time.monotonic()-started
                if (not report['source_aliases_restored_exact'] or not report['external_RNG_restored_exact']) and failure is None:
                    failure = dict(type='WrapperRestoreError', message='External CPU/Python/NumPy RNG or aliases not restored')
                    report['status'] = 'FAILED'; report['error'] = failure
                with report_path.open('x', encoding='utf-8', newline='\n') as stream:
                    json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    del cpu_rng, python_rng, numpy_rng
    gc.collect()
    if failure is not None:
        raise RuntimeError('U40 formal segment failed; partial states/JSON/logs preserved under '+str(output))
    c.check()
    return dict(report_ref=c.ref(report_path), log_ref=c.ref(log_path), seed=202610173101,
        span=core_report['span'], updates_per_arm=spec['segment']['stop']-spec['segment']['start'],
        successful_optimizer_updates=core_report['successful_optimizer_updates'],
        pair_commit_ref=core_report['pair_commit_ref'], order_ref=core_report['order_ref'],
        checkpoint_refs={a: core_report['arms'][a]['checkpoint']['reference'] for a in ('PLAIN', 'FULL')},
        trace_refs={a: core_report['arms'][a]['trace_ref'] for a in ('PLAIN', 'FULL')},
        paired_checkpoint_prefix=core_report['paired_checkpoint_prefix'], paired_prefix_advanced=False,
        new_training=True, formal_main_training_started=True, pilot_only=False,
        qualified=False, AP=False, DEV=False, TEST=False)
