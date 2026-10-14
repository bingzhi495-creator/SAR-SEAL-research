"""UNBOUND / UNEXECUTED: full stage24 DEV independent CPU replay entry.

Ordinary draft only. ROOT must bind and review fresh actual source/spec/admission
after completed stage24 training, independent training CPU qualification, and
completed GPU DEV capture. No current/live scientific inputs were inspected to
write this draft. Running this file is NOT authorized by drafting it.

The CPU uses captured actual GPU F/e/native-L points, never CPU F rendering.
The prelocked 1e-4 residual/combined test, exact strict>0 signs, full-query RLE,
and original official COCO12 are separate mandatory checks over DEV187/all100.
A CPU computational portability failure does not itself invalidate GPU AP and
does not qualify either the CPU attempt or a new scientific method claim.
"""
import argparse
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import types

SCHEMA = 'ROOT-N14-stage24-fixed-fullDEV-CPU-spec-v1'
GPU_SCHEMA = 'ROOT-N14-stage24-fixed-fullDEV-GPU-spec-v1'
ARMS = ('BASE_CONTINUATION', 'MEMORY_ADDON')
THREAD_ENV = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
RAW_KEYS = ('receipt', 'phase', 'admission', 'phase_binding', 'bootstrap')
NUMERIC_SCHEMA = 'N14-fixed-DEV-same-GPU-point-input-CPU-numeric-protocol-v1'
NUMERIC_KEYS = {'schema', 'same_actual_GPU_point_inputs', 'point_chunk',
                'mask_RLE_exact', 'locked_before_CPU_execution',
                'bound_completed_GPU_evaluation_report', 'atol', 'rtol'}
PORTABILITY_MESSAGES = {
    'CPU same-input residual/combined tolerance failed',
    'Even within numeric tolerance, exact point sign-mask is mandatory',
    'Independent CPU full-query fixed-score RLE must exactly match actual GPU export',
}


class ComputationalPortabilityFailure(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    def reject(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=reject)


def ref_shape(ref):
    require(type(ref) is dict and set(ref) == {'path', 'bytes', 'sha256'} and
            type(ref['path']) is str and Path(ref['path']).is_absolute() and
            type(ref['bytes']) is int and ref['bytes'] >= 0 and
            type(ref['sha256']) is str and len(ref['sha256']) == 64 and
            set(ref['sha256']) <= set('0123456789abcdef'), 'Malformed absolute pinned reference')


def reference(path):
    path = Path(path).resolve(strict=True)
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            sha.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha.hexdigest())


def bound(ref):
    ref_shape(ref)
    require(reference(ref['path']) == ref, 'Pinned input changed: ' + ref['path'])
    return Path(ref['path'])


def closed(closure, ref):
    ref_shape(ref)
    require(closure.get(ref['path']) == ref, 'Missing exact SOURCE ref: ' + ref['path'])
    return bound(ref)


def load(closure, ref, name):
    path = closed(closure, ref)
    source = path.read_bytes()
    require(len(source) == ref['bytes'] and hashlib.sha256(source).hexdigest() == ref['sha256'],
            'SOURCE changed before import')
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def terminal_raw_only(ref, phase, code):
    # No hash/import/data/model read may precede the complete RAW/pin gate.
    ref_shape(ref)
    path = Path(ref['path'])
    require(path.name == 'PARENT_RECEIPT.json' and type(phase) is str and phase and
            path.parent.name == phase, 'Original actual RAW locator required')
    raw = read(path)
    require(raw['status'] == 'ACTUAL_OWNED_TERMINAL' and
            type(raw['returncode']) is int and raw['returncode'] == code and
            type(raw['original_identity']) is dict and raw['original_identity'] and
            raw['root_wait_reaped'] is True and
            raw['held_handle_exit'] == dict(wait_state='WAIT_OBJECT_0', actual_exit_code=code) and
            type(raw['held_handle_exit']['actual_exit_code']) is int and
            raw['numeric_birth_absence']['original_birth_absent'] is True and
            raw.get('error') is None and raw.get('abort') is None and
            raw['wait_errors'] is False and raw['GPU_observation_errors'] == [],
            'Actual held exit/reap/original birth absence required before first SHA')
    if 'phase' in raw:
        require(raw['phase'] == phase, 'Original RAW phase differs')
    sha, peak = raw['admission_sha256'], raw['whole_GPU_observed_peak_MiB']
    require(type(sha) is str and len(sha) == 64 and set(sha) <= set('0123456789abcdef') and
            type(peak) in (int, float) and math.isfinite(peak) and 0 <= peak <= 8192 and
            raw['whole_GPU_used_limit_MiB'] == 8192 and
            raw['whole_leaf_VRAM_certified'] is False and
            raw['GPU_scope'] == 'whole GPU observed, not per-process WDDM allocation',
            'Preserved RAW resource/admission scope differs')
    return raw


def pin_check(pin, raw, phase, code):
    require(type(pin) is dict and
            set(pin) == {'phase', 'returncode', 'original_identity', 'admission_sha256'} and
            pin['phase'] == phase and type(pin['returncode']) is int and pin['returncode'] == code and
            pin['original_identity'] == raw['original_identity'] and
            pin['admission_sha256'] == raw['admission_sha256'], 'Actual owner identity/pin differs')


def raw_first(spec):
    extra = spec['additional_preserved_independent_CPU_failure']
    require(type(extra) is dict and set(extra) == {'RAW_ref', 'phase', 'pin', 'returncode'} and
            type(extra['returncode']) is int and extra['returncode'] == 1,
            'Independent failure9 exact RAW/phase/pin/returncode1 required')
    ninth = terminal_raw_only(extra['RAW_ref'], extra['phase'], 1)
    pin_check(extra['pin'], ninth, extra['phase'], 1)
    expected = spec['normal_RAW_count']
    require(type(expected) is int and expected >= 102 and
            spec['expected_normal_RAW_count'] == expected and
            type(spec['total_failed_RAW_count']) is int and spec['total_failed_RAW_count'] == 9 and
            type(spec['total_terminal_identity_pin_count']) is int and
            spec['total_terminal_identity_pin_count'] == expected + 9,
            'Current complete normals and separate failed9 total identity count required')
    rows = spec['RAW_inputs']
    require(type(rows) is list and len(rows) == expected and
            all(type(row) is dict and all(key in row for key in RAW_KEYS) for row in rows) and
            len({row['receipt']['path'] for row in rows}) == expected,
            'Every inherited normal RAW owner must occur exactly once')
    normals = [terminal_raw_only(row['receipt'], row['phase'], 0) for row in rows]
    failed_ref = spec['prior_failed_CPU_receipt']
    failed_phase = Path(failed_ref['path']).parent.name
    failures = [terminal_raw_only(failed_ref, failed_phase, 1)]
    extras = spec['additional_failed_RAW_inputs']
    require(type(extras) is list and len(extras) == 7 and
            all(type(row) is dict and set(row) == {'receipt', 'phase', 'returncode'} and
                type(row['returncode']) is int and row['returncode'] == 1 for row in extras),
            'Original legacy failure8 list is preserved')
    failures += [terminal_raw_only(row['receipt'], row['phase'], 1) for row in extras]
    owners = [dict(row, returncode=0) for row in rows] + [
        dict(receipt=failed_ref, phase=failed_phase, returncode=1)] + extras
    require(len({row['receipt']['path'] for row in owners}) == expected + 8 and
            extra['RAW_ref']['path'] not in {row['receipt']['path'] for row in owners},
            'Normal, legacy failed8 and independent failed9 owners must be distinct')
    pins = spec['terminal_identity_pins']
    require(type(pins) is dict and set(pins) == {row['receipt']['path'] for row in owners} and
            len(pins) == expected + 8, 'Legacy normal+failed8 identity pins required')
    for row, raw in zip(owners, normals + failures):
        pin_check(pins[row['receipt']['path']], raw, row['phase'], row['returncode'])
    # FIRST SHA. All normal, failed8 and failed9 original RAW identities passed.
    require(type(spec['sources']) is list, 'Explicit complete SOURCE list required')
    for ref in spec['sources']:
        ref_shape(ref)
    closure = {ref['path']: ref for ref in spec['sources']}
    require(len(closure) == len(spec['sources']), 'Unique SOURCE paths required')
    for ref in spec['sources']:
        bound(ref)
    for row in rows:
        for key in RAW_KEYS:
            if key != 'phase':
                closed(closure, row[key])
    for ref in [failed_ref, extra['RAW_ref']] + [row['receipt'] for row in extras]:
        closed(closure, ref)
    return normals, closure


def deadline(started):
    require(time.monotonic() - started < 1800, 'CPU DEV deadline exceeded; no retry')


def lookup(value, keys, label):
    require(type(keys) is list and keys and all(type(key) is str and key for key in keys),
            'ROOT must bind an actual admission key path: ' + label)
    for key in keys:
        require(type(value) is dict and key in value, 'Missing actual admission field: ' + label)
        value = value[key]
    return value


def cpu_resource_check(spec, admission, psutil, torch):
    wanted = dict(affinity=[0, 1], thread_limit=2, deadline=1800,
                  automatic_retry=False, cuda_visible_devices='')
    require(set(spec['CPU_resource_fields']) == set(wanted), 'Explicit ROOT CPU resource mapping required')
    for name, value in wanted.items():
        actual = lookup(admission, spec['CPU_resource_fields'][name], name)
        require(type(actual) is type(value) and actual == value, 'Actual CPU admission differs: ' + name)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and
            all(os.environ.get(name) == '2' for name in THREAD_ENV) and
            psutil.Process().cpu_affinity() == [0, 1] and
            spec['deadline_seconds'] == 1800 and spec['automatic_retry'] is False and
            not torch.cuda.is_initialized(), 'Real CPU environment/affinity/cold CUDA differs')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    require(torch.get_num_threads() == torch.get_num_interop_threads() == 2 and
            not torch.cuda.is_initialized(), 'Torch CPU two-thread/cold CUDA invariant failed')
    return wanted


def numeric_check(protocol, completed_ref):
    require(type(protocol) is dict and set(protocol) == NUMERIC_KEYS and
            protocol['schema'] == NUMERIC_SCHEMA and
            protocol['same_actual_GPU_point_inputs'] is True and
            type(protocol['point_chunk']) is int and protocol['point_chunk'] == 2048 and
            protocol['mask_RLE_exact'] is True and protocol['locked_before_CPU_execution'] is True and
            protocol['bound_completed_GPU_evaluation_report'] == completed_ref and
            type(protocol['atol']) in (int, float) and type(protocol['rtol']) in (int, float) and
            protocol['atol'] == protocol['rtol'] == 1e-4,
            'Exactly8 prelocked DEV numeric fields, completed GPU DEV report and fixed1e-4 required')


class ClosedImageSink:
    """Read exactly one admitted physical capture file per requested image."""
    def __init__(self, torch, closure, refs, validator):
        self.torch, self.closure = torch, closure
        self.refs = {ref['path']: ref for ref in refs}
        require(len(self.refs) == 187, 'Exactly187 unique real capture files required')
        self.validator, self.current = validator, None

    def read(self, ref):
        require(self.refs.get(ref['path']) == ref, 'Capture ref outside completed GPU DEV manifest')
        require(self.current is None, 'Read each real image capture only once')
        value = self.torch.load(str(closed(self.closure, ref)), map_location='cpu')
        self.validator(ref, value)
        self.current = value
        return value

    def release(self):
        self.current = None


def write_new(path, value):
    # The original attempt does not read/hash its own new Q. ROOT validates it
    # only after actual held terminal/reap/birth absence. SHA comes from bytes.
    data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
    path = Path(path).resolve()
    with path.open('xb') as stream:
        stream.write(data)
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def replay_one(adapter, torch, np, masks, n10, heads, capture_ref,
               sink, selector_loader, protocol, GPU_report_ref):
    try:
        return adapter.cpu_replay_captured_image(torch, np, masks, n10, heads,
            capture_ref, sink, selector_loader, protocol, GPU_report_ref)
    except RuntimeError as error:
        if str(error) in PORTABILITY_MESSAGES:
            raise ComputationalPortabilityFailure(str(error)) from error
        raise


def completed_GPU_DEV_gate(gpu, gate, spec):
    """Close the separately completed GPU evaluation, never the current Q."""
    closure = gate['closure']
    keys = ('source', 'spec', 'report', 'admission', 'RAW')
    refs = {key: spec['completed_GPU_DEV_' + key] for key in keys}
    for ref in refs.values():
        closed(closure, ref)
    require(spec['GPU_runner_source'] == refs['source'], 'Shared GPU gate must be the completed GPU DEV source')
    parent = read(closed(closure, refs['spec']))
    report = read(closed(closure, refs['report']))
    count = parent['normal_RAW_count']
    require(type(count) is int and count >= 101 and spec['normal_RAW_count'] == count + 1 and
            parent['expected_normal_RAW_count'] == count and
            parent['RAW_inputs'] == spec['RAW_inputs'][:count] and
            spec['RAW_inputs'][count]['receipt'] == refs['RAW'] and
            spec['RAW_inputs'][count]['admission'] == refs['admission'],
            'CPU DEV must append exactly this completed GPU DEV owner to its full normal prefix')
    require(parent['schema'] == GPU_SCHEMA and parent['entry_source'] == refs['source'] and
            Path(parent['GPU_admission_file']).resolve(strict=True) == Path(refs['admission']['path']).resolve(strict=True) and
            parent['phase'] == spec['RAW_inputs'][count]['phase'] and
            report['schema'] == gpu.REPORT_SCHEMA and report['status'] == gpu.STATUS and
            report['source'] == refs['source'] and report['spec'] == refs['spec'] and
            report['GPU_admission'] == refs['admission'] and report['phase'] == parent['phase'],
            'Completed GPU DEV report/source/spec/phase/admission binding differs')
    require(report['RAW_inputs'] == parent['RAW_inputs'] and report['normal_RAW_count'] == count and
            report['terminal_identity_pins'] == parent['terminal_identity_pins'] and
            report['additional_preserved_independent_CPU_failure'] == parent['additional_preserved_independent_CPU_failure'] and
            report['total_failed_RAW_count'] == 9 and report['total_terminal_identity_pin_count'] == count + 9,
            'Completed GPU DEV preserved full RAW/pin population differs')
    for key in ('prior_failed_CPU_receipt', 'additional_failed_RAW_inputs',
                'additional_preserved_independent_CPU_failure'):
        require(parent[key] == spec[key], 'Completed GPU DEV failed history differs: ' + key)
    require(all(spec['terminal_identity_pins'].get(key) == value
                for key, value in parent['terminal_identity_pins'].items()) and
            all(closure.get(ref['path']) == ref for ref in parent['sources']),
            'Completed GPU DEV SOURCE/owner pins must remain fully inherited')
    for key in gpu.STAGE24_REF_KEYS + ('stage24_checkpoints', 'carrier_spec',
            'carrier_evaluation_source', 'evaluation_adapter_source', 'memory_source', 'seed'):
        require(parent[key] == spec[key], 'GPU DEV and CPU DEV fixed stage24 provenance differs: ' + key)
    for key in gpu.STAGE24_REF_KEYS + ('stage24_checkpoints', 'carrier_spec',
            'carrier_evaluation_source', 'evaluation_adapter_source', 'seed'):
        require(report[key] == spec[key], 'Completed GPU DEV report stage24 binding differs: ' + key)
    join = gpu.api(gate['lineage'], 'owner_ref_join')(gate['context'], gate['owners'], refs['RAW'],
        refs['admission'], refs['source'], refs['spec'], '0')
    require(type(join) is int and join == count, 'Actual completed GPU DEV GPU0 owner join must be latest normal ancestor')
    effective, context = gate['evaluation_spec'], gate['context']
    require(report['controls'] == effective['controls'] == ['native_official'] + list(ARMS) and
            report['images'] == 187 and report['raw_GT'] == 380 and report['queries_per_image'] == 100 and
            report['predictions_per_control'] == 18700 and report['original_parent_epochs'] == 24 and
            report['stage_epochs_completed'] == 24 and report['stage_updates_per_arm'] == 39864 and
            report['parameters'] == gate['adapter'].PARAMETERS and report['stage24_checkpoints'] == spec['stage24_checkpoints'] and
            report['DEV'] == context['DEV'] and report['native_reference_predictions'] == context['native_reference_predictions'] and
            report['teacher_checkpoint'] == gate['cache_report']['teacher_checkpoint'] and
            report['packet_refs'] == gate['packets'] and report['selector_manifest'] == gate['selector']['manifest_ref'] and
            report['selector_mode'] == 'reuse' and report['exact_shared_GPU_L_selector'] is True,
            'Historical full DEV population, original native teacher/selector and fixed stage24 differ')
    require(report['capture_schema'] == gate['adapter'].CAPTURE_SCHEMA and
            report['support_formula'] == 'abs(actual_original_GPU_native_L)<=2' and
            report['fixed_native_scores_class_order'] is True and
            all(report[key] is True for key in ('no_GT_query_selection', 'no_NMS', 'no_score_cutoff', 'no_TEST')) and
            all(report[key] is False for key in ('best_DEV_selection', 'native_model_loaded', 'original_JPEG_decoded',
                'optimizer_created', 'weights_updated', 'current_owned_GPU_terminal_self_certified',
                'independent_CPU_evaluation_replayed', 'CPU_computational_portability_established')) and
            report['native_forward_calls'] == 0 and type(report['cleanup']['allocated_after']) is int and
            report['cleanup']['allocated_after'] == 0 and type(report['peak_torch_allocated_bytes']) is int and
            0 <= report['peak_torch_allocated_bytes'] <= 4096 * 1024**2,
            'Completed GPU DEV execution/mask/capture/fixed protocol scope differs')
    # No GPU own-gate is rerun inside CPU: its actual source/spec/admission/RAW
    # owner join and its new full report are checked against this CPU closure.
    manifest = read(closed(closure, report['capture_manifest']))
    require(manifest['schema'] == gpu.CAPTURE_MANIFEST_SCHEMA and
            manifest['source'] == refs['source'] and manifest['spec'] == refs['spec'] and
            manifest['GPU_admission'] == refs['admission'] and manifest['phase'] == parent['phase'] and
            manifest['stage24_GPU_report'] == spec['stage24_GPU_report'] and
            manifest['stage24_CPU_qualification'] == spec['stage24_CPU_qualification'] and
            manifest['images'] == manifest['capture_files'] == 187 and manifest['queries'] == 18700 and
            manifest['capture_schema'] == report['capture_schema'] and
            manifest['current_owned_terminal_self_certified'] is False and
            manifest['actual_GPU_inputs_no_CPU_feature_rerender'] is True and
            manifest['prelocked_budget'] == report['capture_budget'], 'Completed GPU capture manifest binding differs')
    rows = manifest['rows']
    require(type(rows) is list and len(rows) == 187 and
            [row['dataset_index'] for row in rows] == list(range(187)) and
            report['capture_refs'] == [row['capture'] for row in rows] and
            len({row['capture']['path'] for row in rows}) == 187,
            'All187 real whole-image captures in canonical order required')
    for index, row in enumerate(rows):
        ref = row['capture']
        closed(closure, ref)
        require(type(row['dataset_index']) is int and row['dataset_index'] == index and
                row['packet'] == gate['packets'][index] and row['selector'] == gate['selector']['manifest']['rows'][index] and
                row['natural_queries'] == 100 and type(row['U']) is int and row['U'] >= 0 and
                type(row['T']) is int and row['T'] >= 0 and
                type(row['raw_tensor_payload_bytes']) is int and
                row['raw_tensor_payload_bytes'] == 1032 * row['U'] + 28 * row['T'] + 102400 and
                type(row['K_natural_query_order']) is list and len(row['K_natural_query_order']) == 100 and
                all(type(k) is int and k >= 0 for k in row['K_natural_query_order']) and
                sum(row['K_natural_query_order']) == row['T'], 'Capture manifest all100 population/payload differs')
    require(sum(row['raw_tensor_payload_bytes'] for row in rows) == manifest['raw_tensor_payload_bytes'] ==
                report['raw_capture_tensor_payload_bytes'] and
            sum(row['capture']['bytes'] for row in rows) == manifest['actual_serialized_capture_bytes'] ==
                report['actual_serialized_capture_bytes'], 'Real aggregate capture file/payload accounting differs')
    controls = report['controls']
    require(set(report['predictions']) == set(report['control_results']) == set(controls),
            'All three completed GPU native/BASE/MEM controls required')
    predictions = {}
    for control in controls:
        ref = report['predictions'][control]
        require(report['control_results'][control]['predictions'] == ref and
                report['control_results'][control]['COCO']['stats'] == report['control_results'][control]['stats'],
                'Completed GPU control prediction/COCO12 binding differs')
        predictions[control] = read(closed(closure, ref))
        require(type(predictions[control]) is list and len(predictions[control]) == 18700,
                'Completed GPU control must retain every one of18700 queries')
    for key in ('start', 'trace', 'image_diagnostics'):
        closed(closure, report[key])
    diagnostics = read(closed(closure, report['image_diagnostics']))
    trace = [json.loads(line) for line in closed(closure, report['trace']).read_text(encoding='utf-8').splitlines()]
    require(type(diagnostics) is list and len(diagnostics) == 187 and trace == diagnostics,
            'Completed GPU full187 trace/image diagnostics must match')
    for index, (row, diagnostic) in enumerate(zip(rows, diagnostics)):
        require(diagnostic['dataset_index'] == index and diagnostic['image_id'] == row['image_id'] and
                diagnostic['packet'] == row['packet'] and diagnostic['selector'] == row['selector'] and
                diagnostic['capture'] == row['capture'] and diagnostic['U'] == row['U'] and diagnostic['T'] == row['T'] and
                diagnostic['raw_capture_tensor_bytes'] == row['raw_tensor_payload_bytes'] and
                diagnostic['K_natural_query_order'] == row['K_natural_query_order'] and diagnostic['queries'] == 100 and
                diagnostic['GT_used_to_choose_queries_or_residual'] is False and
                diagnostic['fixed_native_scores_class_order'] is True and diagnostic['oldLOG_support_unchanged'] is True and
                diagnostic['exact_GPU_L_selector'] is True and diagnostic['stage_epochs_completed'] == 24,
                'Completed GPU trace/manifest full-query identity differs')
    protocol = read(closed(closure, spec['CPU_DEV_numeric_protocol']))
    numeric_check(protocol, refs['report'])
    return dict(refs=refs, spec=parent, report=report, manifest=manifest,
                rows=rows, predictions=predictions, numeric_protocol=protocol)


def capture_validator(torch, evidence, gate, rows, population, coco):
    lookup_rows = {row['capture']['path']: row for row in evidence['rows']}
    refs = evidence['refs']
    expected_binding = dict(source=refs['source'], spec=refs['spec'], GPU_admission=refs['admission'],
        phase=evidence['spec']['phase'], stage24_GPU_source=gate['stage24_gate']['GPU_source'],
        stage24_GPU_spec=gate['stage24_gate']['GPU_spec'], stage24_GPU_report=gate['stage24_gate']['GPU_report_ref'],
        stage24_CPU_qualification=gate['stage24_gate']['CPU_qualification_ref'])

    def validate(ref, packet):
        row = lookup_rows[ref['path']]
        index = row['dataset_index']
        require(type(packet) is dict and packet['schema'] == gate['adapter'].CAPTURE_SCHEMA and
                packet['binding'] == expected_binding and packet['dataset_index'] == index and
                packet['image_id'] == row['image_id'] == rows[index]['image_id'] and
                packet['native_packet'] == gate['packets'][index] and packet['selector'] == row['selector'] and
                packet['natural_queries'] == 100 and packet['natural_query_ids'] == list(range(100)) and
                packet['U'] == row['U'] and packet['T'] == row['T'] and
                packet['raw_tensor_payload_bytes'] == row['raw_tensor_payload_bytes'] and
                packet['GT_query_selection'] is False and packet['independent_CPU_replayed'] is False and
                packet['current_owned_terminal_self_certified'] is False,
                'Whole-image capture must bind actual completed GPU owner and canonical dataset/native/selector')
        expected_native = population[row['image_id']]
        require(packet['baseline'] == expected_native and
                packet['native_export_query_ids'] == [entry['query_id'] for entry in expected_native] and
                sorted(packet['native_export_query_ids']) == list(range(100)) and
                set(packet['GPU_predictions_by_arm']) == set(ARMS) and
                [entry['K'] for entry in packet['queries']] == row['K_natural_query_order'],
                'All100 capture original score/order/query counts differ')
        shape = packet['identity']['original_shape']
        require(type(shape) in (list, tuple) and len(shape) == 2 and all(type(v) is int and v > 0 for v in shape) and
                tuple(shape) == (coco.imgs[row['image_id']]['height'], coco.imgs[row['image_id']]['width']),
                'Actual positive original-raster shape required')
        for arm in ARMS:
            begin = index * 100
            require(packet['GPU_predictions_by_arm'][arm] == evidence['predictions'][arm][begin:begin + 100],
                    'Capture actual GPU RLEs must equal completed full18700 GPU export')
        # Reject malformed/nonfinite GPU targets as protocol errors before the
        # adapter's allclose message can classify a numerical portability failure.
        for q, entry in enumerate(packet['queries']):
            K = row['K_natural_query_order'][q]
            require(type(entry) is dict and entry['query_id'] == q and entry['K'] == K and
                    set(entry['arms']) == set(ARMS), 'Complete GPU query target record required')
            for arm in ARMS:
                target = entry['arms'][arm]
                require(type(target) is dict and target['query_id'] == q and target['arm'] == arm,
                        'Captured GPU target arm/query identity differs')
                for name in ('GPU_residual', 'GPU_refined_logits'):
                    value = target[name]
                    require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and
                            value.dtype == torch.float32 and tuple(value.shape) == (K,) and
                            not value.requires_grad and bool(torch.isfinite(value).all()),
                            'Captured GPU target must be finite detached FP32 CPU with exact point shape')
        for key in ('native_packet',):
            closed(gate['closure'], packet[key])
    return validate


class BoundedHead:
    """Check own elapsed/RSS before every2048-point CPU head call."""
    def __init__(self, head, started, process, limit):
        self.head, self.started, self.process, self.limit = head, started, process, limit
        self.calls = self.points = 0

    def __getattr__(self, name):
        return getattr(self.head, name)

    def __call__(self, F, *args, **kwargs):
        deadline(self.started)
        require(self.process.memory_info().rss < self.limit, 'CPU process RSS exceeds prelocked replay limit')
        self.calls += 1
        self.points += len(F)
        return self.head(F, *args, **kwargs)


def selector_loader(torch, gpu, gate, sink, started, process, limit):
    def load_selector(row):
        deadline(started)
        require(process.memory_info().rss < limit and sink.current is not None,
                'Selector read needs one current bound image capture and permitted RSS')
        capture = sink.current
        index = capture['dataset_index']
        require(row == gate['selector']['manifest']['rows'][index] == capture['selector'],
                'Selector must be the exact original immutable row for this capture')
        expected = dict(dataset_index=index, image_id=capture['image_id'], identity=capture['identity'],
            native_packet=capture['native_packet'], natural_query_ids=list(range(100)),
            native_export_query_ids=capture['native_export_query_ids'],
            renderer_source=gate['evaluation_spec']['renderer_source'])
        require(all(row[key] == value for key, value in expected.items()),
                'Original selector metadata must equal the captured GPU image identity/query/renderer')
        packet = torch.load(str(closed(gate['closure'], row['carrier'])), map_location='cpu')
        parent = gate['selector']['parent']
        require(type(packet) is dict and packet['schema'] == gate['lineage'].SELECTOR_PACKET_SCHEMA and
                packet['source'] == parent['evaluation_entry_source'] and
                packet['spec'] == parent['evaluation_spec'] and packet['GPU_admission'] == parent['evaluation_GPU_admission'] and
                all(packet[key] == value for key, value in expected.items()) and
                packet['L_digest'] == row['L_digest'] and gpu.digest(packet['native_L']) == row['L_digest'],
                'Closed selector packet actual old GPU source/spec/admission/full100 L bytes differ')
        value = packet['native_L']
        require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and value.dtype == torch.float32 and
                tuple(value.shape) == (100,) + tuple(capture['identity']['original_shape']) and
                bool(torch.isfinite(value).all()), 'Original immutable full100 GPU L must be finite FP32 CPU readback')
        del packet
        return value
    return load_selector


def replay_budget_check(spec, evidence, coco, ids, process, psutil, output):
    budget = spec['CPU_replay_budget']
    keys = {'maximum_image_capture_file_bytes', 'maximum_image_capture_tensor_payload_bytes',
            'maximum_selector_tensor_bytes', 'required_available_RAM_bytes', 'maximum_process_RSS_bytes',
            'auxiliary_output_budget_bytes', 'minimum_free_output_bytes', 'minimum_available_RAM_margin_bytes'}
    require(type(budget) is dict and set(budget) == keys and
            all(type(value) is int and value >= 0 for value in budget.values()),
            'ROOT must prelock exact CPU capture/selector/availableRAM/RSS/output budget fields')
    maximum_payload = max(row['raw_tensor_payload_bytes'] for row in evidence['rows'])
    maximum_file = max(row['capture']['bytes'] for row in evidence['rows'])
    maximum_pixels = max(coco.imgs[i]['height'] * coco.imgs[i]['width'] for i in ids)
    require(budget['maximum_image_capture_file_bytes'] == maximum_file and
            budget['maximum_image_capture_tensor_payload_bytes'] == maximum_payload and
            budget['maximum_selector_tensor_bytes'] == 400 * maximum_pixels and
            budget['minimum_available_RAM_margin_bytes'] >= 1024**3 and
            budget['required_available_RAM_bytes'] >= maximum_payload + 400 * maximum_pixels +
                budget['minimum_available_RAM_margin_bytes'] and
            budget['maximum_process_RSS_bytes'] > process.memory_info().rss + maximum_payload + 400 * maximum_pixels and
            psutil.virtual_memory().available >= budget['required_available_RAM_bytes'] and
            budget['minimum_free_output_bytes'] >= budget['auxiliary_output_budget_bytes'] + 1024**3 and
            psutil.disk_usage(str(output.parent)).free >= budget['minimum_free_output_bytes'],
            'Concrete completed capture sizes, full100 selector and CPU RAM/output prelock are not met')
    return budget


def imported_files(gpu, gate, psutil, torch, np, cv2, masks, COCO, COCOeval, ndimage):
    numerical = gate['common'].imported_files(torch, np, cv2, masks, COCO,
        gate['closure'], gate['admission'], ndimage)
    extra = gpu.imported_files(psutil, torch, np, gate)
    extra['official_COCOeval_python'] = reference(inspect.getsourcefile(COCOeval))
    require(all(gate['closure'].get(ref['path']) == ref and
                gate['admission']['bound_source_sha256'].get(ref['path']) == ref['sha256'] for ref in extra.values()),
            'Actual independent CPU libraries/COCOeval source must be completely SOURCE/admission pinned')
    return dict(numerical=numerical, independent_CPU=extra)


def run(spec_path):
    started = time.monotonic()
    path = Path(spec_path).resolve(strict=True)
    spec = read(path)
    require(spec['schema'] == SCHEMA, 'Fresh independent CPU DEV spec schema required')
    _, closure = raw_first(spec)
    require(Path(spec['entry_source']['path']).resolve(strict=True) == Path(__file__).resolve(strict=True),
            'CPU own source must be this new complete entry')
    closed(closure, spec['entry_source'])
    gpu = load(closure, spec['GPU_runner_source'], '_N14_CPU_DEV_completed_GPU_gate')
    gate = gpu.source_gate(path, spec, cpu=True)
    require(gate['closure'] == closure and gate['source'] == spec['entry_source'] and
            gate['spec_ref'] == reference(path) and
            Path(gate['admission_ref']['path']).resolve(strict=True) ==
                Path(spec['CPU_admission_file']).resolve(strict=True),
            'Shared gate must bind this CPU source/spec/CPUempty admission, not GPU own entry')
    evidence = completed_GPU_DEV_gate(gpu, gate, spec)
    effective, context, adapter = gate['evaluation_spec'], gate['context'], gate['adapter']
    output = Path(spec['output_directory'])
    require(output.is_absolute() and not output.exists() and output.parent.is_dir(),
            'Fresh absolute ROOT-owned CPU evaluation output required')
    for target in (output / 'QUALIFICATION.json', output / 'DETAILS.json', output / 'FAILURE.json'):
        require(str(target) not in closure, 'Current CPU outputs must never be input SOURCE/Q refs')
    import psutil
    import torch
    require(torch.__version__ == '2.0.1+cu118' and not torch.cuda.is_initialized(),
            'Original cold Torch runtime required')
    process = psutil.Process()
    resource = cpu_resource_check(spec, gate['admission'], psutil, torch)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    canonical = gpu.canonical_stage24_gate(torch, gate, spec)
    base_factory = load(closure, context['point_heads_source'], '_N14_CPU_DEV_original_base_factory')
    memory_factory = load(closure, spec['memory_source'], '_N14_CPU_DEV_memory_factory')
    heads, states, head_proofs, canonical = adapter.load_stage24_heads(torch, effective,
        lambda supplied: canonical, lambda ref: closed(closure, ref), base_factory, memory_factory)
    import numpy as np
    import cv2
    import scipy
    from scipy import ndimage
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools import mask as masks
    require(np.__version__ == '1.23.5' and cv2.__version__ == '4.8.0' and scipy.__version__ == '1.10.1',
            'Qualified numerical versions required')
    n10 = load(closure, context['N10_evaluation_source'], '_N14_CPU_DEV_original_N10')
    reader = load(closure, context['reader_source'], '_N14_CPU_DEV_original_reader')
    official = load(closure, context['COCO_source'], '_N14_CPU_DEV_original_official_COCO')
    coco = COCO(str(closed(closure, context['DEV'])))
    ids = list(coco.imgs)
    require(len(ids) == len(set(ids)) == 187 and ids == [row['id'] for row in coco.dataset['images']] and
            len(coco.anns) == 380 and set(coco.cats) == {1}, 'Original full DEV187/GT380/category1 required')
    native = read(closed(closure, context['native_reference_predictions']))
    population = reader.population(coco, native, ids)
    rows = read(closed(closure, gate['prior']['image_rows']))
    prior_trace = [json.loads(line) for line in closed(closure, gate['prior']['trace']).read_text(encoding='utf-8').splitlines() if line.strip()]
    require(rows == prior_trace and len(rows) == len(gate['packets']) == len(evidence['rows']) == 187 and
            [row['image_id'] for row in rows] == ids == [row['image_id'] for row in evidence['rows']] and
            sum(row['raw_GT'] for row in rows) == 380 and evidence['predictions']['native_official'] == native,
            'Complete original native predictions/187 canonical packet/capture/selector order required')
    for values in evidence['predictions'].values():
        reader.population(coco, values, ids)
        require(all({k: v for k, v in a.items() if k != 'segmentation'} ==
                    {k: v for k, v in b.items() if k != 'segmentation'} for a, b in zip(native, values)),
                'GPU completed exports must retain original class/score/query/order exactly')
    budget = replay_budget_check(spec, evidence, coco, ids, process, psutil, output)
    libraries = imported_files(gpu, gate, psutil, torch, np, cv2, masks, COCO, COCOeval, ndimage)
    require(not torch.cuda.is_initialized(), 'Complete independent CPU gates must keep CUDA cold')
    outside = torch.get_rng_state().clone()
    bounded_heads = {arm: BoundedHead(heads[arm], started, process, budget['maximum_process_RSS_bytes']) for arm in ARMS}
    sink = ClosedImageSink(torch, closure, evidence['report']['capture_refs'],
        capture_validator(torch, evidence, gate, rows, population, coco))
    read_selector = selector_loader(torch, gpu, gate, sink, started, process, budget['maximum_process_RSS_bytes'])
    completed = 0
    peak_RSS = process.memory_info().rss
    predictions = {arm: [] for arm in ARMS}
    diagnostics = []
    trace_sha, trace_bytes = hashlib.sha256(), 0
    output.mkdir()
    start_ref = write_new(output / 'START.json', dict(schema='N14-stage24-fixed-DEV-CPU-start-v1',
        source=gate['source'], spec=gate['spec_ref'], CPU_admission=gate['admission_ref'], phase=spec['phase'],
        completed_GPU_DEV_report=evidence['refs']['report'], capture_manifest=evidence['report']['capture_manifest'],
        stage24_GPU_report=spec['stage24_GPU_report'], stage24_CPU_qualification=spec['stage24_CPU_qualification'],
        stage24_checkpoints=spec['stage24_checkpoints'], CPU_DEV_numeric_protocol=spec['CPU_DEV_numeric_protocol'],
        numeric_protocol=evidence['numeric_protocol'], head_state_proofs=head_proofs, stage24_gate=canonical,
        CPU_resources=resource, CPU_replay_budget=budget, no_CPU_feature_rerender=True, no_training=True,
        queries_per_image=100, predictions_per_control=18700, no_TEST=True, best_DEV_selection=False,
        own_terminal_self_certified=False))
    try:
        with (output / 'TRACE.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
            for index, capture_row in enumerate(evidence['rows']):
                deadline(started)
                rss = process.memory_info().rss
                require(rss + capture_row['raw_tensor_payload_bytes'] + 400 *
                    coco.imgs[capture_row['image_id']]['height'] * coco.imgs[capture_row['image_id']]['width'] <
                    budget['maximum_process_RSS_bytes'], 'One captured image/full100 selector would exceed prelocked CPU RSS')
                tick = time.monotonic()
                replay = replay_one(adapter, torch, np, masks, n10, bounded_heads, capture_row['capture'],
                    sink, read_selector, evidence['numeric_protocol'], evidence['refs']['report'])
                peak_RSS = max(peak_RSS, process.memory_info().rss)
                require(set(replay) == set(ARMS) and all(len(replay[arm]) == 100 for arm in ARMS),
                        'Independent CPU must replay both complete100-query controls')
                for arm in ARMS:
                    require(replay[arm] == evidence['predictions'][arm][index * 100:(index + 1) * 100],
                            'Whole-image CPU RLEs must equal completed GPU full18700 export')
                    predictions[arm] += replay[arm]
                diagnostic = dict(dataset_index=index, image_id=capture_row['image_id'], capture=capture_row['capture'],
                    packet=capture_row['packet'], selector=capture_row['selector'], U=capture_row['U'], T=capture_row['T'],
                    raw_tensor_payload_bytes=capture_row['raw_tensor_payload_bytes'], queries=100,
                    residual_and_combined_allclose_fixed1e4=True, strict_greater_than_zero_sign_exact=True,
                    full_fixed_score_RLE_exact=True, actual_GPU_F_e_L_inputs=True, CPU_F_rerendered=False,
                    elapsed_seconds=time.monotonic() - tick)
                diagnostics.append(diagnostic)
                line = json.dumps(diagnostic, allow_nan=False) + '\n'
                stream.write(line)
                encoded = line.encode('utf-8')
                trace_sha.update(encoded)
                trace_bytes += len(encoded)
                stream.flush()
                completed = index + 1
                del replay, diagnostic, line, encoded
                sink.release()
                gc.collect()
                require(not torch.cuda.is_initialized(), 'CPU DEV replay must never initialize CUDA')
                print(json.dumps(dict(N14_CPU_DEV_images_replayed=completed,
                    exact_queries_per_arm=100 * completed)), flush=True)
        require(completed == 187 and all(len(values) == 18700 for values in predictions.values()) and
                all(bounded_heads[arm].points == sum(row['T'] for row in evidence['rows']) for arm in ARMS),
                'Every actual GPU sampled point and all18700 queries per arm must be CPU replayed')
        deadline(started)
        results = adapter.official_COCO_readback(coco, ids, native,
            {arm: evidence['predictions'][arm] for arm in ARMS}, predictions,
            official, COCOeval, n10, evidence['report']['control_results'])
        require(all(len(result['stats']) == 12 for result in results.values()), 'All original official COCO12 values required')
        n10.stats_exact(results['native_official']['stats'], gate['prior_q']['expected_E24_COCO_stats'],
                        'Original strong native COCO12 differs')
        n10.stats_exact(results['native_official']['stats'], gate['prior']['control_results']['native_official']['stats'],
                        'Original prior official native COCO12 differs')
        require(all(torch.equal(value, states[arm][name]) for arm, head in heads.items()
                    for name, value in head.state_dict().items()) and
                all(not p.requires_grad and p.grad is None and p.device.type == 'cpu' and p.dtype == torch.float32
                    for head in heads.values() for p in head.parameters()) and
                torch.equal(outside, torch.get_rng_state()) and not torch.cuda.is_initialized(),
                'Qualified CPU state/grad/outside RNG/cold CUDA changed')
        prediction_refs = {control: write_new(output / (control.upper() + '_PREDICTIONS.json'),
            native if control == 'native_official' else predictions[control]) for control in effective['controls']}
        diagnostic_ref = write_new(output / 'IMAGE_DIAGNOSTICS.json', diagnostics)
        for ref in spec['sources']:
            deadline(started)
            bound(ref)
        require(reference(path) == gate['spec_ref'] and reference(__file__) == gate['source'] and
                reference(spec['CPU_admission_file']) == gate['admission_ref'] and
                imported_files(gpu, gate, psutil, torch, np, cv2, masks, COCO, COCOeval, ndimage) == libraries and
                process.cpu_affinity() == [0, 1] and torch.get_num_threads() == torch.get_num_interop_threads() == 2 and
                not torch.cuda.is_initialized() and process.memory_info().rss < budget['maximum_process_RSS_bytes'],
                'Final own CPU binding/resources/libraries/cold CUDA differ')
        gate['lineage'].final_endpoint_check(gate['parent'])
        deadline(started)
        calls = {arm: bounded_heads[arm].calls for arm in ARMS}
        points = {arm: bounded_heads[arm].points for arm in ARMS}
        trace_ref = dict(path=str((output / 'TRACE.jsonl').resolve()),
                         bytes=trace_bytes, sha256=trace_sha.hexdigest())
        bounded_heads.clear()
        heads.clear()
        sink.release()
        gc.collect()
        require(not torch.cuda.is_initialized(), 'CPU final cleanup must retain cold CUDA')
        details_ref = write_new(output / 'DETAILS.json', dict(schema='N14-stage24-fixed-DEV-CPU-details-v1',
            source=gate['source'], spec=gate['spec_ref'], CPU_admission=gate['admission_ref'], phase=spec['phase'],
            completed_GPU_DEV_report=evidence['refs']['report'], capture_manifest=evidence['report']['capture_manifest'],
            capture_refs=evidence['report']['capture_refs'], CPU_DEV_numeric_protocol=spec['CPU_DEV_numeric_protocol'],
            numeric_protocol=evidence['numeric_protocol'], start=start_ref, trace=trace_ref, image_diagnostics=diagnostic_ref,
            images=187, raw_GT=380, queries_per_image=100, predictions_per_control=18700,
            controls=effective['controls'], predictions=prediction_refs, control_results=results,
            CPU_resources=resource, CPU_replay_budget=budget, actual_imported_files=libraries,
            CPU_point_calls_by_arm=calls, actual_GPU_points_CPU_replayed_by_arm=points,
            sampled_process_RSS_bytes_max=peak_RSS, process_RSS_complete_peak_certified=False,
            elapsed_seconds=time.monotonic() - started, current_owned_CPU_terminal_self_certified=False,
            no_CPU_feature_rerender=True, no_TEST=True, best_DEV_selection=False))
        q = dict(schema='N14-stage24-fixed-fullDEV-independent-CPU-qualification-v1',
            status='ROOT_N14_STAGE24_FIXED_FULL_DEV_CPU_COMPUTATIONAL_PORTABILITY_QUALIFIED_V1',
            source=gate['source'], spec=gate['spec_ref'], CPU_admission=gate['admission_ref'], phase=spec['phase'],
            completed_GPU_DEV_source=evidence['refs']['source'], completed_GPU_DEV_spec=evidence['refs']['spec'],
            completed_GPU_DEV_report=evidence['refs']['report'], completed_GPU_DEV_admission=evidence['refs']['admission'],
            completed_GPU_DEV_RAW=evidence['refs']['RAW'], stage24_GPU_report=spec['stage24_GPU_report'],
            stage24_CPU_qualification=spec['stage24_CPU_qualification'], stage24_checkpoints=spec['stage24_checkpoints'],
            capture_manifest=evidence['report']['capture_manifest'], capture_refs=evidence['report']['capture_refs'],
            CPU_DEV_numeric_protocol=spec['CPU_DEV_numeric_protocol'], details=details_ref,
            original_parent_epochs=24, stage_epochs_completed=24, parameters=adapter.PARAMETERS,
            images=187, raw_GT=380, queries_per_image=100, predictions_per_control=18700,
            same_actual_GPU_point_inputs=True, all18700_residual_and_combined_allclose_fixed1e4=True,
            all18700_strict_greater_than_zero_sign_exact=True, all18700_original_score_order_RLE_exact=True,
            original_official_COCO12_exact=True, CPU_computational_portability_established=True,
            CPU_feature_rerendered=False, CPU_training_executed=False, current_owned_CPU_terminal_self_certified=False,
            normal_RAW_count=spec['normal_RAW_count'], preserved_failed_RAW_count=9,
            total_terminal_identity_pin_count=spec['normal_RAW_count'] + 9,
            GPU_AP_external_validity_determined_by_this_CPU_Q=False, GPU_AP_invalidated_by_CPU_failure=False,
            no_TEST=True, best_DEV_selection=False, task_efficacy_established=False,
            innovation_established=False, generalization_established=False, capacity_confound_not_yet_controlled=True,
            evidence_limits='Independent computational portability of these actual captured GPU inputs and masks on historical full DEV187 at fixed stage24, one seed. '
                'The trained GPU AP and native provenance require their separate actual ROOT terminal/report checks. '
                'No blind TEST, external SOTA, convergence, capacity-controlled mechanism, deployment or innovation claim. '
                'This attempt cannot certify its own terminal; ROOT must qualify actual exit/reap/original-birth absence separately.')
        q_size = len((json.dumps(q, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8'))
        require(sum(p.stat().st_size for p in output.iterdir() if p.is_file()) + q_size <= budget['auxiliary_output_budget_bytes'],
                'Fresh CPU auxiliary output exceeds prelocked output budget')
        q_ref = write_new(output / 'QUALIFICATION.json', q)
        print(json.dumps(dict(status=q['status'], qualification=q_ref, details=details_ref,
            current_owned_CPU_terminal_self_certified=False)), flush=True)
        return q_ref
    except Exception as error:
        category = ('CPU_COMPUTATIONAL_PORTABILITY_FAILURE' if isinstance(error, ComputationalPortabilityFailure)
                    else 'CPU_DEV_PROTOCOL_OR_EXECUTION_FAILURE')
        failure = dict(schema='N14-stage24-fixed-DEV-CPU-failure-v1', status=category,
            source=gate['source'], spec=gate['spec_ref'], CPU_admission=gate['admission_ref'], phase=spec['phase'],
            completed_GPU_DEV_report=evidence['refs']['report'], CPU_DEV_numeric_protocol=spec['CPU_DEV_numeric_protocol'],
            completed_images=completed, complete_population_CPU_qualified=False,
            error_type=type(error).__name__, error=str(error), posterior_tolerance_relaxed=False,
            GPU_AP_invalidity_inferred_from_CPU_failure=False, method_failure_inferred=False,
            preserved_failed_RAW_count=9, current_owned_CPU_terminal_self_certified=False,
            elapsed_seconds=time.monotonic() - started)
        failure_ref = write_new(output / 'FAILURE.json', failure)
        print(json.dumps(dict(status=category, failure=failure_ref, complete_population_CPU_qualified=False)), flush=True)
        raise
    finally:
        sink.release()
        bounded_heads.clear()
        heads.clear()
        gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args()
    require(args.owner_execute, 'UNBOUND_DRAFT: ROOT must close/review and admit a fresh independent CPU DEV attempt')
    run(args.spec)
