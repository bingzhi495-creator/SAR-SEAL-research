"""UNBOUND_DRAFT: one-image CPU/GPU feature/arithmetic compatibility probe.

NOT RUN. No qualification, efficacy, AP, or causal conclusion is established.
ROOT alone reviews/binds a new immutable source and owns one scientific attempt.
The failed CPU source and the existing GPU C are never modified or substituted.
The target is the complete failed image, index25/image113, both (GT39,query1)
and (GT40,query89); GT IDs are provenance labels only and GT is not a numeric input.
Sources containing labels may be SHA-qualified, but no GT is decoded or used.

Required future schema: ROOT-N13-fixed-PR-CPU-GPU-compatibility-image-probe-spec-v1.
Keep original RAW_inputs normal94, old failed8, terminal_identity_pins102 intact.
additional_preserved_independent_CPU_failure must contain exactly RAW_ref, phase,
pin, returncode=1; total_failed_RAW_count=9, total_terminal_identity_pin_count=103.
Its original RAW-only terminal/pin check precedes every SHA or SOURCE import.
Then the original normal94/failed8 pre_gate contract runs, including its first
SOURCE hashes; the actual pinned GPU pre_gate is subsequently inherited again.

Required caller fields beyond the preserved carrier/GPU references and protocol:
  entry_source, phase, GPU_admission_file, output_directory,
  deadline_seconds=1800, minimum_free_output_bytes;
  target_dataset_index=25, target_image_id=113, target_candidates (both complete
  original GPU image_checks candidate refs, in their recorded order);
  packet_field_contract={schema,feature_path,identity_path,feature_shape};
    actual paths are ['native_fields','F256'], ['identity'], shape[1,256,200,200],
    and schema must be confirmed by ROOT from the bound original descriptor;
  weight_contract={checkpoint,state_path,weight_keys}; checkpoint is the frozen
    carrier_GPU_spec.endpoint.checkpoints.LOG_PR, state_path=['state_dict'],
    keys are hidden1/2/3 and final weight,bias in their exact documented order;
  formula_comparison={atol:1e-4,rtol:1e-4}; never widened for this probe;
  legacy_support={source:carrier_GPU_spec,numeric_object_path,refined_support_atol,
    refined_support_rtol}; numeric location/values must equal the original lock.
ROOT supplies actual SOURCE refs and reviews external lineage/N10/renderer/common
ABI; a missing descriptor/API fails explicitly, with no invented fallback.

Four paths: CPU-rendered F with independent CPU functional PR; GPU-rendered F
copied to CPU with the same CPU PR; GPU-rendered F with independent GPU PR;
CPU-rendered F copied to GPU with the same GPU PR. Exact frozen GPU native L,
the original separate ascending2048 oldLOG/halo partitions, eight frozen weights,
and actual stored GPU residual/logit are shared. GT/selection/K never enter PR.
All incompatibilities are recorded at the SAME 1e-4 atol/rtol, never treated as
qualification or used to replace C. Original practical support rule is recorded
as tau=atol+rtol*abs(path_logit), with outside-allowance sign counts; no count cap
or absolute near-zero limit is introduced. It is not a certified error bound.
No process launching, retry, training, dataset evaluation, Q or publishing exists.
"""

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time
import types


SCHEMA = 'ROOT-N13-fixed-PR-CPU-GPU-compatibility-image-probe-spec-v1'
REPORT_SCHEMA = 'N13-fixed-PR-CPU-GPU-compatibility-image-probe-report-v1'
STATUS = 'DIAGNOSTIC_PENDING_ROOT_ANALYSIS'
GPU_SCHEMA = 'N13-fixed-PR-real-diagnostic-GPU-report-v1'
GPU_STATUS = 'N13_FIXED_PR_REAL_DIAGNOSTIC_COMPLETE_PENDING_INDEPENDENT_CPU'
RAW_KEYS = ('receipt', 'phase', 'admission', 'phase_binding', 'bootstrap')
TARGET_IDENTITIES = [dict(image_id=113, gt_id=39, query_id=1), dict(image_id=113, gt_id=40, query_id=89)]
TARGET_PARTITION_COUNTS = {1: (73, 560), 89: (13, 274)}
PACKET_SCHEMA = 'N07-original-detail-all100-native-fields-v1'
WEIGHT_KEYS = ('hidden1.weight', 'hidden1.bias', 'hidden2.weight', 'hidden2.bias',
               'hidden3.weight', 'hidden3.bias', 'final.weight', 'final.bias')
WEIGHT_SHAPES = ((91, 257), (91,), (91, 92), (91,), (91, 92), (91,), (1, 92), (1,))
PATH_NAMES = ('CPU_F_CPU_PR', 'GPU_F_to_CPU_CPU_PR', 'GPU_F_GPU_PR', 'CPU_F_to_GPU_GPU_PR')
THREAD_ENV = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def api(module, name):
    value = getattr(module, name, None)
    require(callable(value), 'ROOT must adapt/review missing external ABI: ' + name)
    return value


def read(path):
    def reject(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=reject)


def ref_shape(ref):
    require(type(ref) is dict and set(ref) == {'path', 'bytes', 'sha256'} and
            type(ref['path']) is str and Path(ref['path']).is_absolute() and
            type(ref['bytes']) is int and ref['bytes'] >= 0 and type(ref['sha256']) is str and
            len(ref['sha256']) == 64 and set(ref['sha256']) <= set('0123456789abcdef'),
            'Malformed absolute pinned reference')


def reference(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def bound(ref):
    ref_shape(ref)
    require(reference(ref['path']) == ref, 'Pinned input changed: ' + ref['path'])
    return Path(ref['path'])


def closed(closure, ref):
    ref_shape(ref)
    require(closure.get(ref['path']) == ref, 'Missing exact complete SOURCE reference: ' + ref['path'])
    return bound(ref)


def load(ref, name):
    path = bound(ref)
    source = path.read_bytes()
    require(len(source) == ref['bytes'] and hashlib.sha256(source).hexdigest() == ref['sha256'], 'SOURCE changed before import')
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def terminal_raw_only(ref, phase, code):
    # Reads only original RAW JSON; no SOURCE hash/import or process probe.
    ref_shape(ref)
    path = Path(ref['path'])
    require(path.name == 'PARENT_RECEIPT.json' and type(phase) is str and phase and path.parent.name == phase,
            'Original RAW locator required')
    raw = read(path)
    require(raw['status'] == 'ACTUAL_OWNED_TERMINAL' and type(raw['returncode']) is int and raw['returncode'] == code and
            type(raw['original_identity']) is dict and raw['original_identity'] and raw['root_wait_reaped'] is True and
            raw['held_handle_exit'] == dict(wait_state='WAIT_OBJECT_0', actual_exit_code=code) and
            type(raw['held_handle_exit']['actual_exit_code']) is int and
            raw['numeric_birth_absence']['original_birth_absent'] is True and raw.get('error') is None and
            raw.get('abort') is None and raw['wait_errors'] is False and raw['GPU_observation_errors'] == [],
            'Actual held exit/reap/original birth absence required before any SHA/import')
    if 'phase' in raw:
        require(raw['phase'] == phase, 'Original RAW phase differs')
    sha, peak = raw['admission_sha256'], raw['whole_GPU_observed_peak_MiB']
    require(type(sha) is str and len(sha) == 64 and set(sha) <= set('0123456789abcdef') and
            type(peak) in (int, float) and math.isfinite(peak) and 0 <= peak <= 8192 and
            raw['whole_GPU_used_limit_MiB'] == 8192 and raw['whole_leaf_VRAM_certified'] is False and
            raw['GPU_scope'] == 'whole GPU observed, not per-process WDDM allocation',
            'Preserved RAW resource/admission scope differs')
    return raw


def check_pin(pin, raw, phase, code):
    require(type(pin) is dict and set(pin) == {'phase', 'returncode', 'original_identity', 'admission_sha256'} and
            pin['phase'] == phase and type(pin['returncode']) is int and pin['returncode'] == code and
            pin['original_identity'] == raw['original_identity'] and pin['admission_sha256'] == raw['admission_sha256'],
            'Original actual owner differs from its immutable identity/admission pin')


def ninth_failed_raw_only(spec):
    extra = spec['additional_preserved_independent_CPU_failure']
    require(type(extra) is dict and set(extra) == {'RAW_ref', 'phase', 'pin', 'returncode'} and
            type(extra['returncode']) is int and extra['returncode'] == 1,
            'Independent failed9 needs exactly RAW_ref/phase/pin/returncode1')
    raw = terminal_raw_only(extra['RAW_ref'], extra['phase'], 1)
    check_pin(extra['pin'], raw, extra['phase'], 1)
    require(type(spec['total_failed_RAW_count']) is int and spec['total_failed_RAW_count'] == 9 and
            type(spec['total_terminal_identity_pin_count']) is int and spec['total_terminal_identity_pin_count'] == 103 and
            extra['RAW_ref']['path'] not in spec['terminal_identity_pins'],
            'Failed9/pins103 must remain separate from the original pins102')
    return raw


def pre_gate(spec, minimum):
    # Inherited RAW/source contract of the original GPU pre_gate, with
    # minimum94 supplied here. Additional failed9 has ALREADY passed RAW-only.
    rows = spec['RAW_inputs']
    expected = spec['normal_RAW_count']
    require(type(expected) is int and expected >= minimum and type(rows) is list and len(rows) == expected and
            len({row['receipt']['path'] for row in rows}) == expected, 'All original normal RAW ancestors required')
    normals = [terminal_raw_only(row['receipt'], row['phase'], 0) for row in rows]
    failed_ref = spec['prior_failed_CPU_receipt']
    failed_phase = Path(failed_ref['path']).parent.name
    failed = terminal_raw_only(failed_ref, failed_phase, 1)
    extra = spec['additional_failed_RAW_inputs']
    require(type(extra) is list and len(extra) == 7 and
            all(type(row) is dict and set(row) == {'receipt', 'phase', 'returncode'} and
                type(row['returncode']) is int and row['returncode'] == 1 for row in extra),
            'Do not alter the original extra failed7 list or preserved failed8')
    failures = [failed] + [terminal_raw_only(row['receipt'], row['phase'], 1) for row in extra]
    failed_paths = [failed_ref['path']] + [row['receipt']['path'] for row in extra]
    require(len(set(failed_paths)) == 8 and not {row['receipt']['path'] for row in rows}.intersection(failed_paths),
            'Original normal94 and failed8 owners must remain distinct')
    all_rows = [dict(row, returncode=0) for row in rows] + [dict(receipt=failed_ref, phase=failed_phase, returncode=1)] + extra
    pins = spec['terminal_identity_pins']
    require(type(pins) is dict and set(pins) == {row['receipt']['path'] for row in all_rows},
            'Original normal-plus-failed8 pins must remain unchanged')
    for row, raw in zip(all_rows, normals + failures):
        check_pin(pins[row['receipt']['path']], raw, row['phase'], row['returncode'])
    require(expected == 94 and len(pins) == 102, 'Original normal94/pins102 required')
    # FIRST HASH: failed9 and all original normal94/failed8 RAW checks passed.
    closure = {ref['path']: ref for ref in spec['sources']}
    require(len(closure) == len(spec['sources']), 'Unique SOURCE closure required')
    for ref in spec['sources']:
        bound(ref)
    require(all(row[key] in closure.values() for row in rows for key in RAW_KEYS if key != 'phase') and
            failed_ref in closure.values() and all(row['receipt'] in closure.values() for row in extra),
            'Original owner metadata and failed8 receipts must be sources-bound')
    return normals, closure


def lookup(value, keys, label):
    require(type(keys) is list and all(type(key) is str and key for key in keys), 'Explicit ROOT descriptor keys required: ' + label)
    for key in keys:
        require(type(value) is dict and key in value, 'ROOT must adapt actual bound descriptor: ' + label + '.' + key)
        value = value[key]
    return value


def source_gate(spec_path, spec):
    require(spec['schema'] == SCHEMA and type(spec['deadline_seconds']) is int and spec['deadline_seconds'] == 1800 and
            type(spec['target_dataset_index']) is int and spec['target_dataset_index'] == 25 and
            type(spec['target_image_id']) is int and spec['target_image_id'] == 113,
            'Exact1800-second complete failed image index25/image113 required')
    preserved_failure = ninth_failed_raw_only(spec)
    normals, closure = pre_gate(spec, 94)
    closed(closure, spec['additional_preserved_independent_CPU_failure']['RAW_ref'])
    for key in ('entry_source', 'diagnostic_GPU_source', 'diagnostic_GPU_spec', 'diagnostic_GPU_report',
                'diagnostic_GPU_RAW', 'diagnostic_GPU_admission', 'carrier_GPU_spec', 'carrier_CPU_spec',
                'carrier_CPU_qualification', 'carrier_GPU_RAW', 'carrier_CPU_RAW', 'carrier_CPU_admission',
                'carrier_evaluation_source', 'carrier_CPU_source'):
        closed(closure, spec[key])
    require(Path(spec['entry_source']['path']).resolve(strict=True) == Path(__file__).resolve(strict=True),
            'Only the new ROOT-reviewed immutable probe source may pass entry identity')
    gpu_source = load(spec['diagnostic_GPU_source'], '_N13_compatibility_probe_original_GPU')
    inherited_normals, inherited_closure = api(gpu_source, 'pre_gate')(spec, 94)
    require(inherited_normals == normals and inherited_closure == closure, 'Actual original GPU pre_gate differs')
    original = read(closed(closure, spec['carrier_GPU_spec']))
    carrier_cpu = read(closed(closure, spec['carrier_CPU_spec']))
    carrier_q = read(closed(closure, spec['carrier_CPU_qualification']))
    gpu_spec = read(closed(closure, spec['diagnostic_GPU_spec']))
    gpu_report = read(closed(closure, spec['diagnostic_GPU_report']))
    require(gpu_spec['schema'] == gpu_source.SCHEMA and gpu_report['schema'] == GPU_SCHEMA and gpu_report['status'] == GPU_STATUS and
            gpu_report['source'] == spec['diagnostic_GPU_source'] and gpu_report['spec'] == spec['diagnostic_GPU_spec'] and
            gpu_report['GPU_admission'] == spec['diagnostic_GPU_admission'] and gpu_report['phase'] == gpu_spec['phase'] and
            gpu_spec['entry_source'] == spec['diagnostic_GPU_source'] and
            gpu_spec['normal_RAW_count'] == gpu_report['normal_RAW_count'] == 93 and
            gpu_spec['RAW_inputs'] == gpu_report['RAW_inputs'] == spec['RAW_inputs'][:93] and
            gpu_spec['terminal_identity_pins'] == gpu_report['terminal_identity_pins'] ==
                {key: value for key, value in spec['terminal_identity_pins'].items() if key != spec['diagnostic_GPU_RAW']['path']} and
            gpu_spec['additional_failed_RAW_inputs'] == spec['additional_failed_RAW_inputs'] and
            gpu_spec['prior_failed_CPU_receipt'] == spec['prior_failed_CPU_receipt'],
            'Original diagnostic GPU provenance/prefix93/failed8 differs')
    require(spec['RAW_inputs'][93]['receipt'] == spec['diagnostic_GPU_RAW'] and
            spec['RAW_inputs'][93]['admission'] == spec['diagnostic_GPU_admission'] and
            spec['RAW_inputs'][93]['phase'] == gpu_spec['phase'] and gpu_report['images'] == 187 and
            gpu_report['matched_pairs'] == 374 and gpu_report['independent_CPU_qualified'] is False and
            gpu_report['current_owned_GPU_terminal_self_certified'] is False and gpu_report['no_TEST'] is True and
            gpu_report['AP_evaluated'] is False, 'Actual diagnostic GPU owner94/report scope differs')
    require(original['seed'] == carrier_cpu['seed'] == carrier_q['seed'] == 202610120701 and
            original['head_endpoint_epoch'] == carrier_cpu['head_endpoint_epoch'] == carrier_q['head_endpoint_epoch'] == 48 and
            carrier_q['status'] == 'ROOT_N13_SAMEK_FIXED_FULLDEV_CPU_QUALIFIED_V1' and
            carrier_q['evaluation_spec'] == spec['carrier_GPU_spec'] and carrier_q['spec'] == spec['carrier_CPU_spec'] and
            carrier_q['source'] == spec['carrier_evaluation_source'] and carrier_q['reader_source'] == spec['carrier_CPU_source'] and
            carrier_q['evaluation_GPU_RAW'] == spec['carrier_GPU_RAW'] and carrier_q['CPU_admission'] == spec['carrier_CPU_admission'],
            'Frozen first701 LOG E48 carrier provenance differs')
    for key in ('carrier_GPU_spec', 'carrier_CPU_spec', 'carrier_CPU_qualification', 'carrier_GPU_RAW',
                'carrier_CPU_RAW', 'carrier_CPU_admission', 'carrier_evaluation_source', 'carrier_CPU_source',
                'diagnostic_sources', 'diagnostic_protocol', 'fixed_population'):
        require(gpu_spec[key] == spec[key], 'Preserved protocol/carrier reference differs: ' + key)
    require(all(ref in closure.values() for payload in (original, carrier_cpu, gpu_spec) for ref in payload['sources']),
            'Complete frozen ancestor SOURCE closure required')
    context = dict(original)
    for key in ('sources', 'RAW_inputs', 'normal_RAW_count', 'prior_failed_CPU_receipt', 'additional_failed_RAW_inputs',
                'terminal_identity_pins', 'phase', 'entry_source', 'GPU_admission_file', 'output_directory', 'deadline_seconds'):
        context[key] = spec[key]
    context['additional_preserved_independent_CPU_failure'] = spec['additional_preserved_independent_CPU_failure']
    context['total_failed_RAW_count'] = 9
    context['total_terminal_identity_pin_count'] = 103
    lineage = load(original['endpoint_lineage_source'], '_N13_compatibility_probe_lineage')
    api(lineage, 'check_recipe')(original)
    gate = api(lineage, 'own_gate')(spec_path, context, __file__, closure, normals)
    join = api(lineage, 'owner_ref_join')
    gi = join(context, gate['owners'], spec['carrier_GPU_RAW'], carrier_q['evaluation_GPU_admission'],
              carrier_q['source'], carrier_q['evaluation_spec'], '0')
    ci = join(context, gate['owners'], spec['carrier_CPU_RAW'], carrier_q['CPU_admission'],
              carrier_q['reader_source'], carrier_q['spec'], '')
    di = join(context, gate['owners'], spec['diagnostic_GPU_RAW'], spec['diagnostic_GPU_admission'],
              spec['diagnostic_GPU_source'], spec['diagnostic_GPU_spec'], '0')
    require(gi < ci < di and di == 93 and original['RAW_inputs'] == spec['RAW_inputs'][:gi] and
            carrier_cpu['RAW_inputs'] == spec['RAW_inputs'][:ci], 'Original chronological actual-owner joins differ')
    gate.update(lineage=lineage, context=context, closure=closure, original=original, gpu_report=gpu_report,
                gpu_spec=gpu_spec, preserved_failure=preserved_failure)
    gate.update(api(lineage, 'endpoint_gate')(context, closure, gate['owners']))
    prior, prior_q = api(lineage, 'prior_gate')(context, closure, gate['owners'])
    require(prior['teacher_checkpoint'] == gate['cache_report']['teacher_checkpoint'], 'Frozen shared teacher differs')
    gate.update(prior=prior, prior_q=prior_q, packets=prior_q['packet_refs'],
                selector=api(lineage, 'selector_parent_gate')(context, closure, gate['owners']))
    image_checks = read(closed(closure, gpu_report['image_checks']))
    require(type(image_checks) is list and len(image_checks) == 187, 'Complete original GPU image-check manifest required')
    target = image_checks[25]
    require(target['dataset_index'] == 25 and target['image_id'] == 113 and target['native_all100_exact'] is True and
            target['selector_exact'] is True and [row['identity'] for row in target['pairs']] == TARGET_IDENTITIES and
            spec['target_candidates'] == [row['candidate'] for row in target['pairs']],
            'Probe must include both actual pairs of the entire failed image, in original order')
    require(all(ref in gpu_report['candidate_packets'] for ref in spec['target_candidates']), 'Target candidates absent from actual GPU report')
    for ref in spec['target_candidates']:
        closed(closure, ref)
    rows = read(closed(closure, prior['image_rows']))
    selector_rows = gate['selector']['manifest']['rows']
    require(len(rows) == len(selector_rows) == len(gate['packets']) == 187 and rows[25]['image_id'] == 113 and
            selector_rows[25]['dataset_index'] == 25 and selector_rows[25]['image_id'] == 113,
            'Original image/native packet/frozen selector index25 differs')
    field_contract = spec['packet_field_contract']
    require(type(field_contract) is dict and set(field_contract) == {'schema', 'feature_path', 'identity_path', 'feature_shape'} and
            field_contract['schema'] == PACKET_SCHEMA and
            field_contract['feature_path'] == ['native_fields', 'F256'] and field_contract['identity_path'] == ['identity'] and
            field_contract['feature_shape'] == [1, 256, 200, 200], 'ROOT-confirmed actual packet descriptor required')
    weight_contract = spec['weight_contract']
    require(type(weight_contract) is dict and set(weight_contract) == {'checkpoint', 'state_path', 'weight_keys'} and
            weight_contract['checkpoint'] == original['endpoint']['checkpoints']['LOG_PR'] and
            weight_contract['state_path'] == ['state_dict'] and weight_contract['weight_keys'] == list(WEIGHT_KEYS),
            'Exact frozen LOG E48 eight-weight checkpoint descriptor required')
    closed(closure, weight_contract['checkpoint'])
    comparison = spec['formula_comparison']
    require(type(comparison) is dict and set(comparison) == {'atol', 'rtol'} and
            type(comparison['atol']) in (int, float) and type(comparison['rtol']) in (int, float) and
            comparison['atol'] == comparison['rtol'] == 1e-4, 'Same original1e-4 atol/rtol; widening is forbidden')
    support = spec['legacy_support']
    require(type(support) is dict and set(support) == {'source', 'numeric_object_path', 'refined_support_atol', 'refined_support_rtol'} and
            support['source'] == spec['carrier_GPU_spec'], 'Actual original numeric_protocol support contract required')
    legacy = lookup(original, support['numeric_object_path'], 'legacy numeric_protocol')
    for key in ('refined_support_atol', 'refined_support_rtol'):
        require(type(support[key]) in (int, float) and math.isfinite(support[key]) and support[key] >= 0 and
                type(legacy) is dict and key in legacy and support[key] == legacy[key], 'Legacy practical support value differs: ' + key)
    gate.update(target=target, row=rows[25], packet_ref=gate['packets'][25], selector_row=selector_rows[25],
                n10=load(context['N10_evaluation_source'], '_N13_compatibility_probe_N10'),
                renderer=load(context['renderer_source'], '_N13_compatibility_probe_renderer'),
                evaluation=load(spec['carrier_evaluation_source'], '_N13_compatibility_probe_evaluation'))
    return gate


def deadline(started, spec):
    require(time.monotonic() - started < spec['deadline_seconds'], 'Prelocked probe deadline exceeded; no automatic retry')


def tensor(t, value, name, shape, dtype, device):
    require(isinstance(value, t.Tensor) and tuple(value.shape) == tuple(shape) and value.dtype == dtype and
            value.layout == t.strided and value.device.type == device, 'Explicit tensor descriptor differs: ' + name)
    require(bool(t.isfinite(value).all().item()), 'Nonfinite tensor: ' + name)
    return value


def independent_pr(t, weights, features, native):
    coarse = native.clamp(-8.0, 8.0).unsqueeze(1) / 8.0
    hidden = t.nn.functional.layer_norm(features, (256,), weight=None, bias=None, eps=1e-5)
    for offset in (0, 2, 4):
        hidden = t.nn.functional.relu(t.nn.functional.linear(t.cat((hidden, coarse), dim=1),
                                                            weights[offset], weights[offset + 1]))
    return t.nn.functional.linear(t.cat((hidden, coarse), dim=1), weights[6], weights[7]).squeeze(1)


def feature_difference(t, CPU_F, GPU_F_CPU, started, spec):
    left, right = CPU_F.reshape(-1), GPU_F_CPU.reshape(-1)
    maximum, at_index, sum_abs, sum_squared, count = 0.0, None, 0.0, 0.0, int(left.numel())
    for start in range(0, count, 1048576):
        deadline(started, spec)
        error = (left[start:start + 1048576] - right[start:start + 1048576]).abs()
        value, index = error.max(dim=0)
        if at_index is None or float(value.item()) > maximum:
            maximum, at_index = float(value.item()), start + int(index.item())
        sum_abs += float(error.double().sum().item())
        sum_squared += float(error.double().square().sum().item())
        del error, value, index
    return dict(shape=list(CPU_F.shape), scalar_count=count, max_abs_error=maximum, max_abs_flat_index=at_index,
                mean_abs_error=sum_abs / count, RMS_error=math.sqrt(sum_squared / count),
                equality_required=False, diagnostic_only=True)


def comparison_counts(t, actual, expected, atol, rtol):
    error = (actual - expected).abs()
    allowed = atol + rtol * expected.abs()
    return dict(max_abs_error=float(error.max().item()), above_same_atol_rtol_count=int((error > allowed).sum().item()),
                compared_points=int(error.numel()))


def validate_cases(t, np, gate, spec, carrier, identity):
    L = carrier['native_L']
    tensor(t, L, 'exact frozen GPU native L', tuple(L.shape), t.float32, 'cpu')
    require(L.ndim == 3 and L.shape[0] == 100, 'Exact frozen GPU L natural100 descriptor differs')
    require(L[0].numel() == 163345, 'ROOT-locked target original raster pixel count differs; no H/W guess is permitted')
    cases = []
    for record, ref in zip(gate['target']['pairs'], spec['target_candidates']):
        candidate = t.load(str(closed(gate['closure'], ref)), map_location='cpu')
        require(candidate['schema'] == 'N13-fixed-PR-real-candidate-v1' and candidate['identity'] == record['identity'] and
                candidate['source'] == spec['diagnostic_GPU_source'] and candidate['spec'] == spec['diagnostic_GPU_spec'] and
                candidate['GPU_admission'] == spec['diagnostic_GPU_admission'] and candidate['phase'] == gate['gpu_spec']['phase'] and
                candidate['native_packet'] == gate['packet_ref'] and candidate['selector'] == gate['selector_row']['carrier'] and
                candidate['head_checkpoint'] == spec['weight_contract']['checkpoint'], 'Actual stored GPU target packet provenance differs')
        z = L[record['identity']['query_id']]
        shape = tuple(z.shape)
        domains = candidate['domains']
        require(set(domains) == {'native_P', 'old_LOG', 'native_width2', 'new_halo', 'Omega'}, 'Stored target domain schema differs')
        for name, mask in domains.items():
            tensor(t, mask, 'stored domain.' + name, shape, t.bool, 'cpu')
        require(t.equal(domains['native_P'], z > 0) and t.equal(domains['old_LOG'], z.abs() <= 2) and
                t.equal(domains['new_halo'], domains['native_width2'] & ~domains['old_LOG']) and
                t.equal(domains['Omega'], domains['old_LOG'] | domains['new_halo']), 'Frozen GPU L and original target domains differ')
        actual_counts = {name: int(mask.sum().item()) for name, mask in domains.items()}
        actual_counts['outside'] = int(z.numel()) - actual_counts['Omega']
        require(actual_counts == record['domain_counts'] and
                (actual_counts['old_LOG'], actual_counts['new_halo']) == TARGET_PARTITION_COUNTS[record['identity']['query_id']],
                'Actual complete failed-image73/560 and13/274 partitions differ from locked GPU metadata')
        stored = candidate['candidates']
        require(stored['point_chunk_size'] == 2048 and stored['refined_binary_rule'] == 'strict_gt_zero',
                'Original separate ascending2048 strict>0 partition contract differs')
        partitions = {}
        for label, domain_name in (('originalLOG', 'old_LOG'), ('addedHalo', 'new_halo')):
            indices = t.from_numpy(np.flatnonzero(domains[domain_name].numpy()).astype(np.int64, copy=False))
            count = int(indices.numel())
            part = stored[label]
            require(type(part) is dict and set(part) == {'flat_indices', 'raw_residual', 'refined_logit', 'binary_candidate', 'chunk_count'},
                    'Actual stored GPU target partition schema differs')
            tensor(t, part['flat_indices'], label + '.indices', (count,), t.int64, 'cpu')
            tensor(t, part['raw_residual'], label + '.GPU_u', (count,), t.float32, 'cpu')
            tensor(t, part['refined_logit'], label + '.GPU_logit', (count,), t.float32, 'cpu')
            tensor(t, part['binary_candidate'], label + '.GPU_binary', (count,), t.bool, 'cpu')
            require(t.equal(indices, part['flat_indices']) and part['chunk_count'] == (count + 2047) // 2048 and
                    t.equal(part['refined_logit'], z.reshape(-1)[indices] + part['raw_residual']) and
                    t.equal(part['binary_candidate'], part['refined_logit'] > 0), 'Stored frozen GPU partition/logit self-consistency differs')
            partitions[label] = part
        cases.append(dict(identity=record['identity'], candidate_ref=ref, L=z, partitions=partitions,
                          logical_domain_counts=record['domain_counts']))
        # C and diagnostic/GT statistics are never consumed by this probe.
        del candidate, stored, domains
    require([case['identity'] for case in cases] == TARGET_IDENTITIES, 'Both complete target-image pairs required')
    return cases


def run_path(t, path_name, features, weights, cases, spec, started):
    device = features.device.type
    feature_points = features.reshape(256, -1).transpose(0, 1)
    result = []
    for case in cases:
        Lflat = case['L'].reshape(-1)
        record = dict(identity=case['identity'], candidate_ref=case['candidate_ref'], path=path_name,
                      feature_device=device, arithmetic_device=device, partitions={})
        for label in ('originalLOG', 'addedHalo'):
            part = case['partitions'][label]
            count = int(part['flat_indices'].numel())
            chunks = []
            for start in range(0, count, 2048):
                deadline(started, spec)
                end = min(count, start + 2048)
                indices_CPU = part['flat_indices'][start:end]
                indices = indices_CPU.to(features.device)
                native = Lflat[indices_CPU].to(features.device)
                residual = independent_pr(t, weights, feature_points[indices], native)
                refined = native + residual
                require(bool(t.isfinite(residual).all().item()) and bool(t.isfinite(refined).all().item()),
                        'Independent probe formula produced nonfinite values')
                residual_CPU, refined_CPU = residual.cpu(), refined.cpu()
                GPU_u, GPU_logit = part['raw_residual'][start:end], part['refined_logit'][start:end]
                residual_check = comparison_counts(t, residual_CPU, GPU_u, 1e-4, 1e-4)
                logit_check = comparison_counts(t, refined_CPU, GPU_logit, 1e-4, 1e-4)
                sign_difference = (refined_CPU > 0) != part['binary_candidate'][start:end]
                tau = spec['legacy_support']['refined_support_atol'] + spec['legacy_support']['refined_support_rtol'] * refined_CPU.abs()
                outside = sign_difference & (refined_CPU.abs() > tau)
                chunks.append(dict(chunk_index=start // 2048, start_ordinal=start, end_ordinal=end,
                    first_flat_index=int(indices_CPU[0].item()), last_flat_index=int(indices_CPU[-1].item()),
                    residual=residual_check, logit=logit_check,
                    logit_sign_differences=int(sign_difference.sum().item()),
                    sign_differences_outside_original_practical_support=int(outside.sum().item()),
                    max_abs_path_logit_on_sign_differences=float(refined_CPU[sign_difference].abs().max().item()) if bool(sign_difference.any().item()) else 0.0))
                del indices, indices_CPU, native, residual, refined, residual_CPU, refined_CPU, GPU_u, GPU_logit, sign_difference, tau, outside
            aggregate = dict(points=count, chunk_count=len(chunks), chunks=chunks,
                logit_sign_differences=sum(chunk['logit_sign_differences'] for chunk in chunks),
                sign_differences_outside_original_practical_support=sum(chunk['sign_differences_outside_original_practical_support'] for chunk in chunks))
            for kind in ('residual', 'logit'):
                aggregate[kind] = dict(max_abs_error=max((chunk[kind]['max_abs_error'] for chunk in chunks), default=0.0),
                    above_same_atol_rtol_count=sum(chunk[kind]['above_same_atol_rtol_count'] for chunk in chunks))
            record['partitions'][label] = aggregate
        result.append(record)
    return result


def write_new(path, value):
    with Path(path).open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def run(spec_path):
    started = time.monotonic()
    path = Path(spec_path).resolve(strict=True)
    spec = read(path)
    gate = source_gate(path, spec)
    output = Path(spec['output_directory'])
    require(output.is_absolute() and not output.exists() and output.parent.is_dir(), 'Fresh ROOT-owned absolute probe output directory required')
    require(type(spec['minimum_free_output_bytes']) is int and spec['minimum_free_output_bytes'] >= 0,
            'Prelocked output-space requirement needed')
    require(all(os.environ.get(key) == '2' for key in THREAD_ENV), 'Pre-import numeric thread environment must be2')
    import psutil
    require(psutil.Process().cpu_affinity() == [0, 1] and psutil.virtual_memory().available >= 16 * 1024**3 and
            psutil.disk_usage(str(output.parent)).free >= spec['minimum_free_output_bytes'], 'Actual CPU01/RAM16GiB/output space required')
    import torch
    require(torch.__version__ == '2.0.1+cu118' and not torch.cuda.is_initialized(), 'Qualified cold Torch runtime required')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    import numpy as np
    import cv2
    import scipy
    from scipy import ndimage
    from pycocotools.coco import COCO
    from pycocotools import mask as masks
    require(np.__version__ == '1.23.5' and cv2.__version__ == '4.8.0' and scipy.__version__ == '1.10.1', 'Qualified numerical versions required')
    # Import-only library qualification; no COCO instance or GT/data evaluation.
    libraries = api(gate['common'], 'imported_files')(torch, np, cv2, masks, COCO, gate['closure'], gate['admission'], ndimage)
    packet = torch.load(str(closed(gate['closure'], gate['packet_ref'])), map_location='cpu')
    descriptor = spec['packet_field_contract']
    require(type(packet) is dict and packet['schema'] == descriptor['schema'] == PACKET_SCHEMA and
            getattr(gate['n10'], 'PACKET_SCHEMA', None) == PACKET_SCHEMA,
            'ROOT-confirmed packet schema must match the actual bound original N10 constant')
    F_source = lookup(packet, descriptor['feature_path'], 'native packet F256')
    identity = lookup(packet, descriptor['identity_path'], 'native packet identity')
    tensor(torch, F_source, 'native packet F256', descriptor['feature_shape'], torch.float32, 'cpu')
    require(type(identity) is dict and identity, 'Original renderer identity descriptor required')
    selector_row = gate['selector_row']
    carrier = torch.load(str(closed(gate['closure'], selector_row['carrier'])), map_location='cpu')
    expected_selector = dict(dataset_index=25, image_id=113, identity=identity, native_packet=gate['packet_ref'],
        natural_query_ids=list(range(100)), renderer_source=gate['context']['renderer_source'])
    require(carrier['schema'] == gate['lineage'].SELECTOR_PACKET_SCHEMA and
            all(carrier[key] == selector_row[key] == value for key, value in expected_selector.items()) and
            carrier['source'] == gate['selector']['report']['source'] and carrier['spec'] == gate['selector']['report']['spec'] and
            carrier['GPU_admission'] == gate['selector']['report']['GPU_admission'] and
            carrier['native_export_query_ids'] == selector_row['native_export_query_ids'] and
            set(carrier['native_export_query_ids']) == set(range(100)) and len(carrier['native_export_query_ids']) == 100 and
            api(gate['evaluation'], 'digest')(carrier['native_L']) == carrier['L_digest'] == selector_row['L_digest'],
            'Exact original packet/renderer/frozen GPU native L selector provenance differs')
    checkpoint = torch.load(str(closed(gate['closure'], spec['weight_contract']['checkpoint'])), map_location='cpu')
    state = lookup(checkpoint, spec['weight_contract']['state_path'], 'frozen LOG head state')
    require(type(state) is dict and set(state) == set(WEIGHT_KEYS), 'Actual exact eight-weight frozen state_dict required')
    weights_CPU = tuple(state[key] for key in WEIGHT_KEYS)
    for key, shape, value in zip(WEIGHT_KEYS, WEIGHT_SHAPES, weights_CPU):
        tensor(torch, value, key, shape, torch.float32, 'cpu')
    require(sum(value.numel() for value in weights_CPU) == 40497, 'Frozen LOG head must have40497 scalars')
    frozen_weights = tuple(value.clone() for value in weights_CPU)
    cases = validate_cases(torch, np, gate, spec, carrier, identity)
    require(not torch.cuda.is_initialized(), 'All provenance/RAW/source gates must precede CUDA initialization')
    deadline(started, spec)
    torch.cuda.init()
    require(torch.cuda.current_device() == 0, 'Actual single-device probe must use GPU0')
    torch.cuda.set_per_process_memory_fraction(min(1.0, 4096 * 1024**2 / torch.cuda.get_device_properties(0).total_memory), 0)
    torch.cuda.reset_peak_memory_stats(0)
    CPU_rng, GPU_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state(0).clone()
    output.mkdir()
    write_new(output / 'START.json', dict(schema='N13-CPU-GPU-image-probe-start-v1', source=gate['source'],
        spec=gate['spec_ref'], GPU_admission=gate['admission_ref'], phase=spec['phase'], target=TARGET_IDENTITIES,
        additional_preserved_independent_CPU_failure=spec['additional_preserved_independent_CPU_failure'],
        total_failed_RAW_count=9, total_terminal_identity_pin_count=103, no_GT_numeric_input=True,
        no_TEST=True, AP_evaluated=False, qualification_generated=False))
    cost, results = {}, {}
    with torch.no_grad():
        tick = time.monotonic()
        CPU_F = api(gate['n10'], 'original_feature')(torch, gate['renderer'], F_source, identity)
        shape = (256, int(carrier['native_L'].shape[1]), int(carrier['native_L'].shape[2]))
        tensor(torch, CPU_F, 'CPU-rendered F_original', shape, torch.float32, 'cpu')
        cost['CPU_original_feature_seconds'] = time.monotonic() - tick
        deadline(started, spec)
        tick = time.monotonic()
        source_GPU = F_source.to('cuda:0')
        GPU_F = api(gate['n10'], 'original_feature')(torch, gate['renderer'], source_GPU, identity)
        tensor(torch, GPU_F, 'GPU-rendered F_original', shape, torch.float32, 'cuda')
        torch.cuda.synchronize(0)
        cost['GPU_original_feature_seconds'] = time.monotonic() - tick
        del source_GPU
        tick = time.monotonic()
        GPU_F_CPU = GPU_F.cpu()
        feature_errors = feature_difference(torch, CPU_F, GPU_F_CPU, started, spec)
        cost['GPU_feature_to_CPU_and_compare_seconds'] = time.monotonic() - tick
        for name, features in ((PATH_NAMES[0], CPU_F), (PATH_NAMES[1], GPU_F_CPU)):
            tick = time.monotonic()
            results[name] = run_path(torch, name, features, weights_CPU, cases, spec, started)
            cost[name + '_seconds'] = time.monotonic() - tick
        tick = time.monotonic()
        weights_GPU = tuple(value.to('cuda:0') for value in weights_CPU)
        results[PATH_NAMES[2]] = run_path(torch, PATH_NAMES[2], GPU_F, weights_GPU, cases, spec, started)
        torch.cuda.synchronize(0)
        cost[PATH_NAMES[2] + '_seconds'] = time.monotonic() - tick
        # Keep only ONE full original-raster feature on GPU at a time, under the
        # original4GiB allocator cap. This does not alter either PR partition.
        del GPU_F
        gc.collect()
        torch.cuda.empty_cache()
        deadline(started, spec)
        tick = time.monotonic()
        CPU_F_GPU = CPU_F.to('cuda:0')
        torch.cuda.synchronize(0)
        cost['CPU_feature_to_GPU_seconds'] = time.monotonic() - tick
        tick = time.monotonic()
        results[PATH_NAMES[3]] = run_path(torch, PATH_NAMES[3], CPU_F_GPU, weights_GPU, cases, spec, started)
        torch.cuda.synchronize(0)
        cost[PATH_NAMES[3] + '_seconds'] = time.monotonic() - tick
        del CPU_F_GPU, weights_GPU
    require(all(torch.equal(before, after) for before, after in zip(frozen_weights, weights_CPU)) and
            torch.equal(CPU_rng, torch.get_rng_state()) and torch.equal(GPU_rng, torch.cuda.get_rng_state(0)),
            'Probe changed frozen weights or CPU/GPU RNG state')
    # The last CPU-path loop binding `features` holds CPU storage only.
    peak = torch.cuda.max_memory_allocated(0)
    gc.collect()
    torch.cuda.synchronize(0)
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(0)
    cleanup = dict(allocated_after=torch.cuda.memory_allocated(0), reserved_after=torch.cuda.memory_reserved(0))
    require(peak <= 4096 * 1024**2 and cleanup['allocated_after'] == 0, 'Actual4GiB allocator cap/cleanup required')
    deadline(started, spec)
    details = dict(schema='N13-CPU-GPU-image-probe-details-v1', source=gate['source'], spec=gate['spec_ref'],
        target_dataset_index=25, target_image_id=113, target_candidates=spec['target_candidates'],
        feature_CPU_GPU_comparison=feature_errors, paths=results, formula_comparison=spec['formula_comparison'],
        original_practical_support=spec['legacy_support'],
        support_rule='tau=legacy_atol+legacy_rtol*abs(this_path_logit); count signs with abs(path_logit)>tau',
        support_rule_for_CPU_paths_is_original_CPU_support_check=True, certified_error_bound=False,
        actual_frozen_GPU_C_modified_or_replaced=False, no_GT_numeric_input=True,
        input_projection_and_arithmetic_interpretation='Four paths provide diagnostic evidence; ROOT analysis is pending.')
    write_new(output / 'DETAILS.json', details)
    report = dict(schema=REPORT_SCHEMA, status=STATUS, source=gate['source'], spec=gate['spec_ref'],
        GPU_admission=gate['admission_ref'], phase=spec['phase'], details=reference(output / 'DETAILS.json'),
        start=reference(output / 'START.json'), RAW_inputs=spec['RAW_inputs'], normal_RAW_count=94,
        prior_failed_CPU_receipt=spec['prior_failed_CPU_receipt'], additional_failed_RAW_inputs=spec['additional_failed_RAW_inputs'],
        terminal_identity_pins=spec['terminal_identity_pins'],
        additional_preserved_independent_CPU_failure=spec['additional_preserved_independent_CPU_failure'],
        total_failed_RAW_count=9, total_terminal_identity_pin_count=103,
        diagnostic_GPU_report=spec['diagnostic_GPU_report'], diagnostic_GPU_source=spec['diagnostic_GPU_source'],
        diagnostic_GPU_spec=spec['diagnostic_GPU_spec'], diagnostic_GPU_RAW=spec['diagnostic_GPU_RAW'],
        diagnostic_GPU_admission=spec['diagnostic_GPU_admission'], carrier_GPU_spec=spec['carrier_GPU_spec'],
        packet=gate['packet_ref'], selector=gate['selector_row']['carrier'], checkpoint=spec['weight_contract']['checkpoint'],
        target_dataset_index=25, target_image_id=113, target_identities=TARGET_IDENTITIES,
        target_candidates=spec['target_candidates'], complete_target_image_checked=True, images=1, pairs=2,
        paths=list(PATH_NAMES), separate_ascending2048_partitions_preserved=True, exact_frozen_GPU_L_used=True,
        feature_CPU_GPU_comparison=feature_errors, formula_comparison=spec['formula_comparison'],
        original_practical_support=spec['legacy_support'], certified_error_bound=False,
        actual_frozen_GPU_C_modified_or_replaced=False, numerical_incompatibility_used_to_relax_tolerances=False,
        qualification_generated=False, no_GT_numeric_input=True, no_TEST=True, AP_evaluated=False,
        training_started=False, root_analysis_pending=True, deployment_gain_established=False, innovation_established=False,
        cost_seconds=cost, total_elapsed_seconds=time.monotonic() - started, peak_allocator_bytes=peak,
        resource_contract=dict(GPU_device=0, CPU_affinity=psutil.Process().cpu_affinity(),
            intraop_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
            deadline_seconds=1800, allocator_limit_bytes=4096 * 1024**2,
            whole_GPU_observation_owned_by_ROOT=True, per_leaf_VRAM_certified=False),
        cleanup=cleanup, actual_imported_files=libraries, current_owned_GPU_terminal_self_certified=False,
        whole_leaf_VRAM_certified=False,
        evidence_limits='Complete failed image index25/image113 only; four CPU/GPU feature/arithmetic paths against the original '
                        'stored GPU residual/logit. Numerical diagnostic only, under identical1e-4 comparison tolerances. '
                        'Original practical near-zero allowance is not a certified error bound; no Q, GT evaluation, AP, '
                        'causal attribution, general method failure, efficacy or innovation conclusion.')
    deadline(started, spec)
    write_new(output / 'REPORT.json', report)
    print(json.dumps(dict(status=STATUS, report=reference(output / 'REPORT.json'), qualification_generated=False)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--owner-execute', action='store_true')
    options = parser.parse_args()
    require(options.owner_execute, 'UNBOUND_DRAFT: ROOT must bind/review the new immutable probe source and own the single attempt')
    run(options.spec)
