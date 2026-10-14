"""UNBOUND_DRAFT: capture actual GPU raw-PR inputs; no execution or Q.

ROOT must bind this helper and a NEW complete GPU runner, review actual external
ABI, own the attempt and preserve old immutable sources/results. Import uses only
the standard library. Torch, original pinned adapter and provenance are explicit
caller arguments. No scientific SOURCE, data, weights or GT are loaded here.

Minimal runner change: construct CapturingAdapter for EACH pair with current
provenance; pass it instead of the original adapter to the unchanged pipeline.
The proxy captures each real raw_pr_residual call, then calls the original raw
function unchanged and restores its temporary Python binding in finally. It adds
only candidates['exact_GPU_point_inputs']; original residual/logit/binary/C are
not replaced. The new runner must include this instrumentation in its bound ABI,
and serialize the augmented CPU-only candidate tree exactly as before.

The ancestry helper preserves spec legacy normal94/failed8/pins102 unchanged.
Failed9 is checked RAW-only first; probe successful95 and optional capture96 are
also checked RAW-only before legacy_pre_gate(spec,94)'s first SOURCE hash. ROOT
then uses the returned NEW context/normal rows for its actual own_gate. The ninth
failure remains separate; extra GPU joins MUST use '0'; CPU joins use ''.
This helper does not implement or certify own_gate or current owner termination.
"""

import hashlib
import json
import math
from pathlib import Path
import sys


SCHEMA = 'N13-fixed-PR-exact-GPU-point-inputs-v1'
CAPTURE_SPEC_SCHEMA = 'ROOT-N13-fixed-PR-exact-GPU-input-capture-GPU-spec-v1'
CPU_SPEC_SCHEMA = 'ROOT-N13-fixed-PR-exact-GPU-input-CPU-spec-v1'
PARTITIONS = ('originalLOG', 'addedHalo')
POINT_CHUNK_SIZE = 2048
RAW_KEYS = ('receipt', 'phase', 'admission', 'phase_binding', 'bootstrap')
PROVENANCE_KEYS = {'source', 'capture_source', 'spec', 'GPU_admission', 'phase', 'identity',
                   'native_packet', 'selector', 'renderer_source', 'N10_source', 'head_checkpoint'}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def ref_shape(ref):
    require(type(ref) is dict and set(ref) == {'path', 'bytes', 'sha256'} and
            type(ref['path']) is str and Path(ref['path']).is_absolute() and type(ref['bytes']) is int and
            ref['bytes'] >= 0 and type(ref['sha256']) is str and len(ref['sha256']) == 64 and
            set(ref['sha256']) <= set('0123456789abcdef'), 'Exact absolute pinned SOURCE reference required')


def json_copy(value):
    def finite(item):
        if type(item) is dict:
            require(all(type(key) is str for key in item), 'JSON keys must be strings')
            for item_value in item.values():
                finite(item_value)
        elif type(item) in (tuple, list):
            for item_value in item:
                finite(item_value)
        else:
            require(item is None or type(item) in (str, bool, int) or type(item) is float and math.isfinite(item),
                    'Finite JSON provenance required')
    finite(value)
    return json.loads(json.dumps(value, allow_nan=False))


def provenance(value):
    require(type(value) is dict and set(value) == PROVENANCE_KEYS, 'Exact current capture provenance fields required')
    for key in PROVENANCE_KEYS - {'phase', 'identity'}:
        ref_shape(value[key])
    require(type(value['phase']) is str and value['phase'], 'Current GPU phase required')
    identity = value['identity']
    require(type(identity) is dict and set(identity) == {'image_id', 'gt_id', 'query_id'} and
            all(type(item) is int and item >= 0 for item in identity.values()), 'Actual fixed pair identity required')
    return json_copy(value)


def tensor_check(t, value, name, shape, dtype, device):
    require(isinstance(value, t.Tensor) and tuple(value.shape) == tuple(shape) and value.dtype == dtype and
            value.layout == t.strided and value.device.type == device, 'Tensor descriptor differs: ' + name)
    if device == 'cuda':
        require(value.device.index == 0, 'Actual capture must use GPU0')
    require(bool(t.isfinite(value).all().item()), 'Nonfinite actual capture tensor: ' + name)


def tensor_digest(t, value):
    require(isinstance(value, t.Tensor) and value.device.type == 'cpu' and value.layout == t.strided,
            'Digest only the new captured CPU tensor after the ROOT execution gate')
    descriptor = dict(shape=list(value.shape), dtype=str(value.dtype), byte_order=sys.byteorder)
    digest = hashlib.sha256()
    digest.update(json.dumps(descriptor, sort_keys=True, separators=(',', ':')).encode('ascii'))
    digest.update(b'\n')
    digest.update(value.detach().contiguous().view(t.uint8).numpy().tobytes(order='C'))
    return dict(descriptor, sha256=digest.hexdigest())


class CapturingAdapter:
    """One-pair proxy; no GT/selection/K is exposed to actual raw PR calls.

    Temporary instrumentation is confined to the supplied in-memory adapter and
    one synchronous call. No old SOURCE file or checkpoint is written. Concurrent
    calls on this module are forbidden by the new runner's reviewed call chain.
    """

    def __init__(self, torch_module, original_adapter, capture_provenance, check_deadline):
        self.t = torch_module
        self.adapter = original_adapter
        self.provenance = provenance(capture_provenance)
        self.check_deadline = check_deadline
        require(callable(check_deadline), 'ROOT must supply the existing1800-second deadline check')
        require(callable(getattr(original_adapter, 'raw_pr_residual', None)) and
                callable(getattr(original_adapter, 'fixed_pr_candidates', None)), 'Actual original adapter ABI required')
        self.used = False

    def fixed_pr_candidates(self, torch_module, weight_tensors, F_original, native_L,
                            originalLOG_domain, addedHalo_domain, *, refined_binary_rule, return_merged=False):
        t = self.t
        require(torch_module is t and not self.used, 'One new capture proxy must be used exactly once per pair')
        self.used = True
        self.check_deadline()
        require(refined_binary_rule == 'strict_gt_zero' and return_merged is True, 'Original strict>0 merged candidate contract required')
        require(isinstance(native_L, t.Tensor) and native_L.ndim == 2, 'Exact frozen GPU native L raster required')
        shape = tuple(native_L.shape)
        tensor_check(t, native_L, 'exact GPU L', shape, t.float32, 'cuda')
        tensor_check(t, F_original, 'actual GPU F_original', (256,) + shape, t.float32, 'cuda')
        for name, domain in (('originalLOG', originalLOG_domain), ('addedHalo', addedHalo_domain)):
            tensor_check(t, domain, name, shape, t.bool, 'cuda')
        require(t.equal(originalLOG_domain, native_L.abs() <= 2) and
                not bool((originalLOG_domain & addedHalo_domain).any().item()), 'Exact original disjoint domains required')
        require(isinstance(weight_tensors, (tuple, list)) and len(weight_tensors) == 8, 'Exact ordered eight frozen weights required')
        F_raster_points = F_original.reshape(256, -1).transpose(0, 1)
        L_flat = native_L.reshape(-1)
        indices = {label: t.sort(t.nonzero(domain.reshape(-1), as_tuple=False).reshape(-1)).values
                   for label, domain in zip(PARTITIONS, (originalLOG_domain, addedHalo_domain))}
        sequence = []
        captured = {label: dict(F=[], L=[], u=[], logit=[], chunks=[]) for label in PARTITIONS}
        for label in PARTITIONS:
            count = int(indices[label].numel())
            for start in range(0, count, POINT_CHUNK_SIZE):
                sequence.append((label, start, min(count, start + POINT_CHUNK_SIZE)))
        original_raw = self.adapter.raw_pr_residual
        call_count = 0

        def observe(actual_t, actual_weights, actual_F_points, actual_L_points):
            nonlocal call_count
            self.check_deadline()
            require(actual_t is t and call_count < len(sequence), 'Unexpected actual GPU raw PR call')
            require(len(actual_weights) == 8 and all(left is right for left, right in zip(actual_weights, weight_tensors)),
                    'Actual raw PR received different weight objects/order')
            label, start, end = sequence[call_count]
            selected = indices[label][start:end]
            tensor_check(t, actual_F_points, 'actual received GPU F_points', (end - start, 256), t.float32, 'cuda')
            tensor_check(t, actual_L_points, 'actual received GPU L_points', (end - start,), t.float32, 'cuda')
            require(t.equal(actual_F_points, F_raster_points[selected]) and t.equal(actual_L_points, L_flat[selected]),
                    'Actual raw PR input differs from the frozen original ascending sampling')
            # Observe before calling the unchanged original arithmetic; no casts.
            F_CPU = actual_F_points.detach().cpu().contiguous().clone()
            L_CPU = actual_L_points.detach().cpu().contiguous().clone()
            residual = original_raw(actual_t, actual_weights, actual_F_points, actual_L_points)
            tensor_check(t, residual, 'original GPU raw residual', (end - start,), t.float32, 'cuda')
            require(t.equal(actual_F_points.detach().cpu(), F_CPU) and t.equal(actual_L_points.detach().cpu(), L_CPU),
                    'Original PR mutated a captured input')
            bucket = captured[label]
            bucket['F'].append(F_CPU)
            bucket['L'].append(L_CPU)
            bucket['u'].append(residual.detach().cpu().contiguous().clone())
            bucket['logit'].append((actual_L_points + residual).detach().cpu().contiguous().clone())
            bucket['chunks'].append(dict(actual_call_ordinal=call_count, chunk_index=start // 2048,
                                         start_ordinal=start, end_ordinal=end))
            call_count += 1
            return residual

        self.adapter.raw_pr_residual = observe
        try:
            result = self.adapter.fixed_pr_candidates(t, weight_tensors, F_original, native_L,
                originalLOG_domain, addedHalo_domain, refined_binary_rule=refined_binary_rule, return_merged=return_merged)
        finally:
            self.adapter.raw_pr_residual = original_raw
        require(call_count == len(sequence) and self.adapter.raw_pr_residual is original_raw,
                'All expected actual PR calls and original adapter binding must be preserved')
        require('exact_GPU_point_inputs' not in result, 'Capture field already exists; never overwrite an existing capture')
        input_packet = dict(schema=SCHEMA, provenance=self.provenance, point_chunk_size=2048,
            refined_binary_rule='strict_gt_zero', capture_contract=dict(actual_GPU_raw_PR_calls_observed=True,
                actual_sampled_inputs_match_original_GPU_F_and_L_exact=True, CPU_copy_has_no_dtype_conversion=True,
                original_PR_arithmetic_called_unchanged=True, original_adapter_binding_restored=True,
                GT_or_selection_or_K_passed_to_PR=False, qualification_self_certified=False))
        for label in PARTITIONS:
            part, bucket = result[label], captured[label]
            count = int(indices[label].numel())
            indices_CPU = indices[label].detach().cpu().contiguous().clone()
            F_CPU = t.cat(bucket['F']) if count else t.empty((0, 256), dtype=t.float32, device='cpu')
            L_CPU = t.cat(bucket['L']) if count else t.empty((0,), dtype=t.float32, device='cpu')
            u_CPU = t.cat(bucket['u']) if count else t.empty((0,), dtype=t.float32, device='cpu')
            logit_CPU = t.cat(bucket['logit']) if count else t.empty((0,), dtype=t.float32, device='cpu')
            require(t.equal(part['flat_indices'].detach().cpu(), indices_CPU) and
                    t.equal(part['raw_residual'].detach().cpu(), u_CPU) and
                    t.equal(part['refined_logit'].detach().cpu(), logit_CPU) and
                    t.equal(part['binary_candidate'], part['refined_logit'] > 0) and
                    part['chunk_count'] == len(bucket['chunks']), 'Saved original GPU partition differs from observed actual calls')
            input_packet[label] = dict(flat_indices=indices_CPU, F_points=F_CPU, native_logit_points=L_CPU,
                chunk_count=len(bucket['chunks']), chunks=bucket['chunks'],
                tensor_digests={name: tensor_digest(t, value) for name, value in
                    (('flat_indices', indices_CPU), ('F_points', F_CPU), ('native_logit_points', L_CPU))})
        result['exact_GPU_point_inputs'] = input_packet
        return result


def check_pin(pin, raw, phase, code):
    require(type(pin) is dict and set(pin) == {'phase', 'returncode', 'original_identity', 'admission_sha256'} and
            pin['phase'] == phase and type(pin['returncode']) is int and pin['returncode'] == code and
            pin['original_identity'] == raw['original_identity'] and pin['admission_sha256'] == raw['admission_sha256'],
            'Additional actual RAW identity/admission pin differs')


def added_normal_shape(node, with_ROOT_qualification):
    required = {'RAW_input', 'pin', 'source', 'spec', 'GPU_admission', 'report'}
    if with_ROOT_qualification:
        required.add('ROOT_qualification')
    require(type(node) is dict and set(node) == required, 'Explicit added successful GPU owner descriptor required')
    row = node['RAW_input']
    require(type(row) is dict and set(row) == set(RAW_KEYS) and type(row['phase']) is str and row['phase'] and
            row['admission'] == node['GPU_admission'], 'Original five-key successful GPU RAW row required')
    for key in required - {'RAW_input', 'pin'}:
        ref_shape(node[key])
    for key in RAW_KEYS:
        if key != 'phase':
            ref_shape(row[key])


def build_current_ancestry(spec, carrier_context, *, terminal_RAW_only, legacy_pre_gate,
                           bound_reference, read_JSON, cpu=False):
    """ROOT invokes this before own_gate; never mutates legacy spec/history.

    Callbacks must be actual pinned ABI reviewed by ROOT. terminal_RAW_only must
    perform original held exit/reap/birth-absence/resource checks without SHA.
    legacy_pre_gate is the preserved original GPU pre_gate(spec,94).
    predecessor_report_contracts supplies actual ROOT-confirmed schema/status for
    probe, probe_ROOT_qualification, and (CPU only) capture. No guessed schema.
    """
    require(spec['schema'] == (CPU_SPEC_SCHEMA if cpu else CAPTURE_SPEC_SCHEMA), 'Exact new ROOT spec schema required')
    require(spec['normal_RAW_count'] == 94 and len(spec['RAW_inputs']) == 94 and
            len(spec['additional_failed_RAW_inputs']) == 7 and len(spec['terminal_identity_pins']) == 102,
            'Preserve original normal94/failed8/pins102 as a distinct legacy record')
    failure = spec['additional_preserved_independent_CPU_failure']
    require(type(failure) is dict and set(failure) == {'RAW_ref', 'phase', 'pin', 'returncode'} and
            type(failure['returncode']) is int and failure['returncode'] == 1, 'Preserved independent failed9 descriptor required')
    failure_raw = terminal_RAW_only(failure['RAW_ref'], failure['phase'], 1)
    check_pin(failure['pin'], failure_raw, failure['phase'], 1)
    nodes = [('probe', spec['successful_compatibility_probe'])]
    if cpu:
        nodes.append(('capture', spec['exact_input_capture_GPU']))
    new_raws = []
    for label, node in nodes:
        added_normal_shape(node, with_ROOT_qualification=(label == 'probe'))
        row = node['RAW_input']
        raw = terminal_RAW_only(row['receipt'], row['phase'], 0)
        check_pin(node['pin'], raw, row['phase'], 0)
        require(raw['admission_sha256'] == node['GPU_admission']['sha256'], 'Actual added GPU admission differs')
        new_raws.append(raw)
    all_extra_paths = [failure['RAW_ref']['path']] + [node['RAW_input']['receipt']['path'] for label, node in nodes]
    require(len(set(all_extra_paths)) == len(all_extra_paths) and
            not set(all_extra_paths).intersection(spec['terminal_identity_pins']), 'Additional actual owners must be distinct from legacy102')
    normals, closure = legacy_pre_gate(spec, 94)  # FIRST SHA after ALL above RAW-only checks.
    require(len(normals) == 94, 'Actual preserved pre_gate normal94 result differs')
    for ref in [failure['RAW_ref']] + [ref for label, node in nodes for key, ref in node.items()
                                      if key not in ('RAW_input', 'pin')] + [node['RAW_input'][key]
                                      for label, node in nodes for key in RAW_KEYS if key != 'phase']:
        ref_shape(ref)
        require(closure.get(ref['path']) == ref, 'Additional immutable owner/reference missing from complete SOURCE closure')
        bound_reference(ref)
    contracts = spec['predecessor_report_contracts']
    for label, node in nodes:
        report = read_JSON(bound_reference(node['report']))
        contract = contracts[label]
        require(type(contract) is dict and set(contract) == {'schema', 'status'} and
                report['schema'] == contract['schema'] and report['status'] == contract['status'] and
                report['source'] == node['source'] and report['spec'] == node['spec'] and
                report['GPU_admission'] == node['GPU_admission'] and report['phase'] == node['RAW_input']['phase'],
                'Actual additional GPU report/source/spec/admission differs')
    probe_Q = read_JSON(bound_reference(spec['successful_compatibility_probe']['ROOT_qualification']))
    Q_contract = contracts['probe_ROOT_qualification']
    require(type(Q_contract) is dict and set(Q_contract) == {'schema', 'status'} and
            probe_Q['schema'] == Q_contract['schema'] and probe_Q['status'] == Q_contract['status'],
            'ROOT-confirmed actual successful probe qualification schema/status required')
    context = dict(carrier_context)
    for key in ('sources', 'prior_failed_CPU_receipt', 'additional_failed_RAW_inputs', 'phase', 'entry_source', 'deadline_seconds'):
        context[key] = spec[key]
    for key in ('GPU_admission_file', 'CPU_admission_file', 'output_directory', 'Q_path', 'details_path'):
        if key in spec:
            context[key] = spec[key]
    context['RAW_inputs'] = list(spec['RAW_inputs']) + [dict(node['RAW_input']) for label, node in nodes]
    context['normal_RAW_count'] = len(context['RAW_inputs'])
    context['terminal_identity_pins'] = dict(spec['terminal_identity_pins'])
    for label, node in nodes:
        context['terminal_identity_pins'][node['RAW_input']['receipt']['path']] = json_copy(node['pin'])
    context['additional_preserved_independent_CPU_failure'] = json_copy(failure)
    all_pins = dict(context['terminal_identity_pins'])
    all_pins[failure['RAW_ref']['path']] = json_copy(failure['pin'])
    context['all_ancestor_identity_pins'] = all_pins
    context['total_failed_RAW_count'] = 9
    context['total_terminal_identity_pin_count'] = len(all_pins)
    require(context['normal_RAW_count'] == (96 if cpu else 95) and len(all_pins) == (105 if cpu else 104),
            'Exact derived normals95/96 and all-pins104/105 required')
    require(spec['normal_RAW_count'] == 94 and len(spec['RAW_inputs']) == 94 and len(spec['terminal_identity_pins']) == 102,
            'Legacy original history was mutated')
    return dict(context=context, normals=list(normals) + new_raws, closure=closure,
                preserved_failed9_RAW=failure_raw, own_gate_cpu_argument=cpu,
                current_owner_termination_self_certified=False)


def join_added_GPU_owners(spec, gate, context, owner_ref_join, cpu=False):
    nodes = [spec['successful_compatibility_probe']]
    if cpu:
        nodes.append(spec['exact_input_capture_GPU'])
    joins = []
    for node in nodes:
        index = owner_ref_join(context, gate['owners'], node['RAW_input']['receipt'], node['GPU_admission'],
                               node['source'], node['spec'], '0')
        joins.append(index)
    require(joins == ([94, 95] if cpu else [94]), 'Actual successful probe95/capture96 chronological GPU joins differ')
    return joins


if __name__ == '__main__':
    raise SystemExit('UNBOUND_DRAFT helper only: ROOT must integrate/review/bind a NEW complete runner; no attempt or Q is launched.')
