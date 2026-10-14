"""UNBOUND / UNEXECUTED N15 same-actual-GPU-input independent CPU replay.

Ordinary draft only: strong original701 E24 LOG PR, fixed374 native.50 pairs
plus6 unmatched GT (not evaluable). No actual scientific inputs were read to
prepare this file. ROOT must close/audit completed GPU0 then bind a fresh CPU
source/spec/admission with empty CUDA visibility before any scientific attempt.

CPU only replays captured GPU F/e/L/C-index. Two independently expressed PR
formulas retain original40497/eight FP32 tensor state, residual AND combined
absolute1e-4 (rtol0), exact strict>0 signs, independent count/array/mask oracle checks. No CPU F
rerender, AP, fullall100 evaluation, training, deployment or innovation claim.
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
from fractions import Fraction

SCHEMA='ROOT-N15-fixed-E24-action-opportunity-CPU-spec-v1'
GPU_SCHEMA='ROOT-N15-fixed-E24-action-opportunity-GPU-spec-v1'
RAW_KEYS=('receipt','phase','admission','phase_binding','bootstrap')
THREAD_ENV=('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')
SEED=202610120701
EPOCH=24
POINT_CHUNK=2048
PARAMETERS=40497
NAMES=('hidden1.weight','hidden1.bias','hidden2.weight','hidden2.bias',
       'hidden3.weight','hidden3.bias','final.weight','final.bias')
SHAPES=((91,257),(91,),(91,92),(91,),(91,92),(91,),(1,92),(1,))
LOCKED={'matched_count':374,'unmatched_gt_count':6,'total_gt_count':380}

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
    require(type(expected) is int and expected >= 104 and
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

def write_new(path, value):
    # The original attempt does not read/hash its own new Q. ROOT validates it
    # only after actual held terminal/reap/birth absence. SHA comes from bytes.
    data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
    path = Path(path).resolve()
    with path.open('xb') as stream:
        stream.write(data)
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def state_check(torch, head, state):
    require(tuple(state) == NAMES and tuple(name for name, _ in head.named_parameters()) == NAMES and
            len(tuple(head.parameters())) == 8 and sum(p.numel() for p in head.parameters()) == PARAMETERS and
            not tuple(head.buffers()) and not head.training, 'Original PR exact8 names/order/40497/frozen ABI required')
    for name, shape in zip(NAMES, SHAPES):
        value, parameter = state[name], dict(head.named_parameters())[name]
        require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and value.dtype == torch.float32 and
                value.layout == torch.strided and tuple(value.shape) == shape and bool(torch.isfinite(value).all()) and
                not parameter.requires_grad and parameter.grad is None and torch.equal(parameter, value),
                'Qualified strongE24 original FP32 state changed: ' + name)


def independent_PR(torch, state, F, L):
    """Explicit original F/L PR formula; no trained forward or gated logits.

    Keep u before addition: (L+u)-L is not an independent residual computation.
    Matches the ROOT supplied original independent_logits arithmetic.
    """
    normalized = torch.nn.functional.layer_norm(F, (256,), weight=None, bias=None, eps=1e-5)
    coarse = L.clamp(-8, 8).unsqueeze(1) / 8
    hidden = normalized
    for layer in ('hidden1', 'hidden2', 'hidden3'):
        hidden = torch.relu(torch.nn.functional.linear(torch.cat((hidden, coarse), dim=1),
            state[layer + '.weight'], state[layer + '.bias']))
    residual = torch.nn.functional.linear(torch.cat((hidden, coarse), dim=1),
        state['final.weight'], state['final.bias']).squeeze(1)
    return residual, L + residual


def cpu_vector(torch, value, shape, dtype, label):
    require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and value.dtype == dtype and
            value.layout == torch.strided and tuple(value.shape) == tuple(shape) and not value.requires_grad and
            bool(torch.isfinite(value).all()), 'Captured finite detached CPU tensor ABI differs: ' + label)


def numerical(torch, first, second, label, strict_sign=False):
    require(first.shape == second.shape and first.dtype == second.dtype == torch.float32 and
            first.device.type == second.device.type == 'cpu' and bool(torch.isfinite(first).all()) and
            bool(torch.isfinite(second).all()), 'Validated numerical comparator tensors required: ' + label)
    if not torch.allclose(first, second, atol=1e-4, rtol=0.0):
        raise ComputationalPortabilityFailure('Fixed absolute1e-4/rtol0 residual/combined comparison failed: ' + label)
    if strict_sign and not torch.equal(first > 0, second > 0):
        raise ComputationalPortabilityFailure('Strict>0 exact C/Omega sign comparison failed: ' + label)
    return float((first - second).abs().max()) if first.numel() else 0.0


def replay_chunks(torch, head, state, shared, pair, positions, gpu_residual, gpu_combined, started, process, rss_limit):
    count = len(positions)
    cpu_vector(torch, positions, (count,), torch.int64, 'selected positions in C')
    cpu_vector(torch, gpu_residual, (count,), torch.float32, 'GPU residual target')
    cpu_vector(torch, gpu_combined, (count,), torch.float32, 'GPU combined target')
    all_indices = pair['C_indices']
    all_L = pair['GPU_native_L']
    union = shared['union_flat_indices']
    selected_indices = all_indices[positions]
    mapping = torch.searchsorted(union, selected_indices)
    require(count == 0 or (int(mapping.max()) < len(union) and torch.equal(union[mapping], selected_indices)),
            'Captured actual GPU F union mapping must contain every selected point')
    native = all_L[positions]
    residuals = [torch.empty_like(native), torch.empty_like(native)]
    combineds = [torch.empty_like(native), torch.empty_like(native)]
    maxima = dict(formula1_residual=0.0, formula1_combined=0.0, formula2_residual=0.0,
                  formula2_combined=0.0, between_formulas_residual=0.0, between_formulas_combined=0.0)
    calls = 0
    q = pair['identity']['query_id']
    with torch.no_grad():
        for first in range(0, count, POINT_CHUNK):
            deadline(started)
            require(process.memory_info().rss < rss_limit, 'CPU prelocked process RSS exceeded')
            last = min(first + POINT_CHUNK, count)
            F = shared['sampled_F'][mapping[first:last]]
            e = shared['e0'][0, q].expand(last - first, 256)
            L = native[first:last]
            output = head(F, e, L, L.new_zeros((last - first, 49)), arm='PR')
            u1, v1 = output['residual'], L + output['residual']
            u2, v2 = independent_PR(torch, state, F, L)
            # For C outside oldOmega, head.logits would silently retain nativeL.
            # Both replay formulas use the explicit ungated nativeL+residual.
            for i, (u, v) in enumerate(((u1, v1), (u2, v2))):
                cpu_vector(torch, u, (last - first,), torch.float32, 'independent CPU residual')
                cpu_vector(torch, v, (last - first,), torch.float32, 'independent CPU combined')
                residuals[i][first:last] = u
                combineds[i][first:last] = v
                key = 'formula%d_' % (i + 1)
                maxima[key + 'residual'] = max(maxima[key + 'residual'],
                    numerical(torch, u, gpu_residual[first:last], key + 'GPU residual'))
                maxima[key + 'combined'] = max(maxima[key + 'combined'],
                    numerical(torch, v, gpu_combined[first:last], key + 'GPU combined', strict_sign=True))
            maxima['between_formulas_residual'] = max(maxima['between_formulas_residual'],
                numerical(torch, u1, u2, 'two independent CPU residual formulas'))
            maxima['between_formulas_combined'] = max(maxima['between_formulas_combined'],
                numerical(torch, v1, v2, 'two independent CPU combined formulas', strict_sign=True))
            calls += 1
            del F, e, L, output, u1, v1, u2, v2, u, v
    return residuals, combineds, maxima, calls


def count_classes(np, native, after, GT, domain):
    classes = dict(a=int(np.count_nonzero(domain & native & ~after & ~GT)),
                   b=int(np.count_nonzero(domain & ~native & after & GT)),
                   c=int(np.count_nonzero(domain & native & ~after & GT)),
                   d=int(np.count_nonzero(domain & ~native & after & ~GT)),
                   z=int(np.count_nonzero(domain & (native == after))))
    require(sum(classes.values()) == int(np.count_nonzero(domain)), 'Physical real-pool action partition must cover exactly')
    return classes


def pixels(np, native, GT):
    return dict(TP=int(np.count_nonzero(native & GT)), FP=int(np.count_nonzero(native & ~GT)),
                FN=int(np.count_nonzero(~native & GT)))


def mask_digest(np, value):
    require(isinstance(value, np.ndarray) and value.dtype == np.bool_, 'Mask digest requires actual Boolean array')
    value = np.ascontiguousarray(value)
    return dict(shape=list(value.shape), dtype='bool', sha256=hashlib.sha256(value.tobytes()).hexdigest())


def oracle_array_crosscheck(np, count_helper, array_helper, native, after, GT, Omega, candidate):
    require(all(isinstance(value, np.ndarray) and value.dtype == np.bool_ and value.shape == native.shape
                for value in (native, after, GT, Omega, candidate)) and bool(GT.any()) and
            not bool(np.any(Omega & ~candidate)), 'Matched nonemptyGT Boolean arrays and Omega subset C required')
    K = int(np.count_nonzero(Omega))
    native_counts = pixels(np, native, GT)
    outputs = {}
    branches = (('atmost_Omega', Omega, after, 'atmost'), ('atmost_C', candidate, after, 'atmost'),
                ('exact_C', candidate, after, 'exact'), ('atmost_GT_C', candidate, GT, 'atmost'))
    for label, domain, action, mode in branches:
        classes = count_classes(np, native, action, GT, domain)
        counted = count_helper.oracle_counts(native_counts['TP'], native_counts['FP'], native_counts['FN'], classes, K, mode)
        selected = (array_helper.exact_k_action_oracle(native, action, GT, domain, K) if mode == 'exact' else
                    array_helper.at_most_k_repair_oracle(native, action, GT, domain, K))
        result = array_helper.apply_fixed_candidate(native, action, selected)
        observed = pixels(np, result, GT)
        selected_classes = count_classes(np, native, action, GT, selected)
        require(selected_classes == counted['selected_counts'] and observed == counted['after_counts'] and
                counted['selected_support'] == int(np.count_nonzero(selected)) and
                counted['intersection'] == observed['TP'] and counted['union'] == sum(observed.values()),
                'Independent count and array/mask oracle results must be exactly equal: ' + label)
        outputs[label] = dict(count_oracle=counted, selected_mask=mask_digest(np, selected), output_mask=mask_digest(np, result))
    actual = array_helper.apply_fixed_candidate(native, after, Omega)
    outputs['baseline_apply_Omega'] = dict(after_counts=pixels(np, actual, GT), output_mask=mask_digest(np, actual))
    outputs['native'] = native_counts
    return outputs


def reconstruct_domains(torch, np, cv2, shared, pair, native_L, fixed_pair):
    """Independent original-pixel Chebyshev8 C; neither GT nor K chooses C."""
    shape = tuple(shared['identity']['original_shape'])
    cpu_vector(torch, native_L, shape, torch.float32, 'immutable original GPU query L')
    Cidx, L = pair['C_indices'], pair['GPU_native_L']
    cpu_vector(torch, Cidx, (len(Cidx),), torch.int64, 'C sorted physical indices')
    cpu_vector(torch, L, (len(Cidx),), torch.float32, 'captured native L at C')
    require(len(Cidx) == 0 or (int(Cidx[0]) >= 0 and int(Cidx[-1]) < native_L.numel() and
            bool((Cidx[1:] > Cidx[:-1]).all())), 'Actual C indices must be sorted unique in original raster')
    native = (native_L > 0).numpy()
    Omega = (native_L.abs() <= 2).numpy()
    if bool(native.any()):
        expanded = cv2.dilate(native.astype(np.uint8), np.ones((17, 17), dtype=np.uint8),
            anchor=(-1, -1), iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(np.bool_)
        candidate = Omega | native | expanded
        fallback = False
    else:
        candidate, fallback = Omega.copy(), True
    expected_indices = torch.from_numpy(np.flatnonzero(candidate).astype(np.int64))
    require(torch.equal(Cidx, expected_indices) and torch.equal(L, native_L.flatten()[Cidx]),
            'CPU independent native-only C and exact actual GPU L mapping must match capture')
    K = int(np.count_nonzero(Omega))
    expected_positions = torch.nonzero(native_L.flatten()[Cidx].abs() <= 2, as_tuple=False).flatten()
    cpu_vector(torch, pair['Omega_C_positions'], (K,), torch.int64, 'ascending original Omega positions in C')
    require(torch.equal(pair['Omega_C_positions'], expected_positions) and K <= len(Cidx),
            'Original K=|Omega|/Omega subset C must hold including K0')
    points = len(Cidx)
    for name in ('GPU_C_residual', 'GPU_C_combined'):
        cpu_vector(torch, pair[name], (points,), torch.float32, name)
    for name in ('GPU_Omega_residual', 'GPU_Omega_combined'):
        cpu_vector(torch, pair[name], (K,), torch.float32, name)
    cpu_vector(torch, pair['GPU_C_positive'], (points,), torch.bool, 'actual GPU C strict>0 bits')
    require(torch.equal(pair['GPU_C_positive'], pair['GPU_C_combined'] > 0),
            'Captured GPU C strictsign must equal actual GPU combined logits')
    # This is a GPU batch self-compatibility qualification, independent of GO.
    overlap = pair['Omega_C_positions']
    numerical(torch, pair['GPU_C_residual'][overlap], pair['GPU_Omega_residual'], 'GPU C/Omega-only residual overlap')
    numerical(torch, pair['GPU_C_combined'][overlap], pair['GPU_Omega_combined'],
              'GPU C/Omega-only combined overlap', strict_sign=True)
    N = native_L.numel()
    require(type(pair['GT_bits_count']) is int and pair['GT_bits_count'] == N and
            pair['GT_bitorder'] == 'little', 'Captured GT bit population/bitorder must be explicit')
    packed = pair['GT_packbits']
    cpu_vector(torch, packed, ((N + 7) // 8,), torch.uint8, 'actual GPU diagnostic GT packbits')
    decoded = np.unpackbits(packed.numpy(), count=N, bitorder='little').astype(np.bool_)
    require(np.array_equal(np.packbits(decoded, bitorder='little'), packed.numpy()) and bool(decoded.any()),
            'Matched GT must be nonempty and packbit padding exactly roundtrip')
    GT = decoded.reshape(shape)
    native_counts = pixels(np, native, GT)
    anchor = {key: fixed_pair['native']['pixel_' + key] for key in ('TP', 'FP', 'FN')}
    require(native_counts == anchor, 'Original fixed native374 pair GT/native pixel anchor differs')
    old_after = native.copy()
    old_after.reshape(-1)[Cidx[overlap].numpy()] = (pair['GPU_Omega_combined'] > 0).numpy()
    old_counts = pixels(np, old_after, GT)
    anchor = {key: fixed_pair['refined']['pixel_' + key] for key in ('TP', 'FP', 'FN')}
    require(old_counts == anchor, 'Exact original strongE24 Omega-only full-binary refined pair anchor differs')
    after = native.copy()
    after.reshape(-1)[Cidx.numpy()] = pair['GPU_C_positive'].numpy()
    return dict(native=native, after=after, GT=GT, Omega=Omega, candidate=candidate,
                K=K, C_count=points, Pempty_fallback=fallback, original_Omega_after=old_after)


def replay_pair(torch, np, cv2, head, state, shared, pair, native_L, fixed_pair,
                count_helper, array_helper, started, process, rss_limit, original_binary):
    domains = reconstruct_domains(torch, np, cv2, shared, pair, native_L, fixed_pair)
    require(np.array_equal(domains['original_Omega_after'], original_binary),
            'Independent original Omega-only fullbinary must exactly equal historical strongE24 actual export')
    all_positions = torch.arange(len(pair['C_indices']), dtype=torch.int64)
    C_u, C_v, C_max, C_calls = replay_chunks(torch, head, state, shared, pair, all_positions,
        pair['GPU_C_residual'], pair['GPU_C_combined'], started, process, rss_limit)
    Omega_u, Omega_v, Omega_max, Omega_calls = replay_chunks(torch, head, state, shared, pair, pair['Omega_C_positions'],
        pair['GPU_Omega_residual'], pair['GPU_Omega_combined'], started, process, rss_limit)
    overlap_max = []
    for i in range(2):
        overlap_max.append(dict(residual=numerical(torch, C_u[i][pair['Omega_C_positions']], Omega_u[i],
                'CPU formula%d C/Omega-only residual overlap' % (i + 1)),
            combined=numerical(torch, C_v[i][pair['Omega_C_positions']], Omega_v[i],
                'CPU formula%d C/Omega-only combined overlap' % (i + 1), strict_sign=True)))
        cpu_after = domains['native'].copy()
        cpu_after.reshape(-1)[pair['C_indices'].numpy()] = (C_v[i] > 0).numpy()
        require(np.array_equal(cpu_after, domains['after']), 'CPU actual C fullbinary must equal captured GPU fixed action')
    oracles = oracle_array_crosscheck(np, count_helper, array_helper, domains['native'], domains['after'],
        domains['GT'], domains['Omega'], domains['candidate'])
    # Same C after-bits drive both primary oracles; Omega-only binary is solely
    # the original baseline anchor after the strict overlap gate above.
    require(oracles['baseline_apply_Omega']['after_counts'] == pixels(np, domains['original_Omega_after'], domains['GT']) and
            oracles['baseline_apply_Omega']['output_mask'] == mask_digest(np, domains['original_Omega_after']),
            'C-on-Omega action must reproduce original Omega-only fullbinary exactly')
    record = dict(identity=pair['identity'], K=domains['K'], C_count=domains['C_count'],
        Pempty_fallback=domains['Pempty_fallback'], independent_oracles=oracles,
        C_numeric_max_absolute_errors=C_max, Omega_numeric_max_absolute_errors=Omega_max,
        CPU_overlap_max_absolute_errors=overlap_max, two_independent_formulas=True,
        C_strict_sign_exact=True, Omega_strict_sign_exact=True, head_point_calls=dict(C=C_calls, Omega=Omega_calls))
    return record, domains


def ratio(metric):
    value = metric['iou_fraction']
    return Fraction(value['numerator'], value['denominator'])


def fraction_dict(value):
    return dict(numerator=value.numerator, denominator=value.denominator)


def actual_branch(np, native, after, GT, selected):
    counts = pixels(np, after, GT)
    union = sum(counts.values())
    fraction = Fraction(counts['TP'], union) if union else Fraction(1)
    classes = count_classes(np, native, after, GT, selected)
    return dict(after_counts=counts, intersection=counts['TP'], union=union, empty_union=(union == 0),
        iou=float(fraction), iou_fraction=fraction_dict(fraction), selected_counts=classes,
        selected_support=int(np.count_nonzero(selected)), actual_flips=sum(classes[name] for name in ('a', 'b', 'c', 'd')),
        selected_noops=classes['z'], helpful_flips=classes['a'] + classes['b'], harmful_flips=classes['c'] + classes['d'])


def independent_diagnostic(np, gpu, replayed, domains, pair, batch):
    native, GT, after, Omega, C = (domains[name] for name in ('native', 'GT', 'after', 'Omega', 'candidate'))
    records = replayed['independent_oracles']
    omega_oracle = records['atmost_Omega']['count_oracle']
    C_oracle = records['atmost_C']['count_oracle']
    exact_C = records['exact_C']['count_oracle']
    GT_C = records['atmost_GT_C']['count_oracle']
    K = domains['K']
    classes_Omega, classes_C = count_classes(np, native, after, GT, Omega), count_classes(np, native, after, GT, C)
    within = dict(TP=int(np.count_nonzero(C & native & GT)), FP=int(np.count_nonzero(C & native & ~GT)),
        FN=int(np.count_nonzero(C & ~native & GT)), TN=int(np.count_nonzero(C & ~native & ~GT)))
    common = native.copy()
    common[Omega] = after[Omega]
    original_actual = actual_branch(np, native, domains['original_Omega_after'], GT, Omega)
    common_actual = actual_branch(np, native, common, GT, Omega)
    baseline = actual_branch(np, native, native, GT, np.zeros(native.shape, dtype=np.bool_))
    main = ratio(C_oracle) - ratio(omega_oracle)
    secondary = ratio(omega_oracle) - ratio(common_actual)
    require(Fraction(1, 2) <= ratio(baseline) <= ratio(omega_oracle) <= ratio(C_oracle) <= ratio(GT_C) and
            ratio(common_actual) <= ratio(exact_C) <= ratio(C_oracle) and main >= 0 and secondary >= 0,
            'Actual fixed374 matched/native/oracle physical feasibility differs')
    minimum_harms = max(0, K - classes_C['a'] - classes_C['b'] - classes_C['z'])
    require(exact_C['harmful_flips'] == minimum_harms, 'Only real noop padding can avoid unavoidable exactK harms')
    return dict(schema=gpu.PAIR_SCHEMA, identity=pair['identity'], budget_K=K, C_count=domains['C_count'], native=baseline,
        native_confusion_within_C=within, classes_Omega=classes_Omega, classes_C=classes_C,
        branches=dict(actual_PR_Omega_original=original_actual, actual_PR_Omega_common_C=common_actual,
            action_oracle_atmostK_Omega_common_C=omega_oracle, action_oracle_atmostK_C=C_oracle,
            action_oracle_exactK_C_with_real_noops=exact_C, GT_replacement_oracle_atmostK_C_same_pool_same_budget=GT_C),
        main_B_delta_fraction=fraction_dict(main), secondary_A_delta_fraction=fraction_dict(secondary),
        exact_C_unavoidable_harms=minimum_harms, actual_Omega_harmful_flips=common_actual['harmful_flips'],
        additional_action_classes_C_minus_Omega={name: classes_C[name] - classes_Omega[name] for name in ('a', 'b', 'c', 'd', 'z')},
        batch_compatibility=batch, array_count_crosscheck_exact=True, GT_privileged_action_selection=True, AP_result=False)


def digest_tensor(value):
    array = value.detach().cpu().contiguous().numpy()
    return dict(shape=list(array.shape), dtype=str(array.dtype), sha256=hashlib.sha256(array.tobytes()).hexdigest())


def immutable_selector(torch, gate, shared, index):
    row = shared['selector']
    require(row == gate['selector']['manifest']['rows'][index] and
            shared['native_packet'] == gate['packets'][index], 'Captured image must use original canonical native packet/selector')
    expected = dict(dataset_index=index, image_id=shared['image_id'], identity=shared['identity'],
        native_packet=shared['native_packet'], natural_query_ids=list(range(100)),
        native_export_query_ids=row['native_export_query_ids'], renderer_source=gate['context']['renderer_source'])
    require(sorted(row['native_export_query_ids']) == list(range(100)) and
            all(row[key] == value for key, value in expected.items()), 'Original selector natural/export query population differs')
    packet = torch.load(str(closed(gate['closure'], row['carrier'])), map_location='cpu')
    producer = gate['selector']['report']
    value = packet['native_L']
    cpu_vector(torch, value, (100,) + tuple(shared['identity']['original_shape']), torch.float32, 'immutable GPU full100 L selector')
    require(packet['schema'] == gate['lineage'].SELECTOR_PACKET_SCHEMA and
            packet['source'] == producer['source'] and packet['spec'] == producer['spec'] and
            packet['GPU_admission'] == producer['GPU_admission'] and
            all(packet[key] == item for key, item in expected.items()) and
            digest_tensor(value) == packet['L_digest'] == row['L_digest'],
            'Actual old GPU selector producer and full100 L bytes/digest differ')
    return value


def image_shared_check(torch, shared, capture_row, gate, index):
    require(type(shared) is dict and shared['dataset_index'] == index and
            shared['image_id'] == capture_row['image_id'] and shared['native_packet'] == gate['packets'][index] and
            shared['selector'] == gate['selector']['manifest']['rows'][index] and
            shared['identity'] == shared['selector']['identity'], 'Captured shared original image provenance differs')
    U = capture_row['U']
    cpu_vector(torch, shared['union_flat_indices'], (U,), torch.int64, 'sampled F physical union index')
    cpu_vector(torch, shared['sampled_F'], (U, 256), torch.float32, 'actual GPU sampled F shared union')
    cpu_vector(torch, shared['e0'], (1, 100, 256), torch.float32, 'actual GPU shared e0')
    union = shared['union_flat_indices']
    shape = shared['identity']['original_shape']
    require(type(shape) in (tuple, list) and len(shape) == 2 and all(type(n) is int and n > 0 for n in shape) and
            (U == 0 or (int(union[0]) >= 0 and int(union[-1]) < shape[0] * shape[1] and
             bool((union[1:] > union[:-1]).all()))), 'Sampled actual GPU F union must be sorted unique inside original image')


def CPU_budget_check(spec, evidence, gate, process, psutil, output):
    budget = spec['CPU_replay_budget']
    keys = {'maximum_image_capture_file_bytes', 'maximum_image_capture_tensor_payload_bytes',
            'maximum_selector_tensor_bytes', 'required_available_RAM_bytes', 'maximum_process_RSS_bytes',
            'auxiliary_output_budget_bytes', 'minimum_free_output_bytes', 'minimum_available_RAM_margin_bytes'}
    require(type(budget) is dict and set(budget) == keys and all(type(n) is int and n >= 0 for n in budget.values()),
            'ROOT must prelock exact CPU capture/selector/RAM/RSS/output budget8 fields')
    rows = evidence['manifest']['rows']
    maximum_pixels = max(row['identity']['original_shape'][0] * row['identity']['original_shape'][1]
                         for row in gate['selector']['manifest']['rows'])
    require(budget['maximum_image_capture_file_bytes'] == max(row['capture']['bytes'] for row in rows) and
            budget['maximum_image_capture_tensor_payload_bytes'] == max(row['raw_tensor_payload_bytes'] for row in rows) and
            budget['maximum_selector_tensor_bytes'] == 400 * maximum_pixels and
            budget['minimum_available_RAM_margin_bytes'] >= 1024**3 and
            budget['required_available_RAM_bytes'] >= budget['maximum_image_capture_tensor_payload_bytes'] +
                budget['maximum_selector_tensor_bytes'] + budget['minimum_available_RAM_margin_bytes'] and
            budget['required_available_RAM_bytes'] >= 16 * 1024**3 and
            psutil.virtual_memory().available >= budget['required_available_RAM_bytes'] and
            budget['maximum_process_RSS_bytes'] > process.memory_info().rss + budget['maximum_image_capture_tensor_payload_bytes'] +
                budget['maximum_selector_tensor_bytes'] and
            budget['minimum_free_output_bytes'] >= budget['auxiliary_output_budget_bytes'] + 1024**3 and
            psutil.disk_usage(str(output.parent)).free >= budget['minimum_free_output_bytes'],
            'Completed actual capture sizes and CPU RAM/RSS/output prelock do not hold')
    return budget


def imported_files(gpu, gate, psutil, torch, np, cv2, masks, COCO, ndimage):
    numerical = gate['common'].imported_files(torch, np, cv2, masks, COCO,
        gate['closure'], gate['admission'], ndimage)
    modules = dict(psutil=psutil, psutil_platform=psutil._psplatform, psutil_binary=psutil._psplatform.cext,
        torch=torch, torch_binary=torch._C, torch_tensor=sys.modules['torch._tensor'], torch_autograd=sys.modules['torch.autograd'],
        torch_random=torch.random, numpy=np, numpy_binary=np.core._multiarray_umath)
    extra = {name: reference(module.__file__) for name, module in modules.items()}
    objects = dict(torch_Module=torch.nn.Module, torch_Linear=torch.nn.Linear, torch_layer_norm=torch.nn.functional.layer_norm,
        torch_linear=torch.nn.functional.linear, torch_serialization=torch.load, stdlib_Fraction=Fraction)
    for name, obj in objects.items():
        # torch.nn.functional.linear is an actual builtin; its binary/module
        # provenance is already above, so do not invent a Python source path.
        if inspect.isbuiltin(obj):
            continue
        source = inspect.getsourcefile(obj)
        if source is not None:
            extra[name] = reference(source)
    require(all(gate['closure'].get(ref['path']) == ref and
            gate['admission']['bound_source_sha256'].get(ref['path']) == ref['sha256'] for ref in extra.values()),
            'Independent CPU actual imported runtime/source/binary not fully admission/SOURCE pinned')
    return dict(numerical=numerical, independent_CPU=extra)


def completed_GPU_gate(gpu, gate, spec):
    closure = gate['closure']
    refs = {name: spec['diagnostic_GPU_' + name] for name in ('source', 'spec', 'report', 'admission', 'RAW')}
    for ref in refs.values():
        closed(closure, ref)
    parent, report = read(closed(closure, refs['spec'])), read(closed(closure, refs['report']))
    count = parent['normal_RAW_count']
    require(parent['schema'] == GPU_SCHEMA and type(count) is int and count >= 103 and
            count == parent['expected_normal_RAW_count'] and spec['normal_RAW_count'] == count + 1 and
            parent['RAW_inputs'] == spec['RAW_inputs'][:count] and
            spec['RAW_inputs'][count]['receipt'] == refs['RAW'] and spec['RAW_inputs'][count]['admission'] == refs['admission'],
            'Current completed diagnosticGPU must be exact latest normal ancestor of CPU attempt')
    require(parent['entry_source'] == refs['source'] and
            Path(parent['GPU_admission_file']).resolve(strict=True) == Path(refs['admission']['path']).resolve(strict=True) and
            parent['phase'] == spec['RAW_inputs'][count]['phase'] and
            report['schema'] == gpu.REPORT_SCHEMA and report['status'] == gpu.STATUS and
            report['source'] == refs['source'] and report['spec'] == refs['spec'] and
            report['GPU_admission'] == refs['admission'] and report['phase'] == parent['phase'] and
            report['RAW_inputs'] == parent['RAW_inputs'] and report['normal_RAW_count'] == count and
            report['terminal_identity_pins'] == parent['terminal_identity_pins'] and
            report['total_failed_RAW_count'] == 9 and report['total_terminal_identity_pin_count'] == count + 9,
            'Completed actual GPU diagnostic report/source/spec/phase/admission/RAW binding differs')
    for key in ('prior_failed_CPU_receipt', 'additional_failed_RAW_inputs', 'additional_preserved_independent_CPU_failure'):
        require(parent[key] == spec[key], 'Preserved completedGPU failure history differs: ' + key)
    require(report['additional_preserved_independent_CPU_failure'] == spec['additional_preserved_independent_CPU_failure'] and
            all(spec['terminal_identity_pins'].get(key) == value for key, value in parent['terminal_identity_pins'].items()) and
            all(closure.get(ref['path']) == ref for ref in parent['sources']), 'CompletedGPU full Source/owner pins must stay inherited')
    for key in gpu.CARRIER_KEYS + ('locked_protocol', 'count_oracle_source', 'array_oracle_source', 'fixed_population'):
        require(parent[key] == spec[key], 'Exact same E24 carrier/prelocked helper/protocol/population required: ' + key)
    require(report['carrier_refs'] == {key: spec[key] for key in gpu.CARRIER_KEYS} and
            report['locked_protocol'] == spec['locked_protocol'] and report['count_oracle_source'] == spec['count_oracle_source'] and
            report['array_oracle_source'] == spec['array_oracle_source'] and report['fixed_population'] == spec['fixed_population'] and
            report['carrier_CPU_qualification'] == spec['carrier_CPU_qualification'] and report['carrier_GPU_report'] == spec['carrier_GPU_report'] and
            report['head_checkpoint'] == gate['context']['endpoint']['checkpoints']['LOG_PR'] and
            report['carrier_seed'] == SEED and report['carrier_epoch'] == EPOCH and report['arm'] == 'LOG_PR' and
            report['parameters'] == PARAMETERS and report['parameter_tensors'] == 8,
            'GPU diagnostic must use the same original strong701 E24 PR checkpoint/proofs')
    joined = gate['lineage'].owner_ref_join(gate['context'], gate['owners'], refs['RAW'], refs['admission'], refs['source'], refs['spec'], '0')
    require(type(joined) is int and joined == count, 'Actual current diagnosticGPU0 owner join must be latest normal')
    require(report['images'] == 187 and report['matched_pairs'] == 374 and report['unmatched_GT'] == 6 and report['raw_GT'] == 380 and
            all(report[key] is True for key in ('unmatched_unevaluable_not_zero', 'native_all18700_exact', 'selector_all18700_exact',
                'original_Omega_full_binary_all374_exact', 'original_Omega_refined_counts_all374_exact',
                'C_Omega_absolute_1e4_and_strict_sign_all374_exact', 'GO_is_privileged_pending_independent_CPU', 'no_TEST', 'no_cross_dataset')) and
            all(report[key] is False for key in ('independent_CPU_qualified', 'current_owned_GPU_terminal_self_certified',
                'whole_leaf_VRAM_certified', 'AP_evaluated', 'training_started', 'optimizer_created', 'native_model_loaded',
                'task_efficacy_established', 'innovation_established')) and
            report['packet_refs'] == gate['packets'] and report['selector_manifest'] == gate['selector']['manifest_ref'] and
            report['native_reference_predictions'] == gate['context']['native_reference_predictions'] and
            report['original_E24_LOG_predictions'] == gate['carrier_report']['predictions'][gpu.CARRIER_LABEL] and
            report['DEV'] == gate['context']['DEV'] and report['cleanup']['allocated_after'] == 0 and
            type(report['peak_allocator_bytes']) is int and 0 <= report['peak_allocator_bytes'] <= 4096 * 1024**2,
            'Completed diagnostic native/binary/numerical/cohort/resource/evidence scope differs')
    manifest = read(closed(closure, report['capture_manifest']))
    binding = dict(source=refs['source'], spec=refs['spec'], GPU_admission=refs['admission'], phase=parent['phase'],
        locked_protocol=spec['locked_protocol'], carrier_CPU_qualification=spec['carrier_CPU_qualification'],
        carrier_GPU_report=spec['carrier_GPU_report'], head_checkpoint=report['head_checkpoint'], fixed_population=spec['fixed_population'])
    require(manifest['schema'] == gpu.CAPTURE_MANIFEST_SCHEMA and
            all(manifest[key] == value for key, value in binding.items()) and manifest['capture_schema'] == report['capture_schema'] == gpu.CAPTURE_SCHEMA and
            manifest['images'] == 187 and manifest['matched_pairs'] == 374 and manifest['unmatched_GT'] == 6 and
            manifest['actual_GPU_inputs_no_CPU_rerender'] is True and manifest['capture_budget'] == report['capture_budget'],
            'Completed actual GPU whole-image capture manifest binding differs')
    rows = manifest['rows']
    require(type(rows) is list and len(rows) == 187 and [row['dataset_index'] for row in rows] == list(range(187)) and
            report['capture_refs'] == [row['capture'] for row in rows] and len({row['capture']['path'] for row in rows}) == 187,
            'All187 real physical whole-image captures must be closed once in canonical order')
    for row in rows:
        closed(closure, row['capture'])
        require(all(type(row[name]) is int and row[name] >= 0 for name in ('U', 'C_points', 'K_points', 'raw_tensor_payload_bytes')),
                'Captured image aggregate counts must be nonnegative true integers')
    require(sum(row['C_points'] for row in rows) == report['logical_C_points'] and
            sum(row['K_points'] for row in rows) == report['logical_Omega_K_points'] and
            sum(row['raw_tensor_payload_bytes'] for row in rows) == manifest['actual_raw_tensor_payload_bytes'] == report['actual_raw_capture_tensor_bytes'] and
            sum(row['capture']['bytes'] for row in rows) == manifest['actual_serialized_capture_bytes'] == report['actual_serialized_capture_bytes'],
            'Actual physical capture aggregate file/payload/point counts differ')
    diagnostics = read(closed(closure, report['pair_diagnostics']))
    population = read(closed(closure, report['population']))
    image_checks = read(closed(closure, report['image_checks']))
    trace = [json.loads(line) for line in closed(closure, report['trace']).read_text(encoding='utf-8').splitlines()]
    start = read(closed(closure, report['start']))
    require(all(start[key] == value for key, value in binding.items()) and start['protocol'] == gate['protocol'] and
            start['head_state_proofs'] == report['head_state_proofs'] and start['capture_budget'] == report['capture_budget'] and
            trace == image_checks and len(image_checks) == 187 and len(diagnostics) == 374,
            'Completed GPU START/proofs/fulltrace/imagechecks/374diagnostics differ')
    for row, image in zip(rows, image_checks):
        require(image['dataset_index'] == row['dataset_index'] and image['image_id'] == row['image_id'] and
                image['capture'] == row['capture'] and image['matched_pairs'] == len(row['pairs']) and
                image['native_all100_exact'] is True and image['selector_exact'] is True and
                image['numerical_qualification_passed'] is True, 'Every actual image must pass GPU binary/numerical qualification')
    for key in ('main_comparison', 'secondary_comparison'):
        require(report[key] == gate['protocol'][key], 'Predata diagnostic comparison changed: ' + key)
    require(report['diagnostic_decision'] == population['primary_B']['diagnostic_decision'], 'GPU provisional decision/population differs')
    for ref in (report['head_checkpoint'], report['original_E24_LOG_predictions'], report['native_reference_predictions'], report['DEV']):
        closed(closure, ref)
    return dict(refs=refs, spec=parent, report=report, manifest=manifest, binding=binding,
                diagnostics=diagnostics, population=population, image_checks=image_checks)


def run(spec_path):
    started = time.monotonic()
    path = Path(spec_path).resolve(strict=True)
    spec = read(path)
    require(spec['schema'] == SCHEMA, 'Fresh independent N15 same-GPU-input CPU schema required')
    _, closure = raw_first(spec)
    closed(closure, spec['entry_source'])
    require(Path(spec['entry_source']['path']).resolve(strict=True) == Path(__file__).resolve(strict=True),
            'CPU must bind its own complete entry source')
    gpu = load(closure, spec['diagnostic_GPU_source'], '_N15_CPU_completed_GPU_gate')
    gate = gpu.source_gate(path, spec, cpu=True)
    require(gate['closure'] == closure and gate['source'] == spec['entry_source'] and gate['spec_ref'] == reference(path) and
            Path(gate['admission_ref']['path']).resolve(strict=True) == Path(spec['CPU_admission_file']).resolve(strict=True),
            'Current ROOT CPUempty source/spec/admission binding differs')
    evidence = completed_GPU_gate(gpu, gate, spec)
    require(gpu.validate_protocol(spec, closure) == gate['protocol'] and
            gate['protocol']['qualification']['CPU_residual_combined_absolute_tolerance'] == 1e-4,
            'Original predata absolute1e-4 protocol must be immutable; no new relative allowance')
    output = Path(spec['output_directory'])
    require(output.is_absolute() and not output.exists() and output.parent.is_dir(), 'Fresh absolute ROOT CPU output required')
    for target in ('QUALIFICATION.json', 'DETAILS.json', 'FAILURE.json'):
        require(str(output / target) not in closure, 'Current runtime CPU outputs are never SOURCE inputs')
    import psutil
    import torch
    require(torch.__version__ == '2.0.1+cu118' and not torch.cuda.is_initialized(), 'Qualified cold Torch runtime required')
    process = psutil.Process()
    resource = cpu_resource_check(spec, gate['admission'], psutil, torch)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    heads, states, proofs = gate['evaluation'].load_heads(torch, gate['context'], gate)
    head, state = heads['LOG_PR'], states['LOG_PR']
    state_check(torch, head, state)
    require(proofs == evidence['report']['head_state_proofs'], 'Original strongE24 actual load_heads proof must equal completedGPU proof')
    import numpy as np
    import cv2
    import scipy
    from scipy import ndimage
    from pycocotools import mask as masks
    from pycocotools.coco import COCO
    require(np.__version__ == '1.23.5' and cv2.__version__ == '4.8.0' and scipy.__version__ == '1.10.1',
            'Qualified original numerical versions required')
    reader = load(closure, gate['context']['reader_source'], '_N15_CPU_original_RLE_reader')
    # Historical prediction RLEs anchor fullbinary only. No COCO evaluator,
    # annotation-to-RLE GT decoder, native model or renderer is constructed.
    native_rows = read(closed(closure, gate['context']['native_reference_predictions']))
    log_rows = read(closed(closure, evidence['report']['original_E24_LOG_predictions']))
    require(len(native_rows) == len(log_rows) == 18700 and
            all({k: v for k, v in left.items() if k != 'segmentation'} ==
                {k: v for k, v in right.items() if k != 'segmentation'} for left, right in zip(native_rows, log_rows)),
            'Historical original all100 class/score/query/export order must stay fixed')
    native_by_key = {(row['image_id'], row['query_id']): row for row in native_rows}
    original_by_key = {(row['image_id'], row['query_id']): row for row in log_rows}
    require(len(native_by_key) == len(original_by_key) == 18700, 'Historical unique original image/query export required')
    prior_rows = read(closed(closure, gate['prior']['image_rows']))
    capture_rows = evidence['manifest']['rows']
    ids = [row['image_id'] for row in prior_rows]
    require(len(ids) == len(set(ids)) == len(capture_rows) == len(gate['packets']) == 187 and
            ids == [row['image_id'] for row in capture_rows] and
            len(gate['selector']['manifest']['rows']) == 187 and sum(row['raw_GT'] for row in prior_rows) == 380,
            'Complete original canonical187 image packet/selector/capture/population required')
    paired = {iid: [row for row in gate['fixed']['rows'] if row['image_id'] == iid and row['query_id'] is not None] for iid in ids}
    unmatched = [dict(image_id=row['image_id'], gt_id=row['gt_id'], status='UNEVALUABLE_NO_QUERY')
                 for row in gate['fixed']['rows'] if row['query_id'] is None]
    require(sum(len(rows) for rows in paired.values()) == 374 and len(unmatched) == 6,
            'Only original374 matched pairs; six unmatched remain independently unevaluable')
    budget = CPU_budget_check(spec, evidence, gate, process, psutil, output)
    libraries = imported_files(gpu, gate, psutil, torch, np, cv2, masks, COCO, ndimage)
    outside = torch.get_rng_state().clone()
    require(not torch.cuda.is_initialized() and 'mmdet' not in sys.modules, 'CPU numerical gate must preserve cold CUDA/no native model')
    output.mkdir()
    start_ref = write_new(output / 'START.json', dict(schema='N15-fixed-E24-action-opportunity-CPU-start-v1',
        source=gate['source'], spec=gate['spec_ref'], CPU_admission=gate['admission_ref'], phase=spec['phase'],
        diagnostic_GPU_report=evidence['refs']['report'], capture_manifest=evidence['report']['capture_manifest'],
        locked_protocol=spec['locked_protocol'], protocol=gate['protocol'], head_checkpoint=evidence['report']['head_checkpoint'],
        head_state_proofs=proofs, CPU_resources=resource, CPU_replay_budget=budget,
        no_CPU_renderer=True, no_original_GT_annotation_decoder=True, captured_GT_packbits_decoded=True,
        no_training=True, AP_evaluated=False,
        current_owned_CPU_terminal_self_certified=False))
    diagnostics, numeric_records = [], []
    completed = 0
    trace_sha, trace_bytes = hashlib.sha256(), 0
    try:
        with (output / 'TRACE.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
            for index, row in enumerate(capture_rows):
                deadline(started)
                require(process.memory_info().rss + row['raw_tensor_payload_bytes'] + budget['maximum_selector_tensor_bytes'] <
                        budget['maximum_process_RSS_bytes'], 'One whole capture/immutable fullL would exceed CPU prelock RSS')
                packet = torch.load(str(closed(closure, row['capture'])), map_location='cpu')
                require(type(packet) is dict and packet['schema'] == gpu.CAPTURE_SCHEMA and packet['binding'] == evidence['binding'] and
                        packet['natural_query_ids'] == list(range(100)) and packet['native_all100_exact'] is True and
                        packet['selector_exact'] is True and packet['candidate_generated_without_GT'] is True and
                        packet['all_pr_pairs_prelocked'] is True and packet['independent_CPU_qualified'] is False and
                        packet['current_terminal_self_certified'] is False and
                        all(packet[key] == row[key] for key in ('U', 'C_points', 'K_points', 'raw_tensor_payload_bytes')),
                        'Whole capture actual GPU binding/population/numerical/scope differs')
                shared = packet['shared']
                image_shared_check(torch, shared, row, gate, index)
                require(packet['native_export_query_ids'] == shared['selector']['native_export_query_ids'] and
                        [item['identity'] for item in packet['pairs']] == row['pairs'] ==
                        [{key: item[key] for key in ('image_id', 'gt_id', 'query_id')} for item in paired[row['image_id']]],
                        'Every captured pair must be the exact original fixed matched query/GT in order')
                z_all = immutable_selector(torch, gate, shared, index)
                union_check = np.zeros(z_all.shape[1] * z_all.shape[2], dtype=np.bool_)
                image_C = image_K = GT_bytes = 0
                for pair, fixed_pair in zip(packet['pairs'], paired[row['image_id']]):
                    deadline(started)
                    q = pair['identity']['query_id']
                    require(type(q) is int and 0 <= q < 100, 'Original natural query must be a nonboolean ID0..99')
                    native_binary = masks.decode(reader.rle_bytes(native_by_key[row['image_id'], q]['segmentation'])).astype(np.bool_)
                    require(np.array_equal(native_binary, (z_all[q] > 0).numpy()), 'Matched native RLE must exactly equal immutable GPU L sign')
                    original_binary = masks.decode(reader.rle_bytes(original_by_key[row['image_id'], q]['segmentation'])).astype(np.bool_)
                    replayed, domains = replay_pair(torch, np, cv2, head, state, shared, pair, z_all[q], fixed_pair,
                        gate['count_oracle'], gate['array_oracle'], started, process, budget['maximum_process_RSS_bytes'], original_binary)
                    require(domains['K'] == prior_rows[index]['band_counts_all100'][q],
                            'Independent original Omega K must equal frozen natural-query band population')
                    require(pair['native_counts'] == pixels(np, domains['native'], domains['GT']), 'Captured native counts must match independent mask accounting')
                    batch = gpu.overlap_check(torch, pair['GPU_C_residual'], pair['GPU_C_combined'],
                        pair['Omega_C_positions'].numpy(), pair['GPU_Omega_residual'], pair['GPU_Omega_combined'])
                    batch.update(original_Omega_full_binary_exact=True, original_Omega_refined_counts_exact=True)
                    require(batch == pair['batch_compatibility'] and batch['qualified'] is True,
                            'Independent saved GPU crossbatch numerical/strictsign/fullbinary evidence must exactly match')
                    diagnostic = independent_diagnostic(np, gpu, replayed, domains, pair, batch)
                    require(diagnostic == pair['diagnostic'] == evidence['diagnostics'][len(diagnostics)],
                            'CPU independent count/array/fullmask/rational pair diagnostic must exactly equal actual GPU')
                    diagnostics.append(diagnostic)
                    numeric_records.append(replayed)
                    union_check[pair['C_indices'].numpy()] = True
                    image_C += domains['C_count']
                    image_K += domains['K']
                    GT_bytes += pair['GT_packbits'].numel()
                    del pair, fixed_pair, native_binary, original_binary, replayed, domains, diagnostic, batch
                require(image_C == row['C_points'] and image_K == row['K_points'] and
                        torch.equal(torch.from_numpy(np.flatnonzero(union_check).astype(np.int64)), shared['union_flat_indices']) and
                        row['raw_tensor_payload_bytes'] == 1032 * row['U'] + 21 * image_C + 16 * image_K + 102400 + GT_bytes,
                        'Actual complete C union/everypair/payload accounting must be exact')
                completed = index + 1
                image_record = dict(dataset_index=index, image_id=row['image_id'], capture=row['capture'], matched_pairs=len(packet['pairs']),
                    C_points=image_C, Omega_K_points=image_K, both_original_PR_formulas_absolute_1e4=True,
                    all_C_and_Omega_strictsign_exact=True, original_Omega_fullbinary_exact=True, independent_oracle_masks_exact=True)
                line = json.dumps(image_record, allow_nan=False) + '\n'
                stream.write(line)
                stream.flush()
                data = line.encode('utf-8')
                trace_sha.update(data)
                trace_bytes += len(data)
                del packet, shared, z_all, union_check, image_record, line, data
                gc.collect()
                require(not torch.cuda.is_initialized() and process.memory_info().rss < budget['maximum_process_RSS_bytes'],
                        'CPU replay must retain cold CUDA and actual RSS limit')
                print(json.dumps(dict(N15_CPU_images_replayed=completed, matched_pairs=len(diagnostics))), flush=True)
        require(completed == 187 and len(diagnostics) == len(numeric_records) == 374, 'Complete187 captures/374 actual pairs required before Q')
        population = gpu.summarize(diagnostics, unmatched)
        require(population == evidence['population'], 'Exact same privileged374 rational distribution/three-gate decision required')
        state_check(torch, head, state)
        require(torch.equal(outside, torch.get_rng_state()) and not torch.cuda.is_initialized() and 'mmdet' not in sys.modules,
                'CPU frozen original weights/outside RNG/cold CUDA/no native model changed')
        for ref in spec['sources']:
            deadline(started)
            bound(ref)
        require(reference(path) == gate['spec_ref'] and reference(__file__) == gate['source'] and
                reference(spec['CPU_admission_file']) == gate['admission_ref'] and
                imported_files(gpu, gate, psutil, torch, np, cv2, masks, COCO, ndimage) == libraries and
                process.cpu_affinity() == [0, 1] and torch.get_num_threads() == torch.get_num_interop_threads() == 2,
                'Final independent CPU own Source/spec/admission/resources/libraries differ')
        gate['lineage'].final_endpoint_check(gate['parent'])
        gpu.validate_protocol(spec, closure)
        deadline(started)
        trace_ref = dict(path=str((output / 'TRACE.jsonl').resolve()), bytes=trace_bytes, sha256=trace_sha.hexdigest())
        pair_ref = write_new(output / 'PAIR_DIAGNOSTICS.json', diagnostics)
        population_ref = write_new(output / 'POPULATION.json', population)
        numeric_ref = write_new(output / 'SAME_INPUT_NUMERICAL_DETAILS.json', numeric_records)
        details_ref = write_new(output / 'DETAILS.json', dict(schema='N15-fixed-E24-action-opportunity-CPU-details-v1',
            source=gate['source'], spec=gate['spec_ref'], CPU_admission=gate['admission_ref'], phase=spec['phase'],
            diagnostic_GPU_report=evidence['refs']['report'], capture_manifest=evidence['report']['capture_manifest'],
            capture_refs=evidence['report']['capture_refs'], carrier_refs=evidence['report']['carrier_refs'],
            locked_protocol=spec['locked_protocol'], head_checkpoint=evidence['report']['head_checkpoint'], head_state_proofs=proofs,
            start=start_ref, trace=trace_ref, pair_diagnostics=pair_ref, population=population_ref, numeric_details=numeric_ref,
            CPU_resources=resource, CPU_replay_budget=budget, actual_imported_files=libraries,
            complete_current_Source_count=len(closure), normal_RAW_count=spec['normal_RAW_count'], preserved_failed_RAW_count=9,
            two_independent_PR_formulas=True, residual_combined_atol=1e-4, residual_combined_rtol=0.0,
            no_CPU_feature_rerender=True, no_original_GT_annotation_decoder=True, captured_GT_packbits_decoded=True,
            AP_evaluated=False, training_started=False,
            unmatched_unevaluable_excluded_from374_mean=True, elapsed_seconds=time.monotonic() - started,
            current_owned_CPU_terminal_self_certified=False))
        q = dict(schema='N15-fixed-E24-action-opportunity-same-GPU-input-CPU-qualification-v1',
            status='ROOT_N15_FIXED_E24_ACTION_OPPORTUNITY_SAME_GPU_INPUT_CPU_QUALIFIED_V1',
            source=gate['source'], spec=gate['spec_ref'], CPU_admission=gate['admission_ref'], phase=spec['phase'],
            diagnostic_GPU_refs=evidence['refs'], locked_protocol=spec['locked_protocol'], fixed_population=spec['fixed_population'],
            capture_manifest=evidence['report']['capture_manifest'], capture_refs=evidence['report']['capture_refs'], details=details_ref,
            carrier_CPU_qualification=spec['carrier_CPU_qualification'], carrier_GPU_report=spec['carrier_GPU_report'],
            head_checkpoint=evidence['report']['head_checkpoint'], carrier_seed=SEED, carrier_epoch=EPOCH, arm='LOG_PR', parameters=40497, parameter_tensors=8,
            images=187, matched_pairs=374, unmatched_GT=6, raw_GT=380, primary_denominator=374,
            all_actual_GPU_point_inputs_replayed=True, two_independent_original_PR_formulas_checked=True,
            C_and_Omega_residual_and_combined_absolute_1e4_rtol0_checked=True, strict_all_C_and_Omega_sign_exact=True,
            independent_C_domain_and_original_Omega_fullbinary_exact=True, independent_count_array_mask_oracles_exact=True,
            privileged_pair_diagnostics_and_rational_GO_gate_exact=True, six_unmatched_unevaluable_no_zero_imputation=True,
            GPU_provisional_diagnostic_decision=population['primary_B']['diagnostic_decision'], population=population_ref,
            same_GPU_input_numerical_qualified=True, current_owned_CPU_terminal_self_certified=False,
            CPU_renderer_qualified=False, AP_evaluated=False, training_started=False, no_TEST=True, no_cross_dataset=True,
            selector_learnability_established=False, task_efficacy_established=False, innovation_established=False,
            normal_RAW_count=spec['normal_RAW_count'], preserved_failed_RAW_count=9, total_terminal_identity_pin_count=spec['normal_RAW_count'] + 9,
            synthetic_validation_is_actual_science_qualification=False,
            evidence_limits='Independent replay of actual captured GPU F/e/L points for strongest701 E24 LOG PR, original374 fixed native.5 pairs plus6 unevaluable. '
                'Absolute1e-4 with rtol0 and strict C/Omega signs; count/array/mask and privileged opportunity distribution exact. '
                'No CPU renderer, fullAP, training, TEST/cross-data, learnable selector, efficacy/innovation/TGRS or general method failure conclusion. '
                'ROOT must separately certify this original CPU terminal/reap/birth absence; synthetic tests never confer actual qualification.')
        q_size = len((json.dumps(q, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8'))
        require(sum(item.stat().st_size for item in output.iterdir() if item.is_file()) + q_size <= budget['auxiliary_output_budget_bytes'],
                'New CPU auxiliary output exceeds prelocked budget')
        q_ref = write_new(output / 'QUALIFICATION.json', q)
        print(json.dumps(dict(status=q['status'], qualification=q_ref, details=details_ref,
            current_owned_CPU_terminal_self_certified=False)), flush=True)
        return q_ref
    except Exception as error:
        status = ('NUMERICAL_QUALIFICATION_FAILURE' if isinstance(error, ComputationalPortabilityFailure)
                  else 'CPU_PROTOCOL_OR_EXECUTION_QUALIFICATION_FAILURE')
        failure_ref = write_new(output / 'FAILURE.json', dict(schema='N15-fixed-E24-action-opportunity-CPU-failure-v1',
            status=status, source=gate['source'], spec=gate['spec_ref'], CPU_admission=gate['admission_ref'], phase=spec['phase'],
            diagnostic_GPU_report=evidence['refs']['report'], locked_protocol=spec['locked_protocol'], completed_images=completed,
            completed_pairs=len(diagnostics), error_type=type(error).__name__, error=str(error),
            complete_same_GPU_input_qualified=False, scientific_STOP=False, GO_evaluated_by_failed_attempt=False,
            posterior_tolerance_relaxed=False, automatic_retry=False, method_failure_inferred=False,
            preserved_failed_RAW_count=9, current_owned_CPU_terminal_self_certified=False))
        print(json.dumps(dict(status=status, failure=failure_ref, complete_same_GPU_input_qualified=False)), flush=True)
        raise
    finally:
        heads.clear()
        gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args()
    require(args.owner_execute, 'UNBOUND_DRAFT: ROOT must close/audit completeGPU and bind one fresh independent CPU numerical attempt')
    run(args.spec)

