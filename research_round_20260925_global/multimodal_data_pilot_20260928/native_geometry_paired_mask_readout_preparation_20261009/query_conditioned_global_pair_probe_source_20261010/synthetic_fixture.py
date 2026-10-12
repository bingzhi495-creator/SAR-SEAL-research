"""Inert owner-gated CPU synthetic fixture; authoring is not qualification.

Only a pinned SOURCE-only spec and four explicit Python SOURCE files are read.
No GT, checkpoints, predictions, query artifacts, receipts, images or loaders.
REAL inherited rectangle geometry checks original E; injected mixed E tests
the readout algebra only. Neither path establishes efficacy/native integration.
Torch is imported only after --owner-execute and CPU environment/affinity gates.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import types


SCHEMA = 'query_conditioned_global_pair_probe_source_only_spec_v1'
MODES = ('raw', 'raw_rms', 'query_rms', 'permuted_query_rms')
ATOL, RTOL = 3e-5, 3e-5


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def pinned_sources(spec):
    """Resolve only predetermined SOURCE paths and verify their exact bytes."""
    own = Path(__file__).resolve(strict=True)
    expected = dict(fixture=own, operator=own.parent / 'readout.py',
        geometry=own.parent.parent / 'paired_context.py',
        old_spatial=own.parent.parent / 'spatial_position_preserving_readout_source_20261009' / 'readout.py')
    require(set(spec) == {'schema', 'inputs', 'sources', 'execution'} and
        spec['schema'] == SCHEMA and spec['inputs'] == 'SOURCE_ONLY', 'SOURCE-only spec schema differs')
    require(spec['execution'] == dict(device='cpu', cpu_affinity=[0, 1], threads=2,
        cuda_visible_devices=''), 'Explicit CPU-only execution contract differs')
    require(set(spec['sources']) == set(expected), 'Exactly four pinned SOURCE references required')
    contents = {}
    for key, expected_path in expected.items():
        ref = spec['sources'][key]
        require(set(ref) == {'path', 'bytes', 'sha256'} and Path(ref['path']).is_absolute(),
            'Each SOURCE reference requires an absolute path, bytes and sha256')
        path = Path(ref['path']).resolve(strict=True)
        require(path == expected_path.resolve(strict=True) and path.suffix == '.py',
            'SOURCE path differs: ' + key)
        data = path.read_bytes()
        require(type(ref['bytes']) is int and len(data) == ref['bytes'] and
            hashlib.sha256(data).hexdigest() == ref['sha256'], 'Pinned SOURCE bytes differ: ' + key)
        contents[key] = (path, data)
    return contents


def load_pinned(item, name):
    # Execute already-verified bytes; do not reread/import a path or write pyc.
    path, data = item
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(data, str(path), 'exec'), module.__dict__)
    return module


def cpu_gate():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
    if os.name == 'nt':
        import ctypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        kernel.GetProcessAffinityMask.argtypes = (ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t))
        process = kernel.GetCurrentProcess()
        require(kernel.SetProcessAffinityMask(process, 3) != 0, 'CPU01 affinity failed')
        actual, system = ctypes.c_size_t(), ctypes.c_size_t()
        require(kernel.GetProcessAffinityMask(process, ctypes.byref(actual), ctypes.byref(system)) != 0
            and actual.value == 3, 'CPU01 affinity was not retained')
    else:
        require(hasattr(os, 'sched_setaffinity'), 'Exact CPU01 affinity unsupported')
        os.sched_setaffinity(0, {0, 1})
        require(os.sched_getaffinity(0) == {0, 1}, 'CPU01 affinity was not retained')


def scalar_bilinear(channel, x, y):
    H, W = len(channel), len(channel[0])
    x0, y0 = math.floor(x), math.floor(y)
    dx, dy = x - x0, y - y0
    return math.fsum(w * channel[yy][xx] if 0 <= xx < W and 0 <= yy < H else 0.0
        for xx, yy, w in ((x0, y0, (1-dx)*(1-dy)), (x0+1, y0, dx*(1-dy)),
                         (x0, y0+1, (1-dx)*dy), (x0+1, y0+1, dx*dy)))


def scalar_normalize(values):
    scale = max(abs(v) for v in values)
    if scale == 0:
        return [0.0] * len(values)
    scaled = [v / scale for v in values]
    denominator = math.sqrt(math.fsum(v*v for v in scaled))
    return [16.0 * v / denominator for v in scaled]


def calibrate(torch, adapter):
    with torch.no_grad():
        adapter.hidden.weight.zero_()
        adapter.final.weight.zero_()
        adapter.hidden.weight[0, 0], adapter.hidden.weight[0, 256] = 0.7, -0.35
        adapter.hidden.weight[1, 1], adapter.hidden.weight[1, 257] = -0.5, 0.4
        adapter.final.weight[0, 0], adapter.final.weight[0, 1] = 0.6, -0.15
        adapter.final.weight[1, 0], adapter.final.weight[1, 1] = 0.25, 0.3


def injected_geometry(torch):
    t = [-1.5 + 3*j/7 for j in range(8)]
    inside = [[4+2*v, 4+0.5*s] for v in t for s in (-1, 1)]
    outside = [[4+2*v, 4+2.5*s] for v in t for s in (-1, 1)]
    inside[0] = outside[0] = [0.0, 0.0]  # An exact all-zero sampled vector.
    selections = [(0, 1, 10, 11, 12, 13, 14, 15), (), (2, 3, 4), tuple(range(16)),
                  (0,), (5,), (1, 9), (8, 10, 12), (2, 14), (3, 6, 11, 15)]
    E = torch.tensor([[[j in slots for j in range(16)] for slots in selections]], dtype=torch.bool)
    Q = len(selections)
    return dict(inside=torch.tensor([[inside]*Q], dtype=torch.float32),
        outside=torch.tensor([[outside]*Q], dtype=torch.float32), eligible=E,
        eligible_counts=E.sum(-1), canonical_eligible_counts=E.sum(-1),
        rho_mass=torch.ones((1, Q)), canonical_theta=torch.zeros((1, Q)),
        used_theta=torch.zeros((1, Q)), sigma_major=torch.full((1, Q), 2.0),
        sigma_minor=torch.ones((1, Q)), centre_xy=torch.tensor([[[4.0, 4.0]]*Q]),
        near_isotropic=torch.zeros((1, Q), dtype=torch.bool),
        degenerate=torch.zeros((1, Q), dtype=torch.bool))


def scalar_reference(torch, F, e0, geometry, mode, permutation):
    """Independent scalar bilinear, calibrated two-unit SiLU and eligible mean."""
    channels, queries = F.detach()[0, :2].tolist(), e0.detach()[0].tolist()
    inside, outside, E = (geometry[k][0].tolist() for k in ('inside', 'outside', 'eligible'))
    means, residuals = [], []
    for q, query in enumerate(queries):
        conditioning = [query[j] for j in permutation] if mode == 'permuted_query_rms' else query
        slots = []
        for xy, other in zip(inside[q], outside[q]):
            halves = [[scalar_bilinear(channel, *point) for channel in channels] + [0.0]*254
                      for point in (xy, other)]
            if mode in ('query_rms', 'permuted_query_rms'):
                halves = [[v*c for v, c in zip(half, conditioning)] for half in halves]
            if mode != 'raw':
                halves = [scalar_normalize(half) for half in halves]
            a, b = halves
            v0, v1 = 0.7*a[0]-0.35*b[0], -0.5*a[1]+0.4*b[1]
            h0, h1 = v0/(1+math.exp(-v0)), v1/(1+math.exp(-v1))
            slots.append((0.6*h0-0.15*h1, 0.25*h0+0.3*h1))
        selected = [j for j, yes in enumerate(E[q]) if yes]
        d = [math.fsum(slots[j][c] for j in selected)/len(selected) if selected else 0.0
             for c in (0, 1)]
        means.append([d+[0.0]*254, d+[0.0]*254])
        residuals.append([[d[0]*channels[0][y][x]+d[1]*channels[1][y][x]
            for x in range(F.shape[-1])] for y in range(F.shape[-2])])
    return torch.tensor([means], dtype=torch.float32), torch.tensor([residuals], dtype=torch.float32)


def normalization_checks(torch, adapter):
    base = torch.tensor([[-2.0, 0.25, 3.0, 0.0]*64], dtype=torch.float32)
    rows = torch.cat((base, base*1e-30, base*1e30, -base, torch.zeros_like(base))).requires_grad_()
    normalized = adapter._normalize_channels(rows)
    require(torch.isfinite(normalized).all().item() and
        torch.allclose(normalized[:3], normalized[:1].expand(3, -1), atol=2e-6, rtol=2e-6) and
        torch.allclose(normalized[3], -normalized[0], atol=2e-6, rtol=2e-6) and
        torch.allclose(normalized[:4].square().mean(-1), torch.ones(4), atol=2e-6, rtol=2e-6) and
        torch.count_nonzero(normalized[4]).item() == 0, 'RMS scale/sign/zero contract failed')
    upstream = torch.arange(256, dtype=torch.float32).remainder(7).reshape(1, -1)
    grad = torch.autograd.grad((normalized*upstream).sum(), rows)[0]
    require(torch.isfinite(grad).all().item() and torch.count_nonzero(grad[4]).item() == 0,
        'Normalizer gradients are not finite with a connected zero derivative')


def execute(torch, operator, geometry_source, old_source):
    rng_before = torch.random.get_rng_state().clone()
    require(operator.MODES == MODES, 'Four explicit modes required')
    adapters = {m: operator.build_adapter(torch, geometry_source, mode=m) for m in MODES}
    altered = {m: operator.build_adapter(torch, geometry_source, mode=m, hidden_seed=2026100912) for m in MODES}
    require(torch.equal(rng_before, torch.random.get_rng_state()), 'Factory changed external CPU RNG')
    reference, other = adapters['raw'].state_dict(), altered['raw'].state_dict()
    for group, expected in ((adapters, reference), (altered, other)):
        for mode, adapter in group.items():
            require(adapter.mode == mode and set(adapter.state_dict()) == {'hidden.weight', 'final.weight'}
                and not list(adapter.named_buffers()) and sum(p.numel() for p in adapter.parameters()) == 393216
                and tuple(adapter.hidden.weight.shape) == (512, 512)
                and tuple(adapter.final.weight.shape) == (256, 512)
                and adapter.hidden.bias is None and adapter.final.bias is None
                and all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in adapter.parameters())
                and torch.count_nonzero(adapter.final.weight).item() == 0
                and all(torch.equal(v, expected[k]) for k, v in adapter.state_dict().items()),
                'Paired cold weights/state contract differs: '+mode)
    require(not torch.equal(reference['hidden.weight'], other['hidden.weight']), 'Hidden seed override ineffective')
    try:
        operator.build_adapter(torch, geometry_source, mode='query')
    except ValueError:
        pass
    else:
        raise RuntimeError('Unknown mode was silently accepted')
    P = adapters['raw'].permutation_indices
    require(type(P) is tuple and len(P) == 256 and sorted(P) == list(range(256))
        and all(i != p for i, p in enumerate(P)) and
        all(a.permutation_indices == P for a in adapters.values()), 'Permutation is not a fixed derangement/bijection')
    visited, cursor = set(), 0
    while cursor not in visited:
        visited.add(cursor)
        cursor = P[cursor]
    require(cursor == 0 and len(visited) == 256, 'Permutation is not one 256-cycle')
    e_values = torch.tensor([[(c-127.5)/129 for c in range(256)]], dtype=torch.float32)
    permuted = e_values.index_select(-1, torch.tensor(P, dtype=torch.long))
    require(torch.equal(e_values.sort(-1).values, permuted.sort(-1).values) and
        torch.allclose(e_values.norm(dim=-1), permuted.norm(dim=-1), atol=1e-6, rtol=1e-6),
        'Permutation changed e0 multiset/norm')
    normalization_checks(torch, adapters['raw_rms'])

    # REAL original geometry: 9x3 positive rectangle, centre(8,4), xx=20/3.
    F = torch.zeros((1, 256, 9, 17), dtype=torch.float32)
    F[:, 0].fill_(-1)
    F[:, 0, 3:6, 4:13] = 1
    F.requires_grad_()
    e0 = torch.zeros((1, 1, 256), dtype=torch.float32)
    e0[:, :, 0] = 4
    # Unused F channels stay zero, so native logits/REAL geometry are unchanged.
    # Nonzero e0 there keeps the permuted-input final-weight path observable.
    e0[:, :, 1:] = 0.5
    e0.requires_grad_()
    valid = torch.ones((1, 1, 9, 17), dtype=torch.float32)
    native = torch.einsum('bqc,bchw->bqhw', e0, F)
    for mode, adapter in adapters.items():
        result = adapter(F, e0, valid, native_logits=native)
        g = result['diagnostics']
        require(torch.equal(result['mask_logits'].view(torch.int32), native.view(torch.int32)) and
            torch.count_nonzero(result['logit_residual']).item() == 0, 'Cold native bit identity failed: '+mode)
        require(g['eligible'].dtype == torch.bool and g['eligible_counts'].tolist() == [[16]]
            and torch.allclose(g['centre_xy'], torch.tensor([[[8.0, 4.0]]]), atol=1e-5, rtol=0)
            and torch.allclose(g['sigma_major'], torch.tensor([[math.sqrt(20/3)]]), atol=1e-5, rtol=0)
            and torch.equal(g['sigma_minor'], torch.ones((1, 1)))
            and torch.allclose(g['used_theta'], torch.zeros((1, 1)), atol=1e-6, rtol=0)
            and all(not v.requires_grad for v in g.values() if torch.is_tensor(v)), 'REAL original geometry/E failed')
        grads = torch.autograd.grad(result['mask_logits'].sum(),
            (F, e0, adapter.hidden.weight, adapter.final.weight), retain_graph=True)
        require(all(torch.isfinite(v).all().item() for v in grads) and grads[0].norm().item() > 0
            and grads[1].norm().item() > 0 and grads[2].norm().item() == 0 and grads[3].norm().item() > 0,
            'Cold native/final gradient paths failed: '+mode)
        zero_e = torch.zeros((1, 2, 256), dtype=torch.float32, requires_grad=True)
        empty = adapter(F, zero_e, valid)
        empty_grads = torch.autograd.grad(empty['logit_residual'].sum(),
            (F, adapter.hidden.weight, adapter.final.weight))
        require(empty['diagnostics']['eligible_counts'].tolist() == [[0, 0]] and
            tuple(empty['mask_logits'].shape) == (1, 2, 9, 17) and
            all(v is not None and torch.isfinite(v).all().item() and torch.count_nonzero(v).item() == 0
                for v in empty_grads), 'REAL all-E0 connected zero failed')

    # INJECTED E is intentionally independent of the original sign predicate.
    F = torch.zeros((1, 256, 9, 9), dtype=torch.float32)
    for y in range(9):
        for x in range(9):
            F[0, 0, y, x], F[0, 1, y, x] = 0.5+0.07*x+0.03*y+0.01*x*y, -0.3+0.04*x-0.08*y
    F[0, :2, 0, 0] = 0
    F.requires_grad_()
    e0 = torch.tensor([[[0.08+0.003*c+0.01*q for c in range(256)] for q in range(10)]],
        dtype=torch.float32, requires_grad=True)
    valid, g = torch.ones((1, 1, 9, 9), dtype=torch.float32), injected_geometry(torch)
    errors, gradient_norms = {}, {}
    for mode, adapter in adapters.items():
        calibrate(torch, adapter)
        adapter._geometry = lambda _F, _e, _v, fixture=g: fixture
        native = torch.einsum('bqc,bchw->bqhw', e0, F)
        result = adapter(F, e0, valid, native_logits=native)
        fields, residual = scalar_reference(torch, F, e0, g, mode, P)
        require(tuple(result['coefficient_fields'].shape) == (1, 10, 2, 256) and
            torch.allclose(result['coefficient_fields'], fields, atol=ATOL, rtol=RTOL) and
            torch.allclose(result['logit_residual'], residual, atol=ATOL, rtol=RTOL) and
            torch.allclose(result['mask_logits'], native.detach()+residual, atol=ATOL, rtol=RTOL),
            'Independent scalar global-pool formula failed: '+mode)
        require(torch.equal(result['coefficient_fields'][:, :, 0], result['coefficient_fields'][:, :, 1])
            and torch.count_nonzero(result['logit_residual'][:, 1]).item() == 0,
            'Global fields differ or an empty query became nonzero')
        upstream = (torch.arange(native.numel(), dtype=torch.float32).reshape_as(native).remainder(7)-3)/11
        aux_grad = torch.autograd.grad((result['logit_residual']*upstream).sum(), e0,
            allow_unused=True, retain_graph=True)[0]
        full_grads = torch.autograd.grad((result['mask_logits']*upstream).sum(),
            (F, e0, adapter.hidden.weight, adapter.final.weight), retain_graph=True)
        native_grad = torch.autograd.grad((native*upstream).sum(), e0, retain_graph=True)[0]
        require(aux_grad is None and torch.equal(full_grads[1], native_grad) and
            all(torch.isfinite(v).all().item() and v.norm().item() > 0 for v in full_grads),
            'Auxiliary e0 detach/native-live/F/two-weight gradients failed: '+mode)
        errors[mode] = dict(fields=(result['coefficient_fields']-fields).abs().max().item(),
            residual=(result['logit_residual']-residual).abs().max().item())
        gradient_norms[mode] = [v.norm().item() for v in full_grads]
        empty_g = dict(g, eligible=torch.zeros_like(g['eligible']), eligible_counts=torch.zeros_like(g['eligible_counts']))
        adapter._geometry = lambda _F, _e, _v, fixture=empty_g: fixture
        empty = adapter(F, e0, valid)
        grads = torch.autograd.grad(empty['logit_residual'].sum(), (F, adapter.hidden.weight, adapter.final.weight))
        require(torch.count_nonzero(empty['logit_residual']).item() == 0 and
            all(v is not None and torch.isfinite(v).all().item() and torch.count_nonzero(v).item() == 0
                for v in grads), 'Calibrated all-E0 lost its connected zero')
        adapter._geometry = lambda _F, _e, _v, fixture=g: fixture

    old = old_source.build_adapter(torch, geometry_source, mode='global')
    old.load_state_dict(adapters['raw'].state_dict(), strict=True)
    old._geometry = lambda _F, _e, _v: g
    old_F, old_e = F.detach().clone().requires_grad_(), e0.detach().clone().requires_grad_()
    new_F, new_e = F.detach().clone().requires_grad_(), e0.detach().clone().requires_grad_()
    old_result, new_result = old(old_F, old_e, valid), adapters['raw'](new_F, new_e, valid)
    for key in ('mask_logits', 'logit_residual', 'coefficient_fields'):
        require(torch.equal(old_result[key], new_result[key]), 'Raw original SPATIALGLOBAL output differs: '+key)
    upstream = (torch.arange(old_result['mask_logits'].numel(), dtype=torch.float32).reshape_as(
        old_result['mask_logits']).remainder(7)-3)/11
    old_grads = torch.autograd.grad((old_result['mask_logits']*upstream).sum(),
        (old_F, old_e, old.hidden.weight, old.final.weight))
    new_grads = torch.autograd.grad((new_result['mask_logits']*upstream).sum(),
        (new_F, new_e, adapters['raw'].hidden.weight, adapters['raw'].final.weight))
    require(all(torch.allclose(a, b, atol=1e-4, rtol=1e-4) for a, b in zip(old_grads, new_grads)),
        'Raw original SPATIALGLOBAL conditional gradients differ')
    require(torch.equal(rng_before, torch.random.get_rng_state()) and not torch.cuda.is_initialized(),
        'Fixture changed external CPU RNG or initialized CUDA')
    return dict(status='QUERY_CONDITIONED_GLOBAL_PAIR_SYNTHETIC_CPU_PASS',
        synthetic_execution_performed=True, native_integration_qualified=False, efficacy_claimed=False,
        input_scope='Pinned SOURCE files and in-memory hand-specified synthetic tensors only',
        real_original_rectangle_E=True, injected_E_is_not_original_predicate_evidence=True,
        all_four_cold_native_bit_identity=True, seeded_state_and_rng_restoration=True,
        permutation_cycle_derangement_multiset_norm=True, normalization_scale_sign_zero_gradients=True,
        scalar_reference_maxabs=errors, calibrated_gradient_norms=gradient_norms,
        auxiliary_e0_detached_native_e0_live=True, all_E0_connected_zero=True,
        raw_original_spatial_global_bit_equal=True,
        raw_original_spatial_global_gradient_maxabs=[(a-b).abs().max().item() for a,b in zip(old_grads,new_grads)],
        tolerances=dict(reference_atol=ATOL, reference_rtol=RTOL, gradient_atol=1e-4, gradient_rtol=1e-4))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args()
    require(args.owner_execute, 'SOURCE preparation is inert; explicit owner execution required')
    require(Path(args.spec).is_absolute() and Path(args.out).is_absolute(), 'Spec/output paths must be absolute')
    spec_path, out = Path(args.spec).resolve(strict=True), Path(args.out).resolve()
    require(spec_path.suffix == '.json' and not out.exists() and out.parent.is_dir(), 'Spec/output path contract differs')
    spec = json.loads(spec_path.read_text(encoding='utf-8-sig'))
    sources = pinned_sources(spec)
    cpu_gate()
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    require(torch.get_num_threads() == 2 and torch.get_num_interop_threads() == 2
        and not torch.cuda.is_initialized(), 'CPU-only Torch gate failed')
    geometry = load_pinned(sources['geometry'], 'synthetic_original_geometry')
    operator = load_pinned(sources['operator'], 'synthetic_query_conditioned_readout')
    old = load_pinned(sources['old_spatial'], 'synthetic_original_spatial_readout')
    report = execute(torch, operator, geometry, old)
    report.update(sources=spec['sources'], execution=spec['execution'], spec_path=str(spec_path))
    with out.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(status=report['status'], out=str(out))))


if __name__ == '__main__':
    main()
