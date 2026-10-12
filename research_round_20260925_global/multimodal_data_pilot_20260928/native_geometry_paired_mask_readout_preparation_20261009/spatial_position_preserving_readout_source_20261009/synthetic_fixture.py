"""Owner-executed CPU synthetic checks; not native training or efficacy.

SOURCE creation/AST compilation does not execute these checks. CLI requires
--source-root (this new directory), --geometry-source (original paired_context
SOURCE), --out (fresh JSON), and --owner-execute. Imports of Torch occur only
after CPU environment/affinity setup. No model, CP, GT, images or data loader.

Two distinct paths are checked: original real geometry/sampling on a hand-
specified 9x17 rectangle, and an explicitly injected four-query geometry/E
fixture for unequal halves, empty/single-half support and axis reversal.
The latter does NOT validate the original E sign predicate. Scalar bilinear,
SiLU and two-field formulas independently reference the injected fixture.
The global comparison uses old paired forward and compares FP32 outputs and
conditional gradients with declared tolerances; it is not a bitwise claim.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_source(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None, 'Source loader unavailable')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_ref(path):
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def scalar_bilinear(channel, x, y):
    """Four-neighbour zero padding; independent list-index reference."""
    H, W = len(channel), len(channel[0])
    x0, y0 = math.floor(x), math.floor(y)
    dx, dy = x - x0, y - y0
    terms = []
    for xx, yy, w in ((x0, y0, (1-dx)*(1-dy)), (x0+1, y0, dx*(1-dy)),
                      (x0, y0+1, (1-dx)*dy), (x0+1, y0+1, dx*dy)):
        terms.append(w * channel[yy][xx] if 0 <= xx < W and 0 <= yy < H else 0.0)
    return math.fsum(terms)


def scalar_phi(fin, fout):
    """The explicitly calibrated two active hidden units, not module calls."""
    v0 = 0.7 * fin[0] - 0.35 * fout[0]
    v1 = -0.5 * fin[1] + 0.4 * fout[1]
    h0, h1 = v0 / (1 + math.exp(-v0)), v1 / (1 + math.exp(-v1))
    return (0.6 * h0 - 0.15 * h1, 0.25 * h0 + 0.3 * h1)


def calibrated_weights(torch, adapter):
    with torch.no_grad():
        adapter.hidden.weight.zero_()
        adapter.final.weight.zero_()
        adapter.hidden.weight[0, 0], adapter.hidden.weight[0, 256] = 0.7, -0.35
        adapter.hidden.weight[1, 1], adapter.hidden.weight[1, 257] = -0.5, 0.4
        adapter.final.weight[0, 0], adapter.final.weight[0, 1] = 0.6, -0.15
        adapter.final.weight[1, 0], adapter.final.weight[1, 1] = 0.25, 0.3


def mock_geometry(torch, geometry_source, Q=4):
    # Same original t/side coordinates, but E is injected and labelled as such.
    t = [-1.5 + 3.0 * j / 7.0 for j in range(8)]
    inside = [[4.0 + 2.0 * value, 4.0 + 0.5 * side]
              for value in t for side in (-1.0, 1.0)]
    outside = [[4.0 + 2.0 * value, 4.0 + 2.5 * side]
               for value in t for side in (-1.0, 1.0)]
    E_list = [[j in (0, 1, 10, 11, 12, 13, 14, 15) for j in range(16)],
              [False] * 16, [j in (2, 3, 4) for j in range(16)], [True] * 16]
    require(Q == len(E_list), 'Fixed four-query fixture only')
    E = torch.tensor([E_list], dtype=torch.bool, device='cpu')
    counts = E.sum(dim=-1)
    return dict(inside=torch.tensor([[inside] * Q], dtype=torch.float32),
        outside=torch.tensor([[outside] * Q], dtype=torch.float32), eligible=E,
        eligible_counts=counts, canonical_eligible_counts=counts,
        rho_mass=torch.ones((1, Q)), canonical_theta=torch.zeros((1, Q)),
        used_theta=torch.zeros((1, Q)), sigma_major=torch.full((1, Q), 2.0),
        sigma_minor=torch.ones((1, Q)), centre_xy=torch.tensor([[[4.0, 4.0]] * Q]),
        near_isotropic=torch.zeros((1, Q), dtype=torch.bool),
        degenerate=torch.zeros((1, Q), dtype=torch.bool))


def scalar_expected(torch, F, e0, geometry, mode):
    # Only two F channels are nonzero. All scalar arithmetic is Python double.
    channel0, channel1 = F.detach()[0, :2].tolist()
    in_xy, out_xy = geometry['inside'][0].tolist(), geometry['outside'][0].tolist()
    E = geometry['eligible'][0].tolist()
    H, W = F.shape[-2:]
    output, globals_list, halves_list = [], [], []
    for q in range(len(E)):
        phi = [scalar_phi([scalar_bilinear(channel0, *xy), scalar_bilinear(channel1, *xy)],
                          [scalar_bilinear(channel0, *other), scalar_bilinear(channel1, *other)])
               for xy, other in zip(in_xy[q], out_xy[q])]
        selected = [j for j, yes in enumerate(E[q]) if yes]
        neg, pos = [j for j in selected if j < 8], [j for j in selected if j >= 8]
        def pool(indices):
            return tuple(math.fsum(phi[j][c] for j in indices) / len(indices) for c in (0, 1)) \
                if indices else (0.0, 0.0)
        global_mean = pool(selected)
        nmean, pmean = (pool(neg), pool(pos)) if neg and pos else (global_mean, global_mean)
        globals_list.append(global_mean)
        halves_list.append((nmean, pmean))
        if mode == 'spatial_reversed':
            nmean, pmean = pmean, nmean
        row = []
        for y in range(H):
            values = []
            for x in range(W):
                f0, f1 = channel0[y][x], channel1[y][x]
                base = float(e0.detach()[0, q, 0]) * f0 + float(e0.detach()[0, q, 1]) * f1
                if mode == 'global':
                    residual = global_mean[0] * f0 + global_mean[1] * f1
                else:
                    alpha = 1.0 / (1.0 + math.exp(-(x - 4.0) / 2.0))
                    residual = (1-alpha) * (nmean[0]*f0+nmean[1]*f1) + \
                               alpha * (pmean[0]*f0+pmean[1]*f1)
                values.append(base + residual)
            row.append(values)
        output.append(row)
    return torch.tensor([output], dtype=torch.float32), globals_list, halves_list


def execute(torch, operator, geometry_source):
    cpu_before = torch.random.get_rng_state().clone()
    adapters = {mode: operator.build_adapter(torch, geometry_source, mode=mode) for mode in operator.MODES}
    require(torch.equal(cpu_before, torch.random.get_rng_state()), 'Factory leaked CPU RNG')
    for mode, adapter in adapters.items():
        require(adapter.mode == mode and sum(p.numel() for p in adapter.parameters()) == 393216 and
            set(dict(adapter.named_parameters())) == {'hidden.weight', 'final.weight'} and
            len(adapter.state_dict()) == 2 and not list(adapter.named_buffers()) and
            tuple(adapter.hidden.weight.shape) == (512, 512) and
            tuple(adapter.final.weight.shape) == (256, 512) and
            adapter.hidden.bias is None and adapter.final.bias is None and
            torch.count_nonzero(adapter.final.weight).item() == 0, 'Cold state contract differs')
    require(all(torch.equal(adapters['global'].hidden.weight, value.hidden.weight)
        for value in adapters.values()), 'Modes have different cold hidden weights')

    # REAL inherited path: a 9x3 rectangle has centre(8,4), xx=20/3, yy=2/3.
    # sigma_minor floors to1; every sampled inside/outside pair passes E.
    real_F = torch.zeros((1, 256, 9, 17), dtype=torch.float32)
    real_F[:, 0].fill_(-1.0)
    real_F[:, 0, 3:6, 4:13] = 1.0
    real_F.requires_grad_()
    real_e = torch.zeros((1, 1, 256), dtype=torch.float32)
    real_e[:, :, 0] = 4.0
    real_e.requires_grad_()
    real_valid = torch.ones((1, 1, 9, 17), dtype=torch.float32)
    native = torch.einsum('bqc,bchw->bqhw', real_e, real_F)
    real_rows = []
    for mode, adapter in adapters.items():
        result = adapter(real_F, real_e, real_valid, native_logits=native)
        g = result['diagnostics']
        require(torch.equal(result['mask_logits'], native) and
            torch.count_nonzero(result['logit_residual']).item() == 0, 'Cold identity failed')
        require(g['eligible_counts'].tolist() == [[16]] and
            torch.allclose(g['centre_xy'], torch.tensor([[[8.0, 4.0]]]), atol=1e-5, rtol=0) and
            torch.allclose(g['sigma_major'], torch.tensor([[math.sqrt(20.0/3.0)]]), atol=1e-5, rtol=0) and
            torch.equal(g['sigma_minor'], torch.ones((1, 1))) and
            torch.allclose(g['used_theta'], torch.zeros((1, 1)), atol=1e-6, rtol=0), 'Real hand-derived geometry failed')
        require(all(not value.requires_grad for value in g.values() if torch.is_tensor(value)) and
            not adapter._alpha(g, real_F, 0, 1).requires_grad, 'Geometry/alpha acquired a gradient')
        grads = torch.autograd.grad(result['mask_logits'].sum(),
            (real_F, real_e, adapter.hidden.weight, adapter.final.weight), retain_graph=True)
        require(all(torch.isfinite(value).all().item() for value in grads) and
            grads[0].norm().item() > 0 and grads[1].norm().item() > 0 and
            grads[2].norm().item() == 0 and grads[3].norm().item() > 0,
            'Cold paths must reach F/e0/final, with hidden gradient exactly zero')
        real_rows.append(dict(mode=mode, eligible_pairs=16, cold_identity=True))

    # REAL E0 path: zero e0 makes rho=0. All queries are retained and zeros are
    # connected; differentiating the residual returns zero tensors, not None.
    zero_e = torch.zeros((1, 2, 256), dtype=torch.float32, requires_grad=True)
    empty = adapters['spatial_halves'](real_F, zero_e, real_valid)
    require(empty['diagnostics']['eligible_counts'].tolist() == [[0, 0]] and
        torch.count_nonzero(empty['logit_residual']).item() == 0 and
        tuple(empty['mask_logits'].shape) == (1, 2, 9, 17), 'Real E0 was dropped or nonzero')
    empty_grads = torch.autograd.grad(empty['logit_residual'].sum(),
        (real_F, adapters['spatial_halves'].hidden.weight, adapters['spatial_halves'].final.weight))
    require(all(value is not None and torch.count_nonzero(value).item() == 0 for value in empty_grads),
            'E0 residual lost its connected zero graph')

    # INJECTED geometry/E: no E predicate claim, real bilinear sampling remains.
    F = torch.zeros((1, 256, 9, 9), dtype=torch.float32)
    for y in range(9):
        for x in range(9):
            F[0, 0, y, x] = 0.5 + 0.07*x + 0.03*y + 0.01*x*y
            F[0, 1, y, x] = -0.3 + 0.04*x - 0.08*y
    F.requires_grad_()
    e0 = torch.zeros((1, 4, 256), dtype=torch.float32)
    e0[:, :, 0], e0[:, :, 1] = 0.13, -0.2
    e0.requires_grad_()
    valid = torch.ones((1, 1, 9, 9), dtype=torch.float32)
    g = mock_geometry(torch, geometry_source)
    outputs, manual_maxabs, calibrated_gradient_norms = {}, {}, {}
    for mode, adapter in adapters.items():
        calibrated_weights(torch, adapter)
        adapter._geometry = lambda _F, _e, _v, fixture=g: fixture
        result = adapter(F, e0, valid)
        expected, global_means, half_means = scalar_expected(torch, F, e0, g, mode)
        maxabs = (result['mask_logits'] - expected).abs().max().item()
        require(torch.allclose(result['mask_logits'], expected, atol=3e-6, rtol=3e-5),
                'Independent scalar two-field formula failed: '+mode)
        require(result['diagnostics']['half_eligible_counts'].tolist() == [[[2, 6], [0, 0], [3, 0], [8, 8]]] and
            result['diagnostics']['global_fallback'].tolist() == [[False, True, True, False]],
            'Half counts or global fallback differ')
        require(torch.equal(result['coefficient_fields'][:, 1], torch.zeros((1, 2, 256))) and
            torch.equal(result['coefficient_fields'][:, 2, 0], result['coefficient_fields'][:, 2, 1]),
            'Empty or single-half fallback differs')
        require(abs(global_means[0][0] - (half_means[0][0][0]+half_means[0][1][0])/2.0) > 1e-6,
            'Unequal-half fixture does not distinguish global eligible mean from half average')
        upstream = (torch.arange(result['mask_logits'].numel(), dtype=torch.float32).reshape_as(
            result['mask_logits']).remainder(7)-3)/11
        grads = torch.autograd.grad((result['mask_logits']*upstream).sum(),
            (F, e0, adapter.hidden.weight, adapter.final.weight), retain_graph=True)
        require(all(torch.isfinite(value).all().item() and value.norm().item() > 0 for value in grads),
            'Calibrated F/e0/two-weight gradients must be finite and reachable')
        manual_maxabs[mode] = maxabs
        calibrated_gradient_norms[mode] = [value.norm().item() for value in grads]
        outputs[mode] = result['mask_logits'].detach()
        flipped = {key: value.clone() if torch.is_tensor(value) else value for key, value in g.items()}
        for key in ('inside', 'outside', 'eligible'):
            flipped[key] = flipped[key].flip(2)
        flipped['used_theta'] = flipped['used_theta'] + math.pi
        flipped['canonical_theta'] = flipped['canonical_theta'] + math.pi
        adapter._geometry = lambda _F, _e, _v, fixture=flipped: fixture
        flip_result = adapter(F, e0, valid)
        require(torch.allclose(flip_result['mask_logits'], outputs[mode], atol=3e-6, rtol=3e-5),
            '180-degree axis reversal with slot permutation changed output: '+mode)
        adapter._geometry = lambda _F, _e, _v, fixture=g: fixture
    require(not torch.allclose(outputs['spatial_halves'], outputs['global'], atol=1e-6, rtol=0) and
        not torch.allclose(outputs['spatial_halves'], outputs['spatial_reversed'], atol=1e-6, rtol=0),
        'Nonzero injected fixture does not distinguish spatial/global/reversed')

    # Old GLOBAL operation: exactly the same eligible coefficient pool and phi.
    # Projection arithmetic differs, so outputs/gradients use stated tolerances.
    old = geometry_source.build_adapter(torch, mode='paired')
    old._geometry = lambda _F, _e, _v: g
    calibrated_weights(torch, old)
    old_F, old_e = F.detach().clone().requires_grad_(), e0.detach().clone().requires_grad_()
    new_F, new_e = F.detach().clone().requires_grad_(), e0.detach().clone().requires_grad_()
    old_result = old(old_F, old_e, valid)
    old_logits = torch.einsum('bqc,bchw->bqhw', old_result['new_coeff'], old_F)
    new_result = adapters['global'](new_F, new_e, valid)
    require(torch.equal(old_result['coefficient_residual'], new_result['coefficient_fields'][:, :, 0]),
        'Global coefficient pool differs from old paired operator')
    require(torch.allclose(old_logits, new_result['mask_logits'], atol=3e-6, rtol=3e-5),
        'Global FP32 masks are not numerically comparable to old coefficient projection')
    upstream = (torch.arange(old_logits.numel(), dtype=torch.float32).reshape_as(old_logits).remainder(7)-3)/11
    old_grads = torch.autograd.grad((old_logits*upstream).sum(),
        (old_F, old_e, old.hidden.weight, old.final.weight))
    new_grads = torch.autograd.grad((new_result['mask_logits']*upstream).sum(),
        (new_F, new_e, adapters['global'].hidden.weight, adapters['global'].final.weight))
    require(all(torch.allclose(a, b, atol=1e-4, rtol=1e-4) for a, b in zip(old_grads, new_grads)),
        'Global conditional gradients are not numerically comparable')
    require(torch.equal(cpu_before, torch.random.get_rng_state()), 'Fixture/factories leaked CPU RNG')
    return dict(status='ROOT_SPATIAL_TWO_FIELD_SYNTHETIC_CPU_PASS',
        scientific_execution_performed=True, native_integration_qualified=False,
        native_training_or_efficacy_claimed=False, fixtures=dict(real_geometry=real_rows,
            real_E0_connected_zero=True, injected_geometry_E=True, manual_formula_maxabs=manual_maxabs,
            calibrated_gradient_norms=calibrated_gradient_norms,
            unequal_half_global_mean=True, empty_half_fallback=True, axis_180_flip=True,
            global_old_output_maxabs=(old_logits-new_result['mask_logits']).abs().max().item(),
            global_old_gradient_maxabs=[(a-b).abs().max().item() for a,b in zip(old_grads,new_grads)]),
        tolerances=dict(manual_and_global_output_atol=3e-6, output_rtol=3e-5,
                        global_gradient_atol=1e-4, global_gradient_rtol=1e-4),
        scope='synthetic real inherited geometry and injected geometry checks; no native head/data/CP')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--geometry-source', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args()
    require(args.owner_execute, 'SOURCE preparation is inert; explicit owner execution required')
    root = Path(args.source_root).resolve(strict=True)
    geometry_path = Path(args.geometry_source).resolve(strict=True)
    out = Path(args.out).resolve()
    require(not out.exists() and out.parent.is_dir(), 'Output must be fresh in an existing directory')
    require(root == Path(__file__).resolve().parent and geometry_path.suffix == '.py', 'Explicit source paths differ')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
    if os.name == 'nt':
        import ctypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        require(kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), 3) != 0, 'CPU01 affinity failed')
    elif hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, {0, 1})
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    require(not torch.cuda.is_initialized(), 'CPU fixture must not initialize CUDA')
    geometry_source = load_source(geometry_path, 'synthetic_original_paired_geometry')
    operator_path = root / 'readout.py'
    operator = load_source(operator_path, 'synthetic_spatial_position_preserving_operator')
    report = execute(torch, operator, geometry_source)
    require(not torch.cuda.is_initialized(), 'Fixture touched CUDA')
    report.update(source=source_ref(Path(__file__).resolve()), operator_source=source_ref(operator_path),
        geometry_source=source_ref(geometry_path), CPU_affinity=[0,1], CUDA_visible_devices='', threads=2)
    with out.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(status=report['status'], out=str(out))))


if __name__ == '__main__':
    main()
