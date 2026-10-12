"""SOURCE-only preparation of a ROOT-executed synthetic CPU operator fixture.

CLI --source-root ABS_C --out FRESH_ABS; C contains paired_context.py and
matched_pointwise_FG_control_source_20261009/control.py. No models, datasets,
checkpoints, GT, predictions, qualifications or receipts are read. Runtime
imports Torch only after verifying the two source files and disabling CUDA.
One real7x11 path uses inherited geometry/bilinear sampling and two queries.
A separate monkeypatched fixture isolates phi-before-pool with known values;
it is explicitly NOT an actual geometry or native-head test. Manual scalar
SiLU/derivatives and a four-neighbor zero-padding formula are references.
PASS means only CPU synthetic operator checks, not native wiring/training,
all100x200 behavior, GPU numeric/performance, efficacy or equal capacity.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from types import MethodType


GEOMETRY_SHA = '7ad19e356342afc15a2a41f1358634b23486c19021e63ccd94cfed03f6841d15'
CONTROL_SHA = 'd531fe766b64b30591d6fa80a2e2f38d60393a029a8779ad7842fc0b96149de3'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def source_module(path, size, digest, name):
    raw = path.read_bytes()
    require(len(raw) == size and hashlib.sha256(raw).hexdigest() == digest, 'Sealed source changed: ' + str(path))
    definition = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(definition)
    definition.loader.exec_module(module)
    return module, dict(path=str(path), bytes=len(raw), sha256=digest)


def scalar_silu(x):
    return x / (1. + math.exp(-x))


def scalar_silu_derivative(x):
    sigmoid = 1. / (1. + math.exp(-x))
    return sigmoid + x * sigmoid * (1. - sigmoid)


def scalar_bilinear(field, x, y):
    """Independent integer-neighbor sum, zero outside, pixel-center indices."""
    h, w = len(field), len(field[0]); x0, y0 = math.floor(x), math.floor(y)
    dx, dy = x - x0, y - y0
    return math.fsum(field[yy][xx] * weight if 0 <= yy < h and 0 <= xx < w else 0.
        for yy, xx, weight in ((y0, x0, (1. - dx) * (1. - dy)),
            (y0, x0 + 1, dx * (1. - dy)), (y0 + 1, x0, (1. - dx) * dy),
            (y0 + 1, x0 + 1, dx * dy)))


def calibrate(torch, adapter, two_channels=True):
    with torch.no_grad():
        adapter.hidden.weight.zero_(); adapter.final.weight.zero_()
        adapter.hidden.weight[0, 0] = 1.; adapter.final.weight[0, 0] = 1.
        if two_channels:
            adapter.hidden.weight[1, 1] = 1.; adapter.final.weight[1, 1] = 1.


def execute(source_root, out_file):
    root, out = Path(source_root), Path(out_file)
    require(root.is_absolute() and root.is_dir() and out.is_absolute() and out.parent.is_dir() and not out.exists(),
            'Absolute source root and fresh existing-parent output required')
    root = root.resolve(strict=True)
    geometry, geometry_ref = source_module(root / 'paired_context.py', 17500, GEOMETRY_SHA, 'sealed_FG_fixture_geometry')
    control, control_ref = source_module(root / 'matched_pointwise_FG_control_source_20261009' / 'control.py',
        6105, CONTROL_SHA, 'sealed_FG_pointwise_fixture_control')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''; os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '2'
    import torch
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    require(not torch.cuda.is_initialized(), 'CPU fixture must not initialize CUDA')
    cpu_rng = torch.get_rng_state().clone()
    adapter = control.build_adapter(torch, geometry)
    require(torch.equal(cpu_rng, torch.get_rng_state()), 'Factory changed external CPU RNG')
    original_fg = geometry.build_adapter(torch, mode='fg_fine')
    require(all(torch.equal(value, original_fg.state_dict()[name])
        for name, value in adapter.state_dict().items()), 'Cold weights differ from original fg_fine')
    del original_fg
    parameters = dict(adapter.named_parameters())
    require(set(parameters) == {'hidden.weight', 'final.weight'} and
            tuple(parameters['hidden.weight'].shape) == (768, 256) and
            tuple(parameters['final.weight'].shape) == (256, 768) and
            sum(p.numel() for p in parameters.values()) == 393216 and len(adapter.state_dict()) == 2 and
            not list(adapter.named_buffers()) and adapter.hidden.bias is None and adapter.final.bias is None and
            all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in parameters.values()) and
            bool((adapter.final.weight == 0).all()), 'Two-weight bias-free CPU initialization differs')
    # REAL inherited geometry/sampling path: horizontal positive rectangle.
    F = torch.zeros((1, 256, 7, 11), dtype=torch.float32, device='cpu')
    F[0, 0].fill_(-2.); F[0, 0, 2:5, 2:9] = 2.; F[0, 1].fill_(1.)
    F.requires_grad_(); e0 = torch.zeros((1, 2, 256), dtype=torch.float32, device='cpu')
    e0[0, 0, 0] = 1.; e0.requires_grad_()
    valid = torch.ones((1, 1, 7, 11), dtype=torch.float32, device='cpu')
    cold = adapter(F, e0, valid)
    require(torch.equal(cold['new_coeff'].view(torch.uint8), e0.view(torch.uint8)) and
            bool((cold['coefficient_residual'] == 0).all()), 'Cold e0+zero identity failed')
    actual = adapter._geometry(F, e0, valid)
    require(int(actual['eligible_counts'][0, 0]) == 16 and int(actual['eligible_counts'][0, 1]) == 0 and
            all(not value.requires_grad and value.grad_fn is None for value in actual.values()),
            'Real supported/E0 queries or detached geometry/E failed')
    require(abs(float(actual['centre_xy'][0, 0, 0]) - 5.) <= 2e-5 and
            abs(float(actual['centre_xy'][0, 0, 1]) - 3.) <= 2e-5 and
            abs(float(actual['sigma_major'][0, 0]) - 2.) <= 2e-5 and
            abs(float(actual['sigma_minor'][0, 0]) - 1.) <= 2e-5 and
            abs(float(actual['used_theta'][0, 0])) <= 2e-5, 'Known rectangle moment reference differs')
    real_eligible = actual['eligible'][0, 0].tolist()
    coords = actual['inside'][0, 0].tolist()
    fields = [F.detach()[0, c].tolist() for c in (0, 1)]
    reference = [math.fsum(scalar_silu(scalar_bilinear(field, x, y))
        for flag, (x, y) in zip(real_eligible, coords) if flag) / sum(real_eligible) for field in fields]
    # Actual inherited sampler, with analytic half-edge and quarter-corner mass.
    edge_xy = [(-.5, 0.), (-.5, -.5), (10.5, 6.5)]
    edge_sample = adapter._sample_shared(F.detach(), torch.tensor([[edge_xy]], dtype=torch.float32, device='cpu'))
    edge_error = max(abs(float(edge_sample[0, 0, j, c]) - scalar_bilinear(fields[c], x, y))
        for j, (x, y) in enumerate(edge_xy) for c in (0, 1))
    require(edge_error <= 2e-6, 'Actual align-false zero-padding sampler differs from scalar edge/corner reference')
    cold['new_coeff'].sum().backward()
    require(F.grad is not None and bool((F.grad == 0).all()) and
            e0.grad is not None and torch.equal(e0.grad, torch.ones_like(e0)) and
            adapter.hidden.weight.grad is not None and bool((adapter.hidden.weight.grad == 0).all()) and
            adapter.final.weight.grad is not None and bool(torch.isfinite(adapter.final.weight.grad).all()) and
            bool((adapter.final.weight.grad != 0).any()), 'Cold zero-path or final-weight learning gradient failed')
    adapter.zero_grad(set_to_none=True); F.grad = None; e0.grad = None
    del cold
    calibrate(torch, adapter)
    live = adapter(F, e0, valid)
    require(max(abs(float(live['coefficient_residual'][0, 0, c]) - reference[c]) for c in (0, 1)) <= 2e-6 and
            bool((live['coefficient_residual'][0, 1] == 0).all()), 'Real sampled output/empty-E reference differs')
    upstream = torch.zeros_like(e0); upstream[:, :, 0] = 1.; upstream[:, :, 1] = -1.
    (live['new_coeff'] * upstream).sum().backward()
    require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) and bool((p.grad != 0).any())
        for p in adapter.parameters()), 'Nonzero path parameter gradients absent, zero or nonfinite')
    require(F.grad is not None and bool(torch.isfinite(F.grad).all()) and bool((F.grad > 0).any()) and bool((F.grad < 0).any()) and
            e0.grad is not None and torch.equal(e0.grad, upstream) and
            bool((e0.grad > 0).any()) and bool((e0.grad < 0).any()), 'Actual live F/e0 gradients or stop-gradient e0 geometry failed')
    # ISOLATED fixture only: explicit -2,+2 features; artificial E does NOT
    # follow M0. These patches live solely on a new synthetic adapter instance.
    synthetic = control.build_adapter(torch, geometry); calibrate(torch, synthetic, two_channels=False)
    E = torch.zeros((1, 2, 16), dtype=torch.bool, device='cpu'); E[0, 0, :2] = True
    def fixture_geometry(self, value, coefficient, fraction):
        return dict(eligible=E.detach(), eligible_counts=E.sum(dim=-1).detach(),
                    inside=torch.zeros((1, 2, 16, 2), dtype=torch.float32, device='cpu'))
    def fixture_sample(self, value, xy):
        left, right = value[:, :, 0, 0], value[:, :, 0, 1]
        return torch.stack([left, right] * 8, dim=1)[:, None].expand(1, 2, 16, 256)
    synthetic._geometry = MethodType(fixture_geometry, synthetic)
    synthetic._sample_shared = MethodType(fixture_sample, synthetic)
    small = torch.zeros((1, 256, 2, 2), dtype=torch.float32, device='cpu')
    small[0, 0, 0, 0] = -2.; small[0, 0, 0, 1] = 2.; small.requires_grad_()
    coefficient = torch.zeros((1, 2, 256), dtype=torch.float32, device='cpu', requires_grad=True)
    s = synthetic(small, coefficient, torch.ones((1, 1, 2, 2), dtype=torch.float32, device='cpu'))
    pointwise_reference = (scalar_silu(-2.) + scalar_silu(2.)) / 2.
    mean_first_reference = scalar_silu((-2. + 2.) / 2.)
    require(abs(float(s['coefficient_residual'][0, 0, 0]) - pointwise_reference) <= 2e-6 and
            abs(pointwise_reference - mean_first_reference) > .5 and bool((s['coefficient_residual'][0, 1] == 0).all()),
            'Pointwise nonlinear interaction degenerated to phi(mean) or changed E0')
    sample = synthetic._sample_shared(small, None)
    weights = (E.to(small.dtype) / E.sum(dim=-1).clamp_min(1)[..., None]).detach()
    mean_first = synthetic._phi((sample * weights[..., None]).sum(dim=2))
    require(abs(float(mean_first[0, 0, 0]) - mean_first_reference) <= 2e-6, 'Actual same-phi mean-before output differs from scalar reference')
    s['new_coeff'][0, 0, 0].backward()
    require(abs(float(small.grad[0, 0, 0, 0]) - scalar_silu_derivative(-2.) / 2.) <= 2e-6 and
            abs(float(small.grad[0, 0, 0, 1]) - scalar_silu_derivative(2.) / 2.) <= 2e-6 and
            float(small.grad[0, 0, 0, 0]) < 0 < float(small.grad[0, 0, 0, 1]), 'Independent signed scalar derivative reference differs')
    require(torch.equal(cpu_rng, torch.get_rng_state()) and not torch.cuda.is_initialized(), 'Synthetic CPU fixture changed RNG or initialized CUDA')
    result = dict(status='ROOT_FG_POINTWISE_SYNTHETIC_CPU_OPERATOR_PASS',
        sources=dict(geometry=geometry_ref, control=control_ref), CPU_threads=2, CUDA_initialized=False,
        external_CPU_RNG_exact_restored=True, parameter_tensors=2, scalars=393216, buffers=0,
        zero_final=True, cold_coefficient_byte_identity=True,
        original_fg_cold_weights_exact=True, cold_hidden_gradient_zero=True,
        cold_final_weight_gradient_live=True, live_two_weight_gradients=True,
        real_path=dict(canvas=[7, 11], queries=2, eligible_counts=actual['eligible_counts'].tolist(),
            manual_bilinear_scalar_output_reference=reference, edge_corner_sampler_maxabs=edge_error,
            signed_F_and_e0_gradients=True, geometry_and_E_detached=True),
        isolated_order_fixture=dict(artificial_geometry_and_sampling=True, pointwise=pointwise_reference,
            mean_before=mean_first_reference, signed_scalar_derivatives_verified=True),
        native_model_wiring_qualified=False, all100_native200_qualified=False,
        GPU_numeric_or_performance_qualified=False, training_or_efficacy_qualified=False)
    with out.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(status=result['status'], out=str(out)), sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True); parser.add_argument('--out', required=True)
    args = parser.parse_args(); execute(args.source_root, args.out)


if __name__ == '__main__':
    main()
