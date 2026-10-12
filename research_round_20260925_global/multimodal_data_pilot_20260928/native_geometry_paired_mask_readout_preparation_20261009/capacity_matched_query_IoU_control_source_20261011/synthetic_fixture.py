"""Small synthetic CPU checks; SOURCE preparation does not execute this file.

Only this fixture and query_iou_control.py are read at future execution. No data,
GT, native model, checkpoint, matching, optimizer or scientific lineage is used.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import types

STATUS = 'CAPACITY_MATCHED_QUERY_IOU_SYNTHETIC_CPU_PASS'
SHAPES = {'hidden1.weight': [32, 260], 'hidden1.bias': [32],
          'hidden2.weight': [32, 32], 'hidden2.bias': [32],
          'final.weight': [1, 32], 'final.bias': [1]}
ORDER = ('native_class_probability', 'binary_area_fraction',
         'official_mask_quality', 'zero_learning_posterior_budget')


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def ref(path, code):
    return dict(path=str(path), bytes=len(code), sha256=hashlib.sha256(code).hexdigest())


def finite_gradients(model):
    result = {}
    for name, parameter in model.named_parameters():
        check(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()), 'gradient ' + name)
        result[name] = dict(norm=float(parameter.grad.norm()), nonzero=int(torch.count_nonzero(parameter.grad)))
    return result


def scalar_forward(model, query, descriptors):
    """Independent scalar affine/SiLU/sigmoid arithmetic, using exported weights."""
    values = list(query) + list(descriptors)
    for layer in (model.hidden1, model.hidden2):
        weights, biases = layer.weight.detach().tolist(), layer.bias.detach().tolist()
        z = [math.fsum(w * x for w, x in zip(row, values)) + b for row, b in zip(weights, biases)]
        values = [x / (1 + math.exp(-x)) for x in z]
    logit = math.fsum(w * x for w, x in zip(model.final.weight[0].detach().tolist(), values))
    logit += float(model.final.bias[0])
    return 1 / (1 + math.exp(-logit))


def numerical_checks(module):
    before_rng = torch.get_rng_state().clone()
    cls = module.CapacityMatchedQueryIoUReadout
    model, repeat, other = cls(seed=111), cls(seed=111), cls(seed=112)
    params = dict(model.named_parameters())
    check(list(params) == list(SHAPES) and len(params) == 6 and
          sum(p.numel() for p in params.values()) == 9441 and not list(model.named_buffers()), 'capacity')
    for name, p in params.items():
        check(list(p.shape) == SHAPES[name] and p.device.type == 'cpu' and
              p.dtype == torch.float32 and p.requires_grad and bool(torch.isfinite(p).all()), 'parameter ' + name)
        check(torch.equal(p, repeat.state_dict()[name]), 'repeat seed ' + name)
    check(not torch.equal(model.hidden1.weight, other.hidden1.weight), 'different seed')
    check(torch.equal(torch.get_rng_state(), before_rng) and not torch.cuda.is_initialized(), 'constructor RNG/CUDA')
    check(tuple(module.DESCRIPTOR_ORDER) == ORDER, 'descriptor order')
    generator = torch.Generator(device='cpu').manual_seed(113)
    queries = (.2 * torch.randn(2, 100, 256, generator=generator)).requires_grad_(True)
    masks = torch.rand(2, 100, 7, 11, generator=generator) > .6
    masks[:, 0] = False
    masks[:, 1] = True
    masks[:, 2] = False
    masks[:, 2, 0, 0] = True
    c = .1 + .8 * torch.rand(2, 100, generator=generator)
    native_quality = torch.rand(2, 100, generator=generator)
    budget = torch.rand(2, 100, generator=generator)
    native_quality[:, 0], budget[:, 0] = 0, 0
    descriptors = module.assemble_native_descriptors(c, masks, native_quality, budget)
    area_oracle = torch.tensor([[sum(int(v) for v in mask.flatten().tolist()) / 77
                                 for mask in image] for image in masks], dtype=torch.float32)
    expected = torch.stack((c, area_oracle, native_quality, budget), dim=-1)
    check(descriptors.shape == (2, 100, 4) and torch.allclose(descriptors, expected, atol=1e-7, rtol=1e-6), 'native descriptors')
    check(bool((descriptors[:, 0, 1:] == 0).all()) and torch.equal(descriptors[:, 0, 0], c[:, 0]), 'empty descriptors preserve c')
    descriptors.requires_grad_(True)
    query_before, desc_before = queries.detach().clone(), descriptors.detach().clone()
    target = torch.tensor([[.1 + .45 * ((q + b) % 7) / 6 for q in range(100)] for b in range(2)])
    cold = model(queries, descriptors)
    check(cold.shape == (2, 100) and torch.equal(cold, torch.full((2, 100), .5)), 'cold all100 output')
    loss = module.all_query_mse(cold, target)
    oracle_loss = math.fsum(math.fsum((.5 - float(t)) ** 2 for t in row) / 100 for row in target) / 2
    check(abs(float(loss) - oracle_loss) <= 1e-7, 'independent image/query MSE')
    loss.backward()
    cold_grad = finite_gradients(model)
    check(all(cold_grad[name]['nonzero'] == 0 for name in SHAPES if not name.startswith('final.')), 'cold upstream zero')
    check(all(cold_grad[name]['nonzero'] > 0 for name in ('final.weight', 'final.bias')), 'cold final gradients')
    bias_oracle = math.fsum(.5 * (.5 - float(t)) for row in target for t in row) / 200
    check(abs(float(model.final.bias.grad[0]) - bias_oracle) <= 1e-7, 'independent cold bias gradient')
    check(queries.grad is None and descriptors.grad is None, 'teacher operands detached')
    warm = cls(seed=111).double()
    with torch.no_grad():
        warm.final.weight.copy_(torch.linspace(.01, .07, 32, dtype=torch.float64)[None])
        warm.final.bias.fill_(-.08)
    q64, d64, t64 = query_before.double(), desc_before.double(), target.double()
    quality = warm(q64, d64)
    check(quality.shape == (2, 100) and bool(((quality >= 0) & (quality <= 1)).all()), 'warm all100/range')
    maximum_error = 0.
    for b, q in ((0, 0), (0, 1), (0, 57), (1, 99)):
        expected_value = scalar_forward(warm, q64[b, q].tolist(), d64[b, q].tolist())
        maximum_error = max(maximum_error, abs(float(quality[b, q]) - expected_value))
    check(maximum_error <= 1e-12, 'independent warm scalar formula')
    quality.retain_grad()
    warm_loss = module.all_query_mse(quality, t64)
    warm_oracle = math.fsum(math.fsum((float(q) - float(t)) ** 2 for q, t in zip(qr, tr)) / 100
                            for qr, tr in zip(quality, t64)) / 2
    check(abs(float(warm_loss) - warm_oracle) <= 1e-12, 'warm scalar MSE')
    warm_loss.backward()
    warm_grad = finite_gradients(warm)
    check(all(row['nonzero'] > 0 for row in warm_grad.values()), 'warm all6 gradients')
    check(torch.allclose(quality.grad, 2 * (quality.detach() - t64) / 200, atol=1e-12, rtol=1e-12), 'all100 analytic MSE gradients')
    permutation = torch.arange(100).roll(17)
    with torch.no_grad():
        permuted = warm(q64[:, permutation], d64[:, permutation])
    check(torch.allclose(permuted, quality.detach()[:, permutation], atol=1e-12, rtol=1e-12), 'query permutation equivariance')
    check(abs(float(module.all_query_mse(permuted, t64[:, permutation])) - warm_oracle) <= 1e-12, 'paired target permutation')
    check(bool(torch.isfinite(module.all_query_mse(quality.detach(), torch.zeros_like(t64)))), 'caller noGT target zeros accepted')
    bad_target = t64.clone()
    bad_target[0, 99] = 1.01
    try:
        module.all_query_mse(quality.detach(), bad_target)
    except ValueError:
        pass
    else:
        raise AssertionError('out-of-range target accepted')
    check(torch.equal(queries.detach(), query_before) and torch.equal(descriptors.detach(), desc_before), 'inputs unchanged')
    check(torch.equal(torch.get_rng_state(), before_rng) and not torch.cuda.is_initialized(), 'whole fixture RNG/CUDA')
    return dict(capacity_9441_six_tensors_no_buffers=True, private_CPU_RNG_preserved=True,
                cold_final_only_and_warm_all6_gradients=True, independent_scalar_MSE_and_gradients=True,
                all100_query_order_and_permutations=True, target_bounds_and_empty_descriptors=True,
                teacher_inputs_detached_unchanged=True, warm_scalar_max_error=maximum_error,
                cold_gradient_norms={k: v['norm'] for k, v in cold_grad.items()},
                warm_gradient_norms={k: v['norm'] for k, v in warm_grad.items()})


def run(out, owner):
    global torch
    check(owner, 'explicit --owner-execute required')
    check(out.is_absolute() and not out.exists() and out.parent.is_dir(), 'fresh absolute output required')
    check(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and 'torch' not in sys.modules, 'CUDA hidden before import')
    here = Path(__file__).resolve()
    control = here.parent / 'query_iou_control.py'
    codes = {'fixture': here.read_bytes(), 'control': control.read_bytes()}
    sources = {name: ref(path, codes[name]) for name, path in (('fixture', here), ('control', control))}
    import psutil
    psutil.Process().cpu_affinity([0, 1])
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    check(sorted(psutil.Process().cpu_affinity()) == [0, 1] and torch.get_num_threads() == 2 and
          torch.get_num_interop_threads() == 2 and not torch.cuda.is_initialized(), 'CPU01/2+2/noCUDA')
    module = types.ModuleType('_synthetic_capacity_matched_query_iou')
    module.__file__ = str(control)
    exec(compile(codes['control'], str(control), 'exec'), module.__dict__)
    checks = numerical_checks(module)
    check(here.read_bytes() == codes['fixture'] and control.read_bytes() == codes['control'], 'SOURCE changed')
    result = dict(status=STATUS, sources=sources, synthetic_execution_performed=True,
                  execution=dict(device='cpu', cpu_affinity=[0, 1], torch_threads=2,
                                 torch_interop_threads=2, cuda_visible_devices='', CUDA_initialized=False),
                  parameter_count=9441, parameter_tensors=6, checks=checks,
                  scientific_data_accessed=False, native_integration_qualified=False,
                  target_generation_qualified=False, calibration_guaranteed=False,
                  efficacy_claimed=False, novelty_claimed=False,
                  limits=['Synthetic CPU scalar head checks only; no model/GT/matching/optimizer test.',
                          'Only 9441 scalar parameters match; six vs eight tensors and computation differ.',
                          'Caller supplies max-GT-IoU targets; duplicates are not suppressed by target definition.',
                          'Constant cold quality=0.5 is not measured IoU; empty descriptors do not hard-force score0.',
                          'No training effectiveness, AP, calibration, or complete prior-method reproduction claim.',
                          'ROOT actual tool terminal is separate execution evidence.'])
    with out.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, allow_nan=False, indent=2)
        handle.write('\n')
    return dict(status=STATUS, out=str(out))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.out, args.owner_execute)))
