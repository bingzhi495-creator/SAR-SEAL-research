"""Inert, explicitly owner-gated CPU fixture for the linear FIT diagnostic.

Authoring/static compilation is not execution or qualification. At execution,
only an explicitly SOURCE_ONLY manifest, pinned Python SOURCE and the inert
SOURCE_CONTRACT are read. All tensors and binary targets are synthetic. There
is no model, image loader, GT/checkpoint/prediction/qualification/receipt read.
The fixture checks rendering, coefficient gradients and finite optimization;
it does not establish real-data feasibility, AP, convergence or an IoU bound.
"""

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import types


SCHEMA = 'ROOT-frozen-feature-linear-readout-synthetic-SOURCE-spec-v1'
STATUS = 'FROZEN_FEATURE_LINEAR_READOUT_SYNTHETIC_CPU_PASS'
EXECUTION = dict(device='cpu', cpu_affinity=[0, 1], torch_threads=2,
    torch_interop_threads=2, cuda_visible_devices='')
SOLVER = dict(name='torch.optim.LBFGS', lr=1, max_iter=40, max_eval=80,
    history_size=10, tolerance_grad=1e-7, tolerance_change=1e-9,
    line_search_fn='strong_wolfe')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def source_item(ref, expected):
    require(type(ref) is dict and set(ref) == {'path', 'bytes', 'sha256'} and
        Path(ref['path']).is_absolute() and type(ref['bytes']) is int and
        ref['bytes'] > 0 and type(ref['sha256']) is str and
        len(ref['sha256']) == 64 and all(c in '0123456789abcdef' for c in ref['sha256']),
        'Malformed pinned SOURCE reference')
    path = Path(ref['path']).resolve(strict=True)
    require(path == expected.resolve(strict=True), 'Pinned SOURCE path differs')
    data = path.read_bytes()
    require(len(data) == ref['bytes'] and hashlib.sha256(data).hexdigest() == ref['sha256'],
        'Pinned SOURCE content differs: ' + path.name)
    return path, data


def pinned_sources(spec):
    own = Path(__file__).resolve(strict=True)
    expected = dict(fixture=own, producer=own.parent / 'diagnose.py',
        contract=own.parent / 'SOURCE_CONTRACT.json')
    require(type(spec) is dict and set(spec) == {'schema', 'inputs', 'sources', 'execution'} and
        spec['schema'] == SCHEMA and spec['inputs'] == 'SOURCE_ONLY' and
        spec['execution'] == EXECUTION and type(spec['sources']) is dict and
        set(spec['sources']) == set(expected), 'SOURCE-only fixture manifest differs')
    items = {key: source_item(spec['sources'][key], path) for key, path in expected.items()}
    contract = json.loads(items['contract'][1].decode('utf-8-sig'))
    require(contract['kind'] == 'INERT_SOURCE_AUTHORING_CONTRACT_NOT_SCIENTIFIC_ADMISSION' and
        contract['schema'] == 'ROOT-frozen-feature-linear-readout-FIT-diagnostic-spec-v1' and
        contract['entry_name'] == 'diagnose.py' and contract['solver'] == SOLVER and
        contract['candidates'] == ['native_best100', 'final'], 'Inert SOURCE contract differs')
    # Only these implementation SOURCE references are followed. Scientific metadata paths
    # appearing in the inert contract are never opened or hashed by this fixture.
    for role, expected_name in (('solver_source', 'lbfgs.py'), ('torch_version_source', 'version.py'),
            ('native_mask_head_SOURCE', 'mask2former_head.py'),
            ('native_fusion_SOURCE', 'maskformer_fusion_head.py'), ('interpolation_SOURCE', 'functional.py')):
        ref = contract[role]
        path = Path(ref['path'])
        require(path.name == expected_name and path.suffix == '.py', 'Solver SOURCE role differs')
        items[role] = source_item(ref, path)
    version_tree = ast.parse(items['torch_version_source'][1].decode('utf-8-sig'), feature_version=8)
    versions = [ast.literal_eval(node.value) for node in version_tree.body
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and
            t.id == '__version__' for t in node.targets)]
    require(versions == ['2.0.1+cu118'], 'Expected Torch version SOURCE differs')
    return items, contract


def load_pinned(item):
    path, data = item
    module = types.ModuleType('synthetic_linear_diagnostic_producer')
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


def scalar_resize_sample(channel, output_y, output_x, output_h, output_w):
    """Independent half-pixel bilinear with edge clamping, align_corners=False."""
    height, width = len(channel), len(channel[0])
    sy = min(max((output_y + 0.5) * height / output_h - 0.5, 0.0), height - 1.0)
    sx = min(max((output_x + 0.5) * width / output_w - 0.5, 0.0), width - 1.0)
    y0, x0 = math.floor(sy), math.floor(sx)
    y1, x1 = min(y0 + 1, height - 1), min(x0 + 1, width - 1)
    dy, dx = sy - y0, sx - x0
    return math.fsum(channel[y][x] * w for y, x, w in (
        (y0, x0, (1-dy)*(1-dx)), (y0, x1, (1-dy)*dx),
        (y1, x0, dy*(1-dx)), (y1, x1, dy*dx)))


def scalar_original_channel(channel, image_shape, original_shape, y, x):
    crop_h, crop_w = image_shape
    original_h, original_w = original_shape
    sy = min(max((y + 0.5) * crop_h / original_h - 0.5, 0.0), crop_h - 1.0)
    sx = min(max((x + 0.5) * crop_w / original_w - 0.5, 0.0), crop_w - 1.0)
    y0, x0 = math.floor(sy), math.floor(sx)
    y1, x1 = min(y0 + 1, crop_h - 1), min(x0 + 1, crop_w - 1)
    dy, dx = sy - y0, sx - x0
    return math.fsum(w * scalar_resize_sample(channel, yy, xx, 800, 800)
        for yy, xx, w in ((y0, x0, (1-dy)*(1-dx)), (y0, x1, (1-dy)*dx),
                         (y1, x0, dy*(1-dx)), (y1, x1, dy*dx)))


def scalar_planes(channels, image_shape, original_shape):
    height, width = original_shape
    return [[[scalar_original_channel(channel, image_shape, original_shape, y, x)
        for x in range(width)] for y in range(height)] for channel in channels]


def scalar_logits(planes, coefficients):
    height, width = len(planes[0]), len(planes[0][0])
    return [[math.fsum(coefficients[k] * plane[y][x] for k, plane in enumerate(planes))
        for x in range(width)] for y in range(height)]


def scalar_bce(logits, target):
    losses = [max(z, 0.0) - z * target[y][x] + math.log1p(math.exp(-abs(z)))
        for y, row in enumerate(logits) for x, z in enumerate(row)]
    return math.fsum(losses) / len(losses)


def scalar_gradient(planes, logits, target):
    count = len(logits) * len(logits[0])
    return [math.fsum((1.0 / (1.0 + math.exp(-z)) - target[y][x]) * plane[y][x]
        for y, row in enumerate(logits) for x, z in enumerate(row)) / count for plane in planes]


def scalar_iou(logits, target):
    intersection = sum(z > 0 and target[y][x] == 1 for y, row in enumerate(logits) for x, z in enumerate(row))
    union = sum(z > 0 or target[y][x] == 1 for y, row in enumerate(logits) for x, z in enumerate(row))
    return intersection / union if union else 1.0


def synthetic_features(torch):
    channels = []
    for kind in range(4):
        rows = []
        for y in range(200):
            row = []
            for x in range(200):
                row.append((1.0, (x - 99.5) / 99.5, (y - 99.5) / 99.5,
                    0.2 * math.sin(x / 13.0) * math.cos(y / 17.0))[kind])
            rows.append(row)
        channels.append(rows)
    feature = torch.zeros((1, 256, 200, 200), dtype=torch.float32)
    feature[0, :4] = torch.tensor(channels, dtype=torch.float32)
    # The scalar reference uses the same rounded FP32 input values, while all
    # interpolation, projection and BCE calculations are independent stdlib math.
    return feature, feature[0, :4].tolist()


def project_render(torch, producer, feature, coefficient, image_shape, original_shape):
    native = producer.project_coefficients(torch, feature, coefficient)
    identity = dict(actual_img_shape=list(image_shape), original_shape=list(original_shape))
    return producer.render_original(torch, native, identity)


def validate_fit(torch, result, original_shape):
    coefficient, logits = result['final_coefficient'], result['final_logits']
    require(tuple(coefficient.shape) == (256,) and tuple(logits.shape) == tuple(original_shape) and
        coefficient.dtype == torch.float32 and logits.dtype == torch.float32 and
        coefficient.device.type == 'cpu' and logits.device.type == 'cpu' and
        torch.isfinite(coefficient).all().item() and torch.isfinite(logits).all().item(),
        'Fit result coefficient/logit contract differs')
    callbacks, calls = result['solver_callbacks'], result['closure_calls']
    require(type(calls) is int and calls > 0 and len(callbacks) == calls and
        result['optimizer_func_evals'] == calls and type(result['optimizer_n_iter']) is int and
        0 <= result['optimizer_n_iter'] <= SOLVER['max_iter'] and
        result['max_eval_is_soft_limit'] is True and
        [row['closure_index'] for row in callbacks] == list(range(1, calls + 1)),
        'Finite LBFGS call/iteration accounting differs')
    # max_eval80 is a soft line-search stopping option. Do not invent an <=80
    # certificate or alter the production solver with a fixture-only hard cap.
    for row in callbacks:
        require(all(isinstance(row[key], (int, float)) and not isinstance(row[key], bool) and
            math.isfinite(row[key]) and row[key] >= 0 for key in ('loss', 'coefficient_norm', 'gradient_norm')),
            'Nonfinite/negative closure scalar')
    require(all(isinstance(result[key], (int, float)) and not isinstance(result[key], bool) and
        math.isfinite(result[key]) and result[key] >= 0 for key in
        ('initial_loss', 'final_loss', 'initial_norm', 'final_norm', 'final_gradient_norm')),
        'Nonfinite/negative fit endpoint scalar')


def execute(torch, producer, contract):
    require(all(callable(getattr(producer, name, None)) for name in
        ('project_coefficients', 'render_original', 'fit_coefficient', 'binary_metrics', 'lowest_best_query')),
        'Producer pure helper interface differs')
    rng_before = torch.random.get_rng_state().clone()
    feature, channels = synthetic_features(torch)
    feature_before = feature.clone()
    require(not feature.requires_grad and feature.grad is None, 'Synthetic F is not frozen')
    coefficients = torch.zeros((1, 2, 256), dtype=torch.float32)
    coefficients[0, 0, :4] = torch.tensor([0.13, 0.7, -0.4, 0.9], dtype=torch.float32)
    coefficients[0, 1, :4] = torch.tensor([-0.11, -0.5, 0.6, -0.3], dtype=torch.float32)
    render_errors = []
    for image_shape, original_shape in (((613, 797), (23, 31)), ((1, 3), (5, 7)), ((800, 800), (7, 11))):
        planes = scalar_planes(channels, image_shape, original_shape)
        expected = torch.tensor([scalar_logits(planes, coefficients[0, q, :4].tolist())
            for q in range(2)], dtype=torch.float32)
        actual = project_render(torch, producer, feature, coefficients, image_shape, original_shape)
        require(tuple(actual.shape) == (2,) + original_shape and actual.dtype == torch.float32 and
            torch.isfinite(actual).all().item() and torch.allclose(actual, expected, atol=2e-5, rtol=2e-5),
            'Independent scalar200->800/crop/original renderer differs')
        render_errors.append(dict(image_shape=list(image_shape), original_shape=list(original_shape),
            maxabs=(actual - expected).abs().max().item()))

    # Generic tiny F is used only to check the actual100 query axis cheaply.
    # All renderer and fit checks above/below retain the exact200 feature canvas.
    many_coefficients = torch.zeros((1, 100, 256), dtype=torch.float32)
    many_coefficients[0, :, 0] = (torch.arange(100, dtype=torch.float32) - 50) / 20
    many_masks = producer.project_coefficients(torch, feature[:, :, 7:9, 11:14].contiguous(), many_coefficients)
    require(tuple(many_masks.shape) == (1, 100, 2, 3) and
        torch.equal(many_masks[0, :, 0, 0], many_coefficients[0, :, 0]), 'Native100 coefficient/query axis differs')
    scores = [0.2] * 100
    require(producer.lowest_best_query(scores) == 0, 'All-tied query selection differs')
    scores[77] = scores[3] = 0.8
    require(producer.lowest_best_query(scores) == 3, 'Tie did not retain lowest actual query ordinal')

    image_shape, original_shape = (613, 797), (23, 31)
    identity = dict(actual_img_shape=list(image_shape), original_shape=list(original_shape))
    planes = scalar_planes(channels, image_shape, original_shape)
    truth_coefficients = [0.03125, 2.0, 0.0, 0.0]
    truth_logits = scalar_logits(planes, truth_coefficients)
    target_list = [[int(value > 0) for value in row] for row in truth_logits]
    target = torch.tensor(target_list, dtype=torch.bool)
    require(0 < target.sum().item() < target.numel() and
        min(abs(value) for row in truth_logits for value in row) > 1e-3,
        'Synthetic truth lacks both classes or a clear threshold margin')
    truth_tensor = torch.zeros((1, 1, 256), dtype=torch.float32)
    truth_tensor[0, 0, :4] = torch.tensor(truth_coefficients, dtype=torch.float32)
    truth_actual = project_render(torch, producer, feature, truth_tensor, image_shape, original_shape)[0]
    require(torch.equal(truth_actual > 0, target), 'Known coefficient does not reproduce scalar binary truth')

    probe = torch.zeros((1, 1, 256), dtype=torch.float32)
    probe[0, 0, :4] = torch.tensor([0.11, -0.09, 0.17, -0.08], dtype=torch.float32)
    probe.requires_grad_(True)
    probe_logits = project_render(torch, producer, feature, probe, image_shape, original_shape)[0]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(probe_logits, target.float(), reduction='mean')
    gradient = torch.autograd.grad(loss, probe)[0]
    scalar_probe = scalar_logits(planes, probe.detach()[0, 0, :4].tolist())
    expected_gradient = scalar_gradient(planes, scalar_probe, target_list)
    reference_gradient = torch.zeros_like(probe)
    reference_gradient[0, 0, :4] = torch.tensor(expected_gradient, dtype=torch.float32)
    require(gradient is not None and torch.isfinite(gradient).all().item() and gradient.norm().item() > 0 and
        torch.allclose(gradient, reference_gradient, atol=2e-6, rtol=2e-5) and
        abs(loss.item() - scalar_bce(scalar_probe, target_list)) <= 2e-6,
        'Independent full-raster BCE/coefficient gradient differs')
    finite_difference_errors = []
    for channel in (0, 1, 2, 3, 255):
        plus, minus = probe.detach().clone(), probe.detach().clone()
        plus[0, 0, channel] += 0.01
        minus[0, 0, channel] -= 0.01
        plus_logits = project_render(torch, producer, feature, plus, image_shape, original_shape)[0]
        minus_logits = project_render(torch, producer, feature, minus, image_shape, original_shape)[0]
        plus_loss = torch.nn.functional.binary_cross_entropy_with_logits(plus_logits, target.float()).item()
        minus_loss = torch.nn.functional.binary_cross_entropy_with_logits(minus_logits, target.float()).item()
        fd = (plus_loss - minus_loss) / 0.02
        expected = gradient[0, 0, channel].item()
        require(abs(fd - expected) <= 3e-5 + 3e-4 * abs(expected), 'Finite-difference coefficient gradient differs')
        finite_difference_errors.append(dict(channel=channel, analytic=expected, finite_difference=fd,
            abs_error=abs(fd - expected)))

    initial = torch.zeros(256, dtype=torch.float32)
    initial_before = initial.clone()
    fit = producer.fit_coefficient(torch, feature, initial, target, identity, dict(SOLVER))
    validate_fit(torch, fit, original_shape)
    final = fit['final_coefficient']
    replay = project_render(torch, producer, feature, final.reshape(1, 1, 256), image_shape, original_shape)[0]
    initial_logits = project_render(torch, producer, feature, initial.reshape(1, 1, 256), image_shape, original_shape)[0]
    initial_iou = scalar_iou(initial_logits.tolist(), target_list)
    final_iou = scalar_iou(replay.tolist(), target_list)
    actual_final_loss = torch.nn.functional.binary_cross_entropy_with_logits(replay, target.float()).item()
    require(torch.equal(replay, fit['final_logits']) and torch.equal(initial, initial_before) and
        torch.count_nonzero(final[4:]).item() == 0 and not torch.equal(final, initial) and
        abs(fit['initial_loss'] - math.log(2)) <= 2e-6 and
        abs(fit['final_loss'] - actual_final_loss) <= 2e-6 and
        fit['final_loss'] < fit['initial_loss'] - 1e-4 and final_iou > initial_iou,
        'Finite pure linear coefficient fit did not reduce BCE/improve synthetic IoU')
    observed_metrics = producer.binary_metrics(torch, replay > 0, target)
    require(abs(observed_metrics['IoU'] - final_iou) <= 1e-12,
        'Producer binary IoU differs from independent scalar full-raster IoU')

    # Zero F must have exactly zero logits for any coefficient. The BCE still
    # has a connected, finite zero coefficient gradient; no extra bias may fit it.
    zero_feature = torch.zeros_like(feature)
    zero_initial = torch.linspace(-1, 1, 256, dtype=torch.float32)
    zero_probe = zero_initial.reshape(1, 1, 256).clone().requires_grad_(True)
    zero_logits = project_render(torch, producer, zero_feature, zero_probe, image_shape, original_shape)[0]
    zero_loss = torch.nn.functional.binary_cross_entropy_with_logits(zero_logits, target.float())
    zero_gradient = torch.autograd.grad(zero_loss, zero_probe)[0]
    require(torch.count_nonzero(zero_logits).item() == 0 and zero_gradient is not None and
        torch.isfinite(zero_gradient).all().item() and torch.count_nonzero(zero_gradient).item() == 0 and
        abs(zero_loss.item() - math.log(2)) <= 2e-6, 'No-bias/connected zero-F gradient failed')
    zero_fit = producer.fit_coefficient(torch, zero_feature, zero_initial, target, identity, dict(SOLVER))
    validate_fit(torch, zero_fit, original_shape)
    require(torch.equal(zero_fit['final_coefficient'], zero_initial) and
        torch.count_nonzero(zero_fit['final_logits']).item() == 0 and
        zero_fit['final_gradient_norm'] == 0 and zero_fit['closure_calls'] == 1 and
        zero_fit['optimizer_n_iter'] == 0, 'Zero-F optimizer changed coefficient or lost stationary accounting')
    empty = torch.zeros_like(target)
    both_empty = producer.binary_metrics(torch, empty, empty)
    require(all(both_empty[key] == 1.0 for key in ('IoU', 'Dice', 'precision', 'recall')) and
        both_empty['prediction_empty'] is True and both_empty['GT_empty'] is True,
        'Both-empty metric convention differs')
    require(torch.equal(feature, feature_before) and feature.grad is None and not feature.requires_grad and
        torch.equal(rng_before, torch.random.get_rng_state()) and not torch.cuda.is_initialized(),
        'Fixture changed frozen F/external CPU RNG or initialized CUDA')
    return dict(status=STATUS, synthetic_execution_performed=True,
        input_scope='Pinned SOURCE and in-memory hand-specified synthetic tensors only',
        real_data_feasibility_qualified=False, native_model_integration_qualified=False,
        AP_computed=False, efficacy_claimed=False, convergence_claimed=False, optimal_IoU_upper_bound_claimed=False,
        exact_feature_canvas=[1, 256, 200, 200], scalar_renderer_errors=render_errors,
        scalar_BCE_abs_error=abs(loss.item() - scalar_bce(scalar_probe, target_list)),
        scalar_gradient_maxabs=(gradient - reference_gradient).abs().max().item(),
        coefficient_finite_difference=finite_difference_errors,
        native100_query_axis_and_lowest_ordinal_ties=True,
        coefficient_only_no_bias=True, zero_F_connected_zero_and_stationary=True,
        frozen_F_unchanged=True, external_CPU_RNG_unchanged=True,
        known_coefficient_binary_truth_IoU=1.0, known_truth_min_abs_logit=min(abs(z) for row in truth_logits for z in row),
        finite_fit=dict(initial_BCE=fit['initial_loss'], final_BCE=fit['final_loss'],
            initial_IoU=initial_iou, final_IoU=final_iou, final_coefficient_norm=fit['final_norm'],
            final_gradient_norm=fit['final_gradient_norm'], closure_calls=fit['closure_calls'],
            optimizer_func_evals=fit['optimizer_func_evals'], optimizer_n_iter=fit['optimizer_n_iter']),
        zero_F_fit=dict(closure_calls=zero_fit['closure_calls'], optimizer_n_iter=zero_fit['optimizer_n_iter']),
        solver=SOLVER, max_eval_is_soft_limit=True, initial_final_metric_evaluations_outside_solver=True,
        GPU_optimizer_bitwise_determinism_claimed=False,
        tolerances=dict(renderer_atol=2e-5, renderer_rtol=2e-5, scalar_BCE_atol=2e-6,
            gradient_atol=2e-6, gradient_rtol=2e-5, finite_difference_step=0.01,
            finite_difference_atol=3e-5, finite_difference_rtol=3e-4))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args()
    require(args.owner_execute, 'SOURCE preparation is inert; explicit owner execution required')
    require('torch' not in sys.modules, 'Torch was imported before the fixture SOURCE gates')
    require(Path(args.spec).is_absolute() and Path(args.out).is_absolute(), 'Manifest/output paths must be absolute')
    spec_path, out = Path(args.spec).resolve(strict=True), Path(args.out).resolve()
    require(spec_path.suffix == '.json' and out.suffix == '.json' and not out.exists() and out.parent.is_dir(),
        'Fresh output contract differs')
    spec = json.loads(spec_path.read_text(encoding='utf-8-sig'))
    sources, contract = pinned_sources(spec)
    require(out != spec_path and all(out != item[0] for item in sources.values()), 'Output aliases SOURCE/manifest')
    cpu_gate()
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    require(torch.__version__ == '2.0.1+cu118' and torch.get_num_threads() == 2 and
        torch.get_num_interop_threads() == 2 and not torch.cuda.is_initialized() and
        torch.cuda.device_count() == 0, 'CPU-only Torch gate differs')
    require(Path(sys.modules['torch.optim.lbfgs'].__file__).resolve(strict=True) == sources['solver_source'][0]
        and Path(sys.modules['torch.version'].__file__).resolve(strict=True) == sources['torch_version_source'][0]
        and Path(torch.nn.functional.__file__).resolve(strict=True) == sources['interpolation_SOURCE'][0],
        'Imported Torch solver/version SOURCE paths differ')
    producer = load_pinned(sources['producer'])
    report = execute(torch, producer, contract)
    report.update(sources=spec['sources'], solver_source=contract['solver_source'],
        torch_version_source=contract['torch_version_source'],
        native_mask_head_SOURCE=contract['native_mask_head_SOURCE'],
        native_fusion_SOURCE=contract['native_fusion_SOURCE'], interpolation_SOURCE=contract['interpolation_SOURCE'],
        execution=EXECUTION,
        spec_path=str(spec_path), torch_version=torch.__version__)
    with out.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(status=report['status'], out=str(out))))


if __name__ == '__main__':
    main()
