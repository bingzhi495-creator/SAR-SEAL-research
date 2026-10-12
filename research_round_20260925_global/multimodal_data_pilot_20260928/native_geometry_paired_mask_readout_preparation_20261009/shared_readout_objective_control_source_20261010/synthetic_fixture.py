"""Inert SOURCE-only authoring; ROOT executes this synthetic CPU fixture.

The fixture checks pure shared-readout-control helpers against independent
stdlib scalar BCE, derivatives and half-pixel interpolation. All inputs are
synthetic. No model, dataset, GT payload, checkpoint, Q or receipt is opened.
Numerical acceptance exists only after the explicit owner execution below.
"""
import argparse
import ast
from collections import Counter
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import types


SCHEMA = 'ROOT-shared-readout-objective-control-synthetic-SOURCE-spec-v1'
STATUS = 'SHARED_READOUT_OBJECTIVE_CONTROL_SYNTHETIC_CPU_PASS'
EXECUTION = dict(device='cpu', cpu_affinity=[0, 1], torch_threads=2,
    torch_interop_threads=2, cuda_visible_devices='')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def source_bytes(row, label):
    require(isinstance(row, dict) and set(row) == {'path', 'bytes', 'sha256'} and
        Path(row['path']).is_absolute() and type(row['bytes']) is int and row['bytes'] > 0 and
        isinstance(row['sha256'], str) and len(row['sha256']) == 64 and
        set(row['sha256']) <= set('0123456789abcdef'), label + ' complete absolute SOURCE ref required')
    path = Path(row['path']).resolve(strict=True)
    require(str(path) == row['path'], label + ' canonical absolute SOURCE path required')
    data = path.read_bytes()
    require(len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256'],
        label + ' SOURCE bytes differ')
    return path, data


def load_verified(row, data, name):
    module = types.ModuleType(name)
    module.__file__ = row['path']
    exec(compile(data, row['path'], 'exec'), module.__dict__)
    return module


def set_cpu_affinity():
    if os.name == 'nt':
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        kernel.SetProcessAffinityMask.restype = ctypes.c_int
        kernel.GetProcessAffinityMask.argtypes = (ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t))
        kernel.GetProcessAffinityMask.restype = ctypes.c_int
        process = kernel.GetCurrentProcess()
        require(kernel.SetProcessAffinityMask(process, 3) != 0, 'CPU01 affinity could not be set')
        actual, system = ctypes.c_size_t(), ctypes.c_size_t()
        require(kernel.GetProcessAffinityMask(process, ctypes.byref(actual), ctypes.byref(system)) != 0 and
            actual.value == 3, 'Actual CPU affinity is not CPU01')
    else:
        os.sched_setaffinity(0, {0, 1})
        require(os.sched_getaffinity(0) == {0, 1}, 'Actual CPU affinity is not CPU01')


def sigmoid(value):
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def scalar_BCE(value, target):
    return max(value, 0.0) - value * target + math.log1p(math.exp(-abs(value)))


def half_pixel_terms(index, output_size, input_size):
    location = (index + 0.5) * input_size / output_size - 0.5
    lower = math.floor(location)
    fraction = location - lower
    return ((min(max(lower, 0), input_size - 1), 1.0 - fraction),
            (min(max(lower + 1, 0), input_size - 1), fraction))


def scalar_interpolate(plane, output_shape, y, x):
    total = 0.0
    for sy, wy in half_pixel_terms(y, output_shape[0], len(plane)):
        for sx, wx in half_pixel_terms(x, output_shape[1], len(plane[0])):
            total += wy * wx * plane[sy][sx]
    return total


def scalar_render(plane, identity):
    """Independent two bilinear operations; crop is defined by metadata only."""
    ih, iw = identity['actual_img_shape']
    oh, ow = identity['original_shape']
    result = []
    for y in range(oh):
        row = []
        for x in range(ow):
            value = 0.0
            for py, wy in half_pixel_terms(y, oh, ih):
                for px, wx in half_pixel_terms(x, ow, iw):
                    value += wy * wx * scalar_interpolate(plane, (800, 800), py, px)
            row.append(value)
        result.append(row)
    return result


def scalar_loss_and_coefficient_gradient(rendered_features, coefficients, targets, indices=None):
    channels = len(rendered_features)
    height, width = len(targets[0]), len(targets[0][0])
    losses, gradients = [], []
    for gt, coefficient in enumerate(coefficients):
        counts = Counter(range(height * width)) if indices is None else Counter(indices[gt])
        denominator = sum(counts.values())
        loss, gradient = 0.0, [0.0] * channels
        for flat, multiplicity in counts.items():
            y, x = divmod(flat, width)
            value = sum(coefficient[c] * rendered_features[c][y][x] for c in range(channels))
            target = int(targets[gt][y][x])
            weight = multiplicity / denominator
            loss += weight * scalar_BCE(value, target)
            for c in range(channels):
                gradient[c] += weight * (sigmoid(value) - target) * rendered_features[c][y][x]
        losses.append(loss)
        gradients.append([value / len(coefficients) for value in gradient])
    return sum(losses) / len(losses), gradients


def assert_close(actual, expected, label, atol=2e-5, rtol=2e-4):
    require(math.isfinite(actual) and math.isfinite(expected) and
        math.isclose(actual, expected, abs_tol=atol, rel_tol=rtol), label + ' differs from independent scalar reference')


def rejects(function, label):
    try:
        function()
    except (RuntimeError, ValueError, TypeError, IndexError):
        return
    raise RuntimeError(label + ' was silently accepted')


def coefficient_and_loss_checks(torch, producer, renderer):
    feature = torch.zeros((1, 256, 200, 200), dtype=torch.float32)
    feature[0, 0].fill_(1.0)
    feature[0, 1] = torch.arange(200, dtype=torch.float32)[None, :] / 199.0 - 0.5
    feature[0, 2] = torch.arange(200, dtype=torch.float32)[:, None] / 199.0 - 0.5
    before = feature.clone()
    identity = dict(actual_img_shape=[613, 797], original_shape=[11, 13])
    planes = feature[0, :3].tolist()
    rendered = [scalar_render(plane, identity) for plane in planes]
    coefficients = torch.zeros((1, 2, 256), dtype=torch.float32)
    coefficients[0, :, :3] = torch.tensor([[-0.15, 0.7, -0.3], [0.25, -0.4, 0.5]], dtype=torch.float32)
    coefficients.requires_grad_()
    target = torch.tensor([[[(x + 2 * y + q) % 4 == 0 for x in range(13)]
        for y in range(11)] for q in range(2)], dtype=torch.bool)
    actual = renderer.render_original(torch, renderer.project_coefficients(torch, feature, coefficients), identity)
    scalar_logits = [[[sum(coefficients[0, q, c].item() * rendered[c][y][x] for c in range(3))
        for x in range(13)] for y in range(11)] for q in range(2)]
    require(torch.allclose(actual.detach(), torch.tensor(scalar_logits, dtype=torch.float32), atol=2e-5, rtol=2e-5),
        'Independent exact200->800/crop/original scalar renderer differs')
    duplicate_rows = [[2, 2, 0, 2] * 3136, [1, 1, 1, 4] * 3136]
    indices = torch.tensor(duplicate_rows, dtype=torch.long)
    details = {}
    for name, sample in (('full', None), ('duplicate12544', indices)):
        loss = producer.per_image_BCE(torch, actual, target, sample)
        gradient = torch.autograd.grad(loss, coefficients, retain_graph=True)[0]
        expected_loss, expected_gradient = scalar_loss_and_coefficient_gradient(rendered,
            coefficients.detach()[0, :, :3].tolist(), target.tolist(), None if sample is None else duplicate_rows)
        assert_close(loss.item(), expected_loss, name + ' BCE')
        require(torch.allclose(gradient[0, :, :3], torch.tensor(expected_gradient, dtype=torch.float32),
            atol=2e-5, rtol=2e-4) and torch.count_nonzero(gradient[0, :, 3:]).item() == 0,
            name + ' independent coefficient gradient differs')
        details[name] = dict(loss=loss.item(), scalar_loss=expected_loss,
            coefficient_gradient_max_abs_error=(gradient[0, :, :3] -
                torch.tensor(expected_gradient, dtype=torch.float32)).abs().max().item())
    unique = torch.tensor([[0, 2], [1, 4]], dtype=torch.long)
    require(abs(producer.per_image_BCE(torch, actual, target, unique).item() - details['duplicate12544']['loss']) > 1e-5,
        'Repeated-index fixture does not distinguish deduplication')
    full_gradient = torch.autograd.grad(producer.per_image_BCE(torch, actual, target), coefficients)[0]
    step, errors = 0.01, []
    for q, channel in ((0, 0), (0, 1), (1, 2)):
        values = []
        for sign in (-1, 1):
            changed = coefficients.detach().clone()
            changed[0, q, channel] += sign * step
            logits = renderer.render_original(torch, renderer.project_coefficients(torch, feature, changed), identity)
            values.append(producer.per_image_BCE(torch, logits, target).item())
        difference = (values[1] - values[0]) / (2 * step)
        assert_close(difference, full_gradient[0, q, channel].item(), 'Coefficient finite difference', atol=3e-5, rtol=3e-3)
        errors.append(abs(difference - full_gradient[0, q, channel].item()))
    require(torch.equal(feature, before) and feature.grad is None and not feature.requires_grad,
        'Frozen synthetic F changed or received a gradient')
    # Averaging all possible one-pixel draws equals full loss AND gradient.
    tiny = torch.tensor([[[-2.0, -0.3, 0.0, 0.4, 1.1, 2.0]]], dtype=torch.float32, requires_grad=True)
    labels = torch.tensor([[[False, True, False, True, False, True]]])
    enumerated = sum(producer.per_image_BCE(torch, tiny, labels, torch.tensor([[pixel]], dtype=torch.long))
        for pixel in range(6)) / 6
    complete = producer.per_image_BCE(torch, tiny, labels)
    require(torch.allclose(enumerated, complete, atol=1e-7, rtol=1e-7) and
        torch.allclose(torch.autograd.grad(enumerated, tiny, retain_graph=True)[0],
            torch.autograd.grad(complete, tiny)[0], atol=1e-7, rtol=1e-7), 'Uniform one-pixel expectation differs')
    # Different rasters/images have equal image weight, regardless of GT or pixels.
    first = torch.tensor([[[0.2, -0.7, 1.1], [0.0, 0.4, -0.2]]], requires_grad=True)
    second = torch.tensor([[[0.3] * 5] * 3, [[-0.9] * 5] * 3], requires_grad=True)
    first_target, second_target = first.detach() > 0.3, second.detach() > 0
    combined = (producer.per_image_BCE(torch, first, first_target) +
        producer.per_image_BCE(torch, second, second_target)) / 2
    expected_first = sum(scalar_BCE(z, int(y)) for z, y in zip(first.detach().flatten().tolist(),
        first_target.flatten().tolist())) / 6
    expected_second = (scalar_BCE(0.3, 1) + scalar_BCE(-0.9, 0)) / 2
    assert_close(combined.item(), (expected_first + expected_second) / 2, 'Equal GT/image/accum2 normalization')
    grad_first, grad_second = torch.autograd.grad(combined, (first, second))
    require(torch.allclose(grad_first, (first.detach().sigmoid() - first_target.float()) / 12, atol=1e-7, rtol=1e-6) and
        torch.allclose(grad_second, (second.detach().sigmoid() - second_target.float()) / 60, atol=1e-7, rtol=1e-6),
        'Different raster/image gradients were incorrectly reweighted')
    empty = torch.empty((0, 2, 3), dtype=torch.float32, requires_grad=True)
    empty_loss = producer.per_image_BCE(torch, empty, torch.empty_like(empty, dtype=torch.bool))
    empty_gradient = torch.autograd.grad(empty_loss, empty)[0]
    require(empty_loss.item() == 0 and empty_gradient.shape == empty.shape and torch.isfinite(empty_gradient).all().item(),
        'Empty GT loss is not a connected finite zero')
    rejects(lambda: producer.per_image_BCE(torch, tiny.detach(), labels, torch.tensor([[6]])), 'Out-of-raster index')
    rejects(lambda: producer.per_image_BCE(torch, tiny.detach(), labels, torch.tensor([[0.0]])), 'Noninteger index')
    ambient = torch.random.get_rng_state().clone()
    private = torch.Generator(device='cpu').manual_seed(3800001)
    sampled = torch.randint(0, 6, (2, 12544), generator=private, device='cpu', dtype=torch.long)
    same = torch.randint(0, 6, (2, 12544), generator=torch.Generator(device='cpu').manual_seed(3800001), dtype=torch.long)
    require(torch.equal(sampled, same) and sampled.unique().numel() <= 6 and sampled.shape == (2, 12544) and
        torch.equal(torch.random.get_rng_state(), ambient), 'Private replacement sampling altered ambient RNG or seed replay')
    details.update(finite_difference_max_absolute_error=max(errors), sampled_draws_per_GT=12544,
        sampled_with_replacement=True, sampler_is_independent_of_learned_logits=True)
    return details


def assignment_checks(producer):
    actual = producer.fixed_assignment_rows([103, 101, 102], [7, 2, 99], [2, 0, 1])
    expected = [dict(annotation_id=103, GT_index=0, query_index=2),
        dict(annotation_id=101, GT_index=1, query_index=99), dict(annotation_id=102, GT_index=2, query_index=7)]
    require(actual == expected and producer.fixed_assignment_rows([103, 101, 102], [99, 7, 2], [1, 2, 0]) == expected,
        'Native GT row/raw annotation mapping or paired permutation differs')
    require(producer.fixed_assignment_rows([], [], []) == [], 'Empty assignment is not empty')
    for values in (([103, 103], [1, 2], [0, 1]), ([103, 101], [1, 1], [0, 1]),
                   ([103, 101], [1, 2], [0, 0]), ([103, 101], [1], [0]),
                   ([103], [100], [0]), ([103], [1], [1])):
        rejects(lambda args=values: producer.fixed_assignment_rows(*args), 'Invalid fixed assignment')
    return dict(raw_annotation_order=[103, 101, 102], mapping=actual,
        native_Hungarian_cost_or_optimality_qualified=False, mapping_helper_only=True)


def raw_gradient_checks(torch, producer, renderer, readout, geometry, contract):
    feature = torch.zeros((1, 256, 9, 17), dtype=torch.float32)
    feature[:, 0].fill_(-1)
    feature[:, 0, 3:6, 4:13] = 1
    e0 = torch.zeros((1, 1, 256), dtype=torch.float32)
    e0[:, :, 0] = 4
    valid = torch.ones((1, 1, 9, 17), dtype=torch.float32)
    native = torch.einsum('bqc,bchw->bqhw', e0, feature)
    feature_before, e0_before = feature.clone(), e0.clone()
    identity = dict(actual_img_shape=[800, 800], original_shape=[9, 17])
    target = torch.zeros((1, 9, 17), dtype=torch.bool)
    state, details = None, {}
    for name in ('raw_uniform_BCE', 'raw_full_BCE'):
        adapter = readout.build_adapter(torch, geometry, mode='raw', hidden_seed=contract['hidden_seed'],
            permutation_seed=contract['permutation_seed'])
        weights = dict(adapter.named_parameters())
        require(set(weights) == {'hidden.weight', 'final.weight'} and len(adapter.state_dict()) == 2 and
            not list(adapter.named_buffers()) and sum(p.numel() for p in weights.values()) == 393216,
            'RAW operator has extra parameters or buffers')
        if state is None:
            state = {key: value.detach().clone() for key, value in adapter.state_dict().items()}
        else:
            require(all(torch.equal(value, state[key]) for key, value in adapter.state_dict().items()),
                'Fresh RAW arms do not share exact initialization')
        indices = None if name == 'raw_full_BCE' else torch.randint(0, 153, (1, 12544),
            generator=torch.Generator(device='cpu').manual_seed(3800002), dtype=torch.long)
        cold = adapter(feature, e0, valid, native_logits=native)
        require(cold['diagnostics']['eligible_counts'].tolist() == [[16]] and
            torch.equal(cold['mask_logits'], native), 'Synthetic real geometry/cold anchor differs')
        original = renderer.render_original(torch, cold['mask_logits'], identity)
        loss = producer.per_image_BCE(torch, original, target, indices)
        gradients = torch.autograd.grad(loss, tuple(weights.values()))
        require(all(torch.isfinite(value).all().item() for value in gradients) and
            gradients[0].norm().item() == 0 and gradients[1].norm().item() > 0, 'Cold RAW two-weight gradient path failed')
        cold_norms = [value.norm().item() for value in gradients]
        # A declared nonzero final weight tests warm connectivity, not optimizer success.
        with torch.no_grad():
            adapter.final.weight.fill_(0.001)
        warm = adapter(feature, e0, valid, native_logits=native)
        original = renderer.render_original(torch, warm['mask_logits'], identity)
        loss = producer.per_image_BCE(torch, original, target, indices)
        gradients = torch.autograd.grad(loss, tuple(weights.values()), retain_graph=True)
        require(all(torch.isfinite(value).all().item() and value.norm().item() > 0 for value in gradients),
            'Warm RAW loss cannot reach both matrices')
        empty_loss = producer.per_image_BCE(torch, original[:0], target[:0])
        empty_gradients = torch.autograd.grad(empty_loss, tuple(weights.values()))
        require(empty_loss.item() == 0 and all(value is not None and torch.count_nonzero(value).item() == 0
            for value in empty_gradients), 'Empty GT image severed RAW parameter connectivity')
        require(feature.grad is None and e0.grad is None and not feature.requires_grad and not e0.requires_grad and
            torch.equal(feature, feature_before) and torch.equal(e0, e0_before), 'Frozen F/e0 changed or received gradients')
        details[name] = dict(cold_gradient_norms=cold_norms, warm_gradient_norms=[v.norm().item() for v in gradients],
            trainable_tensors=2, trainable_scalars=393216, eligible_pairs=16, warm_weight_set_explicitly=True,
            optimizer_trajectory_qualified=False)
    return details


def gain_checks(torch, producer):
    native = torch.tensor([[[0.1, 10.0, -1.0]], [[1.0, 1.0, -1.0]], [[-2.0, 0.0, -1.0]]], dtype=torch.float32)
    before = native.clone()
    original_binary = native > 0
    classes = [0.8, 0.8, 0.9]
    scores = {}
    for gain in (0.25, 1.0, 4.0):
        log_gain = torch.tensor(math.log(gain), dtype=torch.float32, requires_grad=True)
        scaled = producer.score_gain(torch, native, log_gain)
        require(torch.equal(scaled > 0, original_binary), 'Positive gain changed the synthetic binary examples')
        binary = scaled > 0
        quality = (scaled.sigmoid() * binary).flatten(1).sum(1) / (binary.flatten(1).sum(1) + 1e-6)
        actual_scores = quality * torch.tensor(classes, dtype=torch.float32)
        scalar_scores = []
        actual_gain = math.exp(log_gain.detach().item())
        for q, row in enumerate(native.tolist()):
            positives = [value for values in row for value in values if value > 0]
            scalar_scores.append(classes[q] * sum(sigmoid(actual_gain * value) for value in positives) /
                (len(positives) + 1e-6))
        for actual, expected in zip(actual_scores.tolist(), scalar_scores):
            assert_close(actual, expected, 'Independent official mask quality score', atol=1e-6, rtol=1e-6)
        require(actual_scores[2].item() == 0, 'Empty prediction quality is not zero')
        scores[str(gain)] = actual_scores.tolist()
    require(scores['0.25'][0] > scores['0.25'][1] and scores['4.0'][0] < scores['4.0'][1],
        'Gain fixture does not exhibit quality-score ranking change')
    target = torch.tensor([[[False, True, False]], [[True, False, False]], [[False, True, False]]])
    log_gain = torch.tensor(0.2, dtype=torch.float32, requires_grad=True)
    scaled = producer.score_gain(torch, native, log_gain)
    loss = producer.per_image_BCE(torch, scaled, target)
    gradient = torch.autograd.grad(loss, log_gain)[0]
    gain = math.exp(log_gain.detach().item())
    expected = sum((sigmoid(gain * value) - int(label)) * gain * value
        for value, label in zip(native.flatten().tolist(), target.flatten().tolist())) / native.numel()
    assert_close(gradient.item(), expected, 'Analytical log-gain gradient', atol=2e-5, rtol=2e-4)
    values, step = [], 0.01
    for sign in (-1, 1):
        point = log_gain.detach() + sign * step
        values.append(producer.per_image_BCE(torch, producer.score_gain(torch, native, point), target).item())
    finite_difference = (values[1] - values[0]) / (2 * step)
    assert_close(finite_difference, gradient.item(), 'Log-gain finite difference', atol=3e-5, rtol=3e-3)
    empty_native = torch.empty((0, 2, 3), dtype=torch.float32)
    empty_log_gain = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    empty_scaled = producer.score_gain(torch, empty_native, empty_log_gain)
    empty_loss = producer.per_image_BCE(torch, empty_scaled, torch.empty_like(empty_native, dtype=torch.bool))
    empty_gradient = torch.autograd.grad(empty_loss, empty_log_gain)[0]
    require(empty_loss.item() == 0 and torch.isfinite(empty_gradient).item() and empty_gradient.item() == 0,
        'Empty GT gain path is not a connected finite zero')
    rejects(lambda: producer.score_gain(torch, native, torch.tensor([0.0])), 'Nonscalar log gain')
    rejects(lambda: producer.score_gain(torch, native, torch.tensor(1000.0)), 'Overflowed positive gain')
    rejects(lambda: producer.score_gain(torch, native, torch.tensor(-1000.0)), 'Underflowed zero gain')
    require(torch.equal(native, before) and native.grad is None and not native.requires_grad, 'Native original logits were modified')
    return dict(quality_scores=scores, log_gain_gradient=gradient.item(), scalar_gradient=expected,
        finite_difference_gradient=finite_difference, gain_location='after_original_renderer',
        empty_gain_gradient=empty_gradient.item(), connected_empty_gain=True,
        real_FP32_binary_invariance_claimed=False, native_fusion_integration_qualified=False)


def execute(spec_file, output_file):
    require('torch' not in sys.modules, 'Fresh interpreter required before SOURCE gates')
    manifest_path, output = Path(spec_file), Path(output_file)
    require(manifest_path.is_absolute() and output.is_absolute() and manifest_path.suffix == output.suffix == '.json' and
        output.parent.is_dir() and not output.exists(), 'Absolute manifest and fresh JSON output required')
    manifest_path = manifest_path.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    require(set(manifest) == {'schema', 'inputs', 'sources', 'execution'} and manifest['schema'] == SCHEMA and
        manifest['inputs'] == 'SOURCE_ONLY' and manifest['execution'] == EXECUTION and
        set(manifest['sources']) == {'producer', 'fixture', 'contract'}, 'Exact SOURCE-only CPU fixture manifest required')
    sources = manifest['sources']
    folder = Path(__file__).resolve().parent
    for role, filename in (('producer', 'train.py'), ('fixture', 'synthetic_fixture.py'), ('contract', 'SOURCE_CONTRACT.json')):
        require(Path(sources[role]['path']).resolve(strict=True) == folder / filename,
            'Canonical sibling SOURCE required: ' + role)
    require(len({row['path'] for row in sources.values()}) == 3 and
        output.resolve() not in [manifest_path] + [Path(row['path']) for row in sources.values()], 'Fixture path aliases forbidden')
    verified = {role: source_bytes(row, role)[1] for role, row in sources.items()}
    contract = json.loads(verified['contract'].decode('utf-8-sig'))
    require(contract['kind'] == 'INERT_SOURCE_AUTHORING_CONTRACT_NOT_SCIENTIFIC_ADMISSION' and
        contract['fixture_schema'] == SCHEMA and contract['fixture_status'] == STATUS and
        contract['sampled_pixels_per_GT'] == 12544 and contract['sampling_with_replacement'] is True and
        contract['raw_control_mode'] == 'raw', 'Final inert SOURCE contract differs')
    helpers = dict(readout_source=contract['base_role_SOURCE_refs']['readout_source'],
        geometry_source=contract['base_role_SOURCE_refs']['geometry_source'], renderer_source=contract['prior_diag_entry_source'])
    require(contract['helper_SOURCE_refs'] == helpers, 'Explicit canonical fixture helper SOURCE roles differ')
    helper_bytes = {}
    for role, row in helpers.items():
        require(Path(row['path']).suffix == '.py', 'Helper must be SOURCE.py')
        helper_bytes[role] = source_bytes(row, role)[1]
    libraries = contract['library_SOURCE_refs']
    for role, row in libraries.items():
        require(Path(row['path']).suffix == '.py', 'Library pin must refer to SOURCE.py')
        source_bytes(row, role)
    version_ast = ast.parse(source_bytes(libraries['torch_version_source'], 'version_SOURCE')[1])
    version = [ast.literal_eval(node.value) for node in version_ast.body if isinstance(node, ast.Assign) and
        any(isinstance(target, ast.Name) and target.id == '__version__' for target in node.targets)]
    require(version == ['2.0.1+cu118'], 'Pinned Torch version SOURCE differs')
    # No contract scientific payload/receipt/Q/SPEC reference is followed.
    set_cpu_affinity()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '2'
    os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    require(torch.__version__ == '2.0.1+cu118' and torch.get_num_threads() == torch.get_num_interop_threads() == 2 and
        torch.cuda.device_count() == 0 and not torch.cuda.is_initialized(), 'Actual CPU-only Torch execution differs')
    for role, actual in (('torch_version_source', torch.version.__file__), ('interpolation_SOURCE', torch.nn.functional.__file__)):
        require(Path(actual).resolve(strict=True) == Path(libraries[role]['path']).resolve(strict=True),
            'Actual imported library SOURCE path differs: ' + role)
    initial_rng = torch.random.get_rng_state().clone()
    producer = load_verified(sources['producer'], verified['producer'], 'synthetic_control_pure_producer')
    readout = load_verified(helpers['readout_source'], helper_bytes['readout_source'], 'synthetic_control_RAW')
    geometry = load_verified(helpers['geometry_source'], helper_bytes['geometry_source'], 'synthetic_control_geometry')
    renderer = load_verified(helpers['renderer_source'], helper_bytes['renderer_source'], 'synthetic_control_renderer')
    loss_checks = coefficient_and_loss_checks(torch, producer, renderer)
    mapping = assignment_checks(producer)
    raw = raw_gradient_checks(torch, producer, renderer, readout, geometry, contract)
    gain = gain_checks(torch, producer)
    require(torch.equal(torch.random.get_rng_state(), initial_rng) and not torch.cuda.is_initialized(),
        'Fixture changed external CPU RNG or initialized CUDA')
    result = dict(schema=SCHEMA, status=STATUS, sources=sources, spec_path=str(manifest_path), execution=EXECUTION,
        torch_version=str(torch.__version__), helper_SOURCE_refs=helpers, library_SOURCE_refs=libraries,
        synthetic_execution_performed=True, independent_scalar_BCE_and_coefficient_gradients=True,
        duplicate_12544_indices_preserved=True, same_expected_full_BCE_enumeration=True,
        per_GT_then_image_then_accum2_normalization=True, private_CPU_sampling_RNG_restored=True,
        connected_empty_GT=True, connected_empty_gain=True, fixed_assignment_raw_ann_mapping_and_rejections=True,
        raw_cold_warm_two_weight_gradients=True, frozen_F_e0_unchanged_and_no_grad=True,
        gain_after_original_renderer_gradient=True, positive_gain_binary_examples=True,
        quality_score_ranking_change=True, empty_prediction_quality_zero=True, external_CPU_RNG_unchanged=True,
        CUDA_initialized=False, real_data_feasibility_qualified=False, native_model_integration_qualified=False,
        Hungarian_optimality_qualified=False, optimizer_trajectory_qualified=False, AP_computed=False,
        efficacy_claimed=False, convergence_claimed=False, novelty_claimed=False,
        real_FP32_binary_invariance_claimed=False, scalar_loss_checks=loss_checks,
        assignment_checks=mapping, raw_gradient_checks=raw, gain_checks=gain)
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(STATUS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args()
    require(args.owner_execute, 'Inert SOURCE requires explicit ROOT synthetic CPU execution')
    execute(args.spec, args.out)


if __name__ == '__main__':
    main()
