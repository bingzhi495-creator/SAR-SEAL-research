"""Future ROOT-owned CPU synthetic check; SOURCE authoring is not a test run.

Only tiny tensors are used. No model, data, GT file, checkpoint or training entry
is accessed. The scalar reference uses direct 36-vector concatenation; the
candidate uses factorized query chunks. CPU agreement is toleranced, not a GPU
bitwise or memory qualification. Run later with --out freshABS --owner-execute.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import types


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def expect_rejected(function):
    try:
        function()
    except (TypeError, ValueError):
        return
    raise AssertionError('invalid input was accepted')


def reference(torch, model, features, coefficients, logits, descriptors, renderer):
    """Independent direct dense graph, including explicit border neighborhoods."""
    functional = torch.nn.functional
    pixel = renderer(functional.conv2d(features.detach(),
                     model.feature_projection.weight, model.feature_projection.bias))
    query = functional.linear(coefficients.detach(), model.query_projection.weight,
                              model.query_projection.bias)
    native = logits.detach()
    probability = native.sigmoid()
    binary = native > 0
    batch, queries, height, width = native.shape
    contrast_rows = []
    for y in range(height):
        contrast_columns = []
        for x in range(width):
            neighborhood = probability[..., max(0, y-1):min(height, y+2),
                                       max(0, x-1):min(width, x+2)]
            contrast_columns.append(probability[..., y, x] - neighborhood.mean((-2, -1)))
        contrast_rows.append(torch.stack(contrast_columns, -1))
    contrast = torch.stack(contrast_rows, -2)
    inputs = torch.cat((pixel.permute(0, 2, 3, 1)[:, None].expand(-1, queries, -1, -1, -1),
                        query[:, :, None, None].expand(-1, -1, height, width, -1),
                        native[..., None], probability[..., None],
                        binary.to(native.dtype)[..., None], contrast[..., None]), -1)
    hidden = functional.linear(inputs, model.hidden.weight, model.hidden.bias)
    hidden = hidden * hidden.sigmoid()
    support = binary.to(native.dtype)
    foreground = (hidden * support[..., None]).sum((2, 3))
    foreground = foreground / support.sum((2, 3)).clamp_min(1)[..., None]
    background = (hidden * (1-support)[..., None]).sum((2, 3))
    background = background / (1-support).sum((2, 3)).clamp_min(1)[..., None]
    pooled = torch.cat((foreground, background), -1)
    quality = functional.linear(torch.cat((pooled, descriptors.detach()), -1),
                                model.final.weight, model.final.bias).squeeze(-1).sigmoid()
    return quality, pooled, inputs


def run_checks(torch, source):
    state = torch.get_rng_state().clone()
    cls = source.FeaturePooledQueryIoUReadout
    first, second, different = cls(2026100911), cls(2026100911), cls(2026100912)
    require(torch.equal(state, torch.get_rng_state()), 'construction changed ambient CPU RNG')
    require(len(list(first.parameters())) == 8 and sum(p.numel() for p in first.parameters()) == 9477,
            'wrong parameter count')
    require(not list(first.buffers()), 'unexpected registered buffers')
    require(all(torch.equal(first.state_dict()[k], second.state_dict()[k]) for k in first.state_dict()),
            'same seed initialization differs')
    require(not torch.equal(first.hidden.weight, different.hidden.weight), 'different seed ignored')
    require(all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in first.parameters()),
            'constructor did not explicitly use CPU FP32')
    model = first.double()
    features = (torch.arange(2*256*2*3, dtype=torch.float64).reshape(2, 256, 2, 3)/83).sin()
    coefficients = (torch.arange(2*100*256, dtype=torch.float64).reshape(2, 100, 256)/31).cos()
    logits = torch.linspace(-1, 1, 35, dtype=torch.float64).reshape(1, 1, 5, 7).repeat(2, 100, 1, 1)
    logits[:, 0] = -2
    logits[:, 1] = 2
    class_probability = torch.linspace(.1, .9, 200, dtype=torch.float64).reshape(2, 100)
    binary, probability = logits > 0, logits.sigmoid()
    area = binary.to(logits.dtype).sum((-2, -1))
    inside = (binary * probability).sum((-2, -1))
    official = inside / (area + 1e-6)
    budget = inside / (area + ((~binary) * probability).sum((-2, -1)) + 1e-6)
    descriptors = torch.stack((class_probability, area/35, official, budget), -1)
    teacher = [t.requires_grad_() for t in (features, coefficients, logits, descriptors)]
    snapshots = [t.detach().clone() for t in teacher]
    versions = [t._version for t in teacher]

    def renderer(pixel):
        intermediate = torch.nn.functional.interpolate(pixel, (8, 12), mode='bilinear', align_corners=False)
        return torch.nn.functional.interpolate(intermediate[..., :7, :10], (5, 7),
                                              mode='bilinear', align_corners=False)

    targets = torch.linspace(0, .4, 200, dtype=torch.float64).reshape(2, 100).requires_grad_()
    cold = model(*teacher, renderer=renderer)
    require(torch.equal(cold, torch.full_like(cold, .5)), 'zero final layer is not exactly cold .5')
    cold_loss = source.all_query_mse(cold, targets)
    cold_loss.backward()
    for name, parameter in model.named_parameters():
        require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()), 'missing cold gradient')
        if name.startswith('final.'):
            require(bool(torch.count_nonzero(parameter.grad) > 0), 'cold final gradient inactive')
        else:
            require(bool(torch.count_nonzero(parameter.grad) == 0), 'cold upstream gradient not exactly zero')
    require(targets.grad is None, 'MSE sent gradients into targets')
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.final.weight.copy_(torch.linspace(-.13, .17, 68, dtype=torch.float64).reshape(1, 68))
        model.final.bias.fill_(-.03)
    final_inputs = []
    handle = model.final.register_forward_pre_hook(lambda _module, inputs: final_inputs.append(inputs[0].detach().clone()))
    try:
        actual = model(*teacher, renderer=renderer, query_chunk_size=4)
    finally:
        handle.remove()
    expected, pooled, direct_inputs = reference(torch, model, *teacher, renderer)
    require(torch.allclose(actual, expected, rtol=1e-10, atol=1e-12), 'factorized vs direct dense mismatch')
    require(len(final_inputs) == 1 and final_inputs[0].shape == (2, 100, 68), 'wrong actual final input')
    actual_pool = final_inputs[0][..., :64]
    require(torch.allclose(actual_pool, pooled, rtol=1e-10, atol=1e-12), 'actual pooled vectors disagree with reference')
    require(torch.equal(actual_pool[:, 0, :32], torch.zeros_like(actual_pool[:, 0, :32])), 'empty FG pool not zero')
    require(torch.equal(actual_pool[:, 1, 32:], torch.zeros_like(actual_pool[:, 1, 32:])), 'empty BG pool not zero')
    require(torch.equal(final_inputs[0][..., 64:], descriptors.detach()), 'native descriptors changed before final')
    # A fully scalar spot check does not call linear/SiLU from the candidate.
    hidden_pixels = []
    for y in range(5):
        for x in range(7):
            vector = direct_inputs[0, 2, y, x].detach().tolist()
            row = []
            for weight, bias in zip(model.hidden.weight.detach().tolist(), model.hidden.bias.detach().tolist()):
                value = math.fsum(a*b for a, b in zip(weight, vector)) + bias
                row.append(value/(1+math.exp(-value)))
            hidden_pixels.append((bool(binary[0, 2, y, x]), row))
    groups = [[row for selected, row in hidden_pixels if selected == flag] for flag in (True, False)]
    scalar_pool = [math.fsum(row[c] for row in rows)/len(rows) if rows else 0.0
                   for rows in groups for c in range(32)]
    vector = scalar_pool + descriptors[0, 2].detach().tolist()
    value = math.fsum(a*b for a, b in zip(model.final.weight[0].detach().tolist(), vector))
    value += float(model.final.bias)
    require(abs(float(actual[0, 2]) - 1/(1+math.exp(-value))) < 1e-12, 'scalar pooled formula mismatch')
    loss = source.all_query_mse(actual, targets)
    require(torch.allclose(loss, ((actual-targets.detach())**2).mean(1).mean(), rtol=0, atol=0), 'MSE reduction wrong')
    parameters = tuple(model.parameters())
    gradients = torch.autograd.grad(loss, parameters)
    reference_gradients = torch.autograd.grad(((expected-targets.detach())**2).mean(), parameters)
    require(all(bool(torch.isfinite(g).all()) and bool(torch.count_nonzero(g) > 0) for g in gradients),
            'warm gradients do not reach all eight tensors')
    require(all(torch.allclose(a, b, rtol=1e-9, atol=1e-10) for a, b in zip(gradients, reference_gradients)),
            'independent dense gradients disagree')
    # Direct finite difference through the projection and both renderer stages.
    parameter, epsilon = model.feature_projection.weight, 1e-5
    center = float(parameter[0, 0, 0, 0])
    values = []
    for direction in (1, -1):
        with torch.no_grad():
            parameter[0, 0, 0, 0] = center + direction*epsilon
        q, _, _ = reference(torch, model, *teacher, renderer)
        values.append(float(((q-targets.detach())**2).mean()))
    with torch.no_grad():
        parameter[0, 0, 0, 0] = center
    finite_difference = (values[0]-values[1])/(2*epsilon)
    require(abs(finite_difference-float(gradients[0][0, 0, 0, 0])) < 1e-8, 'projection finite difference mismatch')
    norm = math.sqrt(math.fsum(float((g*g).sum()) for g in gradients))
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = gradient.clone()
    measured = torch.nn.utils.clip_grad_norm_(parameters, .01)
    scale = min(1.0, .01/(norm+1e-6))
    require(abs(float(measured)-norm) < 1e-10, 'preclip norm mismatch')
    require(all(torch.allclose(p.grad, g*scale, rtol=1e-10, atol=1e-12)
                for p, g in zip(parameters, gradients)), 'clipping differs from independent scalar scale')
    model.zero_grad(set_to_none=True)
    checkpointed = cls(17).to(device='cpu', dtype=torch.float64)
    checkpointed.load_state_dict(model.state_dict(), strict=True)
    checkpoint_quality = checkpointed(*teacher, renderer=renderer, activation_checkpoint=True)
    source.all_query_mse(checkpoint_quality, targets).backward()
    require(torch.allclose(checkpoint_quality, actual, rtol=1e-10, atol=1e-12), 'checkpoint forward mismatch')
    require(all(torch.allclose(p.grad, g, rtol=1e-9, atol=1e-10)
                for p, g in zip(checkpointed.parameters(), gradients)), 'checkpoint backward mismatch')
    with torch.no_grad():
        for chunk in (1, 7, 100):
            require(torch.allclose(model(*teacher, renderer=renderer, query_chunk_size=chunk), actual,
                                   rtol=1e-10, atol=1e-12), 'query chunk size changed values')
        permutation = torch.arange(99, -1, -1)
        permuted = model(features, coefficients[:, permutation], logits[:, permutation],
                         descriptors[:, permutation], renderer=renderer)
        require(torch.allclose(permuted, actual[:, permutation], rtol=1e-10, atol=1e-12), 'query order not preserved')
    require(all(t.grad is None and torch.equal(t.detach(), s) and t._version == v
                for t, s, v in zip(teacher, snapshots, versions)), 'teacher inputs mutated or received gradients')
    fp32 = cls(19).to(device='cpu', dtype=torch.float32)
    fp32.load_state_dict(model.state_dict(), strict=True)
    fp32_inputs = [t.detach().float() for t in teacher]
    actual32 = fp32(*fp32_inputs, renderer=renderer)
    expected32, _, _ = reference(torch, fp32, *fp32_inputs, renderer)
    require(torch.allclose(actual32, expected32, rtol=1e-5, atol=1e-6), 'FP32 direct dense mismatch')
    bad_logits = logits.detach().clone()
    bad_logits[0, 0, 0, 0] = float('nan')
    bad_descriptors = descriptors.detach().clone()
    bad_descriptors[0, 0, 0] = 1.1
    expect_rejected(lambda: cls(True))
    expect_rejected(lambda: model(*teacher, renderer=renderer, query_chunk_size=True))
    expect_rejected(lambda: model(*teacher, renderer=renderer, activation_checkpoint=1))
    expect_rejected(lambda: model(features.float(), *teacher[1:], renderer=renderer))
    expect_rejected(lambda: model(features, coefficients[:, :99], logits[:, :99], descriptors[:, :99], renderer=renderer))
    expect_rejected(lambda: model(*teacher))
    expect_rejected(lambda: model(*teacher, renderer=lambda p: renderer(p).detach()))
    expect_rejected(lambda: model(features, coefficients, bad_logits, descriptors, renderer=renderer))
    expect_rejected(lambda: model(features, coefficients, logits, bad_descriptors, renderer=renderer))
    expect_rejected(lambda: cls(29).half()(*fp32_inputs, renderer=renderer))
    expect_rejected(lambda: source.all_query_mse(actual, torch.full_like(targets, float('nan'))))
    expect_rejected(lambda: source.all_query_mse(actual, torch.full_like(targets, 1.1)))
    require(torch.equal(state, torch.get_rng_state()), 'forward/backward or checkpoint changed ambient CPU RNG')
    return dict(private_CPU_RNG=True, eight_tensors_9477_no_buffers=True,
                direct_dense_and_scalar_formula=True, native_inner_outer_zero_support=True,
                cold_final_only_warm_all8=True, independent_MSE_gradients=True,
                projection_renderer_finite_difference=True, scalar_clip_check=True,
                nonreentrant_checkpoint_forward_backward=True, query_chunks_and_order=True,
                frozen_teacher_inputs_unchanged_gradNone=True, invalid_inputs_rejected=True,
                native_model_integration_tested=False, GPU_memory_qualified=False,
                efficacy_or_calibration_claimed=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args()
    out = Path(args.out)
    require(args.owner_execute and out.is_absolute() and not out.exists(), 'owner flag and fresh absolute output required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CUDA must be hidden before scientific import')
    require(all(os.environ.get(k) == '2' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')),
            'numeric thread environment must be 2')
    import psutil
    require(sorted(psutil.Process().cpu_affinity()) == [0, 1], 'CPU affinity must be 0,1')
    path = Path(__file__).resolve().with_name('feature_pooled_iou_control.py')
    source_bytes = path.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    import torch
    require(not torch.cuda.is_initialized(), 'CUDA initialized before fixture')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    module = types.ModuleType('fresh_feature_pooled_iou_control')
    module.__file__ = str(path)
    require(hashlib.sha256(path.read_bytes()).hexdigest() == source_sha, 'SOURCE changed before verified-byte load')
    exec(compile(source_bytes, str(path), 'exec'), module.__dict__)
    checks = run_checks(torch, module)
    require(not torch.cuda.is_initialized(), 'fixture initialized CUDA')
    result = dict(schema='query-pixel-feature-pooled-IoU-control-synthetic-CPU-fixture-v1',
                  status='QUERY_PIXEL_FEATURE_POOLED_IOU_CONTROL_SYNTHETIC_CPU_PASS',
                  sources=dict(module=dict(path=str(path), bytes=len(source_bytes), sha256=source_sha)),
                  CPU_execution=dict(affinity=[0, 1], CUDA_VISIBLE_DEVICES='',
                                     torch_threads=torch.get_num_threads(),
                                     torch_interop_threads=torch.get_num_interop_threads()),
                  torch_version=str(torch.__version__), assertions=checks)
    data = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n').encode('utf-8')
    descriptor = os.open(str(out), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'wb') as handle:
        handle.write(data)
    print(json.dumps(dict(status=result['status'], output=str(out)), sort_keys=True))


if __name__ == '__main__':
    main()
