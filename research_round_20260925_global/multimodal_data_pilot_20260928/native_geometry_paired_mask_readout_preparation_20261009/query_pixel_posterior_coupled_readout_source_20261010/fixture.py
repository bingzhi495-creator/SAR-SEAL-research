"""Small synthetic CPU fixture; SOURCE preparation does not execute it."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

SCHEMA = 'ROOT-query-pixel-posterior-readout-synthetic-SOURCE-spec-v1'
STATUS = 'QUERY_PIXEL_POSTERIOR_READOUT_SYNTHETIC_CPU_PASS'
EXECUTION = dict(device='cpu', cpu_affinity=[0, 1], torch_threads=2,
                 torch_interop_threads=2, cuda_visible_devices='')
SHAPES = {'feature_projection.weight': [16, 256, 1, 1], 'feature_projection.bias': [16],
          'query_projection.weight': [16, 256], 'query_projection.bias': [16],
          'hidden.weight': [32, 36], 'hidden.bias': [32], 'final.weight': [1, 32], 'final.bias': [1]}
TOLS = {'float32': dict(rtol=5e-5, atol=5e-6), 'float64': dict(rtol=1e-10, atol=1e-10)}
KEYS = {'posterior', 'corrected_logits', 'bounded_delta'}
POINT_KEYS = KEYS | {'posterior_logits'}

def check(value, message):
    if not value:
        raise AssertionError(message)

def reference(path):
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

def source_gate(spec, out, owner):
    check(owner and spec.is_absolute() and out.is_absolute(), 'owner/absolute paths required')
    check(not out.exists() and out.parent.is_dir(), 'output must be fresh')
    manifest = json.loads(spec.read_text(encoding='utf-8'))
    check(set(manifest) == {'schema', 'inputs', 'sources', 'execution'} and
          manifest['schema'] == SCHEMA and manifest['inputs'] == 'SOURCE_ONLY' and
          manifest['execution'] == EXECUTION, 'manifest differs')
    check(set(manifest['sources']) == {'operator', 'fixture'}, 'two SOURCE refs required')
    here, refs = Path(__file__).resolve(), {}
    for name, path in (('fixture', here), ('operator', here.parent / 'query_pixel_readout.py')):
        ref = manifest['sources'][name]
        check(set(ref) == {'path', 'bytes', 'sha256'} and Path(ref['path']).is_absolute() and
              Path(ref['path']).resolve() == path, 'wrong SOURCE path')
        refs[name] = reference(path)
        check(ref['bytes'] == refs[name]['bytes'] and ref['sha256'] == refs[name]['sha256'], 'SOURCE pin differs')
    check(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and 'torch' not in sys.modules,
          'CUDA hidden/Torch unimported required before SOURCE gate')
    return refs

def same(a, b, label):
    check(a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b), label)

def close(a, b, label):
    check(a.dtype == b.dtype and a.shape == b.shape and bool(torch.isfinite(a).all()) and
          bool(torch.isfinite(b).all()) and torch.allclose(a, b, **TOLS[str(a.dtype).split('.')[-1]]), label)
    return float((a - b).abs().max()) if a.numel() else 0.

def contrast(logits):
    p, result = torch.sigmoid(logits), torch.empty_like(logits)
    h, w = logits.shape[-2:]
    for y in range(h):
        for x in range(w):
            patch = p[..., max(0, y - 1):min(h, y + 2), max(0, x - 1):min(w, x + 2)]
            result[..., y, x] = p[..., y, x] - patch.mean(dim=(-2, -1))
    return result  # Explicit valid-neighbor count, not a padded divisor.

def rows(pixel, query, logits):
    b, q, h, w = logits.shape
    p = pixel.permute(0, 2, 3, 1)[:, None].expand(b, q, h, w, 16)
    e = query[:, :, None, None].expand(b, q, h, w, 16)
    return p.reshape(-1, 16), e.reshape(-1, 16), logits.reshape(-1), contrast(logits).reshape(-1)

def oracle(model, pixel, query, logits, c):
    inputs = torch.cat((pixel, query, logits[:, None], torch.sigmoid(logits)[:, None],
                        (logits > 0).to(logits.dtype)[:, None], c[:, None]), dim=1)
    h = inputs.matmul(model.hidden.weight.t()) + model.hidden.bias
    u = ((h * torch.sigmoid(h)).matmul(model.final.weight.t()) + model.final.bias).squeeze(1)
    delta = model.kappa * torch.tanh(u)
    return dict(posterior_logits=logits + u, posterior=torch.sigmoid(logits + u),
                corrected_logits=logits + delta, bounded_delta=delta)

def budget(p, logits, eps):
    b = (logits > 0).to(p.dtype)
    return (b * p).sum((-2, -1)) / (b.sum((-2, -1)) + ((1 - b) * p).sum((-2, -1)) + eps)

def outputs(result, shape, dense=False):
    check(set(result) == (KEYS | {'budget'} if dense else POINT_KEYS), 'output keys')
    for name, value in result.items():
        check(list(value.shape) == list(shape[:2] if name == 'budget' else shape) and
              value.device.type == 'cpu' and bool(torch.isfinite(value).all()), 'output shape/finite')
        check(not dense or not value.requires_grad, 'dense autograd')
    for name in ('posterior', 'budget') if dense else ('posterior',):
        check(bool(((result[name] >= 0) & (result[name] <= 1)).all()), 'probability range')

def gradients(model):
    result = {}
    for name, p in model.named_parameters():
        check(p.grad is not None and bool(torch.isfinite(p.grad).all()), 'missing/nonfinite grad ' + name)
        result[name] = dict(norm=float(p.grad.norm()), nonzero=int(torch.count_nonzero(p.grad)))
    return result

def numerical_checks(cls):
    rng = torch.get_rng_state().clone()
    model, repeat, other = cls(seed=107), cls(seed=107), cls(seed=108)
    params = dict(model.named_parameters())
    check(list(params) == list(SHAPES) and sum(p.numel() for p in params.values()) == 9441 and
          not list(model.named_buffers()), '9441 scalars/eight tensors/no buffers')
    for name, p in params.items():
        check(list(p.shape) == SHAPES[name] and p.device.type == 'cpu' and
              p.dtype == torch.float32 and p.requires_grad, 'parameter properties')
        same(p, repeat.state_dict()[name], 'same seed')
    check(any(not torch.equal(p, other.state_dict()[n]) for n, p in params.items()
              if not n.startswith('final.')), 'different seed')
    check(bool((model.final.weight == 0).all()) and bool((model.final.bias == 0).all()), 'zero final layer')
    same(torch.get_rng_state(), rng, 'constructor CPU RNG')
    gen = torch.Generator(device='cpu').manual_seed(109)
    f = torch.randn(2, 256, 2, 3, generator=gen) * .2
    e = torch.randn(2, 7, 256, generator=gen) * .2
    l = torch.randn(2, 7, 3, 5, generator=gen) * 1.3
    originals = [v.clone() for v in (f, e, l)]
    projected, q = model.project_features(f), model.project_queries(e)
    close(projected, (f.permute(0, 2, 3, 1).matmul(model.feature_projection.weight[:, :, 0, 0].t()) +
                      model.feature_projection.bias).permute(0, 3, 1, 2), 'Conv projection')
    close(q, e.matmul(model.query_projection.weight.t()) + model.query_projection.bias, 'query projection')
    render = lambda v: torch.nn.functional.interpolate(v, size=(3, 5), mode='bilinear', align_corners=False)
    p = render(projected)  # Project at native resolution, then external synthetic renderer.
    cold, dense = model.point_outputs(*rows(p, q, l)), model.dense_outputs(p, q, l)
    outputs(cold, [l.numel()])
    outputs(dense, l.shape, True)
    for actual, native in ((cold, l.reshape(-1)), (dense, l)):
        same(actual['corrected_logits'], native, 'cold corrected identity')
        same(actual['bounded_delta'], torch.zeros_like(native), 'cold delta')
        same(actual['posterior'], torch.sigmoid(native), 'cold posterior')
    same(cold['posterior_logits'], l.reshape(-1), 'cold stable posterior logits')
    close(dense['budget'], budget(dense['posterior'], l, model.eps), 'cold budget')
    empty_points = model.point_outputs(*(v[:0] for v in rows(p, q, l)))
    outputs(empty_points, [0])
    empty_grad = torch.autograd.grad(empty_points['posterior_logits'].sum(), tuple(model.parameters()),
                                    allow_unused=False, retain_graph=True)
    check(all(bool(torch.isfinite(g).all()) and bool((g == 0).all()) for g in empty_grad), 'connected empty gradients')
    target = ((torch.arange(l.numel()) % 3) == 0).float()  # Synthetic labels, no real GT.
    torch.nn.functional.binary_cross_entropy_with_logits(cold['posterior_logits'], target).backward()
    cold_grad = gradients(model)
    check(all(v['nonzero'] > 0 if n.startswith('final.') else v['nonzero'] == 0
              for n, v in cold_grad.items()), 'constructed cold gradient pattern')
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.final.weight.copy_(torch.linspace(-.12, .12, 32).reshape(1, 32))
        model.final.bias.fill_(.07)
    warm = model.point_outputs(*rows(render(model.project_features(f)), model.project_queries(e), l))
    torch.nn.functional.binary_cross_entropy_with_logits(warm['posterior_logits'], target).backward()
    warm_grad = gradients(model)
    check(all(v['nonzero'] > 0 for v in warm_grad.values()), 'constructed warm gradients')
    for value, original in zip((f, e, l), originals):
        same(value, original, 'frozen native operand')
        check(value.grad is None, 'native operand gradient')
    errors = {}
    for dtype in (torch.float32, torch.float64):
        m = cls(seed=107).to(dtype=dtype)
        m.load_state_dict(model.state_dict(), strict=True)
        p = torch.randn(2, 16, 3, 5, generator=gen, dtype=dtype, requires_grad=True)
        q = torch.randn(2, 7, 16, generator=gen, dtype=dtype, requires_grad=True)
        native = torch.randn(2, 7, 3, 5, generator=gen, dtype=dtype, requires_grad=True)
        expected, point = oracle(m, *rows(p, q, native)), m.point_outputs(*rows(p, q, native))
        err = [close(point[k], expected[k], 'independent point formula') for k in POINT_KEYS]
        expected = {k: v.reshape(native.shape) for k, v in expected.items() if k in KEYS}
        expected['budget'] = budget(expected['posterior'], expected['corrected_logits'], m.eps)
        default, grad_mode = m.dense_outputs(p, q, native), torch.is_grad_enabled()
        for chunk in (1, 2, 4, 7, 11):
            actual = m.dense_outputs(p, q, native, query_chunk_size=chunk)
            outputs(actual, native.shape, True)
            check(bool((actual['bounded_delta'].abs() <= m.kappa).all()), 'kappa bound')
            for k in expected:
                err.append(close(actual[k], expected[k], 'point/dense algebra'))
                close(actual[k], default[k], 'chunk invariance')
        check(torch.is_grad_enabled() == grad_mode, 'no_grad context leak')
        errors[str(dtype)] = max(err)
    edge = cls(seed=107).double()
    with torch.no_grad():
        edge.hidden.weight.zero_()
        edge.hidden.bias.zero_()
        edge.hidden.weight[0, 35], edge.hidden.bias[0] = 1.25, .2
        edge.final.weight.zero_()
        edge.final.weight[0, 0], edge.final.bias[0] = .4, -.1
    for h, w in ((1, 1), (1, 4), (2, 3), (3, 5)):
        native = torch.arange(3 * h * w, dtype=torch.float64).reshape(1, 3, h, w) * .27 - 1.8
        p, q = torch.zeros(1, 16, h, w, dtype=torch.float64), torch.zeros(1, 3, 16, dtype=torch.float64)
        expected, actual = oracle(edge, *rows(p, q, native)), edge.dense_outputs(p, q, native, query_chunk_size=2)
        for k in KEYS:
            close(actual[k], expected[k].reshape(native.shape), 'contrast-only edge replay')
        if (h, w) == (1, 1):
            same(contrast(native), torch.zeros_like(native), '1x1 valid-neighbor contrast')
    constant = cls(seed=107)
    with torch.no_grad():
        constant.final.weight.zero_()
        constant.final.bias.fill_(.75)
    p, q, native = torch.zeros(2, 16, 2, 3), torch.zeros(2, 100, 16), torch.full((2, 100, 2, 3), -.3)
    actual = constant.dense_outputs(p, q, native)
    outputs(actual, native.shape, True)
    close(actual['posterior'], torch.sigmoid(native + .75), 'constant-u posterior')
    close(actual['bounded_delta'], torch.full_like(native, math.tanh(.75)), 'constant-u delta')
    close(actual['corrected_logits'], native + math.tanh(.75), 'constant-u corrected')
    check(bool((actual['corrected_logits'] > 0).all()), 'support must flip')
    close(actual['budget'], budget(actual['posterior'], actual['corrected_logits'], constant.eps), 'final support budget')
    check(bool((actual['budget'] > .5).all()) and bool((budget(actual['posterior'], native, constant.eps) == 0).all()),
          'native vs corrected support distinction')
    empty = constant.dense_outputs(p, q, torch.full_like(native, -20), query_chunk_size=13)
    outputs(empty, native.shape, True)
    check(bool((empty['corrected_logits'] < 0).all()) and bool((empty['posterior'] > 0).all()), 'empty mask case')
    same(empty['budget'], torch.zeros(2, 100), 'empty binary mask score zero')
    small = cls(seed=107, kappa=.25, eps=1e-6)
    small.load_state_dict(constant.state_dict(), strict=True)
    actual = small.point_outputs(torch.zeros(6, 16), torch.zeros(6, 16), torch.linspace(-.4, .4, 6), torch.zeros(6))
    close(actual['bounded_delta'], torch.full((6,), .25 * math.tanh(.75)), 'nondefault kappa')
    check(bool((actual['bounded_delta'].abs() <= .25).all()), 'nondefault bound')
    same(torch.get_rng_state(), rng, 'whole fixture CPU RNG')
    return dict(initialization_RNG=True, cold_identity=True, training_gradients=True,
                independent_formulas=True, point_dense_edge_contrast=True, chunk_all100_empty_budget=True,
                cold_gradient_norms={k: v['norm'] for k, v in cold_grad.items()},
                warm_gradient_norms={k: v['norm'] for k, v in warm_grad.items()}, max_errors=errors)

def run(spec, out, owner):
    global torch
    # Only two fresh SOURCE files are bound; no scientific payload dependencies.
    sources = source_gate(spec, out, owner)
    import psutil
    psutil.Process().cpu_affinity([0, 1])
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    check(sorted(psutil.Process().cpu_affinity()) == [0, 1] and torch.get_num_threads() == 2 and
          torch.get_num_interop_threads() == 2, 'CPU resources')
    check(not torch.cuda.is_initialized(), 'CUDA already initialized')
    rng = torch.get_rng_state().clone()
    loader = importlib.util.spec_from_file_location('_synthetic_posterior_operator', sources['operator']['path'])
    module = importlib.util.module_from_spec(loader)
    code = Path(sources['operator']['path']).read_bytes()
    check(len(code) == sources['operator']['bytes'] and
          hashlib.sha256(code).hexdigest() == sources['operator']['sha256'], 'operator SOURCE changed')
    exec(compile(code, sources['operator']['path'], 'exec'), module.__dict__)
    same(torch.get_rng_state(), rng, 'operator import CPU RNG')
    result = dict(status=STATUS, sources=sources, spec=reference(spec), execution=EXECUTION,
                  synthetic_execution_performed=True, synthetic_inputs_only=True,
                  parameter_count=9441, parameter_tensors=8, torch_version=str(torch.__version__),
                  checks=numerical_checks(module.QueryPixelPosteriorReadout), tolerances=TOLS,
                  calibration_guaranteed=False, expected_IoU_guaranteed=False, novelty_claimed=False,
                  real_data_feasibility_qualified=False, native_integration_qualified=False, efficacy_claimed=False,
                  limits=['Synthetic CPU only; no GT/checkpoint/native forward or optimizer replay.',
                          'Point/dense agreement is toleranced, not bitwise; no GPU numerical guarantee.',
                          'Budget ratio is not generally expected IoU; no calibration or mask improvement guarantee.',
                          'Synthetic nonzero gradients do not guarantee real-image nonzero gradients.',
                          'ROOT actual tool terminal is separate evidence.'])
    check(not torch.cuda.is_initialized(), 'fixture initialized CUDA')
    with out.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, allow_nan=False, indent=2)
        handle.write('\n')
    return dict(status=STATUS, out=str(out))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.spec, args.out, args.owner_execute)))
