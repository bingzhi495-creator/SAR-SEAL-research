"""Synthetic adapter checks only. SOURCE preparation does not execute them."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import types

SCHEMA = 'ROOT-query-pixel-posterior-native-adapter-synthetic-SOURCE-spec-v1'
STATUS = 'QUERY_PIXEL_POSTERIOR_NATIVE_ADAPTER_SYNTHETIC_CPU_PASS'
EXECUTION = dict(device='cpu', cpu_affinity=[0, 1], torch_threads=2,
                 torch_interop_threads=2, cuda_visible_devices='')
IMMUTABLE_PINS = {
    'readout': (11555, '5c309b4d1a2318e8c4d673d343dbe956f1387d4b01180de8cd796d0726457305'),
    'renderer': (45195, 'dd20f440e11b6890f0f797b8f048f6953adf2a8ee8c33e74cfe91874df78018b'),
}
TOLERANCE = dict(rtol=5e-5, atol=5e-6)

def check(value, message):
    if not value:
        raise AssertionError(message)

def source_gate(spec, out, owner):
    check(owner and spec.is_absolute() and out.is_absolute(), 'owner/absolute paths required')
    check(not out.exists() and out.parent.is_dir(), 'fresh output required')
    manifest = json.loads(spec.read_text(encoding='utf-8'))
    check(set(manifest) == {'schema', 'inputs', 'sources', 'execution'} and
          manifest['schema'] == SCHEMA and manifest['inputs'] == 'SOURCE_ONLY' and
          manifest['execution'] == EXECUTION, 'manifest differs')
    here = Path(__file__).resolve()
    paths = dict(fixture=here, adapter=here.parent / 'native_adapter.py',
                 readout=here.parent / 'query_pixel_readout.py',
                 renderer=here.parent.parent / 'frozen_feature_linear_readout_diagnostic_source_20261010' / 'diagnose.py')
    check(set(manifest['sources']) == set(paths), 'four SOURCE roles required')
    refs, code = {}, {}
    for name, path in paths.items():
        ref = manifest['sources'][name]
        check(set(ref) == {'path', 'bytes', 'sha256'} and Path(ref['path']).is_absolute() and
              Path(ref['path']).resolve() == path, 'SOURCE path differs')
        code[name] = path.read_bytes()
        refs[name] = dict(path=str(path), bytes=len(code[name]), sha256=hashlib.sha256(code[name]).hexdigest())
        check(ref['bytes'] == refs[name]['bytes'] and ref['sha256'] == refs[name]['sha256'], 'SOURCE pin differs')
        if name in IMMUTABLE_PINS:
            check((ref['bytes'], ref['sha256']) == IMMUTABLE_PINS[name], 'immutable SOURCE declaration differs')
    check(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and 'torch' not in sys.modules,
          'CUDA hidden/Torch unimported required before SOURCE gate')
    return refs, code

def load_source(name, ref, code):
    module = types.ModuleType('_synthetic_adapter_' + name)
    module.__file__ = ref['path']
    exec(compile(code, ref['path'], 'exec'), module.__dict__)
    return module

def close(a, b, label):
    check(a.shape == b.shape and a.dtype == b.dtype and bool(torch.isfinite(a).all()) and
          bool(torch.isfinite(b).all()) and torch.allclose(a, b, **TOLERANCE), label)
    return float((a - b).abs().max()) if a.numel() else 0.

def manual_render(value, crop, original):
    padded = torch.nn.functional.interpolate(value, size=(800, 800), mode='bilinear', align_corners=False)
    cropped = padded[0, :, :crop[0], :crop[1]][:, None]
    return torch.nn.functional.interpolate(cropped, size=original, mode='bilinear', align_corners=False)[:, 0]

def manual_contrast(logits):
    p, result = logits.sigmoid(), torch.empty_like(logits)
    h, w = logits.shape[-2:]
    for y in range(h):
        for x in range(w):
            patch = p[..., max(0, y - 1):min(h, y + 2), max(0, x - 1):min(w, x + 2)]
            result[..., y, x] = p[..., y, x] - patch.mean((-2, -1))
    return result

def manual_loss(readout, fields, query_ids, masks, indices):
    losses, width = [], masks.shape[-1]
    for row, query_id in enumerate(query_ids.tolist()):
        y, x = indices[row] // width, indices[row] % width
        pixel = fields['projected_pixel'][0, :, y, x].t()
        query = fields['projected_query'][0, query_id].expand(indices.shape[1], 16)
        native = fields['native_logits'][0, query_id, y, x]
        contrast = fields['contrast'][0, query_id, y, x]
        logits = readout.point_outputs(pixel, query, native, contrast)['posterior_logits']
        losses.append(torch.nn.functional.binary_cross_entropy_with_logits(logits, masks[row, y, x].float()))
    return torch.stack(losses).mean(), torch.stack(losses)

def gradient_counts(readout):
    result = {}
    for name, p in readout.named_parameters():
        check(p.grad is not None and p.grad.shape == p.shape and bool(torch.isfinite(p.grad).all()), 'readout gradient')
        result[name] = dict(norm=float(p.grad.norm()), nonzero=int(torch.count_nonzero(p.grad)))
    check(len(result) == 8, 'eight parameter gradients required')
    return result

def numerical_checks(readout_module, adapter, renderer_module):
    rng = torch.get_rng_state().clone()
    readout = readout_module.QueryPixelPosteriorReadout(seed=211)
    gen = torch.Generator(device='cpu').manual_seed(212)
    feature = (torch.randn(1, 256, 200, 200, generator=gen) * .1).requires_grad_()
    query = (torch.randn(1, 100, 256, generator=gen) * .1).requires_grad_()
    axis = torch.linspace(-1, 1, 200)
    spatial = .4 * axis[:, None] + .25 * axis[None, :]
    native = (torch.linspace(-2, 2, 100)[None, :, None, None] + spatial[None, None]).requires_grad_()
    originals = [v.detach().clone() for v in (feature, query, native)]
    crop, original = (613, 757), (7, 11)
    renderer = lambda value, identity: renderer_module.render_original(torch, value, identity)
    fields = adapter.project_native_fields(readout, feature, query, native, crop, original, renderer)
    expected_shapes = dict(projected_pixel=[1, 16, 7, 11], projected_query=[1, 100, 16],
                           native_logits=[1, 100, 7, 11], contrast=[1, 100, 7, 11])
    check(set(fields) == set(expected_shapes), 'field keys')
    for name, value in fields.items():
        check(list(value.shape) == expected_shapes[name] and value.dtype == torch.float32 and
              value.device.type == 'cpu' and bool(torch.isfinite(value).all()), 'field properties')
        check(value.requires_grad == name.startswith('projected_'), 'detached/live fields')
    with torch.no_grad():
        expected_pixel = manual_render(readout.project_features(feature.detach()), crop, original)[None]
        expected_query = readout.project_queries(query.detach())
        expected_native = manual_render(native.detach(), crop, original)[None]
        direct = torch.nn.functional.interpolate(native.detach(), size=original, mode='bilinear', align_corners=False)
    errors = dict(pixel=close(fields['projected_pixel'], expected_pixel, 'project/render/crop pixels'),
                  query=close(fields['projected_query'], expected_query, 'all100 query identity'),
                  native=close(fields['native_logits'], expected_native, 'native full renderer'),
                  contrast=close(fields['contrast'], manual_contrast(expected_native), 'original-raster contrast'))
    check(float((expected_native - direct).abs().max()) > .05, 'renderer test must distinguish omitted crop')
    height, width = original
    y, x = torch.arange(height)[:, None], torch.arange(width)[None, :]
    masks = torch.stack(((x < 5).expand(height, width), torch.zeros(height, width, dtype=torch.bool), (y + x) % 2 == 0))
    query_ids = torch.tensor([99, 7, 42], dtype=torch.int64)
    base = torch.arange(12544, dtype=torch.int64)
    indices = torch.stack((base % 77, (base * 3 + 7) % 77, (base * 5 + 23) % 77))
    indices[0, :6272], indices[1, :4181] = 0, 76
    check(all(torch.unique(row).numel() < row.numel() for row in indices), 'replacement not exercised')
    pair_inputs = [v.clone() for v in (query_ids, masks, indices)]
    cold = adapter.matched_point_loss(readout, fields, query_ids, masks, indices)
    expected, per_GT = manual_loss(readout, fields, query_ids, masks, indices)
    close(cold, expected, 'per-GT original-index BCE reduction')
    permutation = torch.tensor([2, 0, 1])
    reordered = adapter.matched_point_loss(readout, fields, query_ids[permutation], masks[permutation], indices[permutation])
    close(reordered, expected, 'joint query/GT/index row permutation')
    wrong_pairs = adapter.matched_point_loss(readout, fields, query_ids[[1, 0, 2]], masks, indices)
    check(float((wrong_pairs - cold).abs()) > .1, 'query/GT pairing test not discriminating')
    cold.backward(retain_graph=True)
    cold_grad = gradient_counts(readout)
    check(all(v['nonzero'] > 0 if name.startswith('final.') else v['nonzero'] == 0
              for name, v in cold_grad.items()), 'cold gradient pattern')
    readout.zero_grad(set_to_none=True)
    with torch.no_grad():
        readout.final.weight.copy_(torch.linspace(.03, .08, 32).reshape(1, 32))
        readout.final.bias.fill_(.11)
    warm = adapter.matched_point_loss(readout, fields, query_ids, masks, indices)
    expected_warm, _ = manual_loss(readout, fields, query_ids, masks, indices)
    close(warm, expected_warm, 'warm original-index BCE')
    warm.backward()
    warm_grad = gradient_counts(readout)
    check(all(v['nonzero'] > 0 for v in warm_grad.values()), 'constructed warm all8 gradients')
    readout.zero_grad(set_to_none=True)
    zero = adapter.matched_point_loss(readout, fields, torch.empty(0, dtype=torch.int64),
                                      torch.empty(0, height, width, dtype=torch.bool), torch.empty(0, 12544, dtype=torch.int64))
    check(zero.shape == () and zero.requires_grad and bool(torch.isfinite(zero)) and float(zero) == 0., 'G0 connected scalar')
    zero_grad = torch.autograd.grad(zero, tuple(readout.parameters()), allow_unused=False)
    check(all(bool(torch.isfinite(g).all()) and bool((g == 0).all()) for g in zero_grad), 'G0 all8 finite exact-zero gradients')
    for value, before in zip((feature, query, native), originals):
        check(torch.equal(value, before) and value.grad is None, 'teacher operand mutated/acquired gradient')
    for value, before in zip((query_ids, masks, indices), pair_inputs):
        check(torch.equal(value, before), 'query/GT/indices mutated')
    check(torch.equal(torch.get_rng_state(), rng), 'external CPU RNG changed')
    return dict(full200_800_crop_original_renderer=True, shared_original_point_indices=True,
                matched_query_order_and_replacement=True, detached_teacher_unchanged_grad_None=True,
                cold_final_only_gradient=True, warm_all8_gradients=True, G0_all8_connected_zero=True,
                external_CPU_RNG_unchanged=True, query_ids=[99, 7, 42], points_per_GT=12544,
                renderer_max_errors=errors, per_GT_cold_BCE=per_GT.detach().tolist(),
                cold_gradients=cold_grad, warm_gradients=warm_grad)

def run(spec, out, owner):
    global torch
    refs, code = source_gate(spec, out, owner)
    import psutil
    psutil.Process().cpu_affinity([0, 1])
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    check(sorted(psutil.Process().cpu_affinity()) == [0, 1] and torch.get_num_threads() == 2 and
          torch.get_num_interop_threads() == 2 and not torch.cuda.is_initialized(), 'CPU resources/CUDA')
    rng = torch.get_rng_state().clone()
    modules = {name: load_source(name, refs[name], code[name]) for name in ('readout', 'adapter', 'renderer')}
    check(torch.equal(torch.get_rng_state(), rng), 'SOURCE imports changed CPU RNG')
    checks = numerical_checks(modules['readout'], modules['adapter'], modules['renderer'])
    check(not torch.cuda.is_initialized(), 'CUDA initialized')
    result = dict(status=STATUS, sources=refs, execution=EXECUTION, torch_version=str(torch.__version__),
                  synthetic_execution_performed=True, synthetic_inputs_only=True, checks=checks,
                  tolerance=TOLERANCE, native_model_integration_qualified=False, efficacy_claimed=False,
                  calibration_guaranteed=False, novelty_claimed=False,
                  limits=['Synthetic native-shaped CPU tensors and actual pinned renderer SOURCE only.',
                          'No image/GT/checkpoint/assignment payload, native forward or optimizer replay.',
                          'Caller-supplied matching and fixed duplicate indices do not establish assignment or sampling correctness.',
                          'Constructed warm gradients do not guarantee real-image nonzero gradients.',
                          'ROOT actual tool terminal remains separate evidence.'])
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
