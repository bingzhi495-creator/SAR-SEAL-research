"""Caller-bound, synthetic-only joint/native point-loss fixture; do not run at authoring.

Future CLI: --source ROLE ABS_PY SHA256 (five times) --out FRESH_ABS_JSON
--owner-execute. Roles: fixture/native_objective/joint_adapter/readout/renderer.
Only these SOURCE files are read; no model/data/GT/checkpoint/qualification IO.
The renderer SOURCE must expose render_original(torch,tensor,identity).
Synthetic GT is generated here. PASS is not native-model/end-to-end qualification.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import types

ROLES = {'fixture', 'native_objective', 'joint_adapter', 'readout', 'renderer'}


def source_gate(arguments):
    refs, code = {}, {}
    for role, supplied_path, digest in arguments.source:
        path = Path(supplied_path)
        if role not in ROLES or role in refs or not path.is_absolute() or path.suffix != '.py':
            raise ValueError('five unique caller-bound absolute Python SOURCE roles required')
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('caller must supply lowercase SHA256 for every SOURCE')
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError('caller-bound SOURCE differs: ' + role)
        refs[role] = dict(path=str(path.resolve()), bytes=len(data), sha256=digest)
        code[role] = data
    if set(refs) != ROLES or Path(refs['fixture']['path']) != Path(__file__).resolve():
        raise ValueError('all five roles including actual fixture SOURCE must be caller-bound')
    if len({r['path'] for r in refs.values()}) != 5:
        raise ValueError('SOURCE roles must be distinct files')
    return refs, code


def load_verified(role, refs, code):
    module = types.ModuleType('joint_synthetic_' + role)
    module.__file__ = refs[role]['path']
    sys.modules[module.__name__] = module
    exec(compile(code[role], module.__file__, 'exec'), module.__dict__)
    return module


def manual_render(torch, field, image_shape, original_shape):
    """Explicit align_corners=False coordinates, two bilinear stages; no API call."""
    def coordinate(index, source_size, destination_size):
        value = min(max((index + .5) * source_size / destination_size - .5, 0.), source_size - 1.)
        low = math.floor(value)
        return low, min(low + 1, source_size - 1), value - low
    def at800(y, x):
        y0, y1, fy = coordinate(y, 200, 800)
        x0, x1, fx = coordinate(x, 200, 800)
        top = field[0, :, y0, x0] * (1 - fx) + field[0, :, y0, x1] * fx
        bottom = field[0, :, y1, x0] * (1 - fx) + field[0, :, y1, x1] * fx
        return top * (1 - fy) + bottom * fy
    ih, iw = image_shape
    oh, ow = original_shape
    rows = []
    for y in range(oh):
        y0, y1, fy = coordinate(y, ih, oh)
        columns = []
        for x in range(ow):
            x0, x1, fx = coordinate(x, iw, ow)
            top = at800(y0, x0) * (1 - fx) + at800(y0, x1) * fx
            bottom = at800(y1, x0) * (1 - fx) + at800(y1, x1) * fx
            columns.append(top * (1 - fy) + bottom * fy)
        rows.append(torch.stack(columns, dim=-1))
    return torch.stack(rows, dim=-2)


def counted_BCE(torch, logits, query_ids, targets, indices):
    """Independent per-original-pixel multiplicities, preserving repeated draws."""
    per_GT = []
    for g, q in enumerate(query_ids.tolist()):
        multiplicity = torch.bincount(indices[g], minlength=targets[g].numel()).to(logits.dtype)
        z, target = logits[q].reshape(-1), targets[g].reshape(-1).to(logits.dtype)
        losses = torch.nn.functional.softplus(z) - z * target
        per_GT.append((multiplicity * losses).sum() / indices.shape[1])
    return torch.stack(per_GT).mean()


def checks(torch, objective, adapter, readout, renderer):
    rng = torch.get_rng_state().clone()
    model = readout.QueryPixelPosteriorReadout(seed=2026100911, kappa=1.)
    assert torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized()
    parameters = tuple(model.parameters())
    assert len(parameters) == 8 and sum(p.numel() for p in parameters) == 9441 and not tuple(model.buffers())
    gen = torch.Generator(device='cpu').manual_seed(2026101107)
    native_inputs = (torch.randn(1, 256, 200, 200, generator=gen).mul(.2).requires_grad_(),
        torch.randn(1, 100, 256, generator=gen).mul(.2).requires_grad_(),
        (torch.randn(1, 100, 200, 200, generator=gen).mul(.4) +
         (torch.arange(100).float() - 50)[None, :, None, None] / 120).requires_grad_())
    snapshots = tuple(t.detach().clone() for t in native_inputs)
    versions = tuple(t._version for t in native_inputs)
    image_shape, shape = (625, 735), (5, 7)
    query_ids = torch.tensor([99, 0, 37], dtype=torch.int64)
    targets = torch.stack([((torch.arange(35).reshape(shape) + g * 3) % 5 < 2) for g in range(3)])
    indices = (torch.arange(12544)[None, :] * torch.tensor([[1], [2], [3]]) + torch.tensor([[0], [4], [9]])) % 35
    indices[0, :8000] = 0
    indices[1, :3000] = 34
    indices[2, ::3] = 10
    ids_copy, masks_copy, points_copy = query_ids.clone(), targets.clone(), indices.clone()
    assignment_versions = tuple(t._version for t in (query_ids, targets, indices))
    render = lambda x, identity: renderer.render_original(torch, x, identity)
    fields = adapter.project_live_native_fields(model, *native_inputs, image_shape, shape, render)
    assert set(fields) == {'projected_pixel', 'projected_query', 'native_logits', 'contrast'}
    assert all(v.requires_grad and v.grad_fn is not None for v in fields.values())
    assert fields['native_logits'].shape == (1, 100, 5, 7) and fields['projected_query'].shape == (1, 100, 16)
    reference_logits = manual_render(torch, native_inputs[2], image_shape, shape)
    assert torch.allclose(fields['native_logits'][0], reference_logits, rtol=2e-5, atol=2e-6)
    direct_resize = torch.nn.functional.interpolate(native_inputs[2], size=shape, mode='bilinear', align_corners=False)[0]
    assert not torch.allclose(reference_logits, direct_resize, rtol=1e-5, atol=1e-5)
    cold = adapter.matched_joint_point_loss(model, fields, query_ids, targets, indices)
    native = objective.matched_native_point_loss(fields['native_logits'], query_ids, targets, indices)
    manual = counted_BCE(torch, reference_logits, query_ids, targets, indices)
    assert torch.equal(cold, native) and torch.allclose(native, manual, rtol=2e-5, atol=2e-6)
    cold_grads = torch.autograd.grad(cold, native_inputs + parameters, retain_graph=True, allow_unused=False)
    native_M = torch.autograd.grad(native, native_inputs[2], retain_graph=True)[0]
    manual_M = torch.autograd.grad(manual, native_inputs[2], retain_graph=True)[0]
    assert torch.equal(cold_grads[2], native_M)
    assert torch.allclose(native_M, manual_M, rtol=2e-4, atol=2e-7)
    assert all(bool(torch.isfinite(g).all()) for g in cold_grads)
    assert all(torch.count_nonzero(g).item() == 0 for g in cold_grads[:2] + cold_grads[3:9])
    assert all(torch.count_nonzero(g).item() > 0 for g in cold_grads[9:]) and torch.count_nonzero(native_M).item() > 0
    with torch.no_grad():
        model.final.weight.copy_(torch.linspace(.01, .035, 32).reshape(1, 32))
        model.final.bias.fill_(.07)
    del fields, cold, native, manual, cold_grads, native_M, manual_M, reference_logits
    fields = adapter.project_live_native_fields(model, *native_inputs, image_shape, shape, render)
    warm = adapter.matched_joint_point_loss(model, fields, query_ids, targets, indices)
    # Independent projection/render/contrast/concat MLP and multiplicity loss.
    pixel200 = torch.einsum('oc,bchw->bohw', model.feature_projection.weight[:, :, 0, 0], native_inputs[0])
    pixel200 = pixel200 + model.feature_projection.bias[None, :, None, None]
    pixel = manual_render(torch, pixel200, image_shape, shape)
    query = native_inputs[1][0] @ model.query_projection.weight.t() + model.query_projection.bias
    logits = manual_render(torch, native_inputs[2], image_shape, shape)
    probability = logits.sigmoid()
    contrast = torch.empty_like(probability)
    for y in range(shape[0]):
        for x in range(shape[1]):
            contrast[:, y, x] = probability[:, y, x] - probability[:, max(y-1, 0):y+2, max(x-1, 0):x+2].mean((-2, -1))
    packet = torch.cat((pixel.permute(1, 2, 0)[None].expand(100, -1, -1, -1),
        query[:, None, None].expand(-1, 5, 7, -1), logits[..., None], probability[..., None],
        (logits > 0).float()[..., None], contrast[..., None]), dim=-1)
    hidden = packet @ model.hidden.weight.t() + model.hidden.bias
    u = (hidden * hidden.sigmoid()) @ model.final.weight.t() + model.final.bias
    reference = counted_BCE(torch, logits + u[..., 0], query_ids, targets, indices)
    assert torch.allclose(warm, reference, rtol=3e-5, atol=3e-6)
    assert torch.allclose(fields['contrast'][0], contrast, rtol=2e-5, atol=2e-6)
    warm_grads = torch.autograd.grad(warm, native_inputs + parameters + (fields['contrast'],), retain_graph=True, allow_unused=False)
    assert all(bool(torch.isfinite(g).all()) and torch.count_nonzero(g).item() > 0 for g in warm_grads)
    perm = torch.tensor([2, 0, 1])
    reordered = adapter.matched_joint_point_loss(model, fields, query_ids[perm], targets[perm], indices[perm])
    assert torch.allclose(warm, reordered, rtol=1e-6, atol=1e-7)
    reverse = torch.arange(99, -1, -1)
    reordered_fields = {n: v if n == 'projected_pixel' else v[:, reverse] for n, v in fields.items()}
    assert torch.allclose(warm, adapter.matched_joint_point_loss(model, reordered_fields, 99-query_ids, targets, indices), rtol=1e-6, atol=1e-7)
    for loss_fn in (lambda q, g, i: objective.matched_native_point_loss(fields['native_logits'], q, g, i),
                    lambda q, g, i: adapter.matched_joint_point_loss(model, fields, q, g, i)):
        for q, g, i in [(query_ids*0, targets, indices), (query_ids+100, targets, indices),
                        (query_ids.float(), targets, indices), (query_ids, targets.float(), indices),
                        (query_ids, targets, indices[:, :1]), (query_ids, targets, indices-35)]:
            try: loss_fn(q, g, i)
            except (ValueError, TypeError): pass
            else: raise AssertionError('invalid assignment/point packet was accepted')
    bad = fields['native_logits'].detach().clone(); bad[0, 0, 0, 0] = float('nan')
    try: objective.matched_native_point_loss(bad, query_ids, targets, indices)
    except ValueError: pass
    else: raise AssertionError('nonfinite native logits accepted')
    empty_ids, empty_masks, empty_indices = query_ids[:0], targets[:0], indices[:0]
    empty_native = objective.matched_native_point_loss(fields['native_logits'], empty_ids, empty_masks, empty_indices)
    empty_joint = adapter.matched_joint_point_loss(model, fields, empty_ids, empty_masks, empty_indices)
    assert empty_native.item() == empty_joint.item() == 0.
    empty_grad = torch.autograd.grad(empty_joint, native_inputs + parameters, retain_graph=True, allow_unused=False)
    assert all(bool(torch.isfinite(g).all()) and torch.count_nonzero(g).item() == 0 for g in empty_grad)
    empty_native_grad = torch.autograd.grad(empty_native, native_inputs[2], allow_unused=False)[0]
    assert bool(torch.isfinite(empty_native_grad).all()) and torch.count_nonzero(empty_native_grad).item() == 0
    assert all(t._version == v and torch.equal(t.detach(), copy) and t.grad is None for t, v, copy in zip(native_inputs, versions, snapshots))
    assert torch.equal(query_ids, ids_copy) and torch.equal(targets, masks_copy) and torch.equal(indices, points_copy)
    assert tuple(t._version for t in (query_ids, targets, indices)) == assignment_versions
    assert torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized()
    return dict(parameter_scalars=9441, parameter_tensors=8, queries_retained=100,
        cold_native_value_and_M_gradient_exact=True, manual_two_stage_renderer_and_repeat_counts=True,
        cold_F_e0_and_upstream_head_gradient_zero=True, warm_three_native_and_all8_head_gradients=True,
        live_contrast_gradient=True, permutations_and_duplicate_points_preserved=True,
        invalid_packets_rejected=True, empty_native_and_joint_connected_zero=True,
        native_inputs_versions_values_unchanged=True, external_CPU_RNG_unchanged=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', nargs=3, action='append', required=True, metavar=('ROLE', 'ABS_PY', 'SHA256'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args()
    if not args.owner_execute: raise ValueError('explicit --owner-execute required')
    out = Path(args.out)
    if not out.is_absolute() or not out.parent.is_dir() or out.exists(): raise ValueError('fresh absolute output in existing parent required')
    refs, code = source_gate(args)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('caller must hide CUDA before launch')
    import psutil
    import torch
    psutil.Process().cpu_affinity([0, 1])
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    assert not torch.cuda.is_initialized()
    modules = {r: load_verified(r, refs, code) for r in ('native_objective', 'joint_adapter', 'readout', 'renderer')}
    result = checks(torch, modules['native_objective'], modules['joint_adapter'], modules['readout'], modules['renderer'])
    assert psutil.Process().cpu_affinity() == [0, 1] and torch.get_num_threads() == torch.get_num_interop_threads() == 2
    for r, ref in refs.items():
        if Path(ref['path']).read_bytes() != code[r]: raise ValueError('SOURCE changed during synthetic run')
    payload = dict(status='JOINT_NATIVE_POINT_SYNTHETIC_CPU_PASS', sources=refs, checks=result,
        execution=dict(device='cpu', cpu_affinity=[0, 1], torch_threads=2, torch_interop_threads=2,
            cuda_visible_devices='', CUDA_initialized=torch.cuda.is_initialized()), synthetic_execution_performed=True,
        native_model_integration_qualified=False, assignment_generation_qualified=False, scientific_data_accessed=False,
        optimizer_steps=0, AP_computed=False, efficacy_claimed=False,
        limits=['Synthetic gradients are not a native-model forward/backward qualification.',
            'Cold F/e0 direct projection gradients are zero by design, not expected positive each step.',
            'Manual floating renderer/MLP comparisons use tolerances; no GPU bitwise claim.',
            'No native Hungarian, backbone, BatchNorm, optimizer, GT file or production inference is exercised.'])
    with out.open('x', encoding='utf-8') as stream: json.dump(payload, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(status=payload['status'], output=str(out))))


if __name__ == '__main__':
    main()
