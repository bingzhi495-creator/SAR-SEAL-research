"""Deferred injected-Torch mathematical tests; no loader or scientific I/O.

NUMERICAL_VALIDATION_PENDING_ROOT_IDLE_SCI. Saved before the adapter prototype.
No assertion below has been numerically executed by its SOURCE author.
"""

import math

VALIDATION_STATUS = 'NUMERICAL_VALIDATION_PENDING_ROOT_IDLE_SCI'


def run_fixture(torch, adapter_module, readout_class):
    """Future ROOT caller supplies Torch and independently bound SOURCE modules."""
    functional = torch.nn.functional
    seed = 2026100911
    outside = torch.get_rng_state().clone()
    base = readout_class(seed=seed).double()
    control = adapter_module.build_adapter(torch, readout_class, seed=seed).double()
    assert torch.equal(torch.get_rng_state(), outside)
    names = tuple(dict(base.named_parameters()))
    assert names == tuple(dict(control.named_parameters()))
    assert len(names) == 8 and sum(p.numel() for p in control.parameters()) == 9441
    assert not list(control.buffers())
    assert all(torch.equal(a, control.state_dict()[n]) for n, a in base.state_dict().items())
    generator = torch.Generator(device='cpu').manual_seed(43)
    feature = torch.randn((1, 256, 3, 4), generator=generator, dtype=torch.float64, requires_grad=True)
    embedding = torch.randn((1, 100, 256), generator=generator, dtype=torch.float64, requires_grad=True)
    native = torch.randn((1, 100, 3, 4), generator=generator, dtype=torch.float64, requires_grad=True)
    snapshots = [v.detach().clone() for v in (feature, embedding, native)]
    versions = [v._version for v in (feature, embedding, native)]
    identity = dict(actual_img_shape=[5, 7], original_shape=[3, 5])

    def renderer(value, metadata):
        padded = functional.interpolate(value, size=(8, 8), mode='bilinear', align_corners=False)
        h, w = metadata['actual_img_shape']
        return functional.interpolate(padded[:, :, :h, :w], size=metadata['original_shape'],
                                      mode='bilinear', align_corners=False)[0]

    def fields():
        return control.project_native_fields(feature, embedding, native,
                    identity['actual_img_shape'], identity['original_shape'], renderer)

    packet = fields()
    independent_second = functional.linear(feature.permute(0, 2, 3, 1),
                       control.query_projection.weight, control.query_projection.bias).permute(0, 3, 1, 2)
    assert torch.allclose(packet['projected_second_pixel'][0], renderer(independent_second, identity), atol=1e-12, rtol=1e-12)
    changed_embedding = control.project_native_fields(feature, embedding.flip(1), native,
                    identity['actual_img_shape'], identity['original_shape'], renderer)
    assert all(torch.equal(packet[k], changed_embedding[k]) for k in packet)
    query_ids = torch.tensor([99, 7, 42], dtype=torch.int64)
    indices = torch.arange(12544, dtype=torch.int64).remainder(15)[None].expand(3, -1).clone()
    gt = torch.tensor([[[0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [0, 1, 0, 1, 0]],
                       [[1, 1, 1, 0, 0], [1, 0, 1, 1, 0], [0, 1, 1, 0, 1]],
                       [[0, 0, 1, 1, 1], [1, 1, 0, 0, 0], [1, 0, 0, 1, 0]]], dtype=torch.bool)
    target_points = gt.flatten(1).gather(1, indices).to(torch.float64)
    native_points = packet['native_logits'][0, query_ids].flatten(1).gather(1, indices)
    cold = control.matched_point_loss(packet, query_ids, gt, indices)
    reference_cold = (torch.logaddexp(torch.zeros_like(native_points), native_points)-native_points*target_points).mean(1).mean()
    assert torch.allclose(cold, reference_cold, atol=1e-12, rtol=1e-12)
    cold.backward()
    for name, parameter in control.named_parameters():
        assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        if not name.startswith('final.'):
            assert int(torch.count_nonzero(parameter.grad)) == 0
    assert all(float(p.grad.norm()) > 0 for p in control.final.parameters())
    assert embedding.grad is None  # No direct e0 branch; independent native leaf here.
    control.zero_grad(set_to_none=True)
    feature.grad = None; native.grad = None
    with torch.no_grad():
        control.final.weight.copy_(torch.linspace(-.06, .09, 32, dtype=torch.float64)[None])
        control.final.bias.fill_(.025)
    packet = fields()
    point = indices.reshape(-1)
    first = packet['projected_pixel'][0].flatten(1).t()[point]
    second = packet['projected_second_pixel'][0].flatten(1).t()[point]
    logits = packet['native_logits'][0, query_ids].flatten(1).gather(1, indices).reshape(-1)
    contrast = packet['contrast'][0, query_ids].flatten(1).gather(1, indices).reshape(-1)
    direct_input = torch.cat((first, second, logits[:, None], logits.sigmoid()[:, None],
                             (logits > 0).to(logits.dtype)[:, None], contrast[:, None]), 1)
    hidden = functional.linear(direct_input, control.hidden.weight, control.hidden.bias)
    direct_u = functional.linear(hidden*hidden.sigmoid(), control.final.weight, control.final.bias)[:, 0]
    direct_logits = (logits+direct_u).reshape(3, 12544)
    reference = (torch.logaddexp(torch.zeros_like(direct_logits), direct_logits)-direct_logits*target_points).mean(1).mean()
    actual = control.matched_point_loss(packet, query_ids, gt, indices)
    assert torch.allclose(actual, reference, atol=1e-12, rtol=1e-12)
    reordered = control.matched_point_loss(packet, query_ids.flip(0), gt.flip(0), indices.flip(0))
    assert torch.allclose(actual, reordered, atol=1e-12, rtol=1e-12)
    actual.backward()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) and float(p.grad.norm()) > 0
               for p in control.parameters())
    assert feature.grad is not None and float(feature.grad.norm()) > 0
    assert native.grad is not None and float(native.grad.norm()) > 0 and embedding.grad is None
    analytic = float(control.query_projection.weight.grad[0, 0])
    parameter = control.query_projection.weight
    original = float(parameter[0, 0]); step = 1e-5; losses = []
    for delta in (step, -step):
        with torch.no_grad(): parameter[0, 0] = original+delta
        losses.append(float(control.matched_point_loss(fields(), query_ids, gt, indices)))
    with torch.no_grad(): parameter[0, 0] = original
    assert math.isclose((losses[0]-losses[1])/(2*step), analytic, rel_tol=2e-4, abs_tol=1e-8)
    packet = fields()
    dense = control.dense_outputs(packet['projected_pixel'], packet['projected_second_pixel'], packet['native_logits'], 4)
    for chunk in (1, 100):
        other = control.dense_outputs(packet['projected_pixel'], packet['projected_second_pixel'], packet['native_logits'], chunk)
        assert all(torch.allclose(dense[k], other[k], atol=1e-11, rtol=1e-11) for k in dense)
    all_first = packet['projected_pixel'][0].flatten(1).t().repeat(100, 1)
    all_second = packet['projected_second_pixel'][0].flatten(1).t().repeat(100, 1)
    all_logits = packet['native_logits'].reshape(-1)
    all_contrast = packet['contrast'].reshape(-1)
    points = control.point_outputs(all_first, all_second, all_logits, all_contrast)
    for name in ('posterior', 'corrected_logits', 'bounded_delta'):
        assert torch.allclose(points[name].reshape(1, 100, 3, 5), dense[name], atol=1e-11, rtol=1e-11)
    support = (dense['corrected_logits'] > 0).to(torch.float64)
    independent_budget = (support*dense['posterior']).sum((-2, -1))/(support.sum((-2, -1))+((1-support)*dense['posterior']).sum((-2, -1))+control.eps)
    assert torch.allclose(independent_budget, dense['budget'], atol=1e-12, rtol=1e-12)
    permutation = torch.arange(99, -1, -1)
    permuted = control.dense_outputs(packet['projected_pixel'], packet['projected_second_pixel'], packet['native_logits'][:, permutation], 4)
    assert all(torch.allclose(dense[k][:, permutation], permuted[k], atol=1e-11, rtol=1e-11) for k in dense)
    empty = control.matched_point_loss(fields(), query_ids[:0], gt[:0], indices[:0])
    gradients = torch.autograd.grad(empty, (*control.parameters(), feature, native), allow_unused=False)
    assert float(empty) == 0 and all(bool(torch.isfinite(v).all()) and int(torch.count_nonzero(v)) == 0 for v in gradients)
    assert all(torch.equal(v.detach(), saved) for v, saved in zip((feature, embedding, native), snapshots))
    assert versions == [v._version for v in (feature, embedding, native)]
    assert torch.equal(torch.get_rng_state(), outside)
    return dict(status='PIXEL_CAPACITY_MATHEMATICAL_FIXTURE_COMPLETED', parameters=9441, parameter_tensors=8,
                real_native_integration_qualified=False, science_or_AP_qualified=False,
                same_compute_or_effective_capacity_claimed=False)
