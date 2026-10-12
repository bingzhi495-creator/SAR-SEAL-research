"""Deferred injected synthetic CPU fixture; no imports of scientific libraries.

ROOT supplies the bound modules/libraries at future owned CPU execution.
This file reads no images, GT, checkpoints, cache packets, or result files.
"""


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def patch_oracle(gray, indices):
    """Independent scalar row-major, replicate-border, 7x7 intensity oracle."""
    height, width = len(gray), len(gray[0])
    return [[[gray[min(height-1, max(0, index//width+dy))]
                   [min(width-1, max(0, index % width+dx))] / 255.
              for dy in range(-3, 4) for dx in range(-3, 4)]
             for index in row] for row in indices]


def manual_head(torch, parameters, F, e, L, patch, arm):
    """Independent explicit matrix head, population-variance LayerNorm."""
    def normalize(value):
        centered = value.detach()-value.detach().mean(1, keepdim=True)
        return centered / (centered.square().mean(1, keepdim=True)+1e-5).sqrt()
    native = L.detach()
    inputs = torch.cat((normalize(F), normalize(e), native.clamp(-8, 8)[:, None]/8,
                        patch.detach() if arm == 'R' else torch.zeros_like(patch)), 1)
    z1 = inputs @ parameters['hidden1.weight'].t()+parameters['hidden1.bias']
    h1 = torch.where(z1 > 0, z1, torch.zeros_like(z1))
    z2 = h1 @ parameters['hidden2.weight'].t()+parameters['hidden2.bias']
    h2 = torch.where(z2 > 0, z2, torch.zeros_like(z2))
    residual = (h2 @ parameters['final.weight'].t()+parameters['final.bias']).reshape(-1)
    band = native.abs() <= 2
    return torch.where(band, native+residual, native), residual, band


def run_fixture(torch, np, cache, heads, qualifier):
    """Return checks only after actual synthetic CPU math; not a training run."""
    require(not torch.cuda.is_initialized(), 'Fixture must not initialize CUDA')
    external = torch.get_rng_state().clone()
    seed = 202610120701
    head = heads.make_point_head(torch, seed)
    paired = heads.make_point_head(torch, seed)
    names = ('hidden1.weight', 'hidden1.bias', 'hidden2.weight', 'hidden2.bias', 'final.weight', 'final.bias')
    shapes = ((64, 562), (64,), (64, 64), (64,), (1, 64), (1,))
    require(tuple(dict(head.named_parameters())) == names and
            tuple(tuple(v.shape) for v in head.parameters()) == shapes and
            sum(v.numel() for v in head.parameters()) == 40257 and not tuple(head.buffers()), 'Head6/40257 structure differs')
    require(all(torch.equal(a, b) for a, b in zip(head.parameters(), paired.parameters())) and
            torch.equal(external, torch.get_rng_state()), 'Same seed/private CPU RNG differs')
    F = ((torch.arange(7*256, dtype=torch.float32).reshape(7, 256) % 37)-18).requires_grad_()
    e = ((torch.arange(7*256, dtype=torch.float32).reshape(7, 256) % 23)-11).requires_grad_()
    L = torch.tensor([-3., -2., -.5, 0., .75, 2., 3.], requires_grad=True)
    patch = ((torch.arange(7*49, dtype=torch.float32).reshape(7, 49) % 256)/255).requires_grad_()
    target = torch.tensor([1., 0., 0., 0., 1., 1., 0.])
    values = (F, e, L, patch); versions = [v._version for v in values]
    for arm in ('C', 'R'):
        head.zero_grad(set_to_none=True)
        output = head(F, e, L, patch, arm=arm)
        require(set(output) == {'logits', 'residual', 'band'} and torch.equal(output['logits'], L.detach()) and
                bool((output['residual'] == 0).all()) and torch.equal(output['band'], L.detach().abs() <= 2), 'Cold identity/band differs')
        loss = torch.nn.functional.binary_cross_entropy_with_logits(output['logits'], target)
        loss.backward()
        gradients = [v.grad for v in head.parameters()]
        require(all(g is not None and bool(torch.isfinite(g).all()) for g in gradients) and
                all(bool((g == 0).all()) for g in gradients[:4]) and
                any(bool((g != 0).any()) for g in gradients[4:]), 'Cold final-only gradients differ')
    with torch.no_grad():
        head.final.weight.fill_(.01); head.final.bias.fill_(.05)
    for arm in ('C', 'R'):
        head.zero_grad(set_to_none=True)
        before = {name: value.detach().clone().requires_grad_() for name, value in head.named_parameters()}
        actual = head(F, e, L, patch, arm=arm)
        reference, residual, band = manual_head(torch, before, F, e, L, patch, arm)
        require(torch.allclose(actual['logits'], reference, rtol=1e-4, atol=1e-5) and
                torch.allclose(actual['residual'], residual, rtol=1e-4, atol=1e-5) and
                torch.equal(actual['band'], band) and torch.equal(actual['logits'][~band], L.detach()[~band]), 'Warm independent 562 matrix/band differs')
        actual_loss = torch.nn.functional.binary_cross_entropy_with_logits(actual['logits'], target)
        manual_loss = torch.nn.functional.binary_cross_entropy_with_logits(reference, target)
        actual_loss.backward()
        manual_gradients = torch.autograd.grad(manual_loss, tuple(before.values()), allow_unused=False)
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) and
                    torch.allclose(p.grad, g, rtol=1e-3, atol=1e-6)
                    for p, g in zip(head.parameters(), manual_gradients)), 'Independent six-parameter gradients differ')
    require(all(v.grad is None for v in values) and versions == [v._version for v in values], 'Frozen inputs modified or differentiated')
    changed = 1-patch.detach()
    require(torch.equal(head(F, e, L, patch, arm='C')['logits'], head(F, e, L, changed, arm='C')['logits']), 'C still uses direct patch')
    require(not torch.equal(head(F, e, L, patch, arm='R')['logits'], head(F, e, L, changed, arm='R')['logits']), 'Warm R patch path absent')
    head.zero_grad(set_to_none=True)
    head(F[:0], e[:0], L[:0], patch[:0], arm='R')['logits'].sum().backward()
    require(all(p.grad is not None and bool((p.grad == 0).all()) for p in head.parameters()), 'S0 connected-zero gradients differ')
    for bad_args in ((F, e, L, patch, 'X'), (F, e, L, patch-2, 'C'), (F, e, L, patch[:, :48], 'R')):
        try:
            head(*bad_args)
        except (TypeError, ValueError):
            pass
        else:
            raise RuntimeError('Invalid arm/patch was not rejected')

    gray = np.asarray([[0, 17, 255], [51, 99, 201]], dtype=np.uint8)
    indices = torch.tensor([[0, 2, 3, 5, 0]], dtype=torch.int64)
    actual_patch = cache.original_patches(torch, np, gray, indices)
    expected_patch = torch.tensor(patch_oracle(gray.tolist(), indices.tolist()), dtype=torch.float32)
    require(torch.equal(actual_patch, expected_patch) and
            np.array_equal(qualifier.patch_reference(np, gray, indices.numpy()), expected_patch.numpy()), 'Scalar JPEG patch edge/order/duplicates differ')
    full = torch.tensor([[[-3., -2., 0.], [2., 2.1, 7.]], [[3., 4., 5.], [6., 7., 8.]]], dtype=torch.float32)
    actual_indices, counts = cache.sample_points(torch, full, seed)
    generator = torch.Generator(device='cpu'); generator.manual_seed(seed)
    independent = []
    for row in full:
        valid = [i for i, v in enumerate(row.reshape(-1).tolist()) if abs(v) <= 2]
        if valid:
            pick = torch.randint(len(valid), (768,), generator=generator).tolist()
            independent.append([valid[i] for i in pick]+torch.randint(6, (256,), generator=generator).tolist())
        else:
            independent.append(torch.randint(6, (1024,), generator=generator).tolist())
    require(counts == [3, 0] and torch.equal(actual_indices, torch.tensor(independent, dtype=torch.int64)), 'Band768/uniform256 or empty-band fallback differs')
    require(qualifier.uniform_draws_check(torch, actual_indices, counts, seed, 6) == 2, 'Independent uniform replay differs')
    require(torch.equal(external, torch.get_rng_state()) and not torch.cuda.is_initialized(), 'Synthetic fixture changed CPU RNG/CUDA')
    return dict(status='N07_POINT_HEAD_AND_CACHE_COMPONENTS_SYNTHETIC_CPU_PASS',
                checks=dict(head6_40257=True, same_seed_private_CPU_RNG=True, cold_identity=True,
                    cold_final_only_gradient=True, warm_direct_matrix_and_gradients=True,
                    input_detach_and_no_mutation=True, C_zero_patch_R_live_patch=True,
                    S0_connected_zero=True, patch_scalar_edge_order_duplicate_oracle=True,
                    band_and_uniform_sampling=True), no_real_input=True, no_optimizer=True,
                no_AP=True, CPU_only=True, CUDA_initialized=False)
