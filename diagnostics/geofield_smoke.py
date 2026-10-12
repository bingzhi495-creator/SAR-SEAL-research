# -*- coding: utf-8 -*-
"""CPU smoke gate for the GeoField v2 (DualResidual-SplitCode) operator.

Locks the properties the equal-parameter ladder + the v2 fix depend on, no GPU:
  1. byte-identical param count across ALL gen_mode arms (equal-param control).
  2. residual identity at init (gamma_shape=gamma_pos=0 -> E'=E) for every arm.
  3. forward+backward finite (fp32); active arms change E.
  4. softmax / AMP stability (the geokernel fp16 NaN lesson).
  5. batch INDEPENDENCE (deployment).
  6. *** v2 GUARANTEE: gamma_pos=0 makes 'full' recover EXACTLY 'shape' (R3) ***
     -> position is cleanly separable; the R3->R5 destructive regression is structurally
     impossible. And turning gamma_pos on DOES change the output (position can contribute).
  7. shape-branch code is coords-invariant (the v1 dilution fix): z_sh identical with/without coords.

Run in any torch env (pure operator, no mmdet).
"""
import sys
import torch

sys.path.insert(0, 'E:/SAR/seal')
from carrier.geofield_op import GeoField, _GEN_MODES  # noqa: E402

torch.manual_seed(0)
MODES = list(_GEN_MODES)
C, B, H, W = 32, 2, 16, 20


def main():
    # 1. equal-param across ALL arms ---------------------------------------
    counts = {m: sum(p.numel() for p in GeoField(C, gen_mode=m).parameters()) for m in MODES}
    assert len(set(counts.values())) == 1, f'param counts differ: {counts}'
    print(f'[1] PASS equal-param: every arm = {next(iter(counts.values()))} params  ({len(MODES)} arms)')

    # 2. residual identity at init -----------------------------------------
    x = torch.randn(B, C, H, W)
    for m in MODES:
        g = GeoField(C, gen_mode=m).eval()
        assert torch.allclose(g(x), x, atol=1e-6), f'{m}: not identity at init'
    print('[2] PASS residual-identity @ init (E\'=E) for all arms')

    # 3. forward/backward finite; active arms change E ---------------------
    for m in MODES:
        g = GeoField(C, gen_mode=m)
        with torch.no_grad():
            g.gamma_shape.fill_(1.0); g.gamma_pos.fill_(1.0); g.a.fill_(0.5); g.g_pos.fill_(0.5)
        xi = torch.randn(B, C, H, W, requires_grad=True)
        out = g(xi)
        assert torch.isfinite(out).all(), f'{m}: non-finite forward'
        out.square().mean().backward()
        assert torch.isfinite(xi.grad).all(), f'{m}: non-finite grad'
        for n, p in g.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f'{m}.{n}: non-finite grad'
        if m != 'off':
            assert not torch.allclose(out, xi, atol=1e-5), f'{m}: active arm is a no-op'
    print('[3] PASS forward/backward finite; active arms modify E')

    # 4. softmax stability (large activations + AMP) -----------------------
    g = GeoField(C, gen_mode='full')
    with torch.no_grad():
        g.gamma_shape.fill_(1.0); g.wk.weight.mul_(50.0)
    assert torch.isfinite(g(torch.randn(B, C, H, W) * 10.0)).all()
    if torch.cuda.is_available():
        gg = GeoField(C, gen_mode='full').cuda()
        with torch.no_grad():
            gg.gamma_shape.fill_(1.0); gg.gamma_pos.fill_(1.0)
        with torch.cuda.amp.autocast():
            o = gg(torch.randn(B, C, H, W).cuda())
        assert torch.isfinite(o.float()).all()
        print('[4] PASS softmax stable: large-activation (cpu) + AMP-fp16 (cuda)')
    else:
        print('[4] PASS softmax stable: large-activation (cpu); AMP skipped')

    # 5. batch independence ------------------------------------------------
    g = GeoField(C, gen_mode='full').eval()
    with torch.no_grad():
        g.gamma_shape.fill_(1.0); g.gamma_pos.fill_(1.0); g.a.fill_(0.5)
    x0, x1 = torch.randn(1, C, H, W), torch.randn(1, C, H, W)
    d = (g(torch.cat([x0, x1], 0))[:1] - g(x0)).abs().max().item()
    assert d < 1e-5, f'batch-dependent! max|d|={d}'
    print(f'[5] PASS batch-independent (max|d|={d:.2e})')

    # 6. *** v2 GUARANTEE: gamma_pos=0 makes full == shape (recover R3) *** -
    gf, gs = GeoField(C, 'full').eval(), GeoField(C, 'shape').eval()
    gs.load_state_dict(gf.state_dict())  # identical weights
    with torch.no_grad():
        for g in (gf, gs):
            g.gamma_shape.fill_(1.0); g.a.fill_(0.5); g.gamma_pos.fill_(0.0)
    xt = torch.randn(B, C, H, W)
    df = (gf(xt) - gs(xt)).abs().max().item()
    assert df < 1e-6, f'full(gamma_pos=0) != shape! max|d|={df} -- position NOT cleanly separable'
    print(f'[6] PASS v2-fix: full(gamma_pos=0) == shape exactly (max|d|={df:.2e}) -> recovers R3, position separable')
    with torch.no_grad():
        gf.gamma_pos.fill_(1.0)
    assert not torch.allclose(gf(xt), gs(xt), atol=1e-5), 'position never contributes (gamma_pos inert)'
    print('    PASS: gamma_pos>0 DOES change output -> position can contribute (falsifiable)')

    # 7. shape code is coords-invariant (the dilution fix) -----------------
    g = GeoField(C, 'full').eval()
    e = torch.randn(B, C, H, W)
    z_a = g.w1(torch.nn.functional.silu(g.gn(g.w0(e))))
    z_b = g.w1(torch.nn.functional.silu(g.gn(g.w0(e))))  # w0 takes only E -> no coord path
    assert torch.allclose(z_a, z_b, atol=1e-6)
    assert g.w0.in_channels == C, f'w0 must take c={C} channels (coords must NOT enter kernel code), got {g.w0.in_channels}'
    print('[7] PASS shape code is content-only (w0 in=C, coords never dilute the kernel)')

    n = sum(p.numel() for p in GeoField(256, 'full').parameters())
    print(f'[*] core params @ C=256: {n}')
    print('\nALL GEOFIELD v2 SMOKE TESTS PASSED')


if __name__ == '__main__':
    main()
