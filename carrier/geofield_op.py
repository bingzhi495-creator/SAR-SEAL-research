# -*- coding: utf-8 -*-
"""GeoField (v2, DualResidual-SplitCode) -- unified geometry operator (pure torch).

WHY v2.  The v1 single-shared-code design was NET DESTRUCTIVE in the pilot: shape
alone (R3) reached the best AP75 (0.348), but the full operator (R5) dropped to 0.322
(< R3 and < capacity R1). The only R3->R5 difference was feeding coords into the SAME
bottleneck code that drives the oriented kernel: smooth low-frequency coords washed out
the content/texture signal the kernel needs (dilution), and a single residual gate
entangled the harmful position term with the helpful shape term (no independent gating).

v2 fixes this by SPLITTING the operator into two ADDITIVE residual branches, each with its
OWN LayerScale gate, so a harmful principle can go to zero WITHOUT touching the other:

    E' = E + gamma_shape * s_shape (.) E_mix   +   gamma_pos * b_pos

  - SHAPE branch is CONTENT-ONLY: its code z_sh = W1 SiLU(GN(W0(E))) NEVER sees coords,
    so the oriented kernel is position-invariant and cannot be diluted. Its salience keeps
    its own content gate rho_sh = tanh(Wpos_sh z_sh) -- CRITICAL: R3's measured 0.348 used
    this content-rho, so it must be preserved; gamma_pos->0 then recovers EXACTLY R3.
  - POSITION is a separate COORDS-ONLY additive bias b_pos = tanh(Wpos_p z_pos),
    z_pos = Wp1 SiLU(GNp(Wp0(coords))). It never touches the kernel / E_mix.
  - Optional (gen_mode 'full_film') multiplicative trust modulation of s_shape by a
    centered position field (FiLM-style, arXiv:1709.07871), gated by g_pos (default off).

Both branches ENRICH E (fed to the unchanged rank-1 readout); nothing is written to logit
space (the ShapePrior failure). gamma_shape, gamma_pos, a, g_pos all init 0 -> at step 0 the
operator is identity, so it starts byte-identical in output to the carrier refine head.

EQUAL-PARAMETER LADDER (gen_mode; every arm instantiates ALL modules, byte-identical params;
the flag only gates the COMPUTATION via two booleans use_shape / use_pos):
  'off'       R0  : bypass -> stock refine head (sanity).
  'capacity'  R1  : uniform kernel + coords=0 + content gate -> pure capacity.
  'pos'       R2  : + coords-only additive position bias (kernel still uniform).
  'shape'     R3  : oriented kernel + shape-coupled content salience, coords=0
                    -> POSITIONALLY EQUIVALENT to the measured R3=0.348.
  'full'      R5s : shape branch + additive position branch (the v2 fusion).
  'full_film' R5f : + FiLM multiplicative trust modulation (optional 2nd stage).
  'full_rel'  R6  : == 'full' at the head; training-only RKD relation added by the wrapper.
Criteria: fusion fixed/super-additive <=> R5s >= R3 (0.348); fusion>capacity <=> R5s>R1.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# gen_mode -> arm: off=R0 capacity=R1 pos=R2 shape=R3 full=R5s full_film=R5f full_rel=R6
# SCOPE arms (P4 SAR module): shape_scope = shape + measured scatterer-extent prior;
# full_scope = full + the prior. The prior modulates ONLY salience + kernel-axis.
_GEN_MODES = ('off', 'capacity', 'pos', 'shape', 'full', 'full_film', 'full_rel',
              'shape_scope', 'full_scope')


class GeoField(nn.Module):
    """Dual-residual split-code geometry operator (see module docstring).

    Args:
        c: in/out channels. gen_mode: equal-parameter ladder arm.
        d/r/m/k: shape-branch code dim / bottleneck / kernel basis count / kernel size.
        rp/dp: position-branch bottleneck / code dim (coords-only, small).
    """

    def __init__(self, c: int, gen_mode: str = 'full',
                 d: int = 16, r: int = 64, m: int = 8, k: int = 5,
                 rp: int = 16, dp: int = 16):
        super().__init__()
        assert gen_mode in _GEN_MODES, gen_mode
        assert k % 2 == 1, 'kernel size must be odd'
        self.c, self.d, self.m, self.k = int(c), int(d), int(m), int(k)
        self.gen_mode = gen_mode
        self.pad = k // 2

        # ---- shape branch: CONTENT-ONLY (coords NEVER enter w0) ----
        self.w0 = nn.Conv2d(c, r, 1)                 # input c, NOT c+2  (the v1 dilution fix)
        self.gn = nn.GroupNorm(min(8, r), r)
        self.w1 = nn.Conv2d(r, d, 1)                 # -> z_sh
        self.wk = nn.Conv2d(d, m, 1)                 # alpha = wk(z_sh)
        self.basis = nn.Parameter(0.01 * torch.randn(m, k * k))
        self.wpos_sh = nn.Conv2d(d, c, 1)            # shape-gate content rho (was v1 'wpos')

        # ---- position branch: COORDS-ONLY additive bias ----
        self.wp0 = nn.Conv2d(2, rp, 1)
        self.gnp = nn.GroupNorm(min(8, rp), rp)
        self.wp1 = nn.Conv2d(rp, dp, 1)              # -> z_pos
        self.wpos_p = nn.Conv2d(dp, c, 1)            # b_pos = tanh(wpos_p(z_pos))
        self.coord_proj = nn.Conv2d(dp, 1, 1)        # FiLM trust field (full_film only)

        # ---- scalar gates (all init 0 -> identity at step 0) ----
        self.a = nn.Parameter(torch.zeros(1))            # salience temperature
        self.gamma_shape = nn.Parameter(torch.zeros(1))  # shape residual LayerScale
        self.gamma_pos = nn.Parameter(torch.zeros(1))    # position residual LayerScale (independent)
        self.g_pos = nn.Parameter(torch.zeros(1))        # FiLM trust LayerScale (full_film only)

        # ---- SCOPE: measured scatterer-constellation extent prior (gen_mode *_scope) ----
        # Instantiated in EVERY arm (equal-param ladder); only *_scope arms COMPUTE it.
        # Honest physics: PSeg/HRSID are display-stretched JPEGs, so the only measurement
        # cue used is a guard-banded local-contrast (soft-CFAR) statistic, ~invariant to a
        # monotone display stretch. The opened constellation drives salience (fills the dark
        # hull where E~0) and the oriented-kernel axis (averages along the hull spine).
        self.beta = nn.Parameter(torch.tensor(4.0))      # soft-CFAR sharpness
        self.tau = nn.Parameter(torch.tensor(1.5))       # soft-CFAR threshold (sigma units)
        self.rho_a = nn.Parameter(torch.tensor(8.0))     # geodesic-opening amplitude affinity
        self.Waxis = nn.Conv2d(2, m, 1)                  # measured axis -> kernel-basis bias
        self.g_scope = nn.Parameter(torch.zeros(1))      # salience-injection LayerScale (init 0)
        self.g_axis = nn.Parameter(torch.zeros(1))       # kernel-axis-steering LayerScale (init 0)
        nn.init.zeros_(self.Waxis.weight)                # identity at init (axis_bias=0)
        nn.init.zeros_(self.Waxis.bias)
        _wg = torch.tensor([[0.6065, 0.7788, 0.6065],
                            [0.7788, 1.0000, 0.7788],
                            [0.6065, 0.7788, 0.6065]])   # exp(-dist^2/2): opening distance kernel
        self.register_buffer('w_geo', _wg)
        _sob = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8.0
        self.register_buffer('sobel_x', _sob.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', _sob.t().contiguous().view(1, 1, 3, 3))

        # Zero-init ONLY the shape gate (its salience has a sigmoid(a*e_p) baseline, so
        # gamma_shape still gets gradient). The position branch must NOT be zero-init:
        # b_pos = tanh(wpos_p(z_pos)); if wpos_p=0 then b_pos=0 and dloss/dgamma_pos = b_pos = 0,
        # so gamma_pos AND the whole position branch would dead-start (never train). Standard
        # LayerScale = gamma init 0 (identity) with a normally-initialized inner module.
        nn.init.zeros_(self.wpos_sh.weight)
        nn.init.zeros_(self.wpos_sh.bias)

    @staticmethod
    def _coords(b: int, h: int, w: int, device, dtype):
        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([xs, ys], dim=1)  # (B,2,H,W)

    def _scope(self, amp: torch.Tensor):
        """Measured scatterer-constellation extent prior. amp: (B,1,H,W) in [0,1] at
        feature resolution. Returns (G_logit (B,1,H,W), axis_bias (B,M,H,W)).
        All outputs are kept strictly FINITE: even with the gates at 0, a non-finite
        g_logit would poison the output via 0*inf=NaN, so we clamp at every step."""
        amp = torch.nan_to_num(amp, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        # --- Step 1: guard-banded soft-CFAR constellation map C ---
        ko, ki = 13, 5
        no, ni = ko * ko, ki * ki
        so = F.avg_pool2d(amp, ko, 1, ko // 2) * no
        si = F.avg_pool2d(amp, ki, 1, ki // 2) * ni
        qo = F.avg_pool2d(amp * amp, ko, 1, ko // 2) * no
        qi = F.avg_pool2d(amp * amp, ki, 1, ki // 2) * ni
        nann = float(no - ni)
        mu_bg = (so - si) / nann                                         # annulus (guard-banded) mean
        var_bg = torch.relu((qo - qi) / nann - mu_bg * mu_bg)
        sig_bg = torch.sqrt(var_bg + 1e-6)
        t = torch.relu((amp - mu_bg) / (sig_bg + 1e-3))                  # local contrast, sigma units
        beta = self.beta.clamp(1.0, 16.0)
        tau = self.tau.clamp(0.5, 4.0)
        C = torch.sigmoid(beta * (t - tau))                             # ~1 on locally-bright scatterers
        # --- Step 2: geodesic opening grows C INTO the dark hull (measured affinity) ---
        rho = self.rho_a.clamp(min=0.0)
        G = C
        B, _, H, W = G.shape
        for _ in range(3):
            Gp = F.pad(G, (1, 1, 1, 1))
            Ap = F.pad(amp, (1, 1, 1, 1))
            acc = G.new_full(G.shape, -1e9)
            for di in range(3):
                for dj in range(3):
                    da = amp - Ap[:, :, di:di + H, dj:dj + W]
                    a_pq = torch.exp(-rho * da * da) * self.w_geo[di, dj]
                    acc = torch.maximum(acc, Gp[:, :, di:di + H, dj:dj + W] * a_pq)
            G = torch.nan_to_num(acc, nan=0.0).clamp(0.0, 1.0)
        G_logit = torch.logit(G.clamp(1e-4, 1.0 - 1e-4)).clamp(-9.0, 9.0)
        # --- Step 3b: structure tensor of the opened field -> measured axis bias ---
        gx = F.conv2d(G, self.sobel_x, padding=1)
        gy = F.conv2d(G, self.sobel_y, padding=1)
        Jxx = F.avg_pool2d(gx * gx, 5, 1, 2)
        Jyy = F.avg_pool2d(gy * gy, 5, 1, 2)
        Jxy = F.avg_pool2d(gx * gy, 5, 1, 2)
        cos2, sin2 = Jxx - Jyy, 2.0 * Jxy
        # eps INSIDE the sqrt: sqrt'(0)=inf would give NaN gradients in flat regions
        # (cos2=sin2=0). Forward looks finite (sqrt(0)+1e-6) but backward NaNs -> the
        # clip_grad norm becomes NaN -> params NaN -> Hungarian "invalid numeric entries".
        nrm = torch.sqrt(cos2 * cos2 + sin2 * sin2 + 1e-12)
        aniso = (nrm / (Jxx + Jyy + 1e-6)).clamp(0.0, 1.0)
        axis = torch.cat([cos2 / nrm, sin2 / nrm], dim=1)               # (B,2,H,W)
        axis_bias = torch.nan_to_num(aniso * self.Waxis(axis), nan=0.0).clamp(-20.0, 20.0)
        return G_logit, axis_bias

    def forward(self, e: torch.Tensor, amp: torch.Tensor = None) -> torch.Tensor:
        if self.gen_mode == 'off':
            return e
        use_shape = self.gen_mode in ('shape', 'full', 'full_film', 'full_rel',
                                      'shape_scope', 'full_scope')
        use_pos = self.gen_mode in ('pos', 'full', 'full_film', 'full_rel', 'full_scope')
        use_scope = self.gen_mode in ('shape_scope', 'full_scope') and amp is not None

        b, c, h, w = e.shape
        e32 = e.float()

        # ---- SCOPE measured prior (gated; identity contribution at init via g_*=0) ----
        if use_scope:
            if amp.shape[-2:] != (h, w):
                amp = F.interpolate(amp.float(), size=(h, w), mode='area')
            g_logit, axis_bias = self._scope(amp.float())
        else:
            g_logit, axis_bias = None, None

        # ---- shape branch: content-only code -> oriented kernel -> E_mix -> salience ----
        z_sh = self.w1(F.silu(self.gn(self.w0(e))))                      # (B,d,H,W), no coords
        if use_shape:
            alpha = self.wk(z_sh)                                        # (B,M,H,W)
            if use_scope:
                alpha = alpha + self.g_axis * axis_bias                  # measured-axis steering
            klogit = torch.einsum('bmhw,mn->bnhw', alpha.float(), self.basis.float())  # fp32
            kp = torch.softmax(klogit, dim=1)
        else:
            kp = e.new_full((b, self.k * self.k, h, w), 1.0 / (self.k * self.k), dtype=torch.float32)

        epad = F.pad(e32, (self.pad, self.pad, self.pad, self.pad))
        e_mix = torch.zeros_like(e32)
        idx = 0
        for di in range(self.k):
            for dj in range(self.k):
                e_mix = e_mix + kp[:, idx:idx + 1, :, :] * epad[:, :, di:di + h, dj:dj + w]
                idx += 1

        rho_sh = torch.tanh(self.wpos_sh(z_sh))                          # content gate (preserves R3)
        if use_shape:
            e_p = (e_mix - e32).norm(dim=1, keepdim=True)
            s_logit = self.a * e_p + rho_sh
            if use_scope:
                s_logit = s_logit + self.g_scope * g_logit               # fills dark hull (E~0 there)
            s_shape = torch.sigmoid(s_logit)
        else:
            s_shape = torch.sigmoid(rho_sh)

        # ---- position branch: coords-only additive bias (never touches the kernel) ----
        coords = self._coords(b, h, w, e.device, e32.dtype)
        if not use_pos:
            coords = torch.zeros_like(coords)
        z_pos = self.wp1(F.silu(self.gnp(self.wp0(coords))))             # (B,dp,H,W)
        b_pos = torch.tanh(self.wpos_p(z_pos))                           # (B,C,H,W)

        # ---- optional FiLM multiplicative trust (full_film only) ----
        if self.gen_mode == 'full_film':
            m_pos = F.softplus(self.coord_proj(z_pos))                   # (B,1,H,W)
            m_cent = m_pos - m_pos.mean(dim=(2, 3), keepdim=True)        # per-image, batch-independent
            s_shape = (s_shape * (1.0 + self.g_pos * m_cent)).clamp(min=0.0)

        # ---- dual additive residual, three independent LayerScales ----
        e_out = e32 + self.gamma_shape * s_shape * e_mix + self.gamma_pos * b_pos
        return e_out.to(e.dtype)
