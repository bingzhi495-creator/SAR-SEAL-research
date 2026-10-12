# -*- coding: utf-8 -*-
"""GeoFieldMask2FormerHead -- a unified geometry-field mask-feature operator.

DESIGN LINEAGE (principles, NOT modules): we distil four ideas from the user's
validated UNet++ teacher (D:/SAR/UNet++/KD/teacher_model.py) -- CBAM (salience),
DSConv (shape-adaptive sampling), CoordConv (absolute position), BatchFormer
(inter-instance relation) -- and RE-DESIGN them, from first principles, into ONE
native operator in which the four are read-outs of a SINGLE shared per-pixel code.
It is NOT a stack of those blocks.

WHERE IT LIVES.  Mask2Former's mask is a rank-1 readout
    M_q = einsum('bqc,bchw->bqhw', mask_embed(q), E)
over a SHARED per-pixel feature E. There is no per-instance spatial DOF, so where
E~0 (the low-backscatter stern/bow) no query weight recovers the pixel. The
failed ShapePrior (geokernel_m2f_head.py) ADDED a competing rigid prior in LOGIT
space (M_q + g*P) and REGRESSED the baseline -- the prior fought the optimised
readout. GeoField instead ENRICHES the substrate E that the SAME readout reads:
    E' = E + gamma * s ⊙ E_mix ,
fed back to the unchanged einsum. It never writes to logit space, so it cannot
regress the way ShapePrior did; gamma is a LayerScale init 0 so at step 0 the head
is byte-identical to the carrier refine head (no step-0 NaN).

THE OPERATOR (per pixel p, on the 1/2-res refined feature E = hi_features):
  z_p  = W1 · SiLU(GN(W0 · [E ; coords]))            # ONE shared d-dim geometry code
  k_p  = softmax_ij( Σ_m α_{p,m} Bᵐ ),  α = W_k z    # oriented kernel from a FREE
                                                       # low-rank basis {Bᵐ} (learned,
                                                       # NOT a Gaussian/steerable frame)
  E_mix(p) = Σ_ij k_p(ij) · E(p+δij)                  # content-adaptive depthwise mix
  e_p  = ‖E_mix − E‖₂ (over channels)                 # salience = energy of the mix
  ρ_p  = tanh(W_pos z_p)                              # per-channel position phase
  s_p  = σ(a · e_p + ρ_p)                             # SALIENCE IS NON-SEPARABLE from
                                                       # the kernel: no W_gain head;
                                                       # to ablate s you must remove k.
  E'   = E + gamma · s_p ⊙ E_mix

Because salience has no own projection, an "R4 salience-only" arm is structurally
impossible -- that non-separability is the novelty claim, not a weakness.

EQUAL-PARAMETER LADDER (gen_mode; one class, one flag, byte-identical params --
every arm instantiates the WHOLE operator, the flag only changes the COMPUTATION):
  'off'      R0 : bypass (E'=E) -> output-identical to carrier SealM2FHead (sanity).
  'capacity' R1 : coords=const, kernel=isotropic-uniform, s=σ(ρ) (content gate only)
                  -> pure capacity/content routing, NO position, NO shape.
  'pos'      R2 : + coords active (kernel still isotropic) -> isolates position vs.
                  Mask2Former's existing sine PE.
  'shape'    R3 : + oriented free-basis kernel + shape-coupled salience σ(a·e_p+ρ),
                  coords=const -> isolates shape.
  'full'     R5 : coords + oriented kernel + shape-coupled salience (relation OFF).
  'full_rel' R6 : == 'full' at the head; an inter-instance RKD relation loss (z
                  descriptors, training-only, zero inference params) is added by the
                  training wrapper. Head forward is identical to 'full'.
Decisive criteria (multi-seed bootstrap CI, PSeg-SSDD, medium AP75):
  fusion>capacity  ⟺ R5>R1 ;  entanglement  ⟺ R5>max(R2,R3) ;  relation  ⟺ R6>R5.

Inference path is batch-INDEPENDENT and ONNX/TensorRT-friendly: a 1x1 bottleneck,
small linears, a softmax-mixed depthwise conv done as K*K shift-accumulate (no
grid_sample / DCN / Transformer / batch axis); coords are a per-image function.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS

from carrier.seal_m2f_head import SealM2FHead
from carrier.geofield_op import GeoField, _GEN_MODES  # pure-torch operator (reused in Plan B)


@MODELS.register_module()
class GeoFieldMask2FormerHead(SealM2FHead):
    """Carrier refine head + the GeoField operator on the refined feature.

    The ONLY override beyond __init__ is ``_refine_features``: GeoField enriches the
    1/2-res refined feature in place, before the unchanged final einsum readout in
    ``SealM2FHead.forward`` (so detection/aux layers and the inference AMR blend are
    untouched). gen_mode selects the equal-parameter ladder arm.
    """

    def __init__(self, *args,
                 gen_mode: str = 'full',
                 geo_d: int = 16, geo_r: int = 64, geo_m: int = 8, geo_k: int = 5,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert gen_mode in _GEN_MODES, gen_mode
        self.gen_mode = gen_mode
        c = self.mask_embed[-1].out_features  # mask feature channels (=feat_channels)
        self.geofield = GeoField(c, gen_mode=gen_mode, d=geo_d, r=geo_r, m=geo_m, k=geo_k)
        self._amp_full = None

    def forward(self, x, batch_data_samples):
        # SCOPE: stash the frame-level SAR amplitude (post-aug, padded to input size) so
        # _refine_features can feed it to GeoField. Graceful: missing -> None -> SCOPE off.
        self._amp_full = None
        if self.gen_mode in ('shape_scope', 'full_scope'):
            amps = [getattr(ds, 'dense_amp', None) for ds in batch_data_samples]
            if all(a is not None for a in amps) and len(amps) > 0:
                H, W = batch_data_samples[0].batch_input_shape  # padded input size
                dev = x[0].device
                padded = []
                for a in amps:
                    a = a if a.dim() == 3 else a[None]           # (1,h,w)
                    a = a[:, :H, :W]                              # crop if larger
                    h, w = a.shape[-2:]
                    a = F.pad(a, (0, W - w, 0, H - h))           # pad bottom-right with 0 (sea)
                    padded.append(a)
                self._amp_full = torch.stack(padded, 0).to(dev).float()  # (B,1,H,W)
        return super().forward(x, batch_data_samples)

    def _refine_features(self, mask_features):
        hi = super()._refine_features(mask_features)  # carrier refined feature (B,C,2h,2w)
        amp_ds = None
        if self._amp_full is not None:
            amp_ds = F.interpolate(self._amp_full, size=hi.shape[-2:], mode='area')
        return self.geofield(hi, amp=amp_ds)           # GeoField enrichment (gen_mode-gated)


@MODELS.register_module()
class OAEMMask2FormerHead(SealM2FHead):
    """Carrier refine head + OAEM orientation-aware embedding on the multi-scale backbone
    features (before the pixel decoder). Gated residual (gamma init 0) -> step-0 identical
    to the carrier. oaem_in_channels must match the backbone output channels."""

    def __init__(self, *args, oaem_in_channels=(256, 512, 1024, 2048), oaem_k: int = 7,
                 use_oaem: bool = True, use_denodet: bool = False, use_caa: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_oaem, self.use_denodet, self.use_caa = use_oaem, use_denodet, use_caa
        if use_oaem:
            from carrier.oaem_op import OAEM
            self.oaem = nn.ModuleList([OAEM(c, k=oaem_k) for c in oaem_in_channels])
        if use_denodet:
            from carrier.ndm_op import NDM
            self.ndm = nn.ModuleList([NDM(c) for c in oaem_in_channels])
        if use_caa:
            from carrier.caa_op import CAA
            self.caa = nn.ModuleList([CAA(c) for c in oaem_in_channels])

    def forward(self, x, batch_data_samples):
        if self.use_oaem:
            x = [self.oaem[i](f) for i, f in enumerate(x)]      # orientation-aware enrich
        if self.use_denodet:
            x = [self.ndm[i](f) for i, f in enumerate(x)]       # frequency speckle-denoise
        if self.use_caa:
            x = [self.caa[i](f) for i, f in enumerate(x)]       # long-range context anchor (inshore)
        return super().forward(x, batch_data_samples)


@MODELS.register_module()
class DKRMask2FormerHead(SealM2FHead):
    """Carrier refine head + DKR per-instance dynamic-kernel mask branch (D1), in parallel with
    the rank-1 readout. Gated (gamma init 0) -> step-0 byte-identical to A0R. Attacks the high-IoU
    boundary ceiling that the rank-1 readout structurally caps."""

    def __init__(self, *args, dkr_channels: int = 8, dkr_relcoord: bool = False,
                 dkr_scatter: bool = False, dkr_orient: bool = False,
                 dkr_kernel_mode: str = '1x1', **kwargs):
        super().__init__(*args, **kwargs)
        from carrier.dkr_op import DKR
        c = self.mask_embed[-1].out_features
        self.dkr = DKR(c, dkr_channels=dkr_channels, use_relcoord=dkr_relcoord,
                       use_scatter=dkr_scatter, use_orient=dkr_orient, kernel_mode=dkr_kernel_mode)
        self._amp_full = None

    def forward(self, x, batch_data_samples):
        # D3: stash the frame-level SAR amplitude (post-aug) for the scatterer-weighted centre.
        self._amp_full = None
        if self.dkr.use_scatter:
            amps = [getattr(ds, 'dense_amp', None) for ds in batch_data_samples]
            if len(amps) > 0 and all(a is not None for a in amps):
                H, W = batch_data_samples[0].batch_input_shape
                padded = []
                for a in amps:
                    a = a if a.dim() == 3 else a[None]
                    a = a[:, :H, :W]
                    h, w = a.shape[-2:]
                    padded.append(F.pad(a, (0, W - w, 0, H - h)))
                self._amp_full = torch.stack(padded, 0).to(x[0].device).float()
        return super().forward(x, batch_data_samples)

    def _compute_refined_mask(self, query_feat, hi_features):
        decoder_out = self.transformer_decoder.post_norm(query_feat)
        mask_embed = self.mask_embed(decoder_out)
        rank1 = torch.einsum("bqc,bchw->bqhw", mask_embed, hi_features)
        amp_ds = None
        if self._amp_full is not None:
            amp_ds = F.interpolate(self._amp_full, size=hi_features.shape[-2:], mode='area')
        return rank1 + self.dkr(decoder_out, hi_features, coarse_mask=rank1, amp=amp_ds)  # gated (init 0)


@MODELS.register_module()
class GeoStackMask2FormerHead(SealM2FHead):
    """NO-GO fallback: plain STACKING of CoordConv+CBAM+oriented-conv blocks on the refined
    feature (carrier.geostack_op.GeoStack). Diagnostic vs the unified GeoField NO-GO."""

    def __init__(self, *args, geo_m: int = 8, geo_k: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        from carrier.geostack_op import GeoStack
        c = self.mask_embed[-1].out_features
        self.geostack = GeoStack(c, m=geo_m, k=geo_k)

    def _refine_features(self, mask_features):
        hi = super()._refine_features(mask_features)
        return self.geostack(hi)


@MODELS.register_module()
class CoordRefineHead(SealM2FHead):
    """CoordConv at the mask head (teacher puts CoordConv in the head): fold normalized
    coords into the refined mask feature before the rank-1 readout. coord_conv is init to
    identity-on-features + zero-on-coords, so step-0 == baseline; coords are earned."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import torch as _t
        c = self.mask_embed[-1].out_features
        self.coord_conv = nn.Conv2d(c + 2, c, 1)
        with _t.no_grad():
            self.coord_conv.weight.zero_()
            eye = _t.eye(c)
            self.coord_conv.weight[:, :c, 0, 0] = eye   # identity on feature channels
            # weight[:, c:, 0, 0] stays 0 -> coords contribute nothing at init
            self.coord_conv.bias.zero_()

    def _refine_features(self, mask_features):
        hi = super()._refine_features(mask_features)        # (B,C,2h,2w)
        b, c, h, w = hi.shape
        ys = torch.linspace(-1, 1, h, device=hi.device, dtype=hi.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        xs = torch.linspace(-1, 1, w, device=hi.device, dtype=hi.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        return self.coord_conv(torch.cat([hi, xs, ys], dim=1))
