"""Inert two-field mask readout; Torch and the original geometry are injected.

This fresh operator does not modify paired_context.py, a native head, GT,
classification, gates or scoring. Its actual modes are global, spatial_halves
and spatial_reversed; none is an alias recorded as paired. All modes reuse the
original canonical predicted geometry, E, sixteen inside/outside coordinates
and bias-free 512->512->256 SiLU phi (two weights, 393216 scalars, final zero).

Slots 0..7 are four negative t positions with two sides; 8..15 are positive t.
Each half uses its OWN eligible denominator. If either half has no eligible
slot, both fields fall back to the all-eligible mean. All-empty E gives a
connected zero. No query, point or image is removed. Geometry/E/alpha detach;
sampled F, both weights, and the native F/e0 projection remain differentiable.

Spatial logits are M0+(1-alpha)*(delta_neg.F)+alpha*(delta_pos.F), where
alpha=sigmoid(u/sigma_major), u is the predicted PCA-axis pixel coordinate.
Reversed swaps the two fields. Global uses one projection of the all-eligible
mean. Consequently this is not a same-total-compute assertion. The old native
hook returning one coefficient cannot implement the spatial modes.

forward accepts optional actual LIVE native_logits [B,Q,H,W]. If omitted it
projects e0.F. Cold zero residual returns those native values; this has not
been numerically qualified here. M0+delta.F and the old (e0+delta).F are equal
over reals but need not be FP32 bit-identical. Global coefficient/gradient
comparability requires a separate numerical check. An injected native_logits
must retain its original native graph when used in training; this module
cannot establish its provenance. No efficacy, locality, physical evidence,
boundary-learning, novelty, runtime or native-integration claim is made.
"""

MODES = ('global', 'spatial_halves', 'spatial_reversed')
CHANNELS = 256
PAIR_COUNT = 16
HALF_SLOTS = 8
TRAINABLE_TENSORS = 2
TRAINABLE_SCALARS = 393216
INITIALIZATION_SEED = 2026100911


def build_adapter(torch, geometry_source, mode='global'):
    """Build CPU weights with the original restoring factory; no CUDA calls.

    geometry_source is the caller-loaded original paired_context.py module.
    The returned dictionary contains mask_logits, logit_residual, coefficient
    fields [B,Q,2,256], and detached diagnostics. No single new_coeff is exposed
    for spatial modes. The factory has no file loading, I/O or global patches.
    """
    if mode not in MODES:
        raise ValueError('Unknown explicit spatial readout mode: ' + str(mode))
    if not callable(getattr(geometry_source, 'build_adapter', None)) or \
            getattr(geometry_source, 'CHANNELS', None) != CHANNELS or \
            getattr(geometry_source, 'PAIR_COUNT', None) != PAIR_COUNT or \
            getattr(geometry_source, 'LONGITUDINAL_SAMPLES', None) != 8 or \
            getattr(geometry_source, 'LONGITUDINAL_MIN', None) != -1.5 or \
            getattr(geometry_source, 'LONGITUDINAL_MAX', None) != 1.5 or \
            getattr(geometry_source, 'INITIALIZATION_SEED', None) != INITIALIZATION_SEED or \
            getattr(geometry_source, 'TRAINABLE_TENSORS', None) != TRAINABLE_TENSORS or \
            getattr(geometry_source, 'TRAINABLE_SCALARS', None) != TRAINABLE_SCALARS or \
            'paired' not in getattr(geometry_source, 'MODES', ()):
        raise ValueError('Explicit original paired geometry factory required')
    template = geometry_source.build_adapter(torch, mode='paired')
    base_class = type(template)
    del template

    class SpatialPositionPreservingReadout(base_class):
        def __init__(self):
            super().__init__()
            self.mode = mode

        def _pool_fields(self, slot_values, E):
            """All slots execute phi; empty branches remain graph-connected."""
            E = E.detach()
            counts = E.sum(dim=-1)
            weights = (E.to(slot_values.dtype) / counts.clamp_min(1)[..., None]).detach()
            global_mean = (slot_values * weights[..., None]).sum(dim=2)
            global_mean = global_mean * (counts > 0).to(slot_values.dtype)[..., None]
            neg_E, pos_E = E[..., :HALF_SLOTS], E[..., HALF_SLOTS:]
            neg_counts, pos_counts = neg_E.sum(dim=-1), pos_E.sum(dim=-1)
            neg_w = (neg_E.to(slot_values.dtype) / neg_counts.clamp_min(1)[..., None]).detach()
            pos_w = (pos_E.to(slot_values.dtype) / pos_counts.clamp_min(1)[..., None]).detach()
            neg_mean = (slot_values[:, :, :HALF_SLOTS] * neg_w[..., None]).sum(dim=2)
            pos_mean = (slot_values[:, :, HALF_SLOTS:] * pos_w[..., None]).sum(dim=2)
            both = ((neg_counts > 0) & (pos_counts > 0)).detach()
            neg_mean = torch.where(both[..., None], neg_mean, global_mean)
            pos_mean = torch.where(both[..., None], pos_mean, global_mean)
            return global_mean, neg_mean, pos_mean, neg_counts, pos_counts, both

        def _alpha(self, geometry, F, q0, q1):
            # Feature pixel-centre coordinates, no hard domain restriction.
            with torch.no_grad():
                H, W = F.shape[-2:]
                centre = geometry['centre_xy'].detach()[:, q0:q1]
                theta = geometry['used_theta'].detach()[:, q0:q1]
                scale = geometry['sigma_major'].detach()[:, q0:q1]
                xs = torch.arange(W, dtype=F.dtype, device=F.device).reshape(1, 1, 1, W)
                ys = torch.arange(H, dtype=F.dtype, device=F.device).reshape(1, 1, H, 1)
                dx, dy = xs - centre[..., 0, None, None], ys - centre[..., 1, None, None]
                u = torch.cos(theta)[..., None, None] * dx + torch.sin(theta)[..., None, None] * dy
                return torch.sigmoid(u / scale[..., None, None])

        def forward(self, F, e0, valid_fraction, native_logits=None):
            self._validate(F, e0, valid_fraction)
            B, _, H, W = F.shape
            Q = e0.shape[1]
            if native_logits is None:
                native_logits = torch.einsum('bqc,bchw->bqhw', e0, F)
            elif tuple(native_logits.shape) != (B, Q, H, W) or \
                    native_logits.dtype != F.dtype or native_logits.device != F.device or \
                    not bool(torch.isfinite(native_logits).all()):
                raise ValueError('native_logits must be finite FP32 [B,Q,H,W] on the input device')
            geometry = self._geometry(F, e0, valid_fraction)
            E = geometry['eligible'].detach()
            residuals, fields, half_counts, fallbacks = [], [], [], []
            for q0 in range(0, Q, geometry_source.QUERY_CHUNK_SIZE):
                q1 = min(q0 + geometry_source.QUERY_CHUNK_SIZE, Q)
                inside = self._sample_shared(F, geometry['inside'][:, q0:q1])
                outside = self._sample_shared(F, geometry['outside'][:, q0:q1])
                slot_values = self._phi(torch.cat((inside, outside), dim=-1))
                global_mean, neg, pos, nc, pc, both = self._pool_fields(slot_values, E[:, q0:q1])
                if self.mode == 'global':
                    used_neg = used_pos = global_mean
                    residual = torch.einsum('bqc,bchw->bqhw', global_mean, F)
                else:
                    used_neg, used_pos = (pos, neg) if self.mode == 'spatial_reversed' else (neg, pos)
                    neg_logits = torch.einsum('bqc,bchw->bqhw', used_neg, F)
                    pos_logits = torch.einsum('bqc,bchw->bqhw', used_pos, F)
                    alpha = self._alpha(geometry, F, q0, q1)
                    residual = (1.0 - alpha) * neg_logits + alpha * pos_logits
                residuals.append(residual)
                fields.append(torch.stack((used_neg, used_pos), dim=2))
                half_counts.append(torch.stack((nc, pc), dim=-1))
                fallbacks.append(~both)
            logit_residual = torch.cat(residuals, dim=1)
            coefficient_fields = torch.cat(fields, dim=1)
            diagnostics = {key: value.detach() for key, value in geometry.items()}
            diagnostics.update(mode=self.mode, eligibility_frame='canonical',
                half_eligible_counts=torch.cat(half_counts, dim=1).detach(),
                global_fallback=torch.cat(fallbacks, dim=1).detach(),
                alpha_formula='sigmoid(predicted_axis_pixel_coordinate/sigma_major)',
                alpha_and_geometry_detached=True, hard_clipping=False,
                configuration=dict(trainable_tensors=TRAINABLE_TENSORS,
                    trainable_scalars=TRAINABLE_SCALARS, initialization_seed=INITIALIZATION_SEED,
                    negative_slots=(0, 8), positive_slots=(8, 16), slots_per_query=PAIR_COUNT,
                    empty_half='both_fields_all_eligible_mean', empty_E='connected_zero; retain_query',
                    global_projection_count=1, spatial_projection_count=2,
                    global_old_FP32_bit_identity_claimed=False, native_integration_qualified=False))
            return dict(mask_logits=native_logits + logit_residual,
                        logit_residual=logit_residual, coefficient_fields=coefficient_fields,
                        diagnostics=diagnostics)

    adapter = SpatialPositionPreservingReadout()
    expected = {'hidden.weight': (512, 512), 'final.weight': (256, 512)}
    parameters = dict(adapter.named_parameters())
    if set(parameters) != set(expected) or len(adapter.state_dict()) != TRAINABLE_TENSORS or \
            list(adapter.named_buffers()) or \
            any(tuple(parameters[name].shape) != shape for name, shape in expected.items()) or \
            sum(value.numel() for value in parameters.values()) != TRAINABLE_SCALARS:
        raise RuntimeError('Only the two original paired weights may be registered')
    return adapter
