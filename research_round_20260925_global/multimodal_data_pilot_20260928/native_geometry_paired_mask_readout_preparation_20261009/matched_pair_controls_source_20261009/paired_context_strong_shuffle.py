"""Inert geometry-conditioned paired feature readout; Torch is caller-injected.

This is an unqualified, falsifiable engineering candidate, not a new sampling,
PCA, contrast, or dynamic-mask primitive. No imports of scientific libraries,
hooks, forward of a native head, GT access, I/O, evaluation, or main occur here.

The caller passes actual final native F and e0. Geometry and eligibility come
only from stop-gradient M0=e0.F and metadata-derived valid fractions. Sampled
features and the additive coefficient path remain live. The module changes
only the final mask coefficients. It has no classification or scoring code.
Although a single forward leaves its incoming class logits untouched, class
probabilities need not agree across trained arms, and final Hungarian matching
can change when masks change. 'Inside/outside' refers to the current predicted
mask; neither designation proves ship/background, sea, or scattering physics.

paired/shuffled/mean_before/fg_fine use the identical canonical coordinates
and eligibility. axis_aligned shares the centre and eigenvalue scales but
uses theta=0 and its own eligibility; it tests frame plus support coverage,
not an isolated direction effect. No queries or images are removed. Empty
support or empty eligibility gives a connected zero coefficient residual.

Uniform averaging is deliberate. Shuffling is a deterministic one-place
cyclic shift of ONLY the eligible outside sequence, preserving its feature
multiset and uniform marginal weights exactly. Zero/singleton eligibility
cannot give a nontrivial permutation and is reported. Linear paired averaging
would equal a difference of means; the hidden SiLU precedes pair aggregation.

Pixel coordinates use centre indices (x=0,...,W-1). align_corners=False maps
them to grid coordinates 2*(x+.5)/W-1. 'Valid' here means a coordinate within
the feature-centre canvas and a positive bilinearly sampled valid fraction;
fractional boundary cells are retained rather than asserted fully unpadded.
The caller must derive valid_fraction from metadata, never GT.

Prospective strong_shuffled is an equal-parameter, equal-phi-call structural
control, not a new training recipe. Within EACH query and EACH slot parity
(side -1/+1), eligible slots are sorted longitudinally and the outside donor
is shifted by floor(n/2) modulo that side's eligible count. Empty/singleton
sides and all ineligible slots retain their original donors. It never swaps
features across queries, sides or images. Inside coordinates, E, rho and the
outside marginal feature multiset are unchanged. Only pairing BEFORE phi
changes. All original five mode branches and initialization remain copied.
No runtime bitwise equivalence of those modes has been established here,
and an inference-only shuffle drop would not establish training utility.
"""

MODES = ('paired', 'shuffled', 'mean_before', 'axis_aligned', 'fg_fine', 'strong_shuffled')
CHANNELS = 256
PAIR_COUNT = 16
LONGITUDINAL_SAMPLES = 8
LONGITUDINAL_MIN = -1.5
LONGITUDINAL_MAX = 1.5
INSIDE_MINOR_OFFSET = 0.5
OUTSIDE_MINOR_OFFSET = 2.5
SIGMA_FLOOR_PIXELS = 1.0
NEAR_ISOTROPIC_RELATIVE_GAP = 0.05
MASS_EPS = 1e-12
QUERY_CHUNK_SIZE = 8
INITIALIZATION_SEED = 2026100911
TRAINABLE_TENSORS = 2
TRAINABLE_SCALARS = 393216


def build_adapter(torch, mode='paired'):
    """Return an FP32 nn.Module.forward(F, e0, valid_fraction) -> dictionary.

    F: [B,256,H,W], actual final native mask basis, potentially live.
    e0: [B,Q,256], actual native mask coefficients, potentially live.
    valid_fraction: [B,1,H,W], FP32 fractions in [0,1], from image metadata.
    Caller moves the returned module to the same device; this factory touches
    only CPU RNG in a restoring fork and never seeds or initializes CUDA RNG.

    Each mode has exactly 393216 scalars in two bias-free Linear weights.
    Paired modes: 512->512->256. fg_fine: 256->768->256. Hidden weights are
    nonzero native Linear initialization; final weights are all zero. No extra
    parameters or registered buffers are introduced. At cold initialization,
    new_coeff=e0+0 while the final weight still has a differentiable path.

    Outputs are new_coeff, coefficient_residual, and detached diagnostics.
    There are no masks/scores/postprocessing, optimizer, or state side effects.
    Use the native final einsum with new_coeff and the SAME F after this call.
    """
    if mode not in MODES:
        raise ValueError('Unknown paired-context mode: ' + str(mode))

    class GeometryPairedMaskReadout(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mode = mode
            in_features, hidden_features = (256, 768) if mode == 'fg_fine' else (512, 512)
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(INITIALIZATION_SEED)
                self.hidden = torch.nn.Linear(in_features, hidden_features,
                                              bias=False, device='cpu', dtype=torch.float32)
                self.final = torch.nn.Linear(hidden_features, 256,
                                             bias=False, device='cpu', dtype=torch.float32)
                with torch.no_grad():
                    self.final.weight.zero_()
            if sum(p.numel() for p in self.parameters()) != TRAINABLE_SCALARS:
                raise RuntimeError('Unexpected paired-context parameter count')

        def _validate(self, F, e0, valid_fraction):
            if F.ndim != 4 or F.shape[1] != CHANNELS or min(F.shape[2:]) < 1:
                raise ValueError('F must be [B,256,H,W] with a nonempty canvas')
            if e0.ndim != 3 or e0.shape[0] != F.shape[0] or e0.shape[2] != CHANNELS:
                raise ValueError('e0 must be [B,Q,256] with the same batch')
            if e0.shape[1] < 1 or F.shape[0] < 1:
                raise ValueError('Batch and query population must be nonempty')
            if tuple(valid_fraction.shape) != (F.shape[0], 1, F.shape[2], F.shape[3]):
                raise ValueError('valid_fraction must be [B,1,H,W]')
            for name, value in (('F', F), ('e0', e0), ('valid_fraction', valid_fraction)):
                if value.dtype != torch.float32 or value.device != F.device:
                    raise ValueError(name + ' must be FP32 on the same device as F')
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(name + ' contains nonfinite values')
            if not bool(((valid_fraction >= 0) & (valid_fraction <= 1)).all()):
                raise ValueError('valid_fraction must lie in [0,1]')
            if self.hidden.weight.device != F.device or self.hidden.weight.dtype != F.dtype:
                raise ValueError('Move the module to the FP32 input device before forward')

        def _geometry(self, F, e0, valid_fraction):
            # Everything in this function, including the hard sign predicates,
            # is detached. No sampled-feature or e0 additive gradient is cut.
            with torch.no_grad():
                B, _, H, W = F.shape
                Q = e0.shape[1]
                m0_chunks = []
                for q0 in range(0, Q, QUERY_CHUNK_SIZE):
                    m0_chunks.append(torch.einsum('bqc,bchw->bqhw',
                                                  e0.detach()[:, q0:q0 + QUERY_CHUNK_SIZE],
                                                  F.detach()))
                M0 = torch.cat(m0_chunks, dim=1)
                if not bool(torch.isfinite(M0).all()):
                    raise ValueError('Native coefficient projection contains nonfinite values')
                rho = torch.relu(2.0 * torch.sigmoid(M0) - 1.0) * valid_fraction.detach()
                mass = rho.sum(dim=(-2, -1))
                has_support = mass > MASS_EPS
                denominator = mass.clamp_min(MASS_EPS)
                xs = torch.arange(W, device=F.device, dtype=F.dtype).reshape(1, 1, 1, W)
                ys = torch.arange(H, device=F.device, dtype=F.dtype).reshape(1, 1, H, 1)
                mu_x = (rho * xs).sum(dim=(-2, -1)) / denominator
                mu_y = (rho * ys).sum(dim=(-2, -1)) / denominator
                mu_x = torch.where(has_support, mu_x, torch.full_like(mu_x, (W - 1) / 2.0))
                mu_y = torch.where(has_support, mu_y, torch.full_like(mu_y, (H - 1) / 2.0))
                dx = xs - mu_x[..., None, None]
                dy = ys - mu_y[..., None, None]
                xx = (rho * dx.square()).sum(dim=(-2, -1)) / denominator
                xy = (rho * dx * dy).sum(dim=(-2, -1)) / denominator
                yy = (rho * dy.square()).sum(dim=(-2, -1)) / denominator
                trace = xx + yy
                gap = torch.sqrt((xx - yy).square() + 4.0 * xy.square())
                near_isotropic = gap / trace.clamp_min(MASS_EPS) <= NEAR_ISOTROPIC_RELATIVE_GAP
                degenerate = (~has_support) | (trace <= MASS_EPS)
                theta = 0.5 * torch.atan2(2.0 * xy, xx - yy)
                theta = torch.where(degenerate | near_isotropic, torch.zeros_like(theta), theta)
                lambda_major = ((trace + gap) / 2.0).clamp_min(0.0)
                lambda_minor = ((trace - gap) / 2.0).clamp_min(0.0)
                sigma_major = torch.sqrt(lambda_major).clamp_min(SIGMA_FLOOR_PIXELS)
                sigma_minor = torch.sqrt(lambda_minor).clamp_min(SIGMA_FLOOR_PIXELS)
                mu = torch.stack((mu_x, mu_y), dim=-1)
                canonical_inside, canonical_outside = self._coordinates(
                    mu, sigma_major, sigma_minor, theta)
                canonical_E = self._eligibility(M0, valid_fraction.detach(),
                                                 canonical_inside, canonical_outside,
                                                 has_support)
                if self.mode == 'axis_aligned':
                    used_theta = torch.zeros_like(theta)
                    inside, outside = self._coordinates(mu, sigma_major, sigma_minor, used_theta)
                    E = self._eligibility(M0, valid_fraction.detach(), inside, outside, has_support)
                else:
                    used_theta = theta
                    inside, outside, E = canonical_inside, canonical_outside, canonical_E
                return {
                    'inside': inside, 'outside': outside, 'eligible': E,
                    'canonical_eligible_counts': canonical_E.sum(dim=-1),
                    'eligible_counts': E.sum(dim=-1), 'rho_mass': mass,
                    'canonical_theta': theta, 'used_theta': used_theta,
                    'near_isotropic': near_isotropic, 'degenerate': degenerate,
                    'sigma_major': sigma_major, 'sigma_minor': sigma_minor,
                    'centre_xy': mu,
                }

        def _coordinates(self, mu, sigma_major, sigma_minor, theta):
            # Slot ordering is eight ascending t values, each with sides -1,+1.
            t = torch.linspace(LONGITUDINAL_MIN, LONGITUDINAL_MAX,
                               LONGITUDINAL_SAMPLES, device=mu.device, dtype=mu.dtype)
            t = t[:, None].expand(LONGITUDINAL_SAMPLES, 2).reshape(PAIR_COUNT)
            sides = torch.tensor((-1.0, 1.0), device=mu.device, dtype=mu.dtype)
            sides = sides[None, :].expand(LONGITUDINAL_SAMPLES, 2).reshape(PAIR_COUNT)
            major = torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1)
            minor = torch.stack((-torch.sin(theta), torch.cos(theta)), dim=-1)
            axial = t.reshape(1, 1, PAIR_COUNT, 1) * sigma_major[..., None, None] * major[..., None, :]
            transverse = sides.reshape(1, 1, PAIR_COUNT, 1) * sigma_minor[..., None, None] * minor[..., None, :]
            inside = mu[..., None, :] + axial + INSIDE_MINOR_OFFSET * transverse
            outside = mu[..., None, :] + axial + OUTSIDE_MINOR_OFFSET * transverse
            return inside, outside

        def _grid(self, xy, H, W):
            gx = 2.0 * (xy[..., 0] + 0.5) / W - 1.0
            gy = 2.0 * (xy[..., 1] + 0.5) / H - 1.0
            return torch.stack((gx, gy), dim=-1)

        def _sample_shared(self, value, xy):
            # Flatten points into the output grid, never replicate F per query.
            B, C, H, W = value.shape
            Q, S = xy.shape[1:3]
            grid = self._grid(xy, H, W).reshape(B, Q * S, 1, 2)
            sampled = torch.nn.functional.grid_sample(value, grid, mode='bilinear',
                                                       padding_mode='zeros', align_corners=False)
            return sampled[..., 0].reshape(B, C, Q, S).permute(0, 2, 3, 1)

        def _sample_query_scalar(self, value, xy):
            B, Q, H, W = value.shape
            S = xy.shape[2]
            grid = self._grid(xy, H, W).reshape(B * Q, S, 1, 2)
            sampled = torch.nn.functional.grid_sample(value.reshape(B * Q, 1, H, W), grid,
                                                       mode='bilinear', padding_mode='zeros',
                                                       align_corners=False)
            return sampled.reshape(B, Q, S)

        def _eligibility(self, M0, valid_fraction, inside, outside, has_support):
            H, W = M0.shape[-2:]
            in_canvas = ((inside[..., 0] >= 0.0) & (inside[..., 0] <= W - 1.0) &
                         (inside[..., 1] >= 0.0) & (inside[..., 1] <= H - 1.0))
            out_canvas = ((outside[..., 0] >= 0.0) & (outside[..., 0] <= W - 1.0) &
                          (outside[..., 1] >= 0.0) & (outside[..., 1] <= H - 1.0))
            valid_inside = self._sample_shared(valid_fraction, inside)[..., 0] > 0.0
            valid_outside = self._sample_shared(valid_fraction, outside)[..., 0] > 0.0
            inside_M0 = self._sample_query_scalar(M0, inside)
            outside_M0 = self._sample_query_scalar(M0, outside)
            return (has_support[..., None] & in_canvas & out_canvas & valid_inside & valid_outside &
                    (inside_M0 > 0.0) & (outside_M0 <= 0.0))

        def _outside_donor_slots(self, E):
            # Exact cyclic permutation of eligible slots; ineligible donors are
            # irrelevant because their output has zero averaging weight.
            counts = E.sum(dim=-1, keepdim=True)
            slots = torch.arange(PAIR_COUNT, device=E.device, dtype=torch.long)
            slots = slots.reshape(1, 1, PAIR_COUNT).expand_as(E)
            sorted_eligible = torch.where(E, slots, torch.full_like(slots, PAIR_COUNT)).sort(dim=-1).values
            eligible_rank = E.to(torch.long).cumsum(dim=-1) - 1
            donor_rank = (eligible_rank + 1).remainder(counts.clamp_min(1))
            donor_slot = sorted_eligible.gather(-1, donor_rank)
            return donor_slot.clamp_max(PAIR_COUNT - 1)

        def _strong_outside_donor_slots(self, E):
            # Slots 2*t and 2*t+1 are separate -/+ sides. Donors only permute
            # the eligible longitudinal sequence WITHIN a query and side.
            slots = torch.arange(PAIR_COUNT, device=E.device, dtype=torch.long)
            slots = slots.reshape(1, 1, PAIR_COUNT).expand_as(E)
            donors = slots.clone()
            for side in (0, 1):
                side_E = E[..., side::2]
                side_slots = slots[..., side::2]
                n = side_E.sum(dim=-1, keepdim=True)
                sorted_eligible = torch.where(
                    side_E, side_slots, torch.full_like(side_slots, PAIR_COUNT)
                ).sort(dim=-1).values
                eligible_rank = side_E.to(torch.long).cumsum(dim=-1) - 1
                donor_rank = (eligible_rank + n // 2).remainder(n.clamp_min(1))
                shifted = sorted_eligible.gather(-1, donor_rank)
                # The unused n=0 candidate may be sentinel 16; only eligible
                # receivers use shifted, so every exported donor is in 0..15.
                donors[..., side::2] = torch.where(side_E, shifted, side_slots)
            return donors

        def _phi(self, value):
            return self.final(torch.nn.functional.silu(self.hidden(value)))

        def forward(self, F, e0, valid_fraction):
            self._validate(F, e0, valid_fraction)
            geometry = self._geometry(F, e0, valid_fraction)
            E = geometry['eligible']
            counts = geometry['eligible_counts']
            weights = (E.to(F.dtype) / counts.clamp_min(1)[..., None]).detach()
            donor_slots = self._outside_donor_slots(E).detach()
            if self.mode == 'strong_shuffled':
                donor_slots = self._strong_outside_donor_slots(E).detach()
            residual_chunks = []
            for q0 in range(0, e0.shape[1], QUERY_CHUNK_SIZE):
                q1 = min(q0 + QUERY_CHUNK_SIZE, e0.shape[1])
                inside_features = self._sample_shared(F, geometry['inside'][:, q0:q1])
                w = weights[:, q0:q1, :, None]
                if self.mode == 'fg_fine':
                    residual = self._phi((inside_features * w).sum(dim=2))
                else:
                    outside_features = self._sample_shared(F, geometry['outside'][:, q0:q1])
                    if self.mode == 'shuffled':
                        donor = donor_slots[:, q0:q1, :, None].expand_as(outside_features)
                        outside_features = outside_features.gather(2, donor)
                    elif self.mode == 'strong_shuffled':
                        donor = donor_slots[:, q0:q1, :, None].expand_as(outside_features)
                        outside_features = outside_features.gather(2, donor)
                    if self.mode == 'mean_before':
                        mean_inside = (inside_features * w).sum(dim=2)
                        mean_outside = (outside_features * w).sum(dim=2)
                        residual = self._phi(torch.cat((mean_inside, mean_outside), dim=-1))
                    else:
                        joint = torch.cat((inside_features, outside_features), dim=-1)
                        residual = (self._phi(joint) * w).sum(dim=2)
                # Explicitly zero empty queries without bypassing phi's graph.
                residual = residual * (counts[:, q0:q1] > 0).to(F.dtype)[..., None]
                residual_chunks.append(residual)
            coefficient_residual = torch.cat(residual_chunks, dim=1)
            diagnostics = {key: value.detach() for key, value in geometry.items()}
            diagnostics.update({
                'outside_donor_slots': donor_slots,
                'shuffle_can_be_nontrivial': (counts > 1).detach(),
                'mode': self.mode,
                'eligibility_frame': 'axis_aligned' if self.mode == 'axis_aligned' else 'canonical',
                'configuration': {
                    'longitudinal_samples': LONGITUDINAL_SAMPLES,
                    'longitudinal_range': (LONGITUDINAL_MIN, LONGITUDINAL_MAX),
                    'inside_minor_offset': INSIDE_MINOR_OFFSET,
                    'outside_minor_offset': OUTSIDE_MINOR_OFFSET,
                    'sigma_floor_pixels': SIGMA_FLOOR_PIXELS,
                    'near_isotropic_relative_gap': NEAR_ISOTROPIC_RELATIVE_GAP,
                    'mass_eps': MASS_EPS,
                    'query_chunk_size': QUERY_CHUNK_SIZE,
                    'initialization_seed': INITIALIZATION_SEED,
                    'trainable_tensors': TRAINABLE_TENSORS,
                    'trainable_scalars': TRAINABLE_SCALARS,
                    'valid_eligibility': 'inside_feature_centre_canvas_and_sampled_fraction_positive',
                    'shuffle': 'eligible_outside_one_place_cyclic_shift',
                },
            })
            if self.mode == 'strong_shuffled':
                side_counts = torch.stack((E[..., 0::2].sum(dim=-1),
                                           E[..., 1::2].sum(dim=-1)), dim=-1).detach()
                side_nontrivial = (side_counts > 1).detach()
                recipient_slots = torch.arange(PAIR_COUNT, device=E.device, dtype=torch.long)
                recipient_slots = recipient_slots.reshape(1, 1, PAIR_COUNT)
                donor_changed = (E & (donor_slots != recipient_slots)).detach()
                diagnostics.update({
                    'shuffle_can_be_nontrivial': side_nontrivial.any(dim=-1),
                    'strong_shuffle_side_eligible_counts': side_counts,
                    'strong_shuffle_side_shift': side_counts // 2,
                    'strong_shuffle_side_can_be_nontrivial': side_nontrivial,
                    'strong_shuffle_donor_changed': donor_changed,
                    'strong_shuffle_donor_changed_counts': donor_changed.sum(dim=-1),
                })
                diagnostics['configuration'].update({
                    'shuffle': 'eligible_outside_same_side_floor_n_over_2_cyclic_shift',
                    'strong_shuffle_side_parity': '0=-1,1=+1',
                    'strong_shuffle_cross_query': False,
                    'strong_shuffle_cross_image': False,
                })
            return {'new_coeff': e0 + coefficient_residual,
                    'coefficient_residual': coefficient_residual,
                    'diagnostics': diagnostics}

    return GeometryPairedMaskReadout()
