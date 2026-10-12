"""Inert query-conditioned pair-generator probe; caller injects Torch/SOURCE.

All four modes use original detached prediction geometry/sign eligibility,
global eligible pooling, and the SAME unrestricted dense F projection.
raw is the legacy feature-pair generator. The other three normalize each
sampled 256-vector independently before concatenation: raw F, real detached
e0*F, or a preselected channel derangement of e0 multiplied with the SAME F.
This isolates input alternatives under a common overall amplitude control;
channel covariance, sparsity, conditioning and functional capacity may differ.

No extra e0 gradient enters the residual. Native logits retain their supplied
graph, and sampled F plus both MLP weights remain differentiable. A frozen
native training protocol must be established by a separate caller; this
operator does not freeze native parameters or prove unchanged native states.
It has no scientific I/O, GT, training, CUDA initialization or global patch.
No query is removed. No novelty, efficacy, physical SAR, convergence, whole
compute equality or integration claim is made by this standalone SOURCE.
"""

MODES = ('raw', 'raw_rms', 'query_rms', 'permuted_query_rms')
CHANNELS = 256
PAIR_COUNT = 16
TRAINABLE_TENSORS = 2
TRAINABLE_SCALARS = 393216
INITIALIZATION_SEED = 2026100911
PERMUTATION_SEED = 2026101013


def build_adapter(torch, geometry_source, mode='raw',
                  hidden_seed=INITIALIZATION_SEED,
                  permutation_seed=PERMUTATION_SEED):
    """Build CPU-only two-weight operator; private RNG changes are restored.

    forward(F,e0,valid_fraction,native_logits=None) returns final mask logits,
    residual, duplicate global coefficient fields and detached diagnostics.
    A supplied native tensor is shape/dtype/device/finite checked only; its
    head provenance and arithmetic equivalence are the caller's responsibility.
    """
    if mode not in MODES:
        raise ValueError('Unknown query pair-generator mode: ' + str(mode))
    for name, seed in (('hidden_seed', hidden_seed), ('permutation_seed', permutation_seed)):
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError(name + ' must be an explicit nonnegative int64 seed')
    if not callable(getattr(geometry_source, 'build_adapter', None)) or \
            getattr(geometry_source, 'CHANNELS', None) != CHANNELS or \
            getattr(geometry_source, 'PAIR_COUNT', None) != PAIR_COUNT or \
            getattr(geometry_source, 'LONGITUDINAL_SAMPLES', None) != 8 or \
            getattr(geometry_source, 'INITIALIZATION_SEED', None) != INITIALIZATION_SEED or \
            getattr(geometry_source, 'TRAINABLE_TENSORS', None) != TRAINABLE_TENSORS or \
            getattr(geometry_source, 'TRAINABLE_SCALARS', None) != TRAINABLE_SCALARS or \
            'paired' not in getattr(geometry_source, 'MODES', ()):
        raise ValueError('Explicit original paired geometry SOURCE required')
    template = geometry_source.build_adapter(torch, mode='paired')
    base_class = type(template)
    del template

    class QueryConditionedPairGenerator(base_class):
        def __init__(self):
            super().__init__()
            self.mode = mode
            self.hidden_seed = hidden_seed
            self.permutation_seed = permutation_seed
            with torch.random.fork_rng(devices=[]):
                if hidden_seed != INITIALIZATION_SEED:
                    torch.random.default_generator.manual_seed(hidden_seed)
                    torch.nn.init.kaiming_uniform_(self.hidden.weight, a=5**0.5)
                torch.random.default_generator.manual_seed(permutation_seed)
                order = torch.randperm(CHANNELS, device='cpu')
                permutation = torch.empty(CHANNELS, dtype=torch.long, device='cpu')
                permutation[order] = order.roll(-1)
                self.permutation_indices = tuple(permutation.tolist())
            if sorted(self.permutation_indices) != list(range(CHANNELS)) or \
                    any(index == value for index, value in enumerate(self.permutation_indices)):
                raise RuntimeError('The fixed seeded channel cycle must be a derangement')

        def _normalize_channels(self, value):
            """Unit RMS per nonzero vector, zero stays zero; no EPS floor.

            Max scaling prevents finite-vector norm overflow/underflow. Its
            detached scale leaves the real homogeneous normalization derivative
            unchanged; no FP32 bit or extreme-subnormal gradient claim is made.
            """
            if value.dtype != torch.float32 or value.shape[-1] != CHANNELS or \
                    not bool(torch.isfinite(value).all()):
                raise ValueError('Finite FP32 256-channel vectors required')
            scale = value.detach().abs().amax(dim=-1, keepdim=True)
            nonzero = scale > 0
            safe_scale = torch.where(nonzero, scale, torch.ones_like(scale))
            scaled = value / safe_scale
            length = torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)
            safe_length = torch.where(nonzero, length, torch.ones_like(length))
            normalized = (CHANNELS**0.5) * scaled / safe_length
            return torch.where(nonzero, normalized, torch.zeros_like(normalized))

        def _rms_magnitude(self, value):
            with torch.no_grad():
                scale = value.detach().abs().amax(dim=-1, keepdim=True)
                safe = torch.where(scale > 0, scale, torch.ones_like(scale))
                length = torch.linalg.vector_norm(value.detach() / safe, dim=-1, keepdim=True)
                return (length / CHANNELS**0.5 * scale)[..., 0]

        def forward(self, F, e0, valid_fraction, native_logits=None):
            self._validate(F, e0, valid_fraction)
            B, _, H, W = F.shape
            Q = e0.shape[1]
            if native_logits is None:
                native_logits = torch.einsum('bqc,bchw->bqhw', e0, F)
            elif tuple(native_logits.shape) != (B, Q, H, W) or \
                    native_logits.dtype != F.dtype or native_logits.device != F.device or \
                    not bool(torch.isfinite(native_logits).all()):
                raise ValueError('native_logits must be finite FP32 [B,Q,H,W] on input device')
            geometry = self._geometry(F, e0, valid_fraction)
            E = geometry['eligible'].detach()
            if E.dtype != torch.bool or tuple(E.shape) != (B, Q, PAIR_COUNT):
                raise ValueError('Original detached Boolean per-pair eligibility required')
            auxiliary_query = e0.detach()
            if self.mode == 'permuted_query_rms':
                indices = torch.tensor(self.permutation_indices, dtype=torch.long, device=e0.device)
                auxiliary_query = auxiliary_query.index_select(-1, indices)
            residuals, coefficients, before_in, before_out, after_in, after_out = [], [], [], [], [], []
            for q0 in range(0, Q, geometry_source.QUERY_CHUNK_SIZE):
                q1 = min(q0 + geometry_source.QUERY_CHUNK_SIZE, Q)
                inside = self._sample_shared(F, geometry['inside'][:, q0:q1])
                outside = self._sample_shared(F, geometry['outside'][:, q0:q1])
                if self.mode in ('query_rms', 'permuted_query_rms'):
                    query = auxiliary_query[:, q0:q1, None, :]
                    inside, outside = inside * query, outside * query
                before_in.append(self._rms_magnitude(inside))
                before_out.append(self._rms_magnitude(outside))
                if self.mode != 'raw':
                    inside, outside = self._normalize_channels(inside), self._normalize_channels(outside)
                after_in.append(self._rms_magnitude(inside))
                after_out.append(self._rms_magnitude(outside))
                slot_values = self._phi(torch.cat((inside, outside), dim=-1))
                eligible = E[:, q0:q1]
                counts = eligible.sum(dim=-1)
                weights = (eligible.to(slot_values.dtype) / counts.clamp_min(1)[..., None]).detach()
                coefficient = (slot_values * weights[..., None]).sum(dim=2)
                coefficient = coefficient * (counts > 0).to(slot_values.dtype)[..., None]
                residuals.append(torch.einsum('bqc,bchw->bqhw', coefficient, F))
                coefficients.append(coefficient)
            residual = torch.cat(residuals, dim=1)
            coefficient = torch.cat(coefficients, dim=1)
            diagnostics = {key: value.detach() for key, value in geometry.items()}
            diagnostics.update(mode=self.mode, global_pool=True, dense_readout='same_unnormalized_F',
                auxiliary_query_detached=True, input_RMS_normalized=self.mode != 'raw',
                input_conditioning=('real_query' if self.mode == 'query_rms' else
                    'permuted_query' if self.mode == 'permuted_query_rms' else 'raw_features'),
                input_inside_RMS_before=torch.cat(before_in, dim=1).detach(),
                input_outside_RMS_before=torch.cat(before_out, dim=1).detach(),
                input_inside_RMS_after=torch.cat(after_in, dim=1).detach(),
                input_outside_RMS_after=torch.cat(after_out, dim=1).detach(),
                hidden_seed=self.hidden_seed, permutation_seed=self.permutation_seed,
                permutation_kind='seeded_256_cycle_not_all_derangements_uniform',
                no_query_removed=True, empty_E='connected_zero', parameters=TRAINABLE_SCALARS,
                normalization_controls_amplitude_only=True, equal_compute_claimed=False,
                frozen_native_or_native_integration_qualified=False)
            return dict(mask_logits=native_logits + residual, logit_residual=residual,
                coefficient_fields=torch.stack((coefficient, coefficient), dim=2), diagnostics=diagnostics)

    adapter = QueryConditionedPairGenerator()
    expected = {'hidden.weight': (512, 512), 'final.weight': (256, 512)}
    parameters = dict(adapter.named_parameters())
    if set(parameters) != set(expected) or len(adapter.state_dict()) != TRAINABLE_TENSORS or \
            list(adapter.named_buffers()) or \
            any(tuple(parameters[name].shape) != shape for name, shape in expected.items()) or \
            sum(value.numel() for value in parameters.values()) != TRAINABLE_SCALARS:
        raise RuntimeError('Only the two original paired weights may be registered')
    return adapter
