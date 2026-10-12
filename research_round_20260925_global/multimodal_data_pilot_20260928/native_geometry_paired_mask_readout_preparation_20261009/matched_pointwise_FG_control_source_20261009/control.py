"""Inert pointwise foreground control; Torch and sealed geometry are injected.

build_adapter(torch, geometry_source) requires the caller-loaded original
paired_context.py module. Its actual factory creates GeometryPairedMaskReadout
inside build_adapter, not an exported PairedContextGeometryAdapter class.
This fresh factory subclasses that returned class in fg_fine configuration;
it does not change the old module, its MODES, a global mode, or an old instance.
Only forward is replaced. Geometry, M0 detach, coordinates, outside-dependent
eligibility, sigma floors, weights, sampling and validation reuse old methods.

Existing fg_fine: phi(mean_E(F_inside)). This control:
mean_E(phi(F_inside)), evaluating all16 slots before masking/averaging. Its
bias-free 256->768->256 SiLU MLP has2 weights/393216 scalars, initialized by
the original CPU-only restoring RNG factory with its final weight zero.
The two Linear MAC counts per slot sum to393216, equal to paired's
512->512->256. Hidden activation width, input dimension, outside feature
gather, activation work, backward work, effective functions and end-to-end
runtime are different; this is NOT equal capacity or fully equal compute.
Outside M0/valid samples STILL affect E. Only outside256D feature values are
absent from phi. No query/support filtering, GT, class/scoring or postprocess.

All original16 slots execute phi, including E0. E0 residual is multiplied to
connected zero, then new_coeff=e0+residual. State names remain hidden.weight
and final.weight with no new buffers; an attached native model would retain
its native610 plus these2 state tensors, subject to separate qualification.
Cold identity, gradients, numerical equivalence and runtime have NOT been
executed or qualified here. No trainer/preflight/reader or file I/O exists.
This factory has an explicit geometry_source argument: future native wiring
must bind that dependency, rather than assuming the old hook's factory API.
"""


MODE = 'fg_pointwise'
MODES = (MODE,)
TRAINABLE_TENSORS = 2
TRAINABLE_SCALARS = 393216
LINEAR_MACS_PER_SLOT = 393216


def build_adapter(torch, geometry_source, mode=MODE):
    """Return a fresh module with forward(F,e0,valid_fraction) dictionary API.

The original factory is called twice only during CPU construction: once to
obtain its local class, once via the subclass constructor. Each call restores
CPU RNG independently. No CUDA API or global RNG seed is added by this file.
Caller moves the returned module to the FP32 input device, as in old source.
"""
    if mode != MODE:
        raise ValueError('Only the explicit fresh fg_pointwise control is supported')
    if not callable(getattr(geometry_source, 'build_adapter', None)) or \
            getattr(geometry_source, 'CHANNELS', None) != 256 or \
            getattr(geometry_source, 'PAIR_COUNT', None) != 16 or \
            getattr(geometry_source, 'TRAINABLE_TENSORS', None) != TRAINABLE_TENSORS or \
            getattr(geometry_source, 'TRAINABLE_SCALARS', None) != TRAINABLE_SCALARS or \
            'fg_fine' not in getattr(geometry_source, 'MODES', ()):
        raise ValueError('Caller must inject the sealed original paired geometry factory')
    template = geometry_source.build_adapter(torch, mode='fg_fine')
    base_class = type(template)
    del template

    class PointwiseForegroundReadout(base_class):
        def __init__(self):
            super().__init__()
            self.mode = MODE

        def forward(self, F, e0, valid_fraction):
            self._validate(F, e0, valid_fraction)
            geometry = self._geometry(F, e0, valid_fraction)
            E = geometry['eligible']
            counts = geometry['eligible_counts']
            weights = (E.to(F.dtype) / counts.clamp_min(1)[..., None]).detach()
            residual_chunks = []
            for q0 in range(0, e0.shape[1], geometry_source.QUERY_CHUNK_SIZE):
                q1 = min(q0 + geometry_source.QUERY_CHUNK_SIZE, e0.shape[1])
                inside_features = self._sample_shared(F, geometry['inside'][:, q0:q1])
                # phi executes all16 slots; geometry/E do not acquire gradients.
                residual = (self._phi(inside_features) * weights[:, q0:q1, :, None]).sum(dim=2)
                residual = residual * (counts[:, q0:q1] > 0).to(F.dtype)[..., None]
                residual_chunks.append(residual)
            coefficient_residual = torch.cat(residual_chunks, dim=1)
            diagnostics = {key: value.detach() for key, value in geometry.items()}
            diagnostics.update(mode=MODE, eligibility_frame='canonical',
                outside_features_in_phi=False, outside_M0_and_validity_used_for_E=True,
                configuration=dict(readout_order='phi_per_inside_slot_then_uniform_eligible_pool',
                    slots_per_query=geometry_source.PAIR_COUNT, query_chunk_size=geometry_source.QUERY_CHUNK_SIZE,
                    initialization_seed=geometry_source.INITIALIZATION_SEED,
                    trainable_tensors=TRAINABLE_TENSORS, trainable_scalars=TRAINABLE_SCALARS,
                    linear_MACs_per_slot=LINEAR_MACS_PER_SLOT,
                    empty_E='connected_zero_residual; no query removed',
                    geometry_and_eligibility='original inherited canonical methods; stopgrad',
                    equal_total_compute_or_capacity_claimed=False))
            return dict(new_coeff=e0 + coefficient_residual,
                        coefficient_residual=coefficient_residual, diagnostics=diagnostics)

    adapter = PointwiseForegroundReadout()
    expected = {'hidden.weight': (768, 256), 'final.weight': (256, 768)}
    parameters = dict(adapter.named_parameters())
    if set(parameters) != set(expected) or len(adapter.state_dict()) != TRAINABLE_TENSORS or \
            list(adapter.named_buffers()) or \
            any(tuple(parameters[name].shape) != shape for name, shape in expected.items()) or \
            sum(value.numel() for value in parameters.values()) != TRAINABLE_SCALARS:
        raise RuntimeError('Fresh control must have only original fg_fine two-weight state')
    return adapter
