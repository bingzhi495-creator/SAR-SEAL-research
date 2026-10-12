"""Inert passive observation of the existing paired adapter; caller injects Torch.

No model forward, GT, checkpoint access, I/O, scientific import, RNG seeding,
geometry reconstruction, projection, sampling, or output replacement occurs here.
The owner must run an uninstrumented reference and an observed replay on the
same checkpoint/input/metadata/private seed. Only owner-run bitwise checks can
qualify this observation; source inspection cannot establish numerical identity.

Use passive_adapter_capture INSIDE the caller's unchanged native hook/seed scope.
It captures the actual returned diagnostics, including every slot of all 100
queries and queries with no eligible slots. Its forward hook returns None.
Only detached CPU copies survive. F is never saved. Full rho, M0 and raw
covariance entries are not returned by the existing adapter and stay unknown.

Coordinates are feature-centre indices on F200. Together with img_shape,
batch_input_shape and ori_shape they allow later independent position labelling.
GT or lattice-area occupancy labels do not establish feature purity. No GT may
be passed into this helper or into the observed model forward.

Identity snapshots intentionally keep CPU tensors rather than digest summaries:
all ten class/mask outputs, official all100 masks/query IDs/score factors, full
model state, input binding and CPU/CUDA RNG must agree byte for byte. The owner
must release these snapshots after a bounded preflight; they are not exports.
"""

from contextlib import contextmanager


QUERY_COUNT = 100
SLOT_COUNT = 16
READOUT_COUNT = 10
METADATA_KEYS = ('img_shape', 'batch_input_shape', 'ori_shape', 'scale_factor')
GEOMETRY_KEYS = (
    'inside', 'outside', 'eligible', 'canonical_eligible_counts',
    'eligible_counts', 'rho_mass', 'canonical_theta', 'used_theta',
    'near_isotropic', 'degenerate', 'sigma_major', 'sigma_minor', 'centre_xy',
    'outside_donor_slots', 'shuffle_can_be_nontrivial', 'mode',
    'eligibility_frame', 'configuration',
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def cpu_tree(torch, value):
    """Independent detached CPU copies; never retain the original Tensor graph."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device='cpu').contiguous().clone()
    if isinstance(value, dict):
        require(all(isinstance(key, str) for key in value), 'Tree keys must be strings')
        return {key: cpu_tree(torch, item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_tree(torch, item) for item in value)
    if isinstance(value, list):
        return [cpu_tree(torch, item) for item in value]
    require(value is None or type(value) in (str, bool, int, float),
            'Only tensors and ordinary scalar containers may be captured')
    return value


def metadata_snapshot(batch_metadata):
    """Copy only geometry metainfo, excluding every annotation/sample field."""
    require(isinstance(batch_metadata, (list, tuple)) and len(batch_metadata) == 1,
            'The bounded native replay uses one image per forward')
    meta = batch_metadata[0]
    require(isinstance(meta, dict), 'Pass metainfo dictionaries, not data samples')
    result = {}
    for key in METADATA_KEYS:
        if key in meta:
            value = meta[key]
            require(isinstance(value, (list, tuple)) and
                    all(type(item) in (int, float) for item in value),
                    'Geometry metadata must be plain numeric sequences: ' + key)
            result[key] = tuple(value)
    require(tuple(result.get('batch_input_shape', ())) == (800, 800) and
            len(result.get('img_shape', ())) >= 2 and
            len(result.get('ori_shape', ())) >= 2,
            'Actual fixed800 and original/resized image geometry are required')
    for key in ('img_shape', 'ori_shape'):
        require(all(type(item) is int and item > 0 for item in result[key][:2]),
                'Image geometry must have positive integer height/width')
    require(max(result['img_shape'][:2]) <= 800, 'Resized image exceeds its canvas')
    return result


@contextmanager
def passive_adapter_capture(torch, adapter, batch_metadata, *,
                            include_coefficients=False, expected_calls=1):
    """Yield CPU-only capture; install exactly one returning-None forward hook.

    Expected calls is one for applied paired/FG adapters. For native0 or an
    explicitly disabled adapter use expected_calls=0; actual geometry is then
    unavailable, never reconstructed. This helper does not create the native
    paired_readout_context: the owner supplies its existing unchanged scope.
    """
    require(type(include_coefficients) is bool and type(expected_calls) is int and
            expected_calls in (0, 1), 'Explicit zero/one-call contract required')
    require(adapter is not None or expected_calls == 0,
            'A one-call capture requires the actually registered adapter')
    metadata = metadata_snapshot(batch_metadata)
    record = dict(schema='passive-paired-adapter-capture-v1', calls=0,
                  expected_calls=expected_calls, actual_capture_available=False,
                  capture=None, metadata=metadata, hooks_removed=False,
                  forward_hook_returned_None=True, F_saved=False,
                  full_rho_observed=False, M0_observed=False,
                  raw_covariance_observed=False,
                  geometry_recomputed=False, all_queries_retained=True,
                  coordinate_convention='F200 centre-index xy; align_corners=False',
                  GT_position_labels_are_not_feature_purity=True)
    handle = None
    original_hook_ids = tuple(adapter._forward_hooks) if adapter is not None else ()

    def observe(module, inputs, output):
        require(module is adapter and record['calls'] == 0 and expected_calls == 1,
                'Expected exactly one existing adapter call')
        require(isinstance(inputs, tuple) and len(inputs) == 3 and
                isinstance(output, dict), 'Unexpected actual adapter interface')
        F, e0, valid_fraction = inputs
        require(tuple(F.shape) == (1, 256, 200, 200) and
                tuple(e0.shape) == (1, QUERY_COUNT, 256) and
                tuple(valid_fraction.shape) == (1, 1, 200, 200) and
                F.dtype == e0.dtype == valid_fraction.dtype == torch.float32,
                'Expected actual one-image native F/e0/valid-fraction shapes')
        diagnostics = output['diagnostics']
        require(isinstance(diagnostics, dict) and
                all(key in diagnostics for key in GEOMETRY_KEYS),
                'Actual adapter diagnostics contract differs')
        require(tuple(diagnostics['inside'].shape) == (1, QUERY_COUNT, SLOT_COUNT, 2) and
                tuple(diagnostics['outside'].shape) == (1, QUERY_COUNT, SLOT_COUNT, 2) and
                tuple(diagnostics['eligible'].shape) == (1, QUERY_COUNT, SLOT_COUNT) and
                diagnostics['eligible'].dtype == torch.bool and
                tuple(output['coefficient_residual'].shape) == (1, QUERY_COUNT, 256) and
                tuple(output['new_coeff'].shape) == tuple(e0.shape),
                'Returned geometry/residual must preserve every query and slot')
        captured = dict(geometry=cpu_tree(torch, diagnostics),
                        coefficient_residual=cpu_tree(torch, output['coefficient_residual']),
                        actual_valid_fraction=cpu_tree(torch, valid_fraction),
                        F_shape=tuple(F.shape), e0_shape=tuple(e0.shape),
                        native_query_ids=tuple(range(QUERY_COUNT)))
        if include_coefficients:
            captured['e0'] = cpu_tree(torch, e0)
            captured['new_coeff'] = cpu_tree(torch, output['new_coeff'])
        record['capture'] = captured
        record['calls'] += 1
        record['actual_capture_available'] = True
        return None

    try:
        if adapter is not None:
            handle = adapter.register_forward_hook(observe)
        yield record
        require(record['calls'] == expected_calls,
                'Observed adapter invocation count differs from the explicit scope')
    finally:
        if handle is not None:
            handle.remove()
            handle = None
        record['hooks_removed'] = (adapter is None or
                                  tuple(adapter._forward_hooks) == original_hook_ids)
        require(record['hooks_removed'], 'Passive hook cleanup changed existing hooks')


def execution_snapshot(torch, model, *, cuda_device=None):
    """Owner-run before/after evidence; never initialize CUDA or reseed RNG.

    If replay uses an already initialized CUDA device, its index is mandatory.
    Pass the SAME explicit device for all four reference/observed snapshots.
    Take execution snapshots OUTSIDE both native and passive hook scopes and
    outside the restoring private-RNG fork; temporary hook IDs must be absent.
    This captures full registered state, grad/trainability flags, module modes,
    existing hook IDs and generator bytes, but not optimizer or model outputs.
    """
    require(cuda_device is None or type(cuda_device) is int,
            'CUDA device must be None or an explicitly initialized device index')
    rng = dict(CPU=cpu_tree(torch, torch.random.get_rng_state()))
    if cuda_device is not None:
        require(torch.cuda.is_initialized(), 'Do not initialize CUDA for observation')
        rng['CUDA'] = cpu_tree(torch, torch.cuda.get_rng_state(cuda_device))
    return dict(schema='passive-replay-execution-snapshot-v1',
                state=cpu_tree(torch, dict(model.state_dict())), RNG=rng,
                cuda_device=cuda_device,
                parameter_requires_grad={name: parameter.requires_grad
                                         for name, parameter in model.named_parameters()},
                parameter_grad_is_None={name: parameter.grad is None
                                        for name, parameter in model.named_parameters()},
                module_training={name: module.training
                                 for name, module in model.named_modules()},
                hook_ids={name: dict(pre=tuple(module._forward_pre_hooks),
                                     post=tuple(module._forward_hooks))
                          for name, module in model.named_modules()})


def endpoint_snapshot(torch, classes, masks, official, binding):
    """Capture actual ten readouts and actual official fusion, without rerendering.

    official supplies Tensor masks/scores/labels/query_ids/class_probability/
    mask_quality in its ORIGINAL exported order. query_ids must be the already
    owner-verified actual native mapping, not top-k list positions. All100 masks
    include empty masks. binding declares the same CP/input/private seed/mode
    and metainfo; it is compared, not independently verified by this helper.
    """
    require(isinstance(binding, dict) and
            {'checkpoint', 'input_identity', 'head_seed', 'mode', 'metadata'} <= set(binding),
            'Same-checkpoint/input/seed/mode/metadata binding is mandatory')
    require(type(binding['head_seed']) is int, 'Head private seed must be explicit')
    require(isinstance(classes, (tuple, list)) and isinstance(masks, (tuple, list)) and
            len(classes) == len(masks) == READOUT_COUNT,
            'Identity requires every initial-plus-nine readout')
    require(all(tuple(value.shape) == (1, QUERY_COUNT, 2) and value.dtype == torch.float32
                for value in classes) and
            all(tuple(value.shape) == (1, QUERY_COUNT, 200, 200) and value.dtype == torch.float32
                for value in masks), 'Identity requires the actual native FP32 output shapes')
    required = {'masks', 'scores', 'labels', 'query_ids', 'class_probability', 'mask_quality'}
    require(isinstance(official, dict) and set(official) == required and
            all(isinstance(value, torch.Tensor) for value in official.values()),
            'Pass actual official all100 tensors and its verified query mapping')
    require(official['masks'].ndim == 3 and official['masks'].shape[0] == QUERY_COUNT and
            official['masks'].dtype == torch.bool and
            all(tuple(official[key].shape) == (QUERY_COUNT,)
                for key in required - {'masks'}), 'Official query population must stay complete')
    copied = cpu_tree(torch, official)
    require(copied['query_ids'].dtype == torch.int64 and
            sorted(copied['query_ids'].tolist()) == list(range(QUERY_COUNT)) and
            copied['labels'].dtype == torch.int64 and bool((copied['labels'] == 0).all()),
            'Official native query IDs/ship labels must be complete, without filtering')
    for key in ('scores', 'class_probability', 'mask_quality'):
        require(copied[key].dtype == torch.float32 and bool(torch.isfinite(copied[key]).all()) and
                bool(((copied[key] >= 0) & (copied[key] <= 1)).all()),
                'Actual official factor/score must be finite FP32 in [0,1]')
    return dict(schema='passive-replay-endpoint-snapshot-v1',
                binding=cpu_tree(torch, binding), classes=cpu_tree(torch, classes),
                masks=cpu_tree(torch, masks), official=copied)


def require_byte_identity(torch, reference, observed, path='root'):
    """Recursive exact CPU-byte check, including signed zero and full field sets."""
    if isinstance(reference, torch.Tensor):
        require(isinstance(observed, torch.Tensor) and reference.device.type == 'cpu' and
                observed.device.type == 'cpu' and not reference.requires_grad and
                not observed.requires_grad and reference.dtype == observed.dtype and
                tuple(reference.shape) == tuple(observed.shape), 'Tensor interface differs: ' + path)
        lhs = reference.contiguous().reshape(-1).view(torch.uint8)
        rhs = observed.contiguous().reshape(-1).view(torch.uint8)
        require(torch.equal(lhs, rhs), 'Actual tensor bytes differ: ' + path)
    elif isinstance(reference, dict):
        require(isinstance(observed, dict) and set(reference) == set(observed),
                'Dictionary field population differs: ' + path)
        for key in reference:
            require_byte_identity(torch, reference[key], observed[key], path + '.' + key)
    elif isinstance(reference, (list, tuple)):
        require(type(reference) is type(observed) and len(reference) == len(observed),
                'Sequence population differs: ' + path)
        for index, (left, right) in enumerate(zip(reference, observed)):
            require_byte_identity(torch, left, right, path + '[' + str(index) + ']')
    else:
        require(type(reference) is type(observed) and reference == observed,
                'Declared scalar/binding differs: ' + path)


def qualify_passive_identity(torch, reference_before, reference_after,
                             observed_before, observed_after,
                             reference_endpoint, observed_endpoint, capture):
    """Owner-run numerical preflight only; no GT, AP or feature-purity conclusion."""
    require_byte_identity(torch, reference_before, reference_after, 'reference_state_RNG')
    require_byte_identity(torch, observed_before, observed_after, 'observed_state_RNG')
    require_byte_identity(torch, reference_before, observed_before, 'same_start_state_RNG')
    require_byte_identity(torch, reference_after, observed_after, 'same_end_state_RNG')
    require_byte_identity(torch, reference_endpoint, observed_endpoint, 'all10_official_all100')
    require(capture['calls'] == capture['expected_calls'] and capture['hooks_removed'] is True and
            capture['forward_hook_returned_None'] is True and capture['geometry_recomputed'] is False,
            'Passive capture scope/cleanup is incomplete')
    require(capture['actual_capture_available'] is (capture['expected_calls'] == 1),
            'Unavailable geometry must not be labelled as an actual observation')
    CUDA_RNG_observed = reference_before['cuda_device'] is not None
    if CUDA_RNG_observed:
        require('CUDA' in reference_before['RNG'], 'Requested CUDA RNG evidence is missing')
    return dict(status='PASSIVE_REPLAY_BITWISE_IDENTITY_TECHNICALLY_QUALIFIED',
                actual_adapter_calls=capture['calls'], all10_readouts_byte_exact=True,
                official_all100_masks_scores_factors_query_IDs_byte_exact=True,
                full_state_and_CPU_RNG_byte_exact=True,
                CUDA_RNG_observed=CUDA_RNG_observed,
                full_state_and_CPU_CUDA_RNG_byte_exact=CUDA_RNG_observed, hooks_removed=True,
                no_GT_or_performance_claim=True, full_rho_or_M0_observation_claimed=False)
