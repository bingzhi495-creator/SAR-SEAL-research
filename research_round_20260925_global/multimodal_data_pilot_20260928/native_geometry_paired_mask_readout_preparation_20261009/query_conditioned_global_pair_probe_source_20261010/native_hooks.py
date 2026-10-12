"""Inert whole-head bridge for the four query pair-generator input modes.

Torch, the exact operator SOURCE, original geometry SOURCE and native model
are caller-injected. Registration verifies only those two Python SOURCE files.
There is no scientific import, data/checkpoint loading, GT access, training,
global patch or main entry. Native freezing and optimizer policy are separate
caller contracts; this bridge neither changes nor certifies those policies.

Like the original spatial bridge, four scoped module hooks capture actual F
and the tenth original e0 without changing any coefficient output. The whole
head finish hook passes the original LIVE final native mask as native_logits.
It replaces only mask_list[-1]; the class list and first nine mask objects
remain the originals. Direct head.forward(...) bypasses hooks and is rejected
by the completed-forward check. Use the native module call, including the
native self-call made by loss/predict. Native matching and loss are untouched.

Differences from the spatial bridge: modes are raw/raw_rms/query_rms/
permuted_query_rms, the registration prefix is panoptic_head.query_pair_readout.,
hidden/permutation seeds are explicit, both coefficient fields are the same
global eligible mean, and the dense projection uses the same unscaled F.
There is no alpha, spatial-half count or fallback observation. New tensor-free
readout_observation records E/support/counts and input RMS/zero-RMS counts.
An RMS can underflow at extreme subnormal magnitudes, so those counts are
not asserted exact all-channel-zero vector counts. These observations and
requires_grad properties are not a gradient proof.

apply=False retains the registered 612-state/453-parameter model but skips the
adapter, geometry and phi. It is a registered-state residual-off control.
Cold value equality is checked under no_grad only when the current final
weight is entirely zero. Trained nonzero final weights are not subjected to
that assertion. Context hooks and cached live aliases clear in finally.
"""

from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path


MODES = ('raw', 'raw_rms', 'query_rms', 'permuted_query_rms')
REGISTRATION_NAME = 'query_pair_readout'
ADAPTER_PREFIX = 'panoptic_head.query_pair_readout.'
NATIVE_STATE_ENTRIES = 610
NATIVE_PARAMETER_TENSORS = 451
ADDED_PARAMETER_TENSORS = 2
ADDED_PARAMETER_SCALARS = 393216
FULL_STATE_ENTRIES = 612
FULL_PARAMETER_TENSORS = 453
READOUTS = 10
FEATURE_SHAPE = (256, 200, 200)
COEFFICIENT_SHAPE = (100, 256)
CANVAS_SHAPE = (800, 800)
READOUT_SOURCE_BYTES = 10481
READOUT_SOURCE_SHA = '5199354074939e335e0d0b5c19940ad56d637b750843cb38cc9c7213e81b4b2a'
GEOMETRY_SOURCE_BYTES = 17500
GEOMETRY_SOURCE_SHA = '7ad19e356342afc15a2a41f1358634b23486c19021e63ccd94cfed03f6841d15'
INITIALIZATION_SEED = 2026100911
PERMUTATION_SEED = 2026101013
GEOMETRY_MASS_EPS = 1e-12


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _check_head(head):
    _require(type(head).__name__ == 'Mask2FormerHead' and
             type(head).__module__ == 'mmdet.models.dense_heads.mask2former_head',
             'Only the installed native Mask2FormerHead is supported')
    _require(head.num_queries == 100 and head.num_classes == 1 and
             head.num_transformer_feat_level == 3 and head.num_heads == 8 and
             head.num_transformer_decoder_layers == 9 and
             len(head.transformer_decoder.layers) == 9 and head.decoder_embed_dims == 256,
             'Expected native 100-query, three-memory-level, nine-layer head')
    _require(type(head.pixel_decoder).__name__ == 'MSDeformAttnPixelDecoder' and
             type(head.pixel_decoder).__module__ == 'mmdet.models.layers.msdeformattn_pixel_decoder',
             'Expected the installed MSDeformAttnPixelDecoder')
    _require(tuple(head.query_feat.weight.shape) == COEFFICIENT_SHAPE and
             tuple(head.query_embed.weight.shape) == COEFFICIENT_SHAPE and
             head.cls_embed.in_features == 256 and head.cls_embed.out_features == 2,
             'Unexpected query or class interface')
    _require(len(head.mask_embed) == 5 and head.mask_embed[-1].in_features == 256 and
             head.mask_embed[-1].out_features == 256,
             'Expected the native three-Linear mask embedding')


def _provenance(value):
    _require(isinstance(value, dict) and set(('path', 'bytes', 'sha256')) <= set(value),
             'Explicit SOURCE provenance is required')
    _require(isinstance(value['path'], str) and bool(value['path']) and
             Path(value['path']).is_absolute() and Path(value['path']).suffix == '.py' and
             type(value['bytes']) is int and value['bytes'] > 0 and
             isinstance(value['sha256'], str) and len(value['sha256']) == 64 and
             set(value['sha256']) <= set('0123456789abcdef'),
             'SOURCE provenance must be an absolute .py path, byte count and SHA256')
    return {key: value[key] for key in ('path', 'bytes', 'sha256')}


def _source_identity(module, provenance, expected_bytes, expected_sha):
    """Registration-time read of the declared Python SOURCE only."""
    provenance = _provenance(provenance)
    _require(provenance['bytes'] == expected_bytes and provenance['sha256'] == expected_sha,
             'The exact query readout/original geometry SOURCE ref is required')
    module_file = getattr(module, '__file__', None)
    _require(isinstance(module_file, str) and bool(module_file),
             'Injected module has no actual source file')
    path = Path(provenance['path']).resolve(strict=True)
    _require(Path(module_file).resolve(strict=True) == path,
             'Injected module and SOURCE ref differ')
    data = path.read_bytes()
    _require(len(data) == expected_bytes and sha256(data).hexdigest() == expected_sha,
             'Actual injected SOURCE bytes do not match their exact ref')
    return provenance


def _contexts_inactive(head):
    return not any(getattr(head, name, False) for name in
                   ('_paired_readout_hook_active', '_spatial_readout_hook_active',
                    '_query_pair_readout_hook_active'))


def register_query_pair_readout(torch, readout_source, geometry_source, model, mode,
                                readout_provenance, geometry_provenance,
                                hidden_seed=INITIALIZATION_SEED,
                                permutation_seed=PERMUTATION_SEED):
    """Register two CPU weights once on the original strict-loaded native model.

    The caller supplies the exact SOURCE modules and absolute byte/SHA refs,
    then controls device, native freezing, adapter trainability and optimizer.
    File identity does not establish in-memory or native numerical behavior.
    """
    _require(mode in MODES and tuple(getattr(readout_source, 'MODES', ())) == MODES and
             callable(getattr(readout_source, 'build_adapter', None)) and
             getattr(readout_source, 'TRAINABLE_TENSORS', None) == ADDED_PARAMETER_TENSORS and
             getattr(readout_source, 'TRAINABLE_SCALARS', None) == ADDED_PARAMETER_SCALARS and
             getattr(readout_source, 'INITIALIZATION_SEED', None) == INITIALIZATION_SEED and
             getattr(readout_source, 'PERMUTATION_SEED', None) == PERMUTATION_SEED,
             'Actual four-mode query pair-generator factory required')
    _require(all(type(seed) is int and 0 <= seed < 2**63
                 for seed in (hidden_seed, permutation_seed)),
             'Hidden and permutation seeds must be explicit nonnegative int64 values')
    readout_ref = _source_identity(readout_source, readout_provenance,
                                   READOUT_SOURCE_BYTES, READOUT_SOURCE_SHA)
    geometry_ref = _source_identity(geometry_source, geometry_provenance,
                                    GEOMETRY_SOURCE_BYTES, GEOMETRY_SOURCE_SHA)
    head = model.panoptic_head
    _check_head(head)
    _require(not hasattr(head, REGISTRATION_NAME), 'Query pair adapter is already registered')
    _require(_contexts_inactive(head), 'Register outside active readout contexts')
    _require(len(model.state_dict()) == NATIVE_STATE_ENTRIES and
             len(list(model.parameters())) == NATIVE_PARAMETER_TENSORS,
             'Register only on the original native610-state/451-parameter model')
    native_names = tuple(model.state_dict())
    adapter = readout_source.build_adapter(torch, geometry_source, mode=mode,
                                           hidden_seed=hidden_seed,
                                           permutation_seed=permutation_seed)
    parameters = dict(adapter.named_parameters())
    _require(isinstance(adapter, torch.nn.Module) and adapter.mode == mode and
             adapter.hidden_seed == hidden_seed and adapter.permutation_seed == permutation_seed and
             set(parameters) == {'hidden.weight', 'final.weight'} and
             tuple(parameters['hidden.weight'].shape) == (512, 512) and
             tuple(parameters['final.weight'].shape) == (256, 512) and
             len(list(adapter.parameters())) == ADDED_PARAMETER_TENSORS and
             sum(p.numel() for p in adapter.parameters()) == ADDED_PARAMETER_SCALARS and
             all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in adapter.parameters()) and
             not list(adapter.buffers()) and len(adapter.state_dict()) == 2,
             'Query adapter must have exactly two FP32 CPU weights and no buffers')
    adapter.readout_source_provenance = readout_ref
    adapter.geometry_source_provenance = geometry_ref
    added_names = {ADAPTER_PREFIX + 'hidden.weight', ADAPTER_PREFIX + 'final.weight'}
    head.add_module(REGISTRATION_NAME, adapter)
    try:
        _require(len(model.state_dict()) == FULL_STATE_ENTRIES and
                 len(list(model.parameters())) == FULL_PARAMETER_TENSORS and
                 set(model.state_dict()) - set(native_names) == added_names and
                 set(native_names) <= set(model.state_dict()),
                 'Unexpected native/new state boundary after query adapter registration')
    except Exception:
        delattr(head, REGISTRATION_NAME)
        raise
    return adapter


def metadata_valid_fraction(torch, batch_metadata, F):
    """Original metadata-only 4x4 feature-cell valid area; no GT access."""
    try:
        _require(F.ndim == 4 and tuple(F.shape[1:]) == FEATURE_SHAPE and
                 F.dtype == torch.float32, 'Expected actual FP32 F200')
        _require(isinstance(batch_metadata, (list, tuple)) and
                 len(batch_metadata) == F.shape[0] and F.shape[0] > 0,
                 'Metadata population must equal the feature batch')
        heights, widths = [], []
        for meta in batch_metadata:
            _require(isinstance(meta, dict), 'Pass explicit metainfo dictionaries')
            actual, canvas = meta.get('img_shape'), meta.get('batch_input_shape')
            _require(isinstance(actual, (list, tuple)) and len(actual) >= 2 and
                     isinstance(canvas, (list, tuple)) and tuple(canvas) == CANVAS_SHAPE,
                     'Expected keep-ratio img_shape and fixed800 batch_input_shape')
            h, w = actual[:2]
            _require(type(h) is int and type(w) is int and 0 < h <= 800 and 0 < w <= 800,
                     'Actual resized image extent must be integer and within800')
            heights.append(h)
            widths.append(w)
        rows = torch.arange(200, device=F.device, dtype=torch.float32)
        columns = torch.arange(200, device=F.device, dtype=torch.float32)
        h = torch.tensor(heights, device=F.device, dtype=torch.float32)[:, None]
        w = torch.tensor(widths, device=F.device, dtype=torch.float32)[:, None]
        yf = (h - 4.0 * rows[None, :]).clamp(0.0, 4.0) / 4.0
        xf = (w - 4.0 * columns[None, :]).clamp(0.0, 4.0) / 4.0
        return (yf[:, None, :, None] * xf[:, None, None, :]).detach()
    finally:
        F = None


def _readout_observation(torch, adapter, fields, diagnostics, B, device):
    """Validate and copy only compact detached input/support observations."""
    try:
        _require(torch.is_tensor(fields) and tuple(fields.shape) == (B, 100, 2, 256) and
                 fields.dtype == torch.float32 and fields.device == device and
                 bool(torch.isfinite(fields).all()) and
                 torch.equal(fields[:, :, 0], fields[:, :, 1]),
                 'The two global coefficient fields must be identical finite FP32 values')
        conditioning = ('real_query' if adapter.mode == 'query_rms' else
                        'permuted_query' if adapter.mode == 'permuted_query_rms' else 'raw_features')
        _require(isinstance(diagnostics, dict) and diagnostics.get('mode') == adapter.mode and
                 diagnostics.get('global_pool') is True and
                 diagnostics.get('dense_readout') == 'same_unnormalized_F' and
                 diagnostics.get('auxiliary_query_detached') is True and
                 diagnostics.get('input_RMS_normalized') is (adapter.mode != 'raw') and
                 diagnostics.get('input_conditioning') == conditioning and
                 diagnostics.get('hidden_seed') == adapter.hidden_seed and
                 diagnostics.get('permutation_seed') == adapter.permutation_seed and
                 diagnostics.get('parameters') == ADDED_PARAMETER_SCALARS and
                 diagnostics.get('no_query_removed') is True and
                 diagnostics.get('empty_E') == 'connected_zero' and
                 diagnostics.get('permutation_kind') == 'seeded_256_cycle_not_all_derangements_uniform',
                 'Actual query/global/normalization configuration differs')
        shapes = {'eligible': ((B, 100, 16), torch.bool),
                  'eligible_counts': ((B, 100), torch.int64),
                  'canonical_eligible_counts': ((B, 100), torch.int64),
                  'rho_mass': ((B, 100), torch.float32)}
        for key, (shape, dtype) in shapes.items():
            value = diagnostics.get(key)
            _require(torch.is_tensor(value) and tuple(value.shape) == shape and
                     value.dtype == dtype and value.device == device and not value.requires_grad and
                     bool(torch.isfinite(value).all()), 'Invalid detached support observation: ' + key)
        _require(torch.equal(diagnostics['eligible_counts'], diagnostics['eligible'].sum(-1)) and
                 torch.equal(diagnostics['canonical_eligible_counts'], diagnostics['eligible_counts']),
                 'Canonical/global eligible counts differ from actual E')
        rms, zero_counts = {}, {}
        for side in ('inside', 'outside'):
            for stage in ('before', 'after'):
                key = 'input_' + side + '_RMS_' + stage
                value = diagnostics.get(key)
                _require(torch.is_tensor(value) and tuple(value.shape) == (B, 100, 16) and
                         value.dtype == torch.float32 and value.device == device and not value.requires_grad and
                         bool(torch.isfinite(value).all()) and bool((value >= 0).all()),
                         'Invalid detached input RMS observation: ' + key)
                rms[side + '_' + stage] = value.detach().cpu().tolist()
                zero_counts[side + '_' + stage] = (value == 0).sum(-1).detach().cpu().tolist()
        return dict(eligible=diagnostics['eligible'].detach().cpu().tolist(),
            eligible_counts=diagnostics['eligible_counts'].detach().cpu().tolist(),
            canonical_eligible_counts=diagnostics['canonical_eligible_counts'].detach().cpu().tolist(),
            rho_mass=diagnostics['rho_mass'].detach().cpu().tolist(),
            support_nonempty=(diagnostics['rho_mass'] > GEOMETRY_MASS_EPS).detach().cpu().tolist(),
            input_rms=rms, zero_input_rms_counts=zero_counts,
            zero_count_basis='reported_FP32_input_RMS_equals_zero',
            auxiliary_query_detached=True, global_pool=True, dense_readout='same_unnormalized_F',
            input_RMS_normalized=adapter.mode != 'raw', input_conditioning=conditioning,
            hidden_seed=adapter.hidden_seed, permutation_seed=adapter.permutation_seed,
            permutation_kind=diagnostics['permutation_kind'],
            permutation_indices=tuple(adapter.permutation_indices),
            trainable_tensors=ADDED_PARAMETER_TENSORS, parameter_scalars=ADDED_PARAMETER_SCALARS,
            native_freeze_policy_checked=False, forward_gradient_proof=False)
    finally:
        fields = diagnostics = value = None


@contextmanager
def query_pair_readout_context(torch, head, batch_metadata, apply=True):
    """Scope exactly one head __call__; record contains no live tensors.

    with query_pair_readout_context(torch, head, explicit_metadata, apply=True) as record:
        classes, masks = head(features, samples)

    Only caller-supplied img_shape/batch_input_shape metadata is inspected.
    Native samples may contain labels for native loss; this bridge never reads
    those samples. Grad mode is preserved, not forced on or off.
    """
    _check_head(head)
    _require(type(apply) is bool, 'apply must be an explicit bool')
    adapter = getattr(head, REGISTRATION_NAME, None)
    _require(adapter is not None and adapter.mode in MODES,
             'Register the actual query pair adapter, including for residual-off')
    readout_ref = _provenance(adapter.readout_source_provenance)
    geometry_ref = _provenance(adapter.geometry_source_provenance)
    _require(readout_ref['bytes'] == READOUT_SOURCE_BYTES and readout_ref['sha256'] == READOUT_SOURCE_SHA and
             geometry_ref['bytes'] == GEOMETRY_SOURCE_BYTES and geometry_ref['sha256'] == GEOMETRY_SOURCE_SHA,
             'Registered query operator/original geometry SOURCE refs differ')
    _require(_contexts_inactive(head), 'Nested/overlapping readout contexts are unsupported')
    record = dict(forward_calls=0, completed_forwards=0, pixel_decoder_calls=0,
        coefficient_calls=0, passive_coefficient_captures=0, unchanged_early_readouts=0,
        final_mask_replacements=0, adapter_calls=0, apply=apply, mode=adapter.mode,
        adapter_prefix=ADAPTER_PREFIX, readout_source=readout_ref, geometry_source=geometry_ref,
        hidden_seed=adapter.hidden_seed, permutation_seed=adapter.permutation_seed,
        metadata_fields_used=('img_shape', 'batch_input_shape'),
        class_list_object_preserved=False, first_nine_mask_objects_preserved=False,
        native_coefficient_outputs_untouched=True, earlier_gate_paths_untouched=True,
        all_queries_retained=True, readout_observation=None, native_graph_properties=None,
        cold_zero_final_weight=None, cold_native_value_equal=None,
        adapter_geometry_phi_skipped=not apply, cache_cleared_after_forward=False,
        registered_state_not_pruned=True, hooks_removed=False,
        hook_handles_installed=0, hook_handles_removed=0)
    cache = {'active': False, 'F': None, 'e0': None, 'valid_fraction': None}
    handles = []
    head._query_pair_readout_hook_active = True

    def start(module, inputs):
        try:
            _require(not cache['active'] and record['forward_calls'] == 0,
                     'Use one fresh context per native head forward')
            record['forward_calls'] += 1
            record['grad_mode_enabled_at_start'] = torch.is_grad_enabled()
            cache.update(active=True, F=None, e0=None, valid_fraction=None)
        finally:
            module = inputs = None

    def capture_feature(module, inputs, output):
        F = None
        try:
            _require(cache['active'] and record['pixel_decoder_calls'] == 0 and
                     isinstance(output, (tuple, list)) and len(output) == 2,
                     'Expected one native pixel-decoder output tuple')
            F = output[0]
            _require(torch.is_tensor(F) and F.ndim == 4 and tuple(F.shape[1:]) == FEATURE_SHAPE and
                     F.dtype == torch.float32 and F.shape[0] > 0,
                     'Output[0] must be the actual FP32 F200')
            cache['F'] = F
            cache['valid_fraction'] = metadata_valid_fraction(torch, batch_metadata, F)
            record['pixel_decoder_calls'] += 1
            record['feature_shape'] = tuple(F.shape)
        finally:
            F = module = inputs = output = None

    def capture_coefficient(module, inputs, output):
        try:
            _require(cache['active'] and cache['F'] is not None,
                     'Mask embedding must follow the native pixel decoder')
            record['coefficient_calls'] += 1
            _require(record['coefficient_calls'] <= READOUTS and torch.is_tensor(output) and
                     output.ndim == 3 and tuple(output.shape[1:]) == COEFFICIENT_SHAPE and
                     output.shape[0] == cache['F'].shape[0] and output.dtype == torch.float32 and
                     output.device == cache['F'].device,
                     'Unexpected native coefficient call/shape/dtype')
            if record['coefficient_calls'] < READOUTS:
                record['unchanged_early_readouts'] += 1
            else:
                cache['e0'] = output
                record['passive_coefficient_captures'] += 1
                record['coefficient_shape'] = tuple(output.shape)
            return None
        finally:
            module = inputs = output = None

    def finish(module, inputs, output):
        result = replacement = classes = masks = new_masks = returned = native_last = None
        fields = diagnostics = residual = value = None
        try:
            _require(cache['active'] and record['pixel_decoder_calls'] == 1 and
                     record['coefficient_calls'] == READOUTS and record['unchanged_early_readouts'] == 9 and
                     record['passive_coefficient_captures'] == 1 and cache['e0'] is not None,
                     'Native initial-plus-nine passive readouts were not completed')
            _require(isinstance(output, (tuple, list)) and len(output) == 2 and
                     all(isinstance(values, (tuple, list)) and len(values) == READOUTS for values in output),
                     'Expected ten native class outputs and ten native mask outputs')
            classes, masks = output
            B = cache['F'].shape[0]
            for value in classes:
                _require(torch.is_tensor(value) and tuple(value.shape) == (B, 100, 2) and
                         value.dtype == torch.float32 and value.device == cache['F'].device,
                         'All ten native class interfaces must be FP32 Bx100x2')
            for value in masks:
                _require(torch.is_tensor(value) and tuple(value.shape) == (B, 100, 200, 200) and
                         value.dtype == torch.float32 and value.device == cache['F'].device,
                         'All ten native mask interfaces must be FP32 Bx100x200x200')
            native_last = masks[-1]
            record['class_list_object_preserved'] = True
            record['first_nine_mask_objects_preserved'] = True
            record['final_native_mask_shape'] = tuple(native_last.shape)
            record['native_graph_properties'] = dict(feature_requires_grad=cache['F'].requires_grad,
                coefficient_requires_grad=cache['e0'].requires_grad,
                native_final_mask_requires_grad=native_last.requires_grad,
                captured_tensors_detached_by_bridge=False, forward_gradient_proof=False)
            if apply:
                result = adapter(cache['F'], cache['e0'], cache['valid_fraction'], native_logits=native_last)
                _require(isinstance(result, dict) and
                         set(('mask_logits', 'logit_residual', 'coefficient_fields', 'diagnostics')) <= set(result),
                         'Query adapter returned an invalid result dictionary')
                replacement, residual = result['mask_logits'], result['logit_residual']
                for value in (replacement, residual):
                    _require(torch.is_tensor(value) and tuple(value.shape) == tuple(native_last.shape) and
                             value.dtype == native_last.dtype and value.device == native_last.device and
                             bool(torch.isfinite(value).all()), 'Query replacement/residual interface is invalid')
                fields, diagnostics = result['coefficient_fields'], result['diagnostics']
                record['readout_observation'] = _readout_observation(torch, adapter, fields, diagnostics,
                                                                    B, cache['F'].device)
                with torch.no_grad():
                    record['cold_zero_final_weight'] = torch.count_nonzero(adapter.final.weight).item() == 0
                    if record['cold_zero_final_weight']:
                        record['cold_native_value_equal'] = torch.equal(replacement, native_last)
                        _require(record['cold_native_value_equal'] and torch.count_nonzero(residual).item() == 0,
                                 'Zero final weight did not preserve native mask values')
                new_masks = list(masks)
                new_masks[-1] = replacement
                if isinstance(masks, tuple):
                    new_masks = tuple(new_masks)
                returned = (classes, new_masks) if isinstance(output, tuple) else [classes, new_masks]
                _require(returned[0] is classes and all(returned[1][i] is masks[i] for i in range(9)),
                         'Class list or an early mask object was replaced')
                record['adapter_calls'] += 1
                record['final_mask_replacements'] += 1
            _require(record['adapter_calls'] == int(apply) and record['final_mask_replacements'] == int(apply),
                     'Adapter call/replacement count differs')
            record['completed_forwards'] += 1
            return returned
        finally:
            cache.clear()
            cache.update(active=False, F=None, e0=None, valid_fraction=None)
            record['cache_cleared_after_forward'] = True
            result = replacement = classes = masks = new_masks = returned = native_last = None
            fields = diagnostics = residual = value = None
            module = inputs = output = None

    try:
        handles.append(head.register_forward_pre_hook(start))
        handles.append(head.pixel_decoder.register_forward_hook(capture_feature))
        handles.append(head.mask_embed.register_forward_hook(capture_coefficient))
        handles.append(head.register_forward_hook(finish))
        record['hook_handles_installed'] = len(handles)
        yield record
        _require(record['forward_calls'] == 1 and record['completed_forwards'] == 1,
                 'The context must contain one successfully completed native head module call')
    finally:
        removal_error = handle = None
        try:
            for handle in handles:
                try:
                    handle.remove()
                    record['hook_handles_removed'] += 1
                except Exception as error:
                    if removal_error is None:
                        removal_error = error
            record['hooks_removed'] = record['hook_handles_removed'] == len(handles)
            if removal_error is not None:
                raise removal_error
        finally:
            handles.clear()
            cache.clear()
            head._query_pair_readout_hook_active = False
            handle = removal_error = None
