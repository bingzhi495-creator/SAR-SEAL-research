"""Inert final-MASK bridge; Torch and both sealed source modules are injected.

Installed seams actually read: mask2former_head.py:337-377,379-459 has one
pixel-decoder call and initial plus nine _forward_head/mask_embed calls, then
returns (class_list, mask_list). Inherited MaskFormerHead.loss:554 and
predict:589 call self(x,batch_data_samples), so a whole-head output hook covers
both. Direct .forward(...) bypasses Torch module hooks and is unsupported.

Unlike the old single-coefficient hook, this captures live F and the tenth
original e0 without replacing any coefficient. Only after native forward
returns does it replace mask_list[-1] with the actual spatial operator result,
passing the LIVE original mask_list[-1] as native_logits. The class LIST and
the first nine mask OBJECTS stay unchanged. All decoder attention was already
computed on the native path; the tenth attention mask has no subsequent layer.
Native loss/matcher/sampler implementations are not replaced. Final matching,
final losses and later native gradients can change; equality across separately
trained arms, native utility or a boundary-learning cause is not asserted.

All three actual modes are global/spatial_halves/spatial_reversed. Sources are
explicit modules plus absolute byte/SHA refs, not a paired-mode alias. The
registered module adds only hidden.weight/final.weight (393216 scalars), so
native610/451 becomes612/453 without a new buffer. Owner alone controls device,
trainability, optimizer and native initialization. apply=False requires the
registered state but never invokes the adapter/geometry/phi; it is a same-state
native-readout ablation, not a pruned610 or compute-matched deployment claim.

batch_metadata supplies ONLY img_shape and batch_input_shape; GT is never read
here. Small detached half counts/fallbacks are recorded when apply=True; no
dense F/e0/logits/alpha/geometry CPU dump is made. Cache and hooks are cleared
in finally on success/failure. Returned native outputs retain their own live
autograd graph. This SOURCE has no scientific imports, model/data/CP loading,
training/preflight/reader/SPEC, main entry or native runtime qualification.
"""

from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path


MODES = ('global', 'spatial_halves', 'spatial_reversed')
REGISTRATION_NAME = 'spatial_mask_readout'
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
READOUT_SOURCE_BYTES = 9802
READOUT_SOURCE_SHA = '7da9561ed30675fd0d692ce2ad883e0c8cccfddfd9f7913761da0fe4fff0a09d'
GEOMETRY_SOURCE_BYTES = 17500
GEOMETRY_SOURCE_SHA = '7ad19e356342afc15a2a41f1358634b23486c19021e63ccd94cfed03f6841d15'


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _check_head(head):
    # Reject different architectures rather than silently adapting call counts.
    _require(type(head).__name__ == 'Mask2FormerHead' and
             type(head).__module__ == 'mmdet.models.dense_heads.mask2former_head',
             'Only the installed native Mask2FormerHead is supported')
    _require(head.num_queries == 100 and head.num_classes == 1 and
             head.num_transformer_feat_level == 3 and head.num_heads == 8 and
             head.num_transformer_decoder_layers == 9 and
             len(head.transformer_decoder.layers) == 9 and
             head.decoder_embed_dims == 256,
             'Expected native 100-query, three-memory-level, nine-layer head')
    _require(type(head.pixel_decoder).__name__ == 'MSDeformAttnPixelDecoder' and
             type(head.pixel_decoder).__module__ ==
             'mmdet.models.layers.msdeformattn_pixel_decoder',
             'Expected the installed MSDeformAttnPixelDecoder')
    _require(tuple(head.query_feat.weight.shape) == COEFFICIENT_SHAPE and
             tuple(head.query_embed.weight.shape) == COEFFICIENT_SHAPE and
             head.cls_embed.in_features == 256 and head.cls_embed.out_features == 2,
             'Unexpected query or class interface')
    _require(len(head.mask_embed) == 5 and
             head.mask_embed[-1].in_features == 256 and
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
    """Read ONLY the declared Python source, not data or checkpoint payload."""
    provenance = _provenance(provenance)
    _require(provenance['bytes'] == expected_bytes and provenance['sha256'] == expected_sha,
             'The exact separately sealed SOURCE ref is required')
    module_file = getattr(module, '__file__', None)
    _require(isinstance(module_file, str) and bool(module_file), 'Injected module has no actual source file')
    path = Path(provenance['path']).resolve(strict=True)
    _require(Path(module_file).resolve(strict=True) == path, 'Injected module and SOURCE ref differ')
    data = path.read_bytes()
    _require(len(data) == expected_bytes and sha256(data).hexdigest() == expected_sha,
             'Actual injected SOURCE bytes do not match their sealed ref')
    return provenance


def register_spatial_mask_readout(torch, readout_source, geometry_source, model,
                                  mode, readout_provenance, geometry_provenance):
    """Register two CPU weights once on the original strict-loaded native model.

    Explicit modules: readout9802 and original geometry17500. No module is
    loaded implicitly and no old hook/factory/global mode is monkeypatched.
    Caller moves model/adapter to FP32 input device and sets training policy.
    SOURCE byte checks establish declared file identity, not in-memory
    numerical/native behavior; that remains separate owner qualification.
    """
    _require(mode in MODES and tuple(getattr(readout_source, 'MODES', ())) == MODES and
             callable(getattr(readout_source, 'build_adapter', None)) and
             getattr(readout_source, 'TRAINABLE_TENSORS', None) == ADDED_PARAMETER_TENSORS and
             getattr(readout_source, 'TRAINABLE_SCALARS', None) == ADDED_PARAMETER_SCALARS,
             'Actual three-mode spatial readout factory required')
    readout_ref = _source_identity(readout_source, readout_provenance,
                                   READOUT_SOURCE_BYTES, READOUT_SOURCE_SHA)
    geometry_ref = _source_identity(geometry_source, geometry_provenance,
                                    GEOMETRY_SOURCE_BYTES, GEOMETRY_SOURCE_SHA)
    head = model.panoptic_head
    _check_head(head)
    _require(not hasattr(head, REGISTRATION_NAME), 'Spatial adapter is already registered')
    _require(not getattr(head, '_paired_readout_hook_active', False) and
             not getattr(head, '_spatial_readout_hook_active', False), 'Register outside active readout contexts')
    _require(len(model.state_dict()) == NATIVE_STATE_ENTRIES and
             len(list(model.parameters())) == NATIVE_PARAMETER_TENSORS,
             'Register only on the original native610-state/451-parameter model')
    native_names = tuple(model.state_dict())
    adapter = readout_source.build_adapter(torch, geometry_source, mode=mode)
    parameters = dict(adapter.named_parameters())
    _require(isinstance(adapter, torch.nn.Module) and adapter.mode == mode and
             set(parameters) == {'hidden.weight', 'final.weight'} and
             tuple(parameters['hidden.weight'].shape) == (512, 512) and
             tuple(parameters['final.weight'].shape) == (256, 512) and
             len(list(adapter.parameters())) == ADDED_PARAMETER_TENSORS and
             sum(p.numel() for p in adapter.parameters()) == ADDED_PARAMETER_SCALARS and
             all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in adapter.parameters()) and
             not list(adapter.buffers()) and len(adapter.state_dict()) == 2,
             'Spatial adapter must have exactly the two FP32 CPU weights and no buffers')
    adapter.readout_source_provenance = readout_ref
    adapter.geometry_source_provenance = geometry_ref
    added_names = {'panoptic_head.' + REGISTRATION_NAME + '.hidden.weight',
                   'panoptic_head.' + REGISTRATION_NAME + '.final.weight'}
    head.add_module(REGISTRATION_NAME, adapter)
    try:
        _require(len(model.state_dict()) == FULL_STATE_ENTRIES and
                 len(list(model.parameters())) == FULL_PARAMETER_TENSORS and
                 set(model.state_dict()) - set(native_names) == added_names and
                 set(native_names) <= set(model.state_dict()),
                 'Unexpected native/new state boundary after spatial registration')
    except Exception:
        delattr(head, REGISTRATION_NAME)
        raise
    return adapter


def metadata_valid_fraction(torch, batch_metadata, F):
    """Analytic metadata-only 4x4 feature-cell valid area, not GT point support."""
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


@contextmanager
def spatial_readout_context(torch, head, batch_metadata, apply=True):
    """Scope one head __call__ (including a loss/predict-internal self call).

    with spatial_readout_context(torch, head, explicit_metadata) as record:
        classes, masks = head(features, samples)

    Training labels may legitimately be present in samples for native loss,
    but this bridge reads only caller-supplied metadata, never those samples.
    Grad mode is untouched. The record remains tensor-free, including counts.
    The returned outputs keep their required autograd graph; the context has
    no cache of live F/e0/result after head return or after an exception.
    """
    _check_head(head)
    _require(type(apply) is bool, 'apply must be an explicit bool')
    adapter = getattr(head, REGISTRATION_NAME, None)
    _require(adapter is not None and adapter.mode in MODES,
             'Register the actual spatial adapter, including for residual-off')
    readout_ref = _provenance(adapter.readout_source_provenance)
    geometry_ref = _provenance(adapter.geometry_source_provenance)
    _require(readout_ref['bytes'] == READOUT_SOURCE_BYTES and readout_ref['sha256'] == READOUT_SOURCE_SHA and
             geometry_ref['bytes'] == GEOMETRY_SOURCE_BYTES and geometry_ref['sha256'] == GEOMETRY_SOURCE_SHA,
             'Registered spatial/geometry SOURCE refs differ')
    _require(not getattr(head, '_spatial_readout_hook_active', False) and
             not getattr(head, '_paired_readout_hook_active', False),
             'Nested/overlapping spatial or old paired contexts are unsupported')
    record = dict(forward_calls=0, completed_forwards=0, pixel_decoder_calls=0,
        coefficient_calls=0, passive_coefficient_captures=0, unchanged_early_readouts=0,
        final_mask_replacements=0, adapter_calls=0, apply=apply, mode=adapter.mode,
        readout_source=readout_ref, geometry_source=geometry_ref,
        metadata_fields_used=('img_shape', 'batch_input_shape'),
        class_list_object_preserved=False, first_nine_mask_objects_preserved=False,
        native_coefficient_outputs_untouched=True, earlier_gate_paths_untouched=True,
        all_queries_retained=True, half_eligible_counts=None, global_fallback=None,
        adapter_geometry_phi_skipped=not apply, cache_cleared_after_forward=False,
        registered_state_not_pruned=True, hooks_removed=False)
    cache = {'active': False, 'F': None, 'e0': None, 'valid_fraction': None}
    handles = []
    head._spatial_readout_hook_active = True

    def start(module, inputs):
        _require(not cache['active'] and record['forward_calls'] == 0,
                 'Use one fresh context per native head forward')
        record['forward_calls'] += 1
        cache.update(active=True, F=None, e0=None, valid_fraction=None)

    def capture_feature(module, inputs, output):
        _require(cache['active'] and record['pixel_decoder_calls'] == 0 and
                 isinstance(output, (tuple, list)) and len(output) == 2,
                 'Expected one native pixel-decoder output tuple')
        F = output[0]
        _require(torch.is_tensor(F) and F.ndim == 4 and tuple(F.shape[1:]) == FEATURE_SHAPE and
                 F.dtype == torch.float32 and F.shape[0] > 0, 'Output[0] must be live FP32 F200')
        cache['F'] = F
        cache['valid_fraction'] = metadata_valid_fraction(torch, batch_metadata, F)
        record['pixel_decoder_calls'] += 1
        record['feature_shape'] = tuple(F.shape)
        return None

    def capture_coefficient(module, inputs, output):
        _require(cache['active'] and cache['F'] is not None,
                 'Mask embedding must follow the native pixel decoder')
        record['coefficient_calls'] += 1
        _require(record['coefficient_calls'] <= READOUTS and torch.is_tensor(output) and
                 output.ndim == 3 and tuple(output.shape[1:]) == COEFFICIENT_SHAPE and
                 output.shape[0] == cache['F'].shape[0] and output.dtype == torch.float32 and
                 output.device == cache['F'].device, 'Unexpected native coefficient call/shape/dtype')
        if record['coefficient_calls'] < READOUTS:
            record['unchanged_early_readouts'] += 1
        else:
            cache['e0'] = output
            record['passive_coefficient_captures'] += 1
            record['coefficient_shape'] = tuple(output.shape)
        return None  # Never replace any mask_embed output.

    def finish(module, inputs, output):
        result = replacement = classes = masks = new_masks = returned = native_last = None
        fields = diagnostics = halves = fallback = E = value = None
        try:
            _require(cache['active'] and record['pixel_decoder_calls'] == 1 and
                     record['coefficient_calls'] == READOUTS and
                     record['unchanged_early_readouts'] == 9 and
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
            record['class_list_object_preserved'] = True
            record['first_nine_mask_objects_preserved'] = True
            record['final_native_mask_shape'] = tuple(masks[-1].shape)
            if apply:
                native_last = masks[-1]
                result = adapter(cache['F'], cache['e0'], cache['valid_fraction'], native_logits=native_last)
                _require(isinstance(result, dict) and
                         set(('mask_logits', 'coefficient_fields', 'diagnostics')) <= set(result),
                         'Spatial adapter returned an invalid result dictionary')
                replacement = result['mask_logits']
                _require(torch.is_tensor(replacement) and tuple(replacement.shape) == tuple(native_last.shape) and
                         replacement.dtype == native_last.dtype and replacement.device == native_last.device and
                         bool(torch.isfinite(replacement).all()), 'Spatial replacement mask interface is invalid')
                fields, diagnostics = result['coefficient_fields'], result['diagnostics']
                _require(torch.is_tensor(fields) and tuple(fields.shape) == (B, 100, 2, 256) and
                         fields.dtype == torch.float32 and fields.device == cache['F'].device and
                         isinstance(diagnostics, dict) and diagnostics['mode'] == adapter.mode and
                         diagnostics['eligibility_frame'] == 'canonical', 'Actual fields/mode/frame differ')
                halves, fallback, E = diagnostics['half_eligible_counts'], diagnostics['global_fallback'], diagnostics['eligible']
                _require(tuple(halves.shape) == (B, 100, 2) and halves.dtype == torch.int64 and
                         tuple(fallback.shape) == (B, 100) and fallback.dtype == torch.bool and
                         tuple(E.shape) == (B, 100, 16) and E.dtype == torch.bool and
                         not halves.requires_grad and not fallback.requires_grad and not E.requires_grad and
                         torch.equal(halves[..., 0], E[..., :8].sum(dim=-1)) and
                         torch.equal(halves[..., 1], E[..., 8:].sum(dim=-1)) and
                         torch.equal(fallback, (halves == 0).any(dim=-1)), 'Actual half counts/fallback/E differ')
                record['half_eligible_counts'] = halves.detach().cpu().tolist()
                record['global_fallback'] = fallback.detach().cpu().tolist()
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
            return returned  # None for apply=False leaves the native output intact.
        finally:
            cache.clear()
            cache.update(active=False, F=None, e0=None, valid_fraction=None)
            record['cache_cleared_after_forward'] = True
            result = replacement = classes = masks = new_masks = returned = native_last = None
            fields = diagnostics = halves = fallback = E = value = None
            module = inputs = output = None

    try:
        handles.append(head.register_forward_pre_hook(start))
        handles.append(head.pixel_decoder.register_forward_hook(capture_feature))
        handles.append(head.mask_embed.register_forward_hook(capture_coefficient))
        handles.append(head.register_forward_hook(finish))
        yield record
        _require(record['forward_calls'] == 1 and record['completed_forwards'] == 1,
                 'The context must contain one successfully completed native head forward')
    finally:
        try:
            for handle in handles:
                handle.remove()
            record['hooks_removed'] = True
        finally:
            handles.clear()
            cache.clear()
            head._spatial_readout_hook_active = False
