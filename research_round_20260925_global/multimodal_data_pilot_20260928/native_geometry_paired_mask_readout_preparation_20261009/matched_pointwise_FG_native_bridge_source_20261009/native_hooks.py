"""Inert native Mask2Former final-coefficient hook, with caller-injected Torch.

Known source seam: mmdet/models/dense_heads/mask2former_head.py:358-364,
402,430-454; msdeformattn_pixel_decoder.py:245-246. The fixed native setup
has initial supervision plus nine decoder readouts. Only readout ten can
return a replacement mask coefficient; class logits and earlier gates have
already been computed on the original path. This is a per-forward statement,
not equality of class scores or assignments across separately trained arms.

No scientific imports, data/GT access, checkpoint I/O, initialization override,
optimizer policy, numerical qualification, or main entry are provided here.
The owner supplies the sealed paired_context module and its provenance, moves
the registered adapter to the input device, and sets training/freeze policy.
Cold identity and live-feature gradients require owner-run qualification.

Fresh fg_pointwise support requires the actual control.py module as paired_source
and an explicit geometry_source sealed reference. Only that branch verifies
SOURCE bytes and loads the original geometry factory; no implicit path/global
mode alias or old-source edits are used. The proof names both actual sources.
Legacy modes require geometry_source=None and retain the original factory call.
This SOURCE bridge has no runtime/native qualification or training-efficacy claim.
"""

from contextlib import contextmanager


REGISTRATION_NAME = 'paired_mask_readout'
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
POINTWISE_MODE = 'fg_pointwise'
POINTWISE_SOURCE_BYTES = 6105
POINTWISE_SOURCE_SHA = 'd531fe766b64b30591d6fa80a2e2f38d60393a029a8779ad7842fc0b96149de3'
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
             'Explicit sealed paired-source provenance is required')
    _require(isinstance(value['path'], str) and bool(value['path']) and
             type(value['bytes']) is int and value['bytes'] > 0 and
             isinstance(value['sha256'], str) and len(value['sha256']) == 64 and
             set(value['sha256']) <= set('0123456789abcdef'),
             'Provenance must be a path, positive byte count, and SHA256 string')
    return {key: value[key] for key in ('path', 'bytes', 'sha256')}


def _bound_code_source(provenance, source_module=None):
    """Runtime SOURCE-only provenance check; never a data/checkpoint reader."""
    from pathlib import Path
    import hashlib
    source_path = Path(provenance['path'])
    _require(source_path.is_absolute() and source_path.suffix.lower() == '.py',
             'New-mode sources must be explicit absolute Python source references')
    source_path = source_path.resolve(strict=True)
    if source_module is not None:
        module_path = getattr(source_module, '__file__', None)
        _require(isinstance(module_path, str) and
                 Path(module_path).resolve(strict=True) == source_path,
                 'paired_source must be the actual referenced control.py module')
    source_bytes = source_path.read_bytes()
    _require(len(source_bytes) == provenance['bytes'] and
             hashlib.sha256(source_bytes).hexdigest() == provenance['sha256'],
             'Referenced new-mode SOURCE bytes differ')
    return source_path


def _load_geometry_source(provenance):
    """Load only caller-referenced original geometry SOURCE, without Torch imports."""
    import importlib.util
    source_path = _bound_code_source(provenance)
    name = '_native_fg_pointwise_original_geometry_' + provenance['sha256']
    specification = importlib.util.spec_from_file_location(name, str(source_path))
    _require(specification is not None and specification.loader is not None,
             'Referenced geometry SOURCE could not be loaded')
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def register_paired_mask_readout(torch, paired_source, model, mode,
                                 source_provenance, geometry_source=None):
    """Register exactly two new weights; return the CPU-initialized adapter.

    Call once after strict-loading the original native model and before its
    optimizer is built. Weights, device placement, trainability, and any future
    owner-defined reinitialization remain explicit owner responsibilities.
    This helper does not reset the sealed factory seed or freeze native weights.
    Legacy calls leave geometry_source=None. For fg_pointwise, paired_source
    is the real control module and source_provenance references that source;
    geometry_source is the complete sealed original geometry reference, not
    a module/path alias. Only the new branch loads it for the control factory.
    """
    head = model.panoptic_head
    _check_head(head)
    _require(not hasattr(head, REGISTRATION_NAME), 'Adapter is already registered')
    _require(len(model.state_dict()) == NATIVE_STATE_ENTRIES and
             len(list(model.parameters())) == NATIVE_PARAMETER_TENSORS,
             'Register only on the original native 610-state/451-parameter model')
    _require(mode in paired_source.MODES and
             paired_source.TRAINABLE_TENSORS == ADDED_PARAMETER_TENSORS and
             paired_source.TRAINABLE_SCALARS == ADDED_PARAMETER_SCALARS,
             'Unexpected sealed paired-adapter interface')
    provenance = _provenance(source_provenance)
    native_names = tuple(model.state_dict().keys())
    if mode == POINTWISE_MODE:
        _require(geometry_source is not None and
                 getattr(paired_source, 'MODE', None) == POINTWISE_MODE and
                 paired_source.MODES == (POINTWISE_MODE,) and
                 provenance['bytes'] == POINTWISE_SOURCE_BYTES and
                 provenance['sha256'] == POINTWISE_SOURCE_SHA,
                 'fg_pointwise requires the actual sealed pointwise-control module/ref')
        _bound_code_source(provenance, paired_source)
        geometry_provenance = _provenance(geometry_source)
        _require(geometry_provenance['bytes'] == GEOMETRY_SOURCE_BYTES and
                 geometry_provenance['sha256'] == GEOMETRY_SOURCE_SHA,
                 'fg_pointwise requires the sealed original geometry SOURCE ref')
        geometry_module = _load_geometry_source(geometry_provenance)
        adapter = paired_source.build_adapter(torch, geometry_module, mode=mode)
    else:
        _require(geometry_source is None,
                 'Legacy modes must not receive an external geometry source')
        adapter = paired_source.build_adapter(torch, mode=mode)
    _require(isinstance(adapter, torch.nn.Module) and adapter.mode == mode and
             set(dict(adapter.named_parameters())) == {'hidden.weight', 'final.weight'} and
             len(list(adapter.parameters())) == ADDED_PARAMETER_TENSORS and
             sum(p.numel() for p in adapter.parameters()) == ADDED_PARAMETER_SCALARS and
             not list(adapter.buffers()) and len(adapter.state_dict()) == 2,
             'Adapter must have exactly two weights and no registered buffers')
    adapter.paired_source_provenance = provenance
    if mode == POINTWISE_MODE:
        adapter.geometry_source_provenance = geometry_provenance
    head.add_module(REGISTRATION_NAME, adapter)
    added_names = {'panoptic_head.' + REGISTRATION_NAME + '.hidden.weight',
                   'panoptic_head.' + REGISTRATION_NAME + '.final.weight'}
    _require(len(model.state_dict()) == FULL_STATE_ENTRIES and
             len(list(model.parameters())) == FULL_PARAMETER_TENSORS and
             set(model.state_dict()) - set(native_names) == added_names and
             set(native_names) <= set(model.state_dict()),
             'Unexpected native/new state boundary after registration')
    return adapter


def metadata_valid_fraction(torch, batch_metadata, F):
    """Analytic 4x4-cell valid area fractions; metadata only, never GT.

    batch_metadata is a list of ordinary metainfo dictionaries. Each must
    contain img_shape and batch_input_shape. No other sample fields are read.
    The canvas is the native fixed padded 800x800 and F is actual 200x200.
    """
    _require(F.ndim == 4 and tuple(F.shape[1:]) == FEATURE_SHAPE and
             F.dtype == torch.float32, 'Expected actual FP32 F200')
    _require(isinstance(batch_metadata, (list, tuple)) and
             len(batch_metadata) == F.shape[0] and F.shape[0] > 0,
             'Metadata population must equal the feature batch')
    heights, widths = [], []
    for meta in batch_metadata:
        _require(isinstance(meta, dict), 'Pass explicit metainfo dictionaries')
        actual = meta.get('img_shape')
        canvas = meta.get('batch_input_shape')
        _require(isinstance(actual, (list, tuple)) and len(actual) >= 2 and
                 isinstance(canvas, (list, tuple)) and tuple(canvas) == CANVAS_SHAPE,
                 'Expected keep-ratio img_shape and fixed-800 batch_input_shape')
        h, w = actual[:2]
        _require(type(h) is int and type(w) is int and
                 0 < h <= 800 and 0 < w <= 800,
                 'Actual resized image extent must be integer and within 800')
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
def paired_readout_context(torch, head, batch_metadata, apply=True):
    """Scope exactly one native head forward; yield a tensor-free record.

    with paired_readout_context(torch, head, [sample.metainfo], apply=True) as r:
        classes, masks = head(features, stripped_samples)

    apply=False is passive baseline observation and need not register an
    adapter. apply=True uses head.paired_mask_readout, retaining F/e0 gradients
    as supplied by the caller. It never enables/disables grad mode itself.
    Returned native outputs own their normal autograd graph; this context
    retains no such tensors after the successful head forward returns.
    On failure, finally also clears the cache and removes every installed hook.
    """
    _check_head(head)
    _require(type(apply) is bool, 'apply must be an explicit bool')
    adapter = getattr(head, REGISTRATION_NAME, None)
    _require(not apply or adapter is not None, 'Register the adapter before applying it')
    _require(not getattr(head, '_paired_readout_hook_active', False),
             'Nested or overlapping paired-readout contexts are unsupported')
    record = {
        'forward_calls': 0, 'completed_forwards': 0, 'pixel_decoder_calls': 0,
        'coefficient_calls': 0, 'unchanged_early_readouts': 0,
        'final_coefficient_replacements': 0, 'apply': apply,
        'mode': getattr(adapter, 'mode', None),
        'paired_source': dict(adapter.paired_source_provenance) if adapter is not None else None,
        'metadata_fields_used': ('img_shape', 'batch_input_shape'),
        'native_class_path_untouched': True, 'earlier_gate_paths_untouched': True,
        'all_queries_retained': True, 'cache_cleared_after_forward': False,
        'hooks_removed': False,
    }
    if record['mode'] == POINTWISE_MODE:
        record['geometry_source'] = _provenance(adapter.geometry_source_provenance)
    cache = {'active': False, 'F': None, 'valid_fraction': None}
    handles = []
    head._paired_readout_hook_active = True

    def start(module, inputs):
        _require(not cache['active'] and record['forward_calls'] == 0,
                 'Use a new context for every native head forward')
        record['forward_calls'] += 1
        cache.update(active=True, F=None, valid_fraction=None)

    def capture_feature(module, inputs, output):
        _require(cache['active'] and record['pixel_decoder_calls'] == 0 and
                 isinstance(output, (tuple, list)) and len(output) == 2,
                 'Expected exactly one native pixel-decoder output tuple')
        F = output[0]
        _require(F.ndim == 4 and tuple(F.shape[1:]) == FEATURE_SHAPE and
                 F.dtype == torch.float32, 'Native output[0] must be FP32 F200')
        cache['F'] = F  # Live feature, not a clone/detach or memory substitute.
        cache['valid_fraction'] = metadata_valid_fraction(torch, batch_metadata, F)
        record['pixel_decoder_calls'] += 1
        record['feature_shape'] = tuple(F.shape)
        return None

    def replace_final_coefficient(module, inputs, output):
        _require(cache['active'] and cache['F'] is not None,
                 'Mask embedding must follow the native pixel decoder')
        record['coefficient_calls'] += 1
        _require(record['coefficient_calls'] <= READOUTS and
                 output.ndim == 3 and tuple(output.shape[1:]) == COEFFICIENT_SHAPE and
                 output.shape[0] == cache['F'].shape[0] and
                 output.dtype == torch.float32 and output.device == cache['F'].device,
                 'Unexpected native mask coefficient call or shape')
        if record['coefficient_calls'] < READOUTS:
            record['unchanged_early_readouts'] += 1
            return None
        record['coefficient_shape'] = tuple(output.shape)
        if not apply:
            return None
        result = adapter(cache['F'], output, cache['valid_fraction'])
        replacement = result['new_coeff']
        _require(tuple(replacement.shape) == tuple(output.shape) and
                 replacement.dtype == output.dtype and replacement.device == output.device,
                 'Adapter returned an invalid final coefficient interface')
        record['final_coefficient_replacements'] += 1
        return replacement

    def finish(module, inputs, output):
        try:
            _require(cache['active'] and record['pixel_decoder_calls'] == 1 and
                     record['coefficient_calls'] == READOUTS and
                     record['unchanged_early_readouts'] == 9 and
                     record['final_coefficient_replacements'] == int(apply),
                     'Native initial-plus-nine readout contract was not completed')
            _require(isinstance(output, (tuple, list)) and len(output) == 2 and
                     all(isinstance(values, (tuple, list)) and len(values) == READOUTS
                         for values in output),
                     'Expected all ten native class and mask outputs')
            record['completed_forwards'] += 1
        finally:
            cache.clear()
            cache.update(active=False, F=None, valid_fraction=None)
            record['cache_cleared_after_forward'] = True
        return None

    try:
        handles.append(head.register_forward_pre_hook(start))
        handles.append(head.pixel_decoder.register_forward_hook(capture_feature))
        handles.append(head.mask_embed.register_forward_hook(replace_final_coefficient))
        handles.append(head.register_forward_hook(finish))
        yield record
        _require(record['forward_calls'] == 1 and record['completed_forwards'] == 1,
                 'The context must contain one successfully completed native forward')
    finally:
        for handle in handles:
            handle.remove()
        handles.clear()
        cache.clear()
        head._paired_readout_hook_active = False
        record['hooks_removed'] = True
