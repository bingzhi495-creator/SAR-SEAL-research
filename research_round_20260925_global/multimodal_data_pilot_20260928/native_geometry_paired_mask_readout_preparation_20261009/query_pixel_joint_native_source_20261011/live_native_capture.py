"""Passive live native capture; Torch is injected, not imported here.

Installed mmdet SOURCE: detectors/maskformer.py:62-63 calls head.loss, and
dense_heads/maskformer_head.py:554 calls self(...), so head hooks execute.
Mask2FormerHead.forward calls pixel_decoder once and mask_embed ten times;
its output is the two native stage lists. Direct head.forward is unsupported.
No native-model, optimizer, matching or memory qualification is supplied.
"""

from contextlib import contextmanager

_ACTIVE = '_joint_live_native_capture_active'
_PAIRED_ACTIVE = ('_paired_readout_hook_active', '_spatial_readout_hook_active',
                  '_query_pair_readout_hook_active')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _inactive(head):
    _require(not any(bool(getattr(head, name, False)) for name in _PAIRED_ACTIVE),
             'Active paired/spatial/query-pair hooks cannot overlap live native capture')


def _field(torch, name, value, shape, reference=None):
    _require(isinstance(value, torch.Tensor) and value.layout == torch.strided,
             name + ' must be a dense strided Tensor')
    _require(value.dtype == torch.float32 and tuple(value.shape) == shape,
             name + ' must have the exact native FP32 shape')
    _require(reference is None or value.device == reference.device,
             name + ' must share the native feature device')
    _require(value.requires_grad and value.grad_fn is not None,
             name + ' must retain the live native forward graph')
    _require(bool(torch.isfinite(value).all()), name + ' must be finite')


def _inventory(modules):
    return tuple((tuple(module._forward_pre_hooks.items()),
                  tuple(module._forward_hooks.items())) for module in modules)


@contextmanager
def capture_live_final_native_fields(torch, head):
    """Yield {'tensors': live fields, 'forward_counts': counters} for one forward.

    Use around one native model.loss with autograd enabled. After that call,
    tensors contains F256[1,256,200,200], e0[1,100,256], M200[1,100,200,200]
    and final_class_logits[1,100,2], all the original Tensor objects. Hooks return
    None; no outputs are altered, cloned or detached. Passive native-loss hooks
    may coexist. Nested capture or active paired replacement hooks are rejected.

    Add caller-supplied auxiliary loss to native loss before one backward; it may
    occur inside this context. Backbone-only checkpoint replay is compatible;
    replay of the observed head/pixel decoder would violate one-forward counts.
    Copy tensors/counters before exit if needed: all internal maps are cleared,
    while a caller's separate tensor dictionary still owns its live graph.
    Graph properties do not prove nonzero gradients, native30/matching/BN policy,
    end-to-end effectiveness or GPU fit. This helper generates none of these.
    """
    _require(torch.is_grad_enabled(), 'Live native capture requires ambient autograd')
    _require(isinstance(head, torch.nn.Module), 'head must be the native Torch module')
    _require(not bool(getattr(head, _ACTIVE, False)), 'Live native capture cannot reenter')
    _inactive(head)
    _require(hasattr(head, 'pixel_decoder') and hasattr(head, 'mask_embed'),
             'Native pixel_decoder and mask_embed modules are required')
    modules = (head, head.pixel_decoder, head.mask_embed)
    _require(all(isinstance(module, torch.nn.Module) for module in modules) and
             len({id(module) for module in modules}) == 3, 'Three distinct native modules required')
    inventory = _inventory(modules)
    had_marker, previous_marker = _ACTIVE in head.__dict__, head.__dict__.get(_ACTIVE)
    fields, counts, handles = {}, dict(head_forward_started=0, head_forward_finished=0,
                                       pixel_decoder=0, mask_embed=0), []
    record = dict(tensors=fields, forward_counts=counts)

    def start(module, args):
        _inactive(head)
        _require(getattr(head, _ACTIVE, False) is True and
                 counts['head_forward_started'] == counts['head_forward_finished'] == 0,
                 'Exactly one non-reentrant native head __call__ is required')
        counts['head_forward_started'] += 1

    def feature(module, args, output):
        _require(counts['head_forward_started'] == 1 and counts['head_forward_finished'] == 0 and
                 counts['pixel_decoder'] == 0, 'Expected one pixel decoder inside native head __call__')
        _require(isinstance(output, (list, tuple)) and len(output) == 2,
                 'Native pixel decoder must return (mask_features, memories)')
        _field(torch, 'F256', output[0], (1, 256, 200, 200))
        fields['F256'] = output[0]
        counts['pixel_decoder'] += 1

    def embedding(module, args, output):
        _require(counts['head_forward_started'] == 1 and counts['head_forward_finished'] == 0 and
                 counts['pixel_decoder'] == 1 and counts['mask_embed'] < 10,
                 'Expected ten native mask embedding calls after the pixel decoder')
        _field(torch, 'native mask embedding', output, (1, 100, 256), fields['F256'])
        counts['mask_embed'] += 1
        if counts['mask_embed'] == 10:
            fields['e0'] = output

    def finish(module, args, output):
        _inactive(head)
        _require(counts == dict(head_forward_started=1, head_forward_finished=0,
                               pixel_decoder=1, mask_embed=10), 'Native forward counter mismatch')
        _require(isinstance(output, (list, tuple)) and len(output) == 2 and
                 all(isinstance(stages, list) and len(stages) == 10 for stages in output),
                 'Native head must return two unmodified ten-stage lists')
        classes, masks = output
        for name, stages, shape in (('class', classes, (1, 100, 2)),
                                     ('mask', masks, (1, 100, 200, 200))):
            _require(all(isinstance(value, torch.Tensor) and value.layout == torch.strided and
                         value.dtype == torch.float32 and tuple(value.shape) == shape and
                         value.device == fields['F256'].device and value.requires_grad and
                         value.grad_fn is not None for value in stages),
                     'All ten native ' + name + ' fields must preserve shape/device/live graph')
        _field(torch, 'M200', masks[-1], (1, 100, 200, 200), fields['F256'])
        _field(torch, 'final_class_logits', classes[-1], (1, 100, 2), fields['F256'])
        fields.update(M200=masks[-1], final_class_logits=classes[-1])
        counts['head_forward_finished'] += 1

    try:
        setattr(head, _ACTIVE, True)
        handles.append(head.register_forward_pre_hook(start))
        handles.append(head.pixel_decoder.register_forward_hook(feature))
        handles.append(head.mask_embed.register_forward_hook(embedding))
        handles.append(head.register_forward_hook(finish))
        yield record
        _inactive(head)
        _require(counts == dict(head_forward_started=1, head_forward_finished=1,
                               pixel_decoder=1, mask_embed=10) and
                 set(fields) == {'F256', 'e0', 'M200', 'final_class_logits'},
                 'One complete native model.loss forward is required before normal exit')
    finally:
        for handle in handles:
            handle.remove()
        handles.clear()
        fields.clear()
        counts.clear()
        record.clear()
        if had_marker:
            setattr(head, _ACTIVE, previous_marker)
        elif _ACTIVE in head.__dict__:
            delattr(head, _ACTIVE)
        _require(_inventory(modules) == inventory, 'Original native forward hooks were not restored')
