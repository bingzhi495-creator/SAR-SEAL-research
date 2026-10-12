"""One current native loss plus caller-bound final-stage point BCE; inert I/O.

Torch and all helpers are injected. No backward, step, sampling, RNG scope,
fixed-teacher matching, detached native fields, or training qualification.
The caller owns data preprocessing, model/BN policy, resources and provenance.
"""

from contextlib import contextmanager
import types

_GT_ACTIVE = '_joint_native_GT_observer_active'


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _method_identity(method):
    return getattr(method, '__self__', None), getattr(method, '__func__', method)


@contextmanager
def _observe_processed_GT(torch, head, raw_ids, raw_masks, mapping, identity, validator):
    """Observe the original result once; validator must inspect actual GT tensors.

    validator(processed, raw_ids, raw_masks, mapping, identity) returns exactly
    True only after checking actual labels/masks and mapped raw annotation rows
    through the caller-qualified resize/padding transform. Lengths/permutation
    alone are insufficient. This helper does not independently decode raw GT.
    """
    _require(_GT_ACTIVE not in head.__dict__, 'Nested processed-GT observer rejected')
    original = head.preprocess_gt
    had_instance = 'preprocess_gt' in head.__dict__
    previous = head.__dict__.get('preprocess_gt')
    observed = dict(calls=0, correspondence_validator_passed=False)

    def observe(current, *args, **kwargs):
        _require(current is head and observed['calls'] == 0,
                 'Exactly one original preprocess_gt call required')
        observed['calls'] += 1
        result = original(*args, **kwargs)
        _require(isinstance(result, (list, tuple)) and len(result) == 1,
                 'Native preprocessing must return one image')
        gt = result[0]
        _require(isinstance(gt.labels, torch.Tensor) and isinstance(gt.masks, torch.Tensor)
                 and tuple(gt.labels.shape) == (len(raw_ids),)
                 and tuple(gt.masks.shape) == (len(raw_ids), 800, 800)
                 and gt.labels.device == gt.masks.device == raw_masks.device
                 and bool((gt.labels == 0).all())
                 and bool(((gt.masks == 0) | (gt.masks == 1)).all()),
                 'Actual single-class processed GT tensors differ')
        _require(validator(gt, raw_ids, raw_masks, mapping, identity) is True,
                 'Caller must validate actual processed/raw GT correspondence')
        observed['correspondence_validator_passed'] = True
        return result

    try:
        setattr(head, _GT_ACTIVE, True)
        head.preprocess_gt = types.MethodType(observe, head)
        yield observed
        _require(observed == dict(calls=1, correspondence_validator_passed=True),
                 'One validated current processed-GT observation required')
    finally:
        if had_instance:
            head.preprocess_gt = previous
        elif 'preprocess_gt' in head.__dict__:
            delattr(head, 'preprocess_gt')
        if _GT_ACTIVE in head.__dict__:
            delattr(head, _GT_ACTIVE)
        _require(_method_identity(head.preprocess_gt) == _method_identity(original)
                 and ('preprocess_gt' in head.__dict__) is had_instance
                 and (not had_instance or head.__dict__['preprocess_gt'] is previous),
                 'Original processed-GT method/instance state was not restored')


def joint_native_losses(torch, pref, common, old, capture, adapter,
                        nativeobjective, model, data, readout, renderer,
                        identity, raw_annotation_ids, raw_gt_masks,
                        processed_GT_to_raw, validate_GT_correspondence,
                        point_indices, arm, global_micro):
    """Return native_total + auxiliary (lambda=1), both live, without backward.

    arm A: nativeobjective.matched_native_point_loss on rendered native M200.
    arm B: live adapter projections plus matched_joint_point_loss on L+u.
    raw bool[G,Horig,Worig] and int64[G,12544] indices use raw annotation order.
    processed_GT_to_raw[j] is the raw row represented by actual processed row j;
    its mandatory validator checks actual native tensors and image transform.
    Matching is recomputed by this single current model.loss for all ten stages.
    Sampling and global accumulation scaling remain entirely outside this API.
    """
    _require(torch.is_grad_enabled() and arm in ('A', 'B'),
             'Ambient autograd and explicit arm A/B required')
    _require(type(global_micro) is int and global_micro >= 0,
             'Nonnegative current micro index required')
    _require(isinstance(data, dict) and len(data['data_samples']) == 1
             and isinstance(data['inputs'], torch.Tensor), 'One prepared native image required')
    _require(callable(renderer) and callable(validate_GT_correspondence),
             'Caller-bound renderer and actual GT correspondence validator required')
    _require(isinstance(identity, dict), 'Caller-bound native image identity required')
    for key, upper in (('actual_img_shape', 800), ('original_shape', None)):
        dims = identity[key]
        _require(isinstance(dims, (list, tuple)) and len(dims) == 2
                 and all(type(v) is int and v > 0 and (upper is None or v <= upper) for v in dims),
                 'Positive native image dimensions required: ' + key)
    height, width = identity['original_shape']
    raw_ids, mapping = tuple(raw_annotation_ids), tuple(processed_GT_to_raw)
    count = len(raw_ids)
    _require(count <= 100 and all(type(v) is int for v in raw_ids)
             and len(set(raw_ids)) == count and all(type(v) is int for v in mapping)
             and sorted(mapping) == list(range(count)),
             'Unique raw IDs and complete explicit processed-to-raw permutation required')
    _require(isinstance(raw_gt_masks, torch.Tensor) and raw_gt_masks.layout == torch.strided
             and raw_gt_masks.dtype == torch.bool and tuple(raw_gt_masks.shape) == (count, height, width)
             and raw_gt_masks.device == data['inputs'].device, 'Original raw GT rows must be shared-device bool')
    _require(isinstance(point_indices, torch.Tensor) and point_indices.layout == torch.strided
             and point_indices.dtype == torch.int64 and tuple(point_indices.shape) == (count, 12544)
             and point_indices.device == raw_gt_masks.device
             and not bool(((point_indices < 0) | (point_indices >= height * width)).any()),
             'Shared original-raster indices must be int64[G,12544]; replacement is allowed')
    head = model.panoptic_head
    with _observe_processed_GT(torch, head, raw_ids, raw_gt_masks, mapping,
                               identity, validate_GT_correspondence) as GT_observation:
        with common.observe_native_loss(torch, pref, old, None, None, head,
                                        data, 'native0', global_micro) as observation:
            with capture.capture_live_final_native_fields(torch, head) as captured:
                native_losses = model.loss(data['inputs'], data['data_samples'])
                live_native = dict(captured['tensors'])
                forward_counts = dict(captured['forward_counts'])
    _require(set(native_losses) == set(old.ATOMIC_KEYS) and len(native_losses) == 30,
             'Original thirty native loss atoms must remain unchanged')
    native_total, atoms = pref.loss_terms(torch, native_losses)
    matches = observation['matches']
    _require(len(matches) == 10, 'All ten CURRENT native assignments required')
    for row in matches:
        queries, GT = row['positive_query_indices'], row['assigned_GT_indices']
        _require(len(queries) == len(GT) == row['positives'] == count
                 and all(type(q) is int and 0 <= q < 100 for q in queries)
                 and len(set(queries)) == count and all(type(g) is int for g in GT)
                 and sorted(GT) == list(range(count)), 'Every current stage must match all raw GT once')
    query_by_raw = [None] * count
    for query, native_GT in zip(matches[-1]['positive_query_indices'], matches[-1]['assigned_GT_indices']):
        query_by_raw[mapping[native_GT]] = query
    query_ids = torch.tensor(query_by_raw, dtype=torch.int64, device=raw_gt_masks.device)
    if arm == 'A':
        native = renderer(live_native['M200'], identity).unsqueeze(0)
        auxiliary = nativeobjective.matched_native_point_loss(native, query_ids, raw_gt_masks, point_indices)
        fields = dict(native_logits=native)
    else:
        fields = adapter.project_live_native_fields(readout, live_native['F256'], live_native['e0'],
            live_native['M200'], identity['actual_img_shape'], identity['original_shape'], renderer)
        auxiliary = adapter.matched_joint_point_loss(readout, fields, query_ids, raw_gt_masks, point_indices)
    _require(isinstance(auxiliary, torch.Tensor) and auxiliary.ndim == 0
             and auxiliary.requires_grad and bool(torch.isfinite(auxiliary)), 'Finite live auxiliary scalar required')
    total = native_total + auxiliary
    _require(total.requires_grad and bool(torch.isfinite(total)), 'Finite live joint total required')
    metadata = dict(arm=arm, auxiliary_weight=1., native_atoms=30, native_atom_values=atoms,
        current_stage_matches=matches, final_query_by_raw_GT=query_by_raw,
        raw_annotation_ids=list(raw_ids), processed_GT_to_raw=list(mapping),
        processed_GT_observation=dict(GT_observation), native_forward_counts=forward_counts,
        correspondence_scope='caller validated actual processed labels/masks and raw rows/image transform',
        independent_raw_GT_qualification_claimed=False, fixed_teacher_matching_used=False,
        point_indices_sampled_here=False, backward_or_optimizer_step_performed=False,
        auxiliary_accumulation_scaled=False)
    return dict(total=total, native_total=native_total, auxiliary=auxiliary,
                native_losses=native_losses, live_native_fields=live_native,
                auxiliary_fields=fields, query_ids=query_ids, metadata=metadata)
