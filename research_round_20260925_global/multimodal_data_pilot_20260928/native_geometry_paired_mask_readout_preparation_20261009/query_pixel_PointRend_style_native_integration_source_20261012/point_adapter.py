"""SOURCE-only C adapter: live F256/coarse L, direct PointRend-style logits.

No scientific imports, RNG, sampling, optimizer, GT-file IO or residual addition.
The caller binds the original renderer and passive native observers. This is an
integration candidate, not a faithful PointRend pipeline or numerical qualification.
"""


def require(ok, message):
    if not ok:
        raise ValueError(message)


def rendered_fields(torch, F256, M200, identity, renderer):
    require(torch.is_grad_enabled() and callable(renderer), 'Live caller-bound renderer required')
    require(tuple(F256.shape) == (1, 256, 200, 200) and tuple(M200.shape) == (1, 100, 200, 200) and
        F256.dtype == M200.dtype == torch.float32 and F256.device == M200.device and
        F256.requires_grad and M200.requires_grad and bool(torch.isfinite(F256).all()) and
        bool(torch.isfinite(M200).all()), 'Live FP32 native F256/M200 required')
    fine = renderer(F256, identity).unsqueeze(0)
    logits = renderer(M200, identity).unsqueeze(0)
    H, W = identity['original_shape']
    require(tuple(fine.shape) == (1, 256, H, W) and tuple(logits.shape) == (1, 100, H, W) and
        fine.requires_grad and logits.requires_grad and bool(torch.isfinite(fine).all()) and
        bool(torch.isfinite(logits).all()), 'Original raster live renderer fields differ')
    return dict(fine_features=fine, native_logits=logits)


def matched_point_loss(torch, head, fields, query_ids, gt_masks, point_indices):
    """Stable BCE of DIRECT logits; pixel mean then GT mean, no accumulation scale.

    Nonempty supervision has exactly 12544 IID caller-selected integer pixels,
    including duplicates. Empty G, or explicit P=0, returns a connected zero;
    the pure head's R=0/P=0 rejection is not bypassed with fabricated masks.
    """
    fine, native = fields['fine_features'], fields['native_logits']
    require(torch.is_grad_enabled() and fine.ndim == native.ndim == 4 and fine.shape[0] == native.shape[0] == 1 and
        fine.shape[1] == 256 and native.shape[1] == 100 and fine.shape[-2:] == native.shape[-2:] and
        fine.dtype == native.dtype == torch.float32 and fine.device == native.device and
        fine.requires_grad and native.requires_grad and bool(torch.isfinite(fine).all()) and
        bool(torch.isfinite(native).all()), 'Live finite original-grid F256/L required')
    H, W = native.shape[-2:]; G = len(query_ids)
    require(H > 0 and W > 0, 'Positive original raster required')
    require(query_ids.dtype == torch.int64 and tuple(query_ids.shape) == (G,) and
        query_ids.device == native.device and G <= 100 and len(set(query_ids.tolist())) == G and
        not bool(((query_ids < 0) | (query_ids >= 100)).any()), 'Unique all-axis query IDs required')
    require(gt_masks.dtype == torch.bool and tuple(gt_masks.shape) == (G, H, W) and gt_masks.device == native.device and
        point_indices.dtype == torch.int64 and point_indices.ndim == 2 and point_indices.shape[0] == G and
        point_indices.shape[1] in (0, 12544) and point_indices.device == native.device and
        not bool(((point_indices < 0) | (point_indices >= H * W)).any()), 'Original bool GT/int64 replacement indices differ')
    parameters = tuple(head.parameters())
    require(len(parameters) == 8 and sum(p.numel() for p in parameters) == 198402 and not tuple(head.buffers()),
        'C direct point-head eight tensors/198402/no buffers required')
    if G == 0 or point_indices.shape[1] == 0:
        return sum((v[..., :0].sum() for v in (fine, native) + parameters), native.new_zeros(()))
    losses = []
    flat_fine = fine[0].flatten(1); flat_native = native[0].flatten(1)
    for j, q in enumerate(query_ids.tolist()):
        indices = point_indices[j]
        prediction = head(flat_fine[:, indices].unsqueeze(0), flat_native[q, indices].reshape(1, 1, -1))
        target = gt_masks[j].flatten()[indices].to(prediction.dtype).reshape(1, 1, -1)
        losses.append(torch.nn.functional.binary_cross_entropy_with_logits(prediction, target, reduction='mean'))
    loss = torch.stack(losses).mean()
    require(loss.requires_grad and bool(torch.isfinite(loss)), 'Finite connected direct-logit BCE required')
    return loss


def dense_direct(torch, head, fine_fields, native_L, pixel_chunk_size=2048):
    """Inference direct logits, all100 query ordinals, no threshold/score/NMS.

    Caller owns no_grad. Chunking bounds activations, not total compute; there is
    deliberately no native-L residual addition or posterior-head compatibility.
    """
    require(not torch.is_grad_enabled() and type(pixel_chunk_size) is int and pixel_chunk_size > 0,
        'Inference no_grad and positive integer pixel chunk required')
    require(fine_fields.ndim == native_L.ndim == 4 and fine_fields.shape[0] == native_L.shape[0] == 1 and
        fine_fields.shape[1] == 256 and native_L.shape[1] == 100 and fine_fields.shape[-2:] == native_L.shape[-2:] and
        fine_fields.dtype == native_L.dtype == torch.float32 and fine_fields.device == native_L.device and
        bool(torch.isfinite(fine_fields).all()) and bool(torch.isfinite(native_L).all()), 'Dense original raster fields differ')
    H, W = native_L.shape[-2:]
    require(H > 0 and W > 0, 'Positive dense raster required')
    fine = fine_fields[0].flatten(1); native = native_L[0].flatten(1)
    result = native_L.new_empty((100, H * W))
    for q in range(100):
        for first in range(0, H * W, pixel_chunk_size):
            last = min(first + pixel_chunk_size, H * W)
            result[q, first:last] = head(fine[:, first:last].unsqueeze(0), native[q, first:last].reshape(1, 1, -1))[0, 0]
    require(bool(torch.isfinite(result).all()), 'Nonfinite direct dense output')
    return result.reshape(100, H, W)


def raw_GT_validator(torch, np, cv2, coco, data, identity):
    ids = identity['ann_ids']
    parser_ids = [r['annotation_id'] for r in identity['parser_rows'] if r['native_parser_retained']]
    require(len(ids) == len(set(ids)) and parser_ids == ids and 0 <= len(ids) <= 100 and
        coco.get_ann_ids(img_ids=[identity['image_id']]) == ids, 'Actual parser/raw identities differ; no rows may be dropped')
    mapping = tuple(ids.index(a) for a in parser_ids)
    H, W = identity['original_shape']; h, w = identity['actual_img_shape']
    image = coco.load_imgs([identity['image_id']])[0]
    require((image['height'], image['width']) == (H, W), 'Actual original image shape differs')
    masks = [np.zeros((H, W), np.uint8) if a.get('segmentation') == [] else coco.annToMask(a) for a in coco.load_anns(ids)]
    raw = np.stack(masks) if masks else np.zeros((0, H, W), np.uint8)
    require(raw.shape == (len(ids), H, W) and bool(((raw == 0) | (raw == 1)).all()), 'Complete raw binary GT required')
    padded = np.zeros((len(ids), 800, 800), np.int64)
    for j, mask in enumerate(raw):
        padded[j, :h, :w] = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    expected = torch.from_numpy(padded[list(mapping)] if mapping else padded).to(data['inputs'].device)
    sample = data['data_samples'][0]
    require(len(sample.ignored_instances) == 0 and 'gt_sem_seg' not in sample and
        torch.equal(sample.gt_instances.masks.to_tensor(dtype=torch.int64, device=expected.device), expected),
        'Actual resized/padded BitmapMasks disagree with every raw row')
    raw_tensor = torch.from_numpy(raw).to(device=expected.device, dtype=torch.bool)
    def validate(processed, actual_ids, actual_raw, actual_mapping, actual_identity):
        require(tuple(actual_ids) == tuple(ids) and tuple(actual_mapping) == mapping and actual_identity == identity and
            torch.equal(actual_raw, raw_tensor) and processed.labels.dtype == processed.masks.dtype == torch.int64 and
            tuple(processed.labels.shape) == (len(ids),) and bool((processed.labels == 0).all()) and
            torch.equal(processed.masks, expected), 'Actual preprocess_gt/raw correspondence differs')
        return True
    return raw_tensor, mapping, validate


def native_point_losses(torch, pref, common, old, capture, joint, model, data, head, renderer,
                        identity, raw_annotation_ids, raw_gt_masks, mapping, validator, point_indices, global_micro):
    """One current native loss forward + C direct point BCE, lambda1; no backward.

    All ten dynamic assignments and actual processed/raw correspondence remain
    observed through unchanged caller-bound native helpers. Neither the old B
    adapter nor its numerical qualification is used for the C point objective.
    """
    require(torch.is_grad_enabled() and type(global_micro) is int and global_micro >= 0, 'Live current native micro required')
    raw_ids = tuple(raw_annotation_ids); count = len(raw_ids)
    require(len(set(raw_ids)) == count and all(type(v) is int for v in raw_ids) and count <= 100 and
        all(type(v) is int for v in mapping) and sorted(mapping) == list(range(count)) and
        tuple(raw_gt_masks.shape) == (count, *identity['original_shape']) and
        raw_gt_masks.dtype == torch.bool and tuple(point_indices.shape) == (count, 12544) and
        point_indices.dtype == torch.int64 and raw_gt_masks.device == point_indices.device == data['inputs'].device,
        'Complete raw-order original-grid targets/points required')
    native_head = model.panoptic_head
    with joint._observe_processed_GT(torch, native_head, raw_ids, raw_gt_masks, mapping, identity, validator) as GT_observation:
        with common.observe_native_loss(torch, pref, old, None, None, native_head, data, 'native0', global_micro) as observation:
            with capture.capture_live_final_native_fields(torch, native_head) as captured:
                native_losses = model.loss(data['inputs'], data['data_samples'])
                live = dict(captured['tensors']); forward_counts = dict(captured['forward_counts'])
    require(set(native_losses) == set(old.ATOMIC_KEYS) and len(native_losses) == 30, 'Unchanged native thirty loss atoms required')
    native_total, atoms = pref.loss_terms(torch, native_losses)
    matches = observation['matches']; require(len(matches) == 10, 'Ten CURRENT assignments required')
    for row in matches:
        queries, GT = row['positive_query_indices'], row['assigned_GT_indices']
        require(len(queries) == len(GT) == row['positives'] == count and len(set(queries)) == count and
            all(type(q) is int and 0 <= q < 100 for q in queries) and sorted(GT) == list(range(count)), 'Current all-GT one-to-one assignment differs')
    query_by_raw = [None] * count
    for q, native_GT in zip(matches[-1]['positive_query_indices'], matches[-1]['assigned_GT_indices']):
        query_by_raw[mapping[native_GT]] = q
    query_ids = torch.tensor(query_by_raw, device=raw_gt_masks.device, dtype=torch.int64)
    fields = rendered_fields(torch, live['F256'], live['M200'], identity, renderer)
    auxiliary = matched_point_loss(torch, head, fields, query_ids, raw_gt_masks, point_indices)
    total = native_total + auxiliary
    require(total.requires_grad and bool(torch.isfinite(total)), 'Finite joint native+direct-point total required')
    metadata = dict(arm='C', auxiliary_weight=1., native_atoms=30, native_atom_values=atoms,
        current_stage_matches=matches, final_query_by_raw_GT=query_by_raw, raw_annotation_ids=list(raw_ids),
        processed_GT_to_raw=list(mapping), processed_GT_observation=dict(GT_observation), native_forward_counts=forward_counts,
        correspondence_scope='caller validated actual processed labels/masks and every raw row/image transform',
        independent_raw_GT_qualification_claimed=False, fixed_teacher_matching_used=False, point_indices_sampled_here=False,
        backward_or_optimizer_step_performed=False, auxiliary_accumulation_scaled=False,
        direct_point_logits=True, native_L_added_to_point_logits=False, final_e0_direct_head_input=False)
    return dict(total=total, native_total=native_total, auxiliary=auxiliary, native_losses=native_losses,
        live_native_fields=live, auxiliary_fields=fields, query_ids=query_ids, metadata=metadata)
