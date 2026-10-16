# encoding: utf-8
"""UNBOUND N26 FIT-only complete frozen-R quality input capture.

run(torch,np,spec,c,pre) owns no admission/RAW gate and never self-qualifies.
The caller supplies admitted libraries/Control; imports here are stdlib only.
No model code, tensors, datasets or CUDA are touched at module import.
"""
import gc
import hashlib
import json
from pathlib import Path
import time
import types

INPUT_SCHEMA = 'N26-FIT-matched-frozen-R-query-mask-quality-inputs-v1'
REPORT_SCHEMA = 'N26-FIT-frozen-R-complete-mask-quality-capture-report-v1'
POINT_CHUNK = 2048


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def checked_scope(spec, pre, ctx):
    require(pre['images'] == ctx['images'] == 741 and pre['GT'] == ctx['GT'] == 1661
            and pre['dataset_shards'] == [[0, 371], [371, 741]]
            and pre['models'] == 'frozen_R_E24' and pre['no_new_training'] is True,
            'Exact prelocked FIT741/1661/two-shard/no-training scope required')
    require(ctx['dataset_indices'] == list(range(741)), 'Context full canonical FIT order required')
    indices = spec['dataset_indices']
    require(type(indices) is list and all(type(i) is int for i in indices)
            and any(indices == list(range(a, b)) for a, b in pre['dataset_shards']),
            'One complete prelocked dataset shard; no subset, duplicate or new selection')
    return indices


def matched_query_pairs(raw_indices, ann_ids):
    require(type(raw_indices) is list and len(raw_indices) == 100
            and all(type(g) is int and -2 <= g < len(ann_ids) for g in raw_indices)
            and 1 <= len(ann_ids) <= 100 and len(set(ann_ids)) == len(ann_ids),
            'Complete native100 assignment and distinct nonempty raw GT IDs required')
    matched = [(g, q) for q, g in enumerate(raw_indices) if g >= 0]
    require(sorted(g for g, _ in matched) == list(range(len(ann_ids))),
            'Every original real GT must have exactly one matched query')
    by_gt = {g: q for g, q in matched}
    return [(g, ann_ids[g], by_gt[g]) for g in range(len(ann_ids))]


def point_blocks(total):
    require(type(total) is int and total >= 0, 'Nonnegative complete Omega point count required')
    for first in range(0, total, POINT_CHUNK):
        yield first, min(first + POINT_CHUNK, total)


def _read(c, ref):
    return json.loads(Path(c.closed(ref)).read_text(encoding='utf-8-sig'))


def _load(c, ref, name):
    path = Path(c.closed(ref)); data = path.read_bytes()
    require(len(data) == ref['bytes'] and hashlib.sha256(data).hexdigest() == ref['sha256'],
            'Closed module changed while loading: ' + name)
    module = types.ModuleType(name); module.__file__ = str(path)
    exec(compile(data, str(path), 'exec'), module.__dict__)
    return module


def _digest(value):
    array = value.detach().cpu().contiguous().numpy()
    return dict(shape=list(array.shape), dtype=str(array.dtype),
                sha256=hashlib.sha256(array.tobytes()).hexdigest())


def _same_bits(left, right, label):
    actual, expected = _digest(left), _digest(right)
    require(actual == expected, 'N26_BYTE_DRIFT ' + label + ' ' +
            json.dumps(dict(actual=actual, expected=expected), sort_keys=True))


def _tensor(torch, value, dtype, shape, label, device='cpu'):
    require(isinstance(value, torch.Tensor) and value.dtype == dtype
            and tuple(value.shape) == tuple(shape) and value.device.type == device
            and not value.requires_grad and bool(torch.isfinite(value).all()), label)
    return value


def _frozen(model, label):
    require(not model.training and all(not p.requires_grad and p.grad is None
            for p in model.parameters()), label + ' must remain eval/frozen with gradNone')


def _state_unchanged(model, anchor, label, c):
    state = model.state_dict()
    require(tuple(state) == tuple(anchor), label + ' state keys/order changed')
    for ordinal, (name, value) in enumerate(state.items()):
        if ordinal % 16 == 0:
            c.check()
        _same_bits(value, anchor[name], label + ' ' + name)
    _frozen(model, label)


def _rle_record(torch, np, masks, mask):
    array = mask.numpy()
    encoded = masks.encode(np.asfortranarray(array.astype(np.uint8)))
    rle = dict(size=[int(v) for v in encoded['size']], counts=encoded['counts'].decode('ascii'))
    decoded = masks.decode(dict(size=rle['size'], counts=rle['counts'].encode('ascii')))
    require(decoded.shape == array.shape and np.array_equal(decoded != 0, array),
            'Private complete R RLE readback must reproduce every binary pixel')
    yy, xx = np.nonzero(array)
    crop = ([int(yy.min()), int(xx.min()), int(yy.max()) + 1, int(xx.max()) + 1]
            if len(yy) else None)
    return dict(R_mask_RLE=rle, R_mask_digest=_digest(mask), R_mask_area=int(len(yy)),
                crop_yxyx_halfopen=crop, empty=(len(yy) == 0), RLE_roundtrip_exact=True)


def _capture_image(torch, np, cv2, masks, spec, c, ctx, boot, legacy_body,
                   quality, old_spec, n16_report, old_report, index):
    modules = boot['modules']; model, pr = boot['model'], boot['head']
    pref, old, diag, joint, preflight, cache7 = (modules[k] for k in
        ('pref', 'old', 'diag', 'joint', 'preflight', 'cache7'))
    row, old_row = n16_report['rows'][index], old_report['rows'][index]
    ref, L_ref, old_ref = (ctx['existing_packet_refs'][index],
        ctx['existing_native_L_refs'][index], old_report['packet_refs'][index])
    require(row['packet'] == n16_report['packet_refs'][index] == ref
            and row['natural_native_L_carrier'] == n16_report['native_L_refs'][index] == L_ref
            and old_row['packet'] == old_ref and row['dataset_index'] == old_row['dataset_index'] == index,
            'Exact current/all100/legacy packet and canonical row association required')
    packet = torch.load(str(c.closed(ref)), map_location='cpu')
    carrier = torch.load(str(c.closed(L_ref)), map_location='cpu')
    old_packet = torch.load(str(c.closed(old_ref)), map_location='cpu')
    for value in (packet, carrier):
        require(all(value[k] == n16_report[k] for k in ('source', 'spec', 'GPU_admission', 'phase')),
                'Original N16 source/spec/ad/phase joins differ')
    require(packet['schema'] == 'N16-all100-FIT-action-sampled-cache-body-v1'
            and carrier['schema'] == 'N16-actual-GPU-natural100-L-carrier-v1'
            and old_packet['schema'] == cache7.PACKET_SCHEMA
            and all(old_packet[k] == old_report[k] for k in ('source', 'spec', 'phase')),
            'Real original packet schemas/provenance required')
    identity = ctx['identities'][index]
    require(packet['identity'] == carrier['identity'] == old_packet['identity']
            == row['identity'] == old_row['identity'] == identity
            and identity['image_id'] == ctx['image_ids'][index]
            and packet['natural_native_L_carrier'] == L_ref
            and packet['original_cache_packet'] == old_ref
            and packet['teacher_checkpoint'] == old_packet['teacher_checkpoint'] == ctx['teacher_checkpoint']
            and packet['fixed_PR_checkpoint'] == ctx['fixed_PR_checkpoint'], 'Frozen carrier identity/ref drift')
    H, W = identity['original_shape']; G = len(identity['ann_ids'])
    old_L = _tensor(torch, carrier['native_L'], torch.float32, (100, H, W), 'Saved all100 L')
    _tensor(torch, packet['native_e0'], torch.float32, (1, 100, 256), 'Saved all100 e')
    matching_seed = old_spec['matching_loss_seed_base'] + index
    require(matching_seed == packet['matching_loss_seed'] == old_packet['matching_loss_seed']
            == row['matching_loss_seed'] == old_row['matching_loss_seed'], 'Replay original matcher seed only')
    require(boot['dataset'].get_data_info(index)['img_id'] == identity['image_id'], 'Original dataset order changed')
    data, stripped, gray800, gray, observed_identity = old.load_image(
        model, boot['dataset'], boot['coco'], boot['dataset'].get_data_info(index))
    del stripped, gray800, gray
    observed_identity['dataset_index'] = index
    require(observed_identity == identity, 'Original image/parser/GT identity changed')
    processed = pref.check_targets(torch, data, observed_identity)
    require(pref.tree_equal(torch, processed, packet['processed_target_proof']), 'Processed target proof differs')
    raw_GT, mapping, validator = preflight.raw_GT_validator(
        torch, np, cv2, boot['coco'], data, observed_identity)
    raw_cpu = raw_GT.detach().cpu().contiguous()
    _tensor(torch, raw_cpu, torch.bool, (G, H, W), 'Complete true original-grid GT')
    require(G > 0 and bool(raw_cpu.flatten(1).any(1).all())
            and mapping == packet['raw_target_mapping'] and packet['raw_GT_ids'] == identity['ann_ids']
            and pref.digest_tensor(raw_GT) == packet['raw_GT_digest'], 'Original nonempty GT/mapping/digest differs')
    outside = pref.rng_snapshot(torch)
    with torch.no_grad(), torch.random.fork_rng(devices=[0]):
        torch.random.default_generator.manual_seed(matching_seed); torch.cuda.manual_seed(matching_seed)
        native_head = model.panoptic_head
        with joint._observe_processed_GT(torch, native_head, identity['ann_ids'], raw_GT,
                                          mapping, identity, validator) as gt_observation:
            with legacy_body.observe_all100_targets(torch, native_head, old.digest_tensor) as (matches, observations):
                with cache7.observed_fields(torch, native_head) as fields:
                    losses = model.loss(data['inputs'], data['data_samples'])
        require(set(losses) == set(old.ATOMIC_KEYS) and len(losses) == 30,
                'One original teacher loss observation, thirty original atoms required')
        for value in losses.values():
            atoms = value if isinstance(value, (list, tuple)) else [value]
            require(atoms and all(not v.requires_grad and bool(torch.isfinite(v).all()) for v in atoms),
                    'Frozen teacher loss atoms must be finite/no gradient')
        queries = cache7.matched_queries(matches, mapping, G)
        require(matches == packet['native_stage_matches'] == old_packet['native_stage_matches']
                and queries == old_packet['query_ids'].tolist() == old_row['query_ids']
                and pref.tree_equal(torch, observations, packet['all_ten_target_observations'])
                and pref.tree_equal(torch, gt_observation, packet['actual_raw_processed_GT_correspondence']),
                'All original ten matching/GT correspondence stages must remain exact')
        assignment = legacy_body.final_assignment(torch, observations[-1], mapping,
                                                   identity['ann_ids'], native_head.num_classes)
        require(pref.tree_equal(torch, assignment, packet['assignment']), 'Original all100 assignment changed')
        pairs = matched_query_pairs(assignment['raw_GT_index_by_query'].tolist(), identity['ann_ids'])
        require([q for _, _, q in pairs] == queries, 'Original raw-GT canonical query order changed')
        F, e = fields['F'][0], fields['e'][-1]
        classes, masks_stages = fields['output'][0]
        native_class, M = classes[-1], masks_stages[-1]
        _same_bits(e, packet['native_e0'], 'all100 e0')
        _same_bits(native_class, packet['native_class_logits'], 'all100 class')
        _same_bits(e[0, torch.tensor(queries, dtype=torch.int64, device=e.device)],
                   old_packet['query_embeddings'], 'old matched e')
        natural_L = diag.render_original(torch, M, identity)
        _tensor(torch, natural_L, torch.float32, (100, H, W), 'Fresh all100 L', device='cuda')
        natural_cpu = natural_L.detach().cpu().contiguous()
        _same_bits(natural_cpu, old_L, 'all100 native_L; no tolerance fallback')
        legacy_points = natural_cpu[torch.tensor(queries, dtype=torch.int64)].flatten(1).gather(1, old_packet['point_indices'])
        require(torch.allclose(legacy_points, old_packet['native_logits'],
                rtol=old_spec['old_native_L_rtol'], atol=old_spec['old_native_L_atol']),
                'Historical N07 matched-only L retains only its original tolerance')
        del legacy_points
        omegas = {}
        union = np.empty(0, dtype=np.int64)
        for _, _, q in pairs:
            c.check(); idx = np.flatnonzero(np.abs(natural_cpu[q].numpy()) <= 2).astype(np.int64)
            require(len(idx) == packet['K_counts_all100'][q] == row['K_counts_all100'][q], 'Original Omega/K drift')
            omegas[q] = idx; union = np.union1d(union, idx)
        old_unique = packet['unique_flat_indices']
        _tensor(torch, old_unique, torch.int64, (len(old_unique),), 'Original F calibration coordinates')
        require(bool(((old_unique >= 0) & (old_unique < H * W)).all())
                and bool((old_unique[1:] > old_unique[:-1]).all()), 'Original unique F coordinate bounds/order')
        all_flat = np.union1d(union, old_unique.numpy()).astype(np.int64)
        c.check()
        def render(value, ident):
            c.check(); return diag.render_original(torch, value, ident)
        feature_union = legacy_body.sampled_features_unique(torch, render, F, identity, torch.from_numpy(all_flat.copy()))
        positions = np.searchsorted(all_flat, old_unique.numpy())
        _same_bits(feature_union[torch.from_numpy(positions.astype(np.int64))],
                   packet['sampled_F_unique'], 'original sampled F centers; calibration only, not complete R')
        image_features, image_targets, records = [], [], []
        for g, ann_id, q in pairs:
            c.check(); idx = omegas[q]; K = len(idx); Lq = natural_L[q]
            native_mask = (Lq > 0).detach().cpu().contiguous()
            refined = (Lq > 0).flatten().clone(); calls = 0
            positions = np.searchsorted(all_flat, idx)
            require(np.array_equal(all_flat[positions], idx), 'Every complete Omega point must have actual F')
            for first, stop in point_blocks(K):
                c.check(); flat = torch.from_numpy(idx[first:stop].copy()).to(Lq.device)
                f = feature_union[torch.from_numpy(positions[first:stop].astype(np.int64))].to(Lq.device)
                query_e = e[0, q].unsqueeze(0).expand(stop-first, 256)
                logits = Lq[flat // W, flat % W]
                result = pr(f, query_e, logits, logits.new_zeros((stop-first, 49)), arm='PR')
                residual = result['residual']; combined = logits + residual
                _tensor(torch, residual, torch.float32, (stop-first,), 'Actual complete Omega u', device='cuda')
                _tensor(torch, combined, torch.float32, (stop-first,), 'Actual FP32 L+u', device='cuda')
                refined[flat] = combined > 0; calls += 1
                del flat, f, query_e, logits, result, residual, combined
            Rmask = refined.reshape(H, W).detach().cpu().contiguous(); del refined, Lq
            outside_omega = np.ones(H * W, dtype=np.bool_); outside_omega[idx] = False
            require(np.array_equal(Rmask.numpy().reshape(-1)[outside_omega], native_mask.numpy().reshape(-1)[outside_omega]),
                    'R outside Omega must retain native P, including K0')
            rle_meta = _rle_record(torch, np, masks, Rmask)
            mask_features = quality.make_mask_features(torch, Rmask)
            _tensor(torch, mask_features, torch.float32, (784,), 'Predicted-mask-only 784 input')
            features = torch.cat((e[0, q].detach().cpu().contiguous(), mask_features)).contiguous()
            _tensor(torch, features, torch.float32, (1040,), 'Original e0 plus predicted-mask features only')
            target = quality.matched_mask_iou(np, Rmask.numpy(), raw_cpu[g].numpy())
            require(isinstance(target, np.float64) and np.isfinite(target) and 0 <= target <= 1,
                    'Matched true-GT original-grid FP64 IoU target required')
            records.append(dict(dataset_index=index, image_id=identity['image_id'], identity=identity,
                GT_index=g, GT_annotation_id=ann_id, query_id=q, original_shape=[H,W],
                matching_loss_seed=matching_seed, Omega_points=K, PR_chunk_calls=calls,
                PR_point_chunk=POINT_CHUNK, last_PR_chunk_points=(K % POINT_CHUNK or POINT_CHUNK) if K else 0,
                native_mask_digest=_digest(native_mask), e0_digest=_digest(e[0,q]),
                matched_GT_digest=_digest(raw_cpu[g]), quality_target_FP64=float(target),
                complete_Omega_not_sampled_PR=True, GT_used_in_mask_features=False, **rle_meta))
            image_features.append(features); image_targets.append(float(target))
        # No GPU tensor escapes this function; the setup model/head are released by run.
    require(pref.tree_equal(torch, outside, pref.rng_snapshot(torch)), 'Matching must restore outside CPU/CUDA RNG')
    _frozen(model, 'Teacher'); _frozen(pr, 'R')
    image_audit = dict(dataset_index=index, image_id=identity['image_id'], samples=G,
        native_packet=ref, native_L_carrier=L_ref, original_N07_packet=old_ref,
        one_teacher_loss_forward=True, all_ten_matching_stages_exact=True,
        all100_e_class_L_byteexact=True, old_sampled_F_centers_byteexact=True,
        old_N07_L_rtol=old_spec['old_native_L_rtol'], old_N07_L_atol=old_spec['old_native_L_atol'],
        original_matching_rng_restored=True, matched_GT_annotation_ids=list(identity['ann_ids']),
        Omega_points=sum(len(v) for v in omegas.values()), PR_chunk_calls=sum(r['PR_chunk_calls'] for r in records))
    return image_features, image_targets, records, image_audit


def run(torch, np, spec, c, pre):
    import cv2
    from scipy import ndimage
    from pycocotools import mask as masks
    started = time.monotonic(); c.check()
    require(not torch.cuda.is_initialized(), 'CPU-only setup must precede first CUDA initialization')
    ctx = _read(c, spec['context'])
    require(ctx['schema'] == 'ROOT-N26-quality-input-context-v1', 'Explicit N26 quality-input context required')
    indices = checked_scope(spec, pre, ctx)
    require(len(ctx['existing_packet_refs']) == len(ctx['existing_native_L_refs']) == len(ctx['identities'])
            == len(ctx['image_ids']) == 741, 'Complete original carrier metadata required')
    old_spec = _read(c, ctx['N16_GPU_spec']); n16_report = _read(c, ctx['N16_GPU_report'])
    old_report = _read(c, ctx['original_fit_cache_report'])
    require(n16_report['spec'] == ctx['N16_GPU_spec'] and n16_report['images'] == old_report['images'] == 741
            and n16_report['raw_GT'] == old_report['raw_GT'] == 1661
            and n16_report['packet_refs'] == ctx['existing_packet_refs']
            and n16_report['native_L_refs'] == ctx['existing_native_L_refs'], 'Original complete report refs/counts differ')
    setup = _load(c, spec['setup_source'], '_N26_capture_setup')
    quality = _load(c, spec['quality_source'], '_N26_quality_features')
    legacy_body = _load(c, old_spec['cache_body_source'], '_N26_original_N16_cache_body')
    cpu_outside = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        boot = setup.setup(torch, np, cv2, ndimage, ctx, c, spec)
    require(not torch.cuda.is_initialized(), 'Setup must not initialize CUDA')
    required = {'modules','model','head','dataset','coco','context','anchor','pr_anchor','loading'}
    require(type(boot) is dict and required <= set(boot), 'Exact setup return API required')
    model, head = boot['model'], boot['head']; loading = boot['loading']
    json.dumps(loading, allow_nan=False)  # Metadata only; do not retain a hidden model/tensor alias.
    _frozen(model, 'Teacher'); _frozen(head, 'R')
    require(boot['context']['checkpointmap']['native0'] == ctx['teacher_checkpoint']
            and len(boot['dataset']) == 741 and len(boot['coco'].anns) == 1661, 'Same teacher and full FIT required')
    require(spec['allocator_limit_MiB'] == 4096, 'Locked four-GiB allocator required')
    torch.cuda.init()
    torch.cuda.set_per_process_memory_fraction(4096 * 1024**2 / torch.cuda.get_device_properties(0).total_memory, 0)
    torch.cuda.reset_peak_memory_stats(0)
    cuda_outside = torch.cuda.get_rng_state(0).cpu().clone()
    model.to('cuda:0'); head.to('cuda:0')
    output = Path(spec['output_directory']); output.mkdir()
    binding = dict(source=c.ref(Path(__file__)), spec=c.execution_spec_ref,
        GPU_admission=c.execution_admission_ref, phase=spec['phase'], context=spec['context'], prelock=spec['prelock'],
        setup_source=spec['setup_source'], quality_source=spec['quality_source'],
        teacher_checkpoint=ctx['teacher_checkpoint'], fixed_PR_checkpoint=ctx['fixed_PR_checkpoint'])
    all_features, all_targets, records, images = [], [], [], []
    for completed, index in enumerate(indices, 1):
        c.check()
        features, targets, current, audit = _capture_image(torch, np, cv2, masks, spec, c,
            ctx, boot, legacy_body, quality, old_spec, n16_report, old_report, index)
        all_features.extend(features); all_targets.extend(targets); records.extend(current); images.append(audit)
        del features, targets, current, audit
        gc.collect(); c.check()
        print(json.dumps(dict(N26_capture_completed_images=completed, scope_images=len(indices))), flush=True)
    expected_GT = sum(len(ctx['identities'][i]['ann_ids']) for i in indices)
    require(len(records) == len(all_features) == len(all_targets) == expected_GT
            and len({(r['image_id'],r['GT_annotation_id']) for r in records}) == expected_GT
            and len({(r['image_id'],r['query_id']) for r in records}) == expected_GT,
            'All real matched samples retained once, including low-quality/empty/K0')
    _state_unchanged(model, boot['anchor'], 'Native teacher', c)
    _state_unchanged(head, boot['pr_anchor'], 'Strong R', c)
    require(torch.equal(torch.cuda.get_rng_state(0).cpu(), cuda_outside), 'Global CUDA RNG changed')
    require(torch.equal(torch.get_rng_state(), cpu_outside), 'Global CPU RNG changed')
    peak = int(torch.cuda.max_memory_allocated(0)); model.cpu(); head.cpu()
    del model, head, boot
    gc.collect(); torch.cuda.synchronize()
    torch._C._cuda_clearCublasWorkspaces(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    cleanup = dict(allocated_after=int(torch.cuda.memory_allocated(0)), reserved_after=int(torch.cuda.memory_reserved(0)))
    require(cleanup['allocated_after'] == cleanup['reserved_after'] == 0 and peak <= 4096 * 1024**2,
            'All captured CUDA aliases/workspaces must be released within allocator bound')
    c.check()
    payload = dict(schema=INPUT_SCHEMA, **binding, dataset_indices=list(indices),
        features=torch.stack(all_features).contiguous(), targets_FP64=torch.tensor(all_targets,dtype=torch.float64),
        records=[{k:v for k,v in r.items() if k != 'R_mask_RLE'} for r in records],
        input_order='original_e0_256_then_predicted_mask_crop784', matched_only=True,
        unmatched_zero_target_created=False, qualified=False, training_executed=False, AP_evaluated=False)
    _tensor(torch, payload['features'], torch.float32, (expected_GT,1040), 'Saved complete small input matrix')
    _tensor(torch, payload['targets_FP64'], torch.float64, (expected_GT,), 'Saved FP64 matched targets')
    target = output / 'FIT_QUALITY_INPUTS.pth'
    with target.open('xb') as stream: torch.save(payload, stream)
    replay = torch.load(str(target), map_location='cpu', weights_only=True)
    _same_bits(replay['features'], payload['features'], 'saved feature readback')
    _same_bits(replay['targets_FP64'], payload['targets_FP64'], 'saved target readback')
    require({k:v for k,v in replay.items() if k not in ('features','targets_FP64')}
            == {k:v for k,v in payload.items() if k not in ('features','targets_FP64')}, 'Saved small-payload metadata readback differs')
    inputs_ref = c.ref(target)
    audit_ref = c.write_json(output/'AUDIT.json', dict(schema='N26-complete-R-mask-matched-quality-audit-v1',
        **binding, dataset_indices=list(indices), images=images, records=records,
        private_RLE_not_for_publication=True, GT_only_in_targets=True, qualified=False))
    report = dict(schema=REPORT_SCHEMA, status='N26_FIT_SCOPE_CAPTURE_COMPLETE_PENDING_ROOT_AND_INDEPENDENT_CPU',
        **binding, dataset_indices=list(indices), images=len(images), samples=expected_GT,
        full_FIT_images=741, full_FIT_GT=1661, inputs=inputs_ref, audit=audit_ref,
        one_teacher_forward_per_image=True, complete_Omega_R=True, PR_chunk=POINT_CHUNK,
        all100_e_class_L_byteexact=True, old_sampled_F_centers_byteexact=True,
        all_ten_matcher_GT_correspondence_exact=True, original_N07_L_tolerance_only=True,
        original_parameters_byteexact_unchanged=True, all_parameter_gradients_None=True,
        global_CPU_CUDA_RNG_restored=True, RLE_roundtrip_exact=True, GT_used_in_mask_features=False,
        unmatched_zero_target_created=False, full_F_or_L_saved=False, qualified=False,
        ROOT_actual_GPU0_verified=False, independent_CPU_verified=False,
        training_executed=False, AP_evaluated=False, cleanup=cleanup,
        torch_peak_allocated_bytes=peak, elapsed_seconds=time.monotonic()-started,
        original_model_loading=loading)
    report_ref = c.write_json(output/'CAPTURE_REPORT.json', report)
    c.check()
    return dict(report_ref=report_ref)
