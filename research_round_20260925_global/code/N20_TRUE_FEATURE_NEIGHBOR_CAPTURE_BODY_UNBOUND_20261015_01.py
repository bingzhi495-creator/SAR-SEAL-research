"""UNBOUND N20 numerical capture body; no IO, admission, CLI or scientific run.

ROOT supplies source-qualified modules, frozen teacher and qualified N16 packet.
Only one existing native loss forward is observed. No sampling, label creation,
PR forward, optimizer, backward, COCO or CPU renderer is invoked here.
ROOT confirmed the coordinate module's exact two-key NumPy INT64 mapping ABI.
"""

import time


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def _frozen(model):
    require(not model.training and all(not p.requires_grad and p.grad is None
                                     for p in model.parameters()),
            'Qualified teacher must remain frozen/eval with no gradients')


def _cpu_tensor(torch, value, dtype, shape, name):
    require(isinstance(value, torch.Tensor) and value.device.type == 'cpu'
            and value.dtype == dtype and tuple(value.shape) == tuple(shape)
            and value.is_contiguous() and not value.requires_grad,
            name + ': exact contiguous detached CPU tensor required')
    require(bool(torch.isfinite(value).all()), name + ': finite values required')


def _bytes_equal(torch, actual, expected, name):
    require(actual.dtype == expected.dtype and tuple(actual.shape) == tuple(expected.shape)
            and actual.device.type == expected.device.type == 'cpu',
            name + ': dtype/shape/device mismatch')
    require(torch.equal(actual.contiguous().view(torch.uint8),
                        expected.contiguous().view(torch.uint8)),
            name + ': byte mismatch (no numeric tolerance permitted)')


def capture_image(torch, np, cv2, modules, model, dataset, coco, spec, context,
                  packet, row, index, coordinate_module, old_cache_body):
    """Return (row, payload); ROOT alone binds refs and grants qualification.

    modules contains pref/old/diag/joint/preflight/cache7. The tensor payload is
    extra_F[E,256], extra_flat_indices[E], neighbor_to_combined[U,9]. Old center
    F, samples, targets and assignments stay in the referenced N16 packet.
    """
    started = time.monotonic()
    require(isinstance(modules, dict), 'Explicit ROOT-qualified named modules required')
    pref, old, diag, joint, preflight, cache7 = (
        modules[k] for k in ('pref', 'old', 'diag', 'joint', 'preflight', 'cache7'))
    require(type(index) is int and index == packet['dataset_index'] == row['dataset_index']
            and packet['identity'] == row['identity'], 'Exact qualified FIT row/packet/index')
    require(packet['teacher_checkpoint'] == context['checkpointmap']['native0'],
            'Actual saved N16 outer teacher reference must match frozen native0')
    matching_seed = packet['matching_loss_seed']
    require(type(matching_seed) is int and matching_seed == row['matching_loss_seed'],
            'Replay old matching seed; do not define a new matching recipe')
    sample = packet['samples']
    require(sample['seed'] == row['point_seed'] and sample['replacement'] is True
            and sample['duplicates_preserved'] is True,
            'Reuse old private sample seed/order/replacement metadata')
    identity = packet['identity']
    H, W = map(int, identity['original_shape'])
    require(H > 0 and W > 0, 'Original positive spatial dimensions required')
    unique = packet['unique_flat_indices']
    require(isinstance(unique, torch.Tensor) and unique.ndim == 1,
            'Old unique physical center vector required')
    U = len(unique)
    flats, sq, inverse = sample['flat_indices'], sample['query_ids'], packet['sample_to_unique']
    require(isinstance(flats, torch.Tensor) and flats.ndim == 1,
            'Old flat sample vector required')
    N = len(flats)
    _cpu_tensor(torch, unique, torch.int64, (U,), 'Old unique centers')
    _cpu_tensor(torch, flats, torch.int64, (N,), 'Old sampled flats')
    _cpu_tensor(torch, sq, torch.int64, (N,), 'Old sampled query IDs')
    _cpu_tensor(torch, inverse, torch.int64, (N,), 'Old sample-to-unique')
    old_F = packet['sampled_F_unique']
    _cpu_tensor(torch, old_F, torch.float32, (U, 256), 'Old sampled F centers')
    require(N == 4096 and row['points'] == N and row['unique_sampled_F_points'] == U
            and row['natural_queries'] == 100 and packet['natural_query_ids'] == list(range(100)),
            'Old complete sample population/natural-query identity')
    require(bool(((flats >= 0) & (flats < H * W)).all())
            and bool(((sq >= 0) & (sq < 100)).all()), 'Original sample coordinates/range')
    actual_unique, actual_inverse = torch.unique(flats, sorted=True, return_inverse=True)
    require(torch.equal(unique, actual_unique) and torch.equal(inverse, actual_inverse),
            'Exact old unique order/inverse including repeated samples')

    # ROOT-confirmed dictionary ABI; no tuple or alternate return is accepted.
    coordinates = coordinate_module.build_coordinates(np, unique.numpy(), (H, W))
    require(isinstance(coordinates, dict)
            and set(coordinates) == {'extra_flat_indices', 'neighbor_to_combined'},
            'Coordinate module must return the confirmed two-key mapping')
    extra_np, map_np = coordinates['extra_flat_indices'], coordinates['neighbor_to_combined']
    require(isinstance(extra_np, np.ndarray) and extra_np.dtype == np.dtype('int64')
            and extra_np.ndim == 1 and isinstance(map_np, np.ndarray)
            and map_np.dtype == np.dtype('int64') and map_np.shape == (U, 9),
            'Physical extra/map NumPy INT64 ABI')
    E = len(extra_np)
    require(np.all((extra_np >= 0) & (extra_np < H * W))
            and (E < 2 or np.all(extra_np[1:] > extra_np[:-1]))
            and not np.intersect1d(unique.numpy(), extra_np).size,
            'Extra coordinates must be sorted distinct and exclude old centers')
    require(np.all((map_np >= 0) & (map_np < U + E))
            and np.array_equal(map_np[:, 4], np.arange(U, dtype=np.int64)),
            'Combined mapping bounds and exact center identity column4')
    extra = torch.from_numpy(extra_np.copy(order='C')).contiguous()
    neighbor_map = torch.from_numpy(map_np.copy(order='C')).contiguous()
    combined_indices = torch.cat((unique, extra)).contiguous()
    # Validate the actual mapping against original-grid replicate coordinates;
    # ROOT's independent CPU qualifier must derive this separately as well.
    y, x = unique.numpy() // W, unique.numpy() % W
    expected_neighbors = np.stack([
        np.clip(y + dy, 0, H - 1) * W + np.clip(x + dx, 0, W - 1)
        for dy in (-1, 0, 1) for dx in (-1, 0, 1)], axis=1)
    require(np.array_equal(combined_indices.numpy()[map_np], expected_neighbors),
            'Exact NW/N/NE/W/C/E/SW/S/SE original-grid replicate mapping')
    require(np.array_equal(np.setdiff1d(np.unique(expected_neighbors), unique.numpy()), extra_np),
            'Extra vector is exactly neighbor union minus qualified old centers')
    extra_F = torch.empty((E, 256), dtype=torch.float32, device='cpu')
    _frozen(model)
    outside = pref.rng_snapshot(torch)
    forwards, renderer_calls = 0, 0
    try:
        with torch.no_grad(), torch.random.fork_rng(devices=[0]):
            info = dataset.get_data_info(index)
            data, stripped, gray800, gray, observed_identity = old.load_image(model, dataset, coco, info)
            del stripped, gray800, gray
            observed_identity['dataset_index'] = index
            require(observed_identity == identity, 'Original parser/image/GT identity changed')
            processed = pref.check_targets(torch, data, observed_identity)
            require(pref.tree_equal(torch, processed, packet['processed_target_proof']),
                    'Exact original processed target proof')
            raw_GT, mapping, validator = preflight.raw_GT_validator(
                torch, np, cv2, coco, data, observed_identity)
            require(mapping == packet['raw_target_mapping']
                    and identity['ann_ids'] == packet['raw_GT_ids']
                    and pref.digest_tensor(raw_GT) == packet['raw_GT_digest'],
                    'Exact old raw GT IDs/mapping/digest; no new assignment')
            head = model.panoptic_head
            torch.random.default_generator.manual_seed(matching_seed)
            torch.cuda.manual_seed(matching_seed)
            with joint._observe_processed_GT(torch, head, identity['ann_ids'], raw_GT,
                                            mapping, identity, validator) as gt_observation:
                with old_cache_body.observe_all100_targets(
                        torch, head, old.digest_tensor) as (matches, targets):
                    with cache7.observed_fields(torch, head) as fields:
                        losses = model.loss(data['inputs'], data['data_samples'])
                        forwards += 1
            require(len(losses) == 30 and set(losses) == set(old.ATOMIC_KEYS)
                    == set(packet['native_loss_atoms']), 'Exact thirty original native loss atom keys')
            for value in losses.values():
                atoms = list(value) if isinstance(value, (tuple, list)) else [value]
                require(atoms and all(not atom.requires_grad and bool(torch.isfinite(atom).all())
                                      for atom in atoms), 'Frozen finite original native loss atoms')
            require(matches == packet['native_stage_matches'] and len(matches) == 10
                    and pref.tree_equal(torch, targets, packet['all_ten_target_observations']),
                    'Exact original ten matching stages and observed target metadata')
            require(pref.tree_equal(torch, gt_observation,
                                    packet['actual_raw_processed_GT_correspondence']),
                    'Exact old raw/processed GT observation; no relabeling')
            F, e = fields['F'][0], fields['e'][-1]
            classes, mask_stages = fields['output'][0]
            native_class = classes[-1]
            require(F.device.type == 'cuda' and F.device.index == 0 and F.dtype == torch.float32
                    and F.ndim == 4 and F.shape[0] == 1 and F.shape[1] == 256,
                    'Actual native CUDA0 F256 feature carrier required')
            _bytes_equal(torch, e.detach().cpu(), packet['native_e0'], 'All100 native e')
            _bytes_equal(torch, native_class.detach().cpu(), packet['native_class_logits'],
                         'All100 native class logits')
            device_indices = combined_indices.to(F.device)
            if U + E:
                for first in range(0, 256, 16):
                    original = diag.render_original(torch, F[:, first:first + 16], identity)
                    renderer_calls += 1
                    require(original.device.type == 'cuda' and original.device.index == 0
                            and original.dtype == torch.float32 and tuple(original.shape) == (16, H, W)
                            and bool(torch.isfinite(original).all()),
                            'Original channel16 two-interpolation GPU renderer ABI')
                    gathered = original.flatten(1).index_select(1, device_indices).T.detach().cpu().contiguous()
                    _bytes_equal(torch, gathered[:U], old_F[:, first:first + 16],
                                 'Actual re-rendered old F center block')
                    extra_F[:, first:first + 16].copy_(gathered[U:])
                    del original, gathered
            require(forwards == 1 and renderer_calls == (16 if U + E else 0),
                    'One native loss forward and one renderer per channel16 block')
    finally:
        require(pref.tree_equal(torch, outside, pref.rng_snapshot(torch)),
                'External CPU/CUDA RNG must be restored even on failed capture')
        _frozen(model)
    _cpu_tensor(torch, extra_F, torch.float32, (E, 256), 'Actual GPU extra F readback')
    _cpu_tensor(torch, extra, torch.int64, (E,), 'Extra flat coordinates')
    _cpu_tensor(torch, neighbor_map, torch.int64, (U, 9), 'Neighbor-to-combined map')
    input_digests = {name: old.digest_tensor(value) for name, value in (
        ('unique_flat_indices', unique), ('sample_to_unique', inverse),
        ('sample_flat_indices', flats), ('sample_query_ids', sq))}
    payload = dict(schema='N20-true-F-neighbor-capture-body-v1', dataset_index=index,
        identity=identity, teacher_checkpoint=packet['teacher_checkpoint'],
        matching_loss_seed=matching_seed, point_seed=sample['seed'],
        original_unique_count=U, extra_count=E, sampled_points=N,
        input_index_digests=input_digests, center_column=4,
        neighbor_order=['NW', 'N', 'NE', 'W', 'C', 'E', 'SW', 'S', 'SE'],
        combined_order='old unique order then sorted extra union-minus-centers',
        extra_F=extra_F, extra_flat_indices=extra, neighbor_to_combined=neighbor_map)
    output_row = dict(dataset_index=index, image_id=identity['image_id'], identity=identity,
        natural_queries=100, points=N, unique_centers=U, original_unique_count=U,
        raw_GT=len(raw_GT), extra_count=E,
        matching_loss_seed=matching_seed, point_seed=sample['seed'],
        one_teacher_loss_forward=True, all_ten_matching_stages=10,
        old_e_class_targets_GT_exact=True, old_F_centers_byte_exact=True,
        original_samples_and_order_reused=True, original_global_RNG_restored=True,
        actual_GPU_renderer_channel16_calls=renderer_calls,
        extra_feature_bytes=E * 256 * 4,
        raw_tensor_payload_bytes=E * 256 * 4 + E * 8 + U * 9 * 8,
        no_resampling=True, no_new_labels=True, no_PR_forward=True,
        no_training=True, no_AP=True, no_DEV=True,
        CPU_renderer_equivalence_claimed=False, CPU_qualification_pending=True,
        actual_source_terminal_qualified=False,
        elapsed_seconds=time.monotonic() - started)
    return output_row, payload
