"""UNBOUND DEV extra-F capture body; no IO/front/teacher/PR/scout or CLI.

ROOT supplies closed original native packet and N16 shared member. modules has
n10/reader/renderer; packet_gate is the OLD packet_check gate (in particular its
report is the original prior report, never an N20 report). It must already bind
the original packet/source/teacher lineage. Native packet layout is intentionally
opaque here: only packet_check's existing fields/identity/qids return ABI is used.
No new matching, sampling, labels, seeds, selector or AP computation is created.
"""

import hashlib
import time


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def _parent_metadata(row, index, native_packet_ref, shared):
    """Pure metadata association; ROOT separately authenticates physical refs."""
    require(type(index) is int and index == row['dataset_index'] == shared['dataset_index']
            and row['image_id'] == shared['image_id']
            and row['identity'] == shared['identity'], 'Exact original whole-image identity/index')
    require(shared['schema'] == 'N16-actual-GPU-image-shared-inputs-v1'
            and shared['actual_GPU_F_captured'] is True and shared['CPU_F_rerendered'] is False
            and shared['native_packet'] == row['native_packet'] == native_packet_ref
            and shared['selector'] == row['selector']
            and shared['natural_query_ids'] == list(range(100))
            and shared['native_export_query_ids'] == row['qids']
            and sorted(row['qids']) == list(range(100)), 'Original actual-GPU natural100 carrier/order')
    for ref in (row['bundle'], native_packet_ref):
        require(type(ref) is dict and set(ref) == {'path', 'bytes', 'sha256'},
                'Parent bundle/native packet must be real strict physical refs, not member refs')
    member = row['shared_actual_image_inputs']
    require(type(member) is str and member == 'i%03d/SHARED_ACTUAL_INPUTS.pth' % index,
            'Original shared member is an internal string ID in its physical parent bundle')
    return dict(parent_bundle=row['bundle'], parent_shared_member=member,
                native_packet=native_packet_ref, identity=shared['identity'])


def _cpu_tensor(torch, value, dtype, shape, label):
    require(isinstance(value, torch.Tensor) and value.layout == torch.strided
            and value.device.type == 'cpu' and value.dtype == dtype
            and tuple(value.shape) == tuple(shape) and value.is_contiguous()
            and not value.requires_grad and bool(torch.isfinite(value).all()),
            label + ': finite contiguous detached CPU tensor required')


def _same_bytes(torch, left, right, label):
    require(left.dtype == right.dtype and tuple(left.shape) == tuple(right.shape)
            and left.device.type == right.device.type == 'cpu'
            and torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)),
            label + ': actual input bytes differ; no numeric tolerance')


def capture_image(torch, np, modules, packet, row, index, native_packet_ref,
                  shared, packet_gate, coordinates, *, coco, packet_row, check=None):
    """Return (small row, extra-only payload); ROOT binds source/ad/report refs.

    combined=[old shared F_unique in original order, extra_F in sorted extra-flat
    order]. The only new tensors are extra_F[E,256], extra_flat_indices[E] and
    neighbor_to_combined[U,9]. Centers retain the OLD captured bytes for use.
    coco is the original closed DEV COCO object required by packet_check; pass
    it explicitly, without inventing a coco member in the old packet gate.
    packet_row is the original prior image_rows entry for packet_check; row
    remains the N16 manifest row. ROOT binds both to the same original packet.
    """
    started = time.monotonic()
    meta = _parent_metadata(row, index, native_packet_ref, shared)
    require(type(modules) is dict and all(k in modules for k in ('n10', 'reader', 'renderer'))
            and (check is None or callable(check)), 'Explicit original modules and optional resource callback')
    require(type(packet_row) is dict, 'Original prior packet-check row is separate from the N16 manifest row')
    n10, reader, renderer = (modules[k] for k in ('n10', 'reader', 'renderer'))
    require('report' in packet_gate, 'Old n10 packet gate must retain its original prior report')
    require(torch.cuda.is_initialized() and torch.cuda.current_device() == 0,
            'Outer admitted actual CUDA0 before invoking this numerical body')
    if check is not None:
        check()
    identity = shared['identity']
    H, W = map(int, identity['original_shape'])
    require(H > 0 and W > 0, 'Positive original raster dimensions')
    union, old_F = shared['union_flat_indices'], shared['F_unique']
    U = len(union)
    _cpu_tensor(torch, union, torch.int64, (U,), 'Original union flat indices')
    _cpu_tensor(torch, old_F, torch.float32, (U, 256), 'Original actual GPU union F')
    _cpu_tensor(torch, shared['e0'], torch.float32, (100, 256), 'Original e100')
    _cpu_tensor(torch, shared['native_L'], torch.float32, (100, H, W), 'Original full natural100 L')
    require(shared['U'] == U and bool(((union >= 0) & (union < H * W)).all())
            and bool((union[1:] > union[:-1]).all()), 'Original sorted unique union bounds/order')
    require(len(shared['C_indices']) == len(shared['C_counts']) == len(shared['K_counts']) == 100,
            'Retain every natural query C/K')
    merged = np.empty(0, dtype=np.int64)
    for q in range(100):
        indices = shared['C_indices'][q]
        _cpu_tensor(torch, indices, torch.int64, (shared['C_counts'][q],), 'Original query C indices')
        require(bool(((indices >= 0) & (indices < H * W)).all())
                and bool((indices[1:] > indices[:-1]).all())
                and 0 <= shared['K_counts'][q] <= len(indices), 'Physical original C/K bounds')
        merged = np.union1d(merged, indices.numpy())
    require(np.array_equal(merged, union.numpy()), 'Shared F coverage is exactly all100 C union')
    coord = coordinates.build_coordinates(np, union.numpy(), (H, W))
    require(type(coord) is dict and set(coord) == {'extra_flat_indices', 'neighbor_to_combined'},
            'Frozen N20 coordinate mapping ABI')
    extra_np, map_np = coord['extra_flat_indices'], coord['neighbor_to_combined']
    require(extra_np.dtype == map_np.dtype == np.int64 and extra_np.ndim == 1
            and map_np.shape == (U, 9) and len(extra_np) <= min(8 * U, H * W - U)
            and np.array_equal(map_np[:, 4], np.arange(U, dtype=np.int64)), 'Exact N20 nine-neighbor/center mapping')
    E = len(extra_np)
    extra = torch.from_numpy(extra_np.copy(order='C')).contiguous()
    neighbor_map = torch.from_numpy(map_np.copy(order='C')).contiguous()
    extra_F = torch.empty((E, 256), dtype=torch.float32, device='cpu')
    outside_CPU = torch.get_rng_state().clone()
    outside_CUDA = torch.cuda.get_rng_state(0).cpu().clone()
    with torch.no_grad():
        fields, observed_identity, qids = n10.packet_check(
            torch, reader, coco, packet, packet_row, index, native_packet_ref, packet_gate)
        require(observed_identity == identity and qids == shared['native_export_query_ids'],
                'Original packet hook identity and native export order must remain exact')
        F = fields['F256'].to('cuda:0')
        e, M = fields['e0'].to('cuda:0'), fields['M200'].to('cuda:0')
        require(F.dtype == e.dtype == M.dtype == torch.float32 and F.ndim == 4
                and F.shape[0] == 1 and F.shape[1] == 256 and min(F.shape[2:]) > 0
                and tuple(e.shape) == (1, 100, 256)
                and torch.equal(torch.einsum('bqc,bchw->bqhw', e, F), M),
                'Original source F256/e100/M=eF packet hook; no guessed native spatial size')
        _same_bytes(torch, e[0].detach().cpu(), shared['e0'], 'Original all100 e')
        L = renderer.render_original(torch, M, identity)
        _same_bytes(torch, L.detach().cpu(), shared['native_L'], 'Original all100 native L')
        del L, e, M
        # Preserve the existing two-stage/channel16 original_feature implementation;
        # do not replace it with a direct point_sample or a CPU renderer.
        original = n10.original_feature(torch, renderer, F, identity)
        require(original.dtype == torch.float32 and original.device.type == 'cuda'
                and original.device.index == 0 and tuple(original.shape) == (256, H, W),
                'Actual original GPU feature raster from the existing qualified hook')
        for first in range(0, U, 2048):
            if check is not None:
                check()
            idx = union[first:first + 2048].to(original.device)
            observed = original[:, idx // W, idx % W].T.detach().cpu().contiguous()
            _same_bytes(torch, observed, old_F[first:first + len(idx)], 'Old actual union F center block')
            del idx, observed
        for first in range(0, E, 2048):
            if check is not None:
                check()
            idx = extra[first:first + 2048].to(original.device)
            extra_F[first:first + len(idx)].copy_(original[:, idx // W, idx % W].T.detach().cpu())
            del idx
        del original, F, fields
    require(torch.equal(outside_CPU, torch.get_rng_state())
            and torch.equal(outside_CUDA, torch.cuda.get_rng_state(0).cpu()), 'External CPU/CUDA RNG and seeds unchanged')
    _cpu_tensor(torch, extra_F, torch.float32, (E, 256), 'Actual GPU extra F readback')
    payload = dict(schema='N20-DEV-true-feature-neighbor-extra-capture-body-v1', **meta,
        dataset_index=index, image_id=row['image_id'], selector=shared['selector'],
        natural_query_ids=list(range(100)), native_export_query_ids=list(qids),
        original_unique_count=U, extra_count=E, center_column=4,
        neighbor_order=['NW', 'N', 'NE', 'W', 'C', 'E', 'SW', 'S', 'SE'],
        combined_order='old shared union order then sorted extra union-minus-centers',
        original_union_flat_sha256=hashlib.sha256(union.numpy().tobytes(order='C')).hexdigest(),
        extra_F=extra_F, extra_flat_indices=extra, neighbor_to_combined=neighbor_map)
    result_row = dict(dataset_index=index, image_id=row['image_id'], identity=identity,
        parent_bundle=row['bundle'], parent_shared_member=row['shared_actual_image_inputs'],
        native_packet=native_packet_ref, natural_queries=100, candidate_points_all100=sum(shared['C_counts']),
        original_unique_count=U, extra_count=E, extra_feature_bytes=E * 256 * 4,
        raw_tensor_payload_bytes=E * 1032 + U * 9 * 8, old_F_centers_byte_exact=True,
        old_native_L_e_byte_exact=True, original_packet_identity_order_exact=True,
        external_CPU_CUDA_RNG_unchanged=True, seeds_reseeded=False,
        teacher_forward_calls=0, PR_forward_calls=0, scout_forward_calls=0,
        matching_or_sampling_executed=False, selector_modified=False, AP_evaluated=False,
        CPU_renderer_used=False, ROOT_source_terminal_qualified=False,
        elapsed_seconds=time.monotonic() - started)
    return result_row, payload
