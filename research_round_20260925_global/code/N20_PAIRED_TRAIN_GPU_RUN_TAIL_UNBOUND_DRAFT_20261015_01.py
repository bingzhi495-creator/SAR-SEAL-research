"""UNBOUND N20 GPU training tail; concatenate after ROOT's qualified front.

No admission front or CLI is supplied here, and importing this file does not run
training. ROOT supplies source_gate(path, spec), require/read/closed/bound/load/
reference/write/api. __file__ must be the final frozen concatenated entry.

Required gate fields:
  closure, source, spec_ref, admission_ref, admission, common, lineage, parent;
  parent_cache_report (decoded qualified original N16 report),
  parent_cache_report_ref (its strict physical ref),
  neighbor_report (decoded independently qualified N20 GPU capture report),
  neighbor_report_ref (its strict physical ref), protocol (decoded N20 lock),
  training_binding_refs (ROOT-validated, nonempty name -> strict physical ref
  mapping for completed neighbor GPU/CPU qualification and ROOT result).
The front must establish all actual owners, CPU qualification, complete741 rows,
prefix/pins, full Source closure and exact lock equality before returning.

Required spec fields used by this tail:
  output_directory, phase, entry_source, GPU_admission_file, sources;
  normal_RAW_count, total_failed_RAW_count, total_terminal_identity_pin_count;
  deadline_seconds=1800, automatic_retry=False, allocator_limit_MiB=4096,
  whole_GPU_limit_MiB=8192, maximum_process_RSS_bytes=16GiB;
  initialization_seed=202610150201, expected_N20_initialization_digest;
  scout_module_source (N20 create_head), base_scout_source (N16 module),
  paired_training_body_source (N20), original_paired_training_body_source (N16),
  N20_locked_protocol (new design), locked_protocol (unchanged N16 cache lock).
All *_source/protocol values above are strict physical refs already in closure.
No N17 checkpoint, four-tensor shape, initialization digest or schema is reused.
"""

import gc
import hashlib
import json
import os
from pathlib import Path
import time


N20_TRAIN_ARMS = ('LOCAL_ERROR', 'CENTER_ERROR')
N20_TRAIN_SEED = 202610150201
N20_TRAIN_ORDER_SEED = 202610140160001
N20_TRAIN_THREAD_ENV = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                        'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
N20_TRAIN_REPORT_SCHEMA = 'N20-paired-local-center-feature-error-training-GPU-report-v1'
N20_TRAIN_STATUS = 'N20_FULLFIT_PAIRED_FEATURE_ERROR_E8_E24_COMPLETE_PENDING_INDEPENDENT_CPU'


def n20_training_state_digest(torch, state):
    """Same sorted name/shape/dtype header + raw FP32 bytes algorithm as N17."""
    whole = hashlib.sha256()
    tensors = []
    for name, tensor in sorted(state.items()):
        require(isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32
                and bool(torch.isfinite(tensor).all()), 'Finite FP32 state to hash')
        info = dict(name=name, shape=list(tensor.shape), dtype=str(tensor.dtype))
        header = json.dumps(info, sort_keys=True, separators=(',', ':')).encode('utf8')
        raw = tensor.detach().cpu().contiguous().numpy().tobytes(order='C')
        whole.update(len(header).to_bytes(8, 'little'))
        whole.update(header)
        whole.update(raw)
        tensors.append(dict(info, raw_bytes_sha256=hashlib.sha256(raw).hexdigest()))
    return whole.hexdigest(), tensors


def n20_verify_training_initialization(torch, state, expected):
    require(type(expected) is str and len(expected) == 64
            and all(c in '0123456789abcdef' for c in expected),
            'ROOT must prelock the new N20 whole-state SHA256')
    shapes = {'spatial_weight': (9, 256), 'hidden.weight': (32, 513),
              'hidden.bias': (32,), 'output.weight': (3, 32), 'output.bias': (3,)}
    require(state['initialization_seed'] == N20_TRAIN_SEED
            and state['order_seed'] == N20_TRAIN_ORDER_SEED
            and state['completed_epoch'] == 0 and set(state['heads']) == set(N20_TRAIN_ARMS)
            and state['updates'] == {arm: 0 for arm in N20_TRAIN_ARMS},
            'First N20 private initialization and original shared order only')
    proofs = {}
    for arm in N20_TRAIN_ARMS:
        head = state['heads'][arm]
        values = head.state_dict()
        require(set(values) == set(shapes) and len(tuple(head.parameters())) == 5
                and sum(p.numel() for p in head.parameters()) == 18851
                and not tuple(head.buffers())
                and all(tuple(values[k].shape) == shape for k, shape in shapes.items())
                and bool((values['spatial_weight'] == 0).all()),
                'Five-tensor18851 N20 adapter has zero spatial initialization')
        digest, tensors = n20_training_state_digest(torch, values)
        require(digest == expected, 'Actual N20 initial state differs from prelocked digest')
        proofs[arm] = dict(combined_digest=digest, tensors=tensors)
    left, right = (state['heads'][arm] for arm in N20_TRAIN_ARMS)
    require(all(torch.equal(value, right.state_dict()[key])
                for key, value in left.state_dict().items())
            and all(a.data_ptr() != b.data_ptr()
                    for a, b in zip(left.parameters(), right.parameters())),
            'Paired N20 states must be equal and own independent storage')
    state['expected_first_initialization_digest'] = expected
    return proofs


def n20_load_training_image(torch, index, gate, spec, check):
    """Read two qualified sparse packets; do not open the full native-L carrier."""
    check()
    closure = gate['closure']
    old_report, report = gate['parent_cache_report'], gate['neighbor_report']
    old_row, row = old_report['rows'][index], report['rows'][index]
    old_ref, neighbor_ref = old_report['packet_refs'][index], report['packet_refs'][index]
    require(type(index) is int and index == old_row['dataset_index'] == row['dataset_index']
            and old_row['packet'] == old_ref and row['packet'] == neighbor_ref
            and row['parent_packet'] == old_ref and row['image_id'] == old_row['image_id']
            and row['identity'] == old_row['identity']
            and row['points'] == old_row['points'] == 4096
            and row['unique_centers'] == old_row['unique_sampled_F_points'],
            'Qualified sparse neighbor/old packet rows must bind one exact FIT image')
    packet = torch.load(str(closed(closure, old_ref)), map_location='cpu')
    neighbor = torch.load(str(closed(closure, neighbor_ref)), map_location='cpu')
    require(type(packet) is dict and packet['dataset_index'] == index
            and packet['identity'] == old_row['identity']
            and packet['protocol'] == spec['locked_protocol'],
            'Original qualified thin packet identity and N16 protocol must remain exact')
    for key in ('source', 'spec', 'GPU_admission', 'phase',
                'fixed_PR_checkpoint', 'teacher_checkpoint'):
        require(packet[key] == old_report[key], 'Original packet/report binding differs: ' + key)
    require(type(neighbor) is dict and neighbor['schema'] == 'N20-true-F-neighbor-capture-body-v1'
            and neighbor['dataset_index'] == index and neighbor['identity'] == packet['identity']
            and neighbor['parent_packet'] == old_ref
            and neighbor['protocol'] == spec['N20_locked_protocol']
            and neighbor['teacher_checkpoint'] == packet['teacher_checkpoint']
            and neighbor['matching_loss_seed'] == packet['matching_loss_seed'] == row['matching_loss_seed']
            and neighbor['point_seed'] == packet['samples']['seed'] == row['point_seed']
            and neighbor['original_unique_count'] == row['unique_centers']
            and neighbor['extra_count'] == row['extra_count']
            and neighbor['sampled_points'] == 4096 and neighbor['center_column'] == 4,
            'Actual qualified N20 extra-F payload must bind old sample order and protocol')
    for key in ('source', 'spec', 'GPU_admission', 'phase'):
        require(neighbor[key] == report[key], 'N20 capture packet/report binding differs: ' + key)
    sample = packet['samples']
    query, flat = sample['query_ids'], sample['flat_indices']
    H, W = map(int, packet['identity']['original_shape'])
    require(query.device.type == flat.device.type == 'cpu'
            and query.dtype == flat.dtype == torch.int64
            and tuple(query.shape) == tuple(flat.shape) == (4096,)
            and bool(((query >= 0) & (query < 100)).all())
            and bool(((flat >= 0) & (flat < H * W)).all()),
            'Exactly original4096 natural-query/point slots, including duplicates')
    # The body validates sparse tensor shapes/ranges and gathers the original
    # sample_to_unique order. All other payload metadata remains outside its ABI.
    loaded = dict(packet=packet, neighbor_packet={
        'extra_F': neighbor['extra_F'],
        'neighbor_to_combined': neighbor['neighbor_to_combined']})
    check()
    return loaded


def run(spec_path):
    started = time.monotonic()
    path = Path(spec_path).resolve(strict=True)
    require(Path(spec_path).is_absolute(), 'Absolute N20 training specification required')
    spec = read(path)
    gate = source_gate(path, spec)
    closure = gate['closure']
    output = Path(spec['output_directory'])
    require(output.is_absolute() and not output.exists() and not output.is_symlink()
            and output.parent.is_dir(), 'Fresh owned N20 paired training output required')
    require(spec['initialization_seed'] == N20_TRAIN_SEED
            and spec['deadline_seconds'] == 1800 and spec['automatic_retry'] is False
            and spec['allocator_limit_MiB'] == 4096 and spec['whole_GPU_limit_MiB'] == 8192
            and spec['maximum_process_RSS_bytes'] == 16 * 1024**3,
            'N20 first-seed fixed resource recipe; no retry')
    required_refs = ('scout_module_source', 'base_scout_source', 'paired_training_body_source',
                     'original_paired_training_body_source', 'N20_locked_protocol', 'locked_protocol')
    for key in required_refs:
        closed(closure, spec[key])
    for report_key in ('parent_cache_report', 'neighbor_report'):
        ref = gate[report_key + '_ref']
        require(read(closed(closure, ref)) == gate[report_key],
                'Decoded cache report must equal its closed physical JSON: ' + report_key)
        report = gate[report_key]
        require(report['images'] == 741 and report['sampled_points'] == 3035136
                and len(report['rows']) == len(report['packet_refs']) == 741,
                'Both qualified cache reports must contain all FIT741 sample packets')
    bindings = gate['training_binding_refs']
    require(type(bindings) is dict and bool(bindings)
            and not set(bindings).intersection({'source', 'spec', 'GPU_admission', 'phase',
                'N20_locked_protocol', 'locked_protocol', 'parent_cache_report', 'neighbor_report',
                'scout_module_source', 'base_scout_source', 'paired_training_body_source',
                'original_paired_training_body_source', 'initialization_seed', 'order_seed',
                'expected_N20_initialization_digest', 'frozen_teacher_checkpoint', 'frozen_PR_checkpoint'}),
            'Explicit completed-cache qualification refs must not replace own training binding')
    for ref in bindings.values():
        closed(closure, ref)
    import psutil
    process = psutil.Process()
    require(process.cpu_affinity() == [0, 1] and os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
            and all(os.environ.get(key) == '2' for key in N20_TRAIN_THREAD_ENV),
            'GPU0/CPU01/two-thread actual own environment')
    import torch
    require(torch.__version__ == '2.0.1+cu118' and not torch.cuda.is_initialized(),
            'Original Torch, cold CUDA after complete source/admission gates required')
    import numpy as np
    import cv2
    from scipy import ndimage
    from pycocotools import mask as masks
    from pycocotools.coco import COCO
    require(np.__version__ == '1.23.5' and cv2.__version__ == '4.8.0',
            'Closed original numerical libraries required')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    cv2.setNumThreads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    scout = load(spec['scout_module_source'], '_N20_closed_five_tensor_feature_adapter')
    base = load(spec['base_scout_source'], '_N20_closed_original_scout_factory')
    body = load(spec['paired_training_body_source'], '_N20_closed_paired_feature_error_body')
    old_body = load(spec['original_paired_training_body_source'], '_N20_closed_original_paired_body')
    require(tuple(body.ARMS) == N20_TRAIN_ARMS and body.INIT_SEED == N20_TRAIN_SEED
            and body.ORDER_SEED == N20_TRAIN_ORDER_SEED and body.EPOCHS == 24
            and tuple(body.ENDPOINTS) == (8, 24), 'Exact N20 paired fixed24 body contract')
    libraries = gate['common'].imported_files(
        torch, np, cv2, masks, COCO, closure, gate['admission'], ndimage)
    require(not torch.cuda.is_initialized(), 'Actual imports and source-qualified modules precede CUDA')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            'Exactly one admitted CUDA device')
    torch.cuda.set_device(0)
    torch.cuda.init()
    torch.cuda.set_per_process_memory_fraction(
        min(1., 4096 * 1024**2 / torch.cuda.get_device_properties(0).total_memory), 0)
    torch.cuda.reset_peak_memory_stats(0)
    outside_CPU = torch.get_rng_state().clone()
    outside_CUDA = torch.cuda.get_rng_state(0).cpu().clone()

    def check():
        require(time.monotonic() - started < 1800 and process.memory_info().rss < 16 * 1024**3
                and psutil.virtual_memory().available >= 16 * 1024**3
                and process.cpu_affinity() == [0, 1] and os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
                and all(os.environ.get(key) == '2' for key in N20_TRAIN_THREAD_ENV)
                and torch.get_num_threads() == torch.get_num_interop_threads() == 2
                and torch.cuda.current_device() == 0 and torch.cuda.memory_reserved(0) <= 4096 * 1024**2,
                'N20 bounded resources/deadline/environment; no retry')

    check()
    state = api(body, 'make_paired_state')(
        torch, scout, base, old_body, device='cuda:0', seed=N20_TRAIN_SEED)
    initial = n20_verify_training_initialization(
        torch, state, spec['expected_N20_initialization_digest'])
    require(torch.equal(outside_CPU, torch.get_rng_state())
            and torch.equal(outside_CUDA, torch.cuda.get_rng_state(0).cpu()),
            'Private paired constructor must restore outside CPU/CUDA RNG')
    old_report = gate['parent_cache_report']
    binding = dict(source=gate['source'], spec=gate['spec_ref'], GPU_admission=gate['admission_ref'],
        phase=spec['phase'], N20_locked_protocol=spec['N20_locked_protocol'], locked_protocol=spec['locked_protocol'],
        parent_cache_report=gate['parent_cache_report_ref'], neighbor_report=gate['neighbor_report_ref'],
        scout_module_source=spec['scout_module_source'], base_scout_source=spec['base_scout_source'],
        paired_training_body_source=spec['paired_training_body_source'],
        original_paired_training_body_source=spec['original_paired_training_body_source'],
        initialization_seed=N20_TRAIN_SEED, order_seed=N20_TRAIN_ORDER_SEED,
        expected_N20_initialization_digest=spec['expected_N20_initialization_digest'],
        frozen_teacher_checkpoint=old_report['teacher_checkpoint'], frozen_PR_checkpoint=old_report['fixed_PR_checkpoint'])
    binding.update(bindings)
    output.mkdir()
    write(output / 'START.json', dict(schema='N20-paired-feature-error-training-start-v1', **binding,
        initial_states=initial, parameters_per_arm=18851, tensors_per_arm=5, arms=list(N20_TRAIN_ARMS),
        teacher_or_PR_loaded=False, full_native_L_loaded=False, fixed_epochs=24,
        updates_each_arm=17784, point_exposure_each_arm=72843264, AMP=False,
        DEV_AP_evaluated=False, qualified=False))
    trace_path = output / 'UPDATE_TRACE.jsonl'
    checkpoints, endpoint_files = {}, {}
    loads = 0
    with trace_path.open('x', encoding='utf8', newline='\n') as trace:
        def observer(event):
            check()
            if event['event'] == 'paired_image_update':
                trace.write(json.dumps(event, separators=(',', ':'), allow_nan=False) + '\n')
                if event['dataset_index'] % 32 == 0:
                    trace.flush()

        def loader(index):
            nonlocal loads
            loaded = n20_load_training_image(torch, index, gate, spec, check)
            loads += 1
            return loaded

        def checkpoint(epoch, endpoint, snapshot):
            check()
            require(epoch in (8, 24) and str(epoch) not in checkpoints
                    and snapshot['completed_epoch'] == epoch and snapshot['parameters_per_arm'] == 18851
                    and set(snapshot['models']) == set(snapshot['optimizers']) == set(N20_TRAIN_ARMS)
                    and snapshot['expected_first_initialization_digest'] == spec['expected_N20_initialization_digest'],
                    'Only complete N20 five-tensor E8/E24 snapshots')
            snapshot = dict(snapshot, metadata=dict(binding, endpoint_epoch=epoch),
                external_CPU_RNG=outside_CPU.clone(), external_CUDA_RNG=outside_CUDA.clone())
            filename = output / ('E%02d_COMPLETE_PAIRED_SNAPSHOT.pth' % epoch)
            with filename.open('xb') as stream:
                torch.save(snapshot, stream)
            checkpoints[str(epoch)] = reference(filename)
            endpoint_path = output / ('E%02d_FIT_ENDPOINT.json' % epoch)
            write(endpoint_path, dict(binding, endpoint=endpoint))
            endpoint_files[str(epoch)] = reference(endpoint_path)
            check()

        result = api(body, 'train_until')(torch, state, loader, last_epoch=24,
            image_count=741, observer=observer, checkpoint_callback=checkpoint)
        trace.flush()
    require(result['completed_epoch'] == 24 and result['updates'] == {arm: 17784 for arm in N20_TRAIN_ARMS}
            and state['sampled_points'] == 72843264 and state['visited_images'] == 17784
            and state['empty_images'] == 0 and len(state['history']) == len(state['epoch_orders']) == 24
            and set(checkpoints) == set(endpoint_files) == {'8', '24'} and loads == 741 * 26,
            'Complete same-sample paired fixed24 trajectory and E8/E24 endpoints required')
    for ref in spec['sources']:
        check()
        bound(ref)
    require(reference(path) == gate['spec_ref'] and reference(__file__) == gate['source']
            and reference(spec['GPU_admission_file']) == gate['admission_ref']
            and torch.equal(outside_CPU, torch.get_rng_state())
            and torch.equal(outside_CUDA, torch.cuda.get_rng_state(0).cpu())
            and gate['common'].imported_files(torch, np, cv2, masks, COCO,
                closure, gate['admission'], ndimage) == libraries,
            'Final Source/spec/admission/imported libraries/external RNG unchanged')
    gate['lineage'].final_endpoint_check(gate['parent'])
    check()
    peak = torch.cuda.max_memory_allocated(0)
    for optimizer in state['optimizers'].values():
        optimizer.zero_grad(set_to_none=True)
        optimizer.state.clear()
    state['optimizers'].clear()
    del optimizer
    for head in state['heads'].values():
        head.cpu()
    del head
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(0)
    cleanup = dict(allocated_after=torch.cuda.memory_allocated(0), reserved_after=torch.cuda.memory_reserved(0))
    require(cleanup['allocated_after'] == cleanup['reserved_after'] == 0,
            'All model/optimizer CUDA tensors must be released after training')
    check()
    write(output / 'REPORT.json', dict(schema=N20_TRAIN_REPORT_SCHEMA, status=N20_TRAIN_STATUS, **binding,
        START=reference(output / 'START.json'), checkpoints=checkpoints, FIT_endpoints=endpoint_files,
        UPDATE_TRACE=reference(trace_path), history=state['history'], epoch_orders=state['epoch_orders'],
        images=741, natural_queries=74100, sampled_points_per_epoch=3035136, epochs=24,
        paired_updates_per_arm=17784, point_exposure_per_arm=72843264, parameters_per_arm=18851,
        tensors_per_arm=5, arms=list(N20_TRAIN_ARMS), initial_states=initial,
        normal_RAW_count=spec['normal_RAW_count'], total_failed_RAW_count=spec['total_failed_RAW_count'],
        total_terminal_identity_pin_count=spec['total_terminal_identity_pin_count'], actual_imported_files=libraries,
        torch_peak_allocated_bytes=peak, cleanup=cleanup, qualified=False, ROOT_actual_GPU0_verified=False,
        current_owned_terminal_self_certified=False, CPU_endpoint_qualification_pending=True,
        teacher_or_PR_loaded=False, full_native_L_loaded=False, DEV_AP_evaluated=False,
        best_endpoint_selected=False, old_cache_Source_or_protocol_modified=False,
        additional_FIT_annotations=False, new_primitive_claimed=False,
        old_point_query_label_weight_order_preserved=True, elapsed_seconds=time.monotonic() - started))
    print(json.dumps(dict(status=N20_TRAIN_STATUS, report=reference(output / 'REPORT.json'), qualified=False)), flush=True)
