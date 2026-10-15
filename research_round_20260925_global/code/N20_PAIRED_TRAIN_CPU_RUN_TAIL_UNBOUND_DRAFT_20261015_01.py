# UNBOUND_DRAFT: N20 endpoint CPU run tail only; no RAW/admission front or CLI.
"""Concatenate after the frozen ROOT-owned typed front and N20 CPU FRONT.
ROOT supplies require/read/closed/bound/load/reference/write/source_gate; all
module load() calls receive strict three-field refs after closed() membership.
Source/actual-terminal qualification belongs to ROOT, never this pending Q.
"""
import gc
import json
import os
from pathlib import Path
import time

N20_CPU_ARMS = ('LOCAL_ERROR', 'CENTER_ERROR')
N20_CPU_SEED = 202610150201
N20_CPU_ORDER_SEED = 202610140160001
N20_CPU_THREAD_ENV = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                      'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
N20_CPU_Q_SCHEMA = 'N20-paired-local-center-feature-error-training-endpoint-CPU-qualification-v1'
N20_CPU_STATUS = 'N20_PAIRED_FEATURE_ERROR_E8_E24_ENDPOINT_CPU_COMPLETE_PENDING_ROOT_ACTUAL0'


def n20_load_endpoint_CPU_image(torch, index, gate, spec, check):
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



def n20_read_actual_update_trace(path, check):
    """Only the saved paired_image_update substream, with no invented events."""
    records = []
    with path.open('r', encoding='utf8') as stream:
        for line in stream:
            check()
            require(bool(line.strip()), 'No blank line in the original update trace')
            records.append(json.loads(line))
            require(len(records) <= 17784, 'No extra original paired updates')
    require(len(records) == 17784, 'Full fixed24 original17784 paired updates required')
    return records


def run(spec_path):
    started = time.monotonic()
    require(Path(spec_path).is_absolute(), 'Absolute independent CPU specification required')
    path = Path(spec_path).resolve(strict=True)
    spec = read(path)
    gate = source_gate(path, spec)
    closure = gate['closure']
    output = Path(spec['output_directory'])
    require(output.is_absolute() and not output.exists() and not output.is_symlink()
            and output.parent.is_dir(), 'Fresh owned CPU output with existing parent required')
    require(spec['deadline_seconds'] == 1800 and spec['automatic_retry'] is False
            and spec['maximum_process_RSS_bytes'] == 16 * 1024**3
            and spec['initialization_seed'] == N20_CPU_SEED,
            'Original first-seed CPU resources and fixed recipe required')
    source_keys = ('scout_module_source', 'base_scout_source', 'paired_training_body_source',
                   'original_paired_training_body_source', 'qualifier_body_source')
    for key in source_keys:
        closed(closure, spec[key])
    for report_key in ('parent_cache_report', 'neighbor_report', 'training_report'):
        require(read(closed(closure, gate[report_key + '_ref'])) == gate[report_key],
                'Decoded report must equal its exact closed physical JSON: ' + report_key)
    report = gate['training_report']
    require(gate['training_GPU_record']['report'] == gate['training_report_ref']
            and set(gate['training_endpoints']) == {8, 24},
            'Original completed training report and actual inner E8/E24 endpoints required')
    cache_rows = gate['parent_cache_report']['rows']
    require(len(cache_rows) == len(gate['neighbor_report']['rows']) == 741,
            'Complete qualified sparse FIT741 inputs required')

    import psutil
    process = psutil.Process()
    require(process.cpu_affinity() == [0, 1] and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and all(os.environ.get(key) == '2' for key in N20_CPU_THREAD_ENV),
            'Independent CPU01/two threads/empty CUDA actual environment')
    import torch
    require(torch.__version__ == '2.0.1+cu118' and not torch.cuda.is_initialized(),
            'Original Torch with cold CUDA required after the complete front')
    import numpy as np
    import cv2
    from scipy import ndimage
    from pycocotools import mask as masks
    from pycocotools.coco import COCO
    require(np.__version__ == '1.23.5' and cv2.__version__ == '4.8.0',
            'Closed original numerical-library versions required')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    cv2.setNumThreads(2)
    # These modules are already closed and bound by the front. The paired bodies
    # are provenance only here: no training/optimizer step is called.
    scout = load(spec['scout_module_source'], '_N20_CPU_closed_feature_adapter')
    base = load(spec['base_scout_source'], '_N20_CPU_closed_original_scout_factory')
    paired = load(spec['paired_training_body_source'], '_N20_CPU_closed_training_body_provenance')
    original = load(spec['original_paired_training_body_source'], '_N20_CPU_closed_original_training_provenance')
    body = load(spec['qualifier_body_source'], '_N20_CPU_closed_independent_endpoint_body')
    require(tuple(paired.ARMS) == tuple(body.ARMS) == N20_CPU_ARMS
            and paired.INIT_SEED == body.INIT_SEED == N20_CPU_SEED
            and paired.ORDER_SEED == body.ORDER_SEED == N20_CPU_ORDER_SEED
            and paired.EPOCHS == 24 and tuple(paired.ENDPOINTS) == (8, 24)
            and callable(getattr(original, 'train_one_epoch', None))
            and callable(getattr(scout, 'create_head', None))
            and callable(getattr(base, 'make_scout', None))
            and body.ABS_LOGITS == body.ABS_MEAN_CE == 1e-4,
            'Exact same four model sources and independent two-formula body ABI')
    libraries = gate['common'].imported_files(
        torch, np, cv2, masks, COCO, closure, gate['admission'], ndimage)
    require(not torch.cuda.is_initialized(), 'Source-qualified imports must keep CUDA cold')
    outside_CPU = torch.get_rng_state().clone()
    peak_RSS = 0
    minimum_available_RAM = None

    def check():
        nonlocal peak_RSS, minimum_available_RAM
        rss = process.memory_info().rss
        available = psutil.virtual_memory().available
        peak_RSS = max(peak_RSS, rss)
        minimum_available_RAM = available if minimum_available_RAM is None else min(minimum_available_RAM, available)
        require(time.monotonic() - started < spec['deadline_seconds']
                and rss < spec['maximum_process_RSS_bytes'] and available >= 16 * 1024**3
                and process.cpu_affinity() == [0, 1] and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
                and all(os.environ.get(key) == '2' for key in N20_CPU_THREAD_ENV)
                and torch.get_num_threads() == torch.get_num_interop_threads() == 2
                and cv2.getNumThreads() == 2 and not torch.cuda.is_initialized(),
                'CPU bounded resources/deadline/affinity/thread/cold-CUDA invariant')

    check()
    expected_digest = spec['expected_N20_initialization_digest']
    initialization_checks = body.validate_report(
        torch, scout, base, report, gate['training_start'], expected_digest)
    trace = n20_read_actual_update_trace(closed(closure, report['UPDATE_TRACE']), check)
    snapshots = {}
    trained_digests = {}
    for epoch in (8, 24):
        check()
        ref = report['checkpoints'][str(epoch)]
        # Snapshot is the torch file's top-level dictionary, not ['snapshot'].
        snapshot = torch.load(str(closed(closure, ref)), map_location='cpu')
        body.validate_snapshot_report(torch, snapshot, report, epoch)
        body.validate_trace(trace[:epoch * 741], snapshot, cache_rows)
        snapshots[epoch] = snapshot
        trained_digests[str(epoch)] = {
            arm: body.state_digest(torch, snapshot['models'][arm])[0] for arm in N20_CPU_ARMS}
        check()
    body.validate_snapshot_pair(torch, snapshots[8], snapshots[24])
    output.mkdir()
    input_trace_path = output / 'CPU_INPUT_READS.jsonl'
    endpoints = {}
    loads = 0
    with input_trace_path.open('x', encoding='utf8', newline='\n') as input_trace:
        for epoch in (8, 24):
            check()
            next_index = 0

            def observer(event):
                check()
                require(event['event'] == 'before_endpoint_CPU_image'
                        and event['epoch'] == epoch and event['dataset_index'] == next_index,
                        'CPU endpoint must visit each original image exactly once in canonical order')

            def loader(index):
                nonlocal loads, next_index
                require(type(index) is int and index == next_index and 0 <= index < 741,
                        'Canonical complete FIT741 same-input endpoint traversal')
                loaded = n20_load_endpoint_CPU_image(torch, index, gate, spec, check)
                input_trace.write(json.dumps(dict(epoch=epoch, dataset_index=index,
                    image_id=cache_rows[index]['image_id'],
                    parent_packet=cache_rows[index]['packet'],
                    neighbor_packet=gate['neighbor_report']['rows'][index]['packet']),
                    separators=(',', ':'), allow_nan=False) + '\n')
                loads += 1
                next_index += 1
                return loaded

            endpoints[str(epoch)] = body.validate_endpoint(
                torch, scout, base, snapshots[epoch], gate['training_endpoints'][epoch],
                loader, cache_rows, epoch, observer=observer,
                expected_initialization_digest=expected_digest)
            require(next_index == 741, 'Only complete actual741-image CPU endpoint qualifies')
            require(all(body.state_digest(torch, snapshots[epoch]['models'][arm])[0] ==
                        trained_digests[str(epoch)][arm] for arm in N20_CPU_ARMS),
                    'Trained checkpoint tensors must remain unchanged during numerical reads')
            input_trace.flush()
            check()
    require(loads == 1482 and set(endpoints) == {'8', '24'},
            'Both actual fixed endpoints, each full741 images/3035136 same input slots')
    del trace, snapshots
    gc.collect()
    for ref in spec['sources']:
        check()
        bound(ref)
    require(reference(path) == gate['spec_ref'] and reference(__file__) == gate['source']
            and reference(spec['CPU_admission_file']) == gate['admission_ref']
            and torch.equal(outside_CPU, torch.get_rng_state())
            and gate['common'].imported_files(torch, np, cv2, masks, COCO,
                closure, gate['admission'], ndimage) == libraries,
            'Final complete Source/spec/current CPU entry/admission/imports/CPU RNG unchanged')
    gate['lineage'].final_endpoint_check(gate['parent'])
    check()
    binding = dict(source=gate['source'], spec=gate['spec_ref'], CPU_admission=gate['admission_ref'],
        phase=spec['phase'], training_GPU_record=gate['training_GPU_record'],
        training_GPU_report=gate['training_report_ref'],
        parent_cache_report=gate['parent_cache_report_ref'], neighbor_report=gate['neighbor_report_ref'],
        N20_locked_protocol=spec['N20_locked_protocol'], locked_protocol=spec['locked_protocol'],
        qualifier_body_source=spec['qualifier_body_source'],
        scout_module_source=spec['scout_module_source'], base_scout_source=spec['base_scout_source'],
        paired_training_body_source=spec['paired_training_body_source'],
        original_paired_training_body_source=spec['original_paired_training_body_source'],
        initialization_seed=N20_CPU_SEED, order_seed=N20_CPU_ORDER_SEED,
        expected_N20_initialization_digest=expected_digest)
    binding.update(gate['training_binding_refs'])
    checks = dict(initialization_digest_and_all_five_tensors_exact=True,
        both_E8_E24_full_snapshot_and_report_bindings_exact=True,
        all_five_AdamW_states_moments_and_exact_steps_5928_17784=True,
        private_orders_rng_history_and_saved_paired_update_substream_exact=True,
        complete_same_input_FIT741_CPU_two_formulas_absolute1e4=True,
        recorded_GPU_endpoint_mean_weighted_CE_absolute1e4=True,
        original_source_spec_admission_imports_rng_and_cpu_resources_unchanged=True)
    details_path = output / 'DETAILS.json'
    write(details_path, dict(schema='N20-paired-feature-error-endpoint-CPU-details-v1',
        **binding, checks=checks, initialization_checks=initialization_checks,
        endpoints=endpoints, trained_state_digests=trained_digests,
        checkpoints=report['checkpoints'], FIT_endpoints=report['FIT_endpoints'],
        UPDATE_TRACE=report['UPDATE_TRACE'], CPU_input_reads=reference(input_trace_path),
        saved_paired_updates=17784, E8_saved_trace_prefix_updates=5928,
        trace_scope='Only the saved paired_image_update substream; no missing before/endpoint event replay claimed',
        actual_imported_files=libraries, qualified=False, ROOT_actual_CPU0_verified=False,
        GPU_pointwise_logits_qualified=False, optimizer_trajectory_replayed=False,
        selected_indices_qualified=False, mask_RLE_qualified=False, AP_evaluated=False,
        same_actual_GPU_input_F_reused=True, extra_F_independently_rerendered=False,
        teacher_or_PR_loaded=False, full_native_L_loaded=False,
        scope='Fixed E8/E24 CPU state/moments/orders/trace and same-input two formulas plus recorded GPU aggregate CE only'))
    check()
    q_path = output / 'Q.json'
    write(q_path, dict(schema=N20_CPU_Q_SCHEMA, status=N20_CPU_STATUS, **binding,
        details=reference(details_path), checks=checks, endpoints=endpoints,
        images=741, natural_queries=74100, sampled_points_per_endpoint=3035136,
        CPU_packet_pair_reads=loads, total_CPU_endpoint_sampled_points=2 * 3035136,
        fixed_endpoints=[8, 24], parameters_per_arm=18851, tensors_per_arm=5,
        E8_updates_per_arm=5928, E24_updates_per_arm=17784, arms=list(N20_CPU_ARMS),
        normal_RAW_count=spec['normal_RAW_count'], total_failed_RAW_count=spec['total_failed_RAW_count'],
        total_terminal_identity_pin_count=spec['total_terminal_identity_pin_count'],
        qualified=False, ROOT_actual_CPU0_verified=False, original_owned_terminal_self_certified=False,
        GPU_pointwise_logits_qualified=False, optimizer_trajectory_replayed=False,
        selected_indices_qualified=False, mask_RLE_qualified=False, AP_evaluated=False,
        teacher_or_PR_loaded=False, full_native_L_loaded=False,
        peak_observed_process_RSS_bytes=peak_RSS, minimum_observed_available_RAM_bytes=minimum_available_RAM,
        actual_imported_files=libraries, elapsed_seconds=time.monotonic() - started))
    check()
    print(json.dumps(dict(status=N20_CPU_STATUS, Q=reference(q_path), qualified=False)), flush=True)
