"""Inert independent ROOT CPU consumer for one-image passive replay.

Current owned terminal and complete admission/source chain precede scientific
hashes/imports. Actual official RLEs/factors are decoded independently. Full
paired CP state is independently checked using the qualified CPU state reader.
Ten dense floating outputs are NOT exported or independently replayed here:
their equality is source-bound GPU RAM byte-check evidence plus equal digests.
There is no GT-mask analysis, inference, matching, AP or scientific promotion.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

SCHEMA = 'ROOT-native-geometry-paired-passive-replay-spec-v1'
OWNER_SHA = 'ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
HELPER_SHA = '54d3b9fb0866fe6e88cea7272b7667fc70997644c2c8732762529968a32f5cb4'
MODES = ('native0', 'paired', 'fg_fine')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def reference(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def bound(row):
    require(isinstance(row, dict) and Path(row['path']).is_absolute() and
            reference(row['path']) == {key: row[key] for key in ('path', 'bytes', 'sha256')},
            'Pinned reference differs')
    return Path(row['path'])


def load_source(row, name):
    loader = importlib.util.spec_from_file_location(name, bound(row))
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


def terminal(receipt, phase):
    require(receipt['status'] == 'ACTUAL_OWNED_TERMINAL' and receipt['original_identity'] is not None and
            type(receipt['returncode']) is int and receipt['returncode'] == 0 and
            receipt['root_wait_reaped'] is True and receipt['held_handle_exit'] ==
            dict(wait_state='WAIT_OBJECT_0', actual_exit_code=0) and
            receipt['numeric_birth_absence']['original_birth_absent'] is True and
            receipt['abort'] is None and receipt['error'] is None and receipt['wait_errors'] is False and
            receipt['GPU_observation_errors'] == [], 'STOP before scientific access: owned actual0/held reap required')
    if 'phase' in receipt:
        require(receipt['phase'] == phase, 'Owned phase differs')
    peak = receipt['whole_GPU_observed_peak_MiB']
    require(type(peak) in (int, float) and math.isfinite(peak) and 0 <= peak <= 8192 and
            receipt['whole_GPU_used_limit_MiB'] == 8192 and receipt['whole_leaf_VRAM_certified'] is False and
            receipt['GPU_scope'] == 'whole GPU observed, not per-process WDDM allocation', 'Owned resource/scope differs')


def source_gate(spec_path, spec, receipt):
    require(spec['schema'] == SCHEMA and spec['phase'].startswith('native_geometry_paired_passive_replay_') and
            spec['modes'] == list(MODES) and set(spec['checkpointmap']) == set(spec['qualificationmap']) == set(MODES) and
            type(spec['expected_epoch']) is int and spec['expected_epoch'] in range(8, 25, 2) and
            spec['dataset_index'] == 0 and spec['expected_cases'] == 1 and spec['evaluation_images'] == 187 and
            spec['CPU_affinity'] == [0, 1] and spec['allocator_limit_MiB'] == 4096 and
            spec['whole_GPU_owner_limit_MiB'] == 8192 and spec['deadline_seconds'] == 900 and
            spec['automatic_retry'] is False and spec['cached_backbone_replay'] is True,
            'Actual fixed one-case/even-epoch/resource contract differs')
    require(Path(spec['GPU_admission_file']).is_absolute(), 'Absolute admission string required')
    adm_ref = reference(spec['GPU_admission_file']); adm = read(spec['GPU_admission_file'])
    entry = reference(Path(__file__).with_name('probe.py')); reader = reference(__file__); spec_ref = reference(spec_path)
    owner = reference(spec_path.parent / 'finite_point_cache_execution_20261005_v2' / 'run_owned_gpu_limited.py')
    require(receipt['admission_sha256'] == adm_ref['sha256'] and owner['sha256'] == OWNER_SHA and
            adm['driver_sha256'] == OWNER_SHA and adm['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and
            adm['automatic_retry'] is False and adm['phase'] == spec['phase'] and adm['allowed_cpus'] == [0, 1] and
            adm['cuda_visible_devices'] == '0' and adm['qualified_leaf_deadline_seconds'] == 900 and
            adm['whole_GPU_used_limit_MiB'] == 8192 and Path(adm['entry']).resolve(strict=True) == Path(entry['path']) and
            adm['entry_args'] == ['--spec', str(spec_path)] and spec['entry_source'] == entry and spec['CPU_reader'] == reader,
            'Current actual receipt/admission/entry/reader/SPEC linkage differs')
    require(adm['bound_source_sha256'].get(str(spec_path)) == spec_ref['sha256'], 'Actual SPEC not admission-bound')
    for row in spec['sources']:
        bound(row)
        require(adm['bound_source_sha256'].get(row['path']) == row['sha256'], 'Source closure not admission-bound')
    for key in ('entry_source', 'CPU_reader', 'helper_source', 'evaluation_source', 'evaluation_CPU_source',
                'training_source', 'initializer_source', 'micro_preflight_source', 'native_helpers', 'adapter_source', 'hooks_source'):
        require(spec[key] in spec['sources'], 'Executing source role omitted: ' + key)
    require(spec['helper_source']['bytes'] == 16788 and spec['helper_source']['sha256'] == HELPER_SHA,
            'Immutable helper differs')
    return entry, reader, spec_ref, adm_ref, owner


def digest_array(array):
    return dict(shape=list(array.shape), dtype=str(array.dtype),
                sha256=hashlib.sha256(array.tobytes(order='C')).hexdigest())


def digest_contract(row, shape, dtype):
    require(isinstance(row, dict) and set(row) == {'shape', 'dtype', 'sha256'} and
            row['shape'] == list(shape) and row['dtype'] == dtype and
            isinstance(row['sha256'], str) and len(row['sha256']) == 64 and
            set(row['sha256']) <= set('0123456789abcdef'), 'Tensor digest interface differs')


def array_packet(np, row, shape, dtype):
    require(set(row) == {'shape', 'dtype', 'sha256', 'values'}, 'Actual small tensor packet required')
    digest_contract({key: row[key] for key in ('shape', 'dtype', 'sha256')}, shape, dtype)
    array = np.asarray(row['values'], dtype=np.dtype(dtype))
    require(tuple(array.shape) == tuple(shape) and bool(np.isfinite(array).all()) and
            digest_array(array) == {key: row[key] for key in ('shape', 'dtype', 'sha256')},
            'Independent small-array shape/finite/actual bytes differ')
    return array


def check_capture(np, capture, metadata):
    require(capture['schema'] == 'passive-paired-adapter-capture-v1' and capture['calls'] == capture['expected_calls'] == 1 and
            capture['actual_capture_available'] is True and capture['hooks_removed'] is True and
            capture['forward_hook_returned_None'] is True and capture['F_saved'] is False and
            capture['full_rho_observed'] is capture['M0_observed'] is capture['raw_covariance_observed'] is False and
            capture['geometry_recomputed'] is False and capture['all_queries_retained'] is True and
            capture['metadata'] == metadata and capture['GT_position_labels_are_not_feature_purity'] is True,
            'Actual passive capture scope differs')
    packet = capture['capture']; geometry = packet['geometry']
    require(packet['native_query_ids'] == list(range(100)) and packet['F_shape'] == [1, 256, 200, 200] and
            packet['e0_shape'] == [1, 100, 256] and geometry['mode'] == 'paired' and
            geometry['eligibility_frame'] == 'canonical', 'All100 actual paired frame differs')
    parsed = {}
    shapes = dict(inside=((1, 100, 16, 2), 'float32'), outside=((1, 100, 16, 2), 'float32'),
                  eligible=((1, 100, 16), 'bool'), outside_donor_slots=((1, 100, 16), 'int64'),
                  centre_xy=((1, 100, 2), 'float32'))
    for key in ('rho_mass', 'canonical_theta', 'used_theta', 'sigma_major', 'sigma_minor'):
        shapes[key] = ((1, 100), 'float32')
    for key in ('canonical_eligible_counts', 'eligible_counts'):
        shapes[key] = ((1, 100), 'int64')
    for key in ('near_isotropic', 'degenerate', 'shuffle_can_be_nontrivial'):
        shapes[key] = ((1, 100), 'bool')
    for key, (shape, dtype) in shapes.items():
        parsed[key] = array_packet(np, geometry[key], shape, dtype)
    E = parsed['eligible']; counts = E.sum(axis=-1, dtype=np.int64)
    require(np.array_equal(counts, parsed['eligible_counts']) and np.array_equal(counts, parsed['canonical_eligible_counts']) and
            np.array_equal(counts > 1, parsed['shuffle_can_be_nontrivial']) and
            np.array_equal(parsed['canonical_theta'], parsed['used_theta']) and
            bool((parsed['rho_mass'] >= 0).all()) and bool((parsed['sigma_major'] >= 1).all()) and
            bool((parsed['sigma_minor'] >= 1).all()) and
            bool(((parsed['outside_donor_slots'] >= 0) & (parsed['outside_donor_slots'] < 16)).all()),
            'Actual retained eligibility/count/frame consistency differs')
    for key in ('inside', 'outside'):
        require(bool(((parsed[key][E] >= 0) & (parsed[key][E] <= 199)).all()),
                'An eligible actual coordinate is outside its native centre canvas')
    delta = array_packet(np, packet['coefficient_residual'], (1, 100, 256), 'float32')
    array_packet(np, packet['e0'], (1, 100, 256), 'float32')
    array_packet(np, packet['new_coeff'], (1, 100, 256), 'float32')
    require(bool((delta[counts == 0] == 0).all()), 'Empty eligibility query must retain a zero residual')
    fraction = array_packet(np, packet['actual_valid_fraction'], (1, 1, 200, 200), 'float32')
    ih, iw = metadata['img_shape'][:2]
    index = np.arange(200, dtype=np.float32)
    yf = np.clip(np.float32(ih) - 4 * index, 0, 4) / 4
    xf = np.clip(np.float32(iw) - 4 * index, 0, 4) / 4
    expected = (yf[:, None] * xf[None, :])[None, None].astype(np.float32)
    require(fraction.tobytes() == expected.tobytes(), 'Independent metadata-only valid fraction differs')
    return dict(queries=100, slots_per_query=16, eligible_pairs=int(counts.sum()),
                E0_queries=int((counts == 0).sum()), all_E0_retained=True,
                actual_arrays_hashes_verified=True, M0_sign_predicates_independently_recomputed=False,
                GT_occupancy_or_feature_purity_computed=False)


def check_official(np, masks, rows, digests, original_shape):
    require(isinstance(rows, list) and len(rows) == 100 and
            sorted(row['query_id'] for row in rows) == list(range(100)), 'All100 official rows/query IDs required')
    binary, scores, labels, qids, probabilities, qualities = [], [], [], [], [], []
    for row in rows:
        require(set(row) == {'query_id', 'label', 'score', 'class_probability', 'mask_quality', 'segmentation'} and
                type(row['query_id']) is int and row['label'] == 0 and
                row['segmentation']['size'] == original_shape and isinstance(row['segmentation']['counts'], str),
                'Actual official row/category/original mask geometry differs')
        rle = dict(row['segmentation']); rle['counts'] = rle['counts'].encode('ascii')
        image = masks.decode(rle)
        require(list(image.shape) == original_shape and bool(((image == 0) | (image == 1)).all()) and
                int(masks.area(rle)) == int(image.sum()), 'Independent original-shape RLE decode/area differs')
        c, quality, score = (np.float32(row[key]) for key in ('class_probability', 'mask_quality', 'score'))
        require(all(math.isfinite(float(value)) and 0 <= value <= 1 for value in (c, quality, score)),
                'Official FP32 factors/scores out of range')
        product = float(c) * float(quality); u = 2. ** -24
        bound_error = (u / (1 - u)) * abs(product) + 2 * float(np.finfo(np.float32).tiny)
        require(abs(float(score) - product) <= bound_error, 'Official product outside legal FP32/FTZ rounding range')
        binary.append(image.astype(np.bool_)); scores.append(score); labels.append(row['label']); qids.append(row['query_id'])
        probabilities.append(c); qualities.append(quality)
    arrays = dict(masks=np.stack(binary), scores=np.asarray(scores, dtype=np.float32),
                  labels=np.asarray(labels, dtype=np.int64), query_ids=np.asarray(qids, dtype=np.int64),
                  class_probability=np.asarray(probabilities, dtype=np.float32), mask_quality=np.asarray(qualities, dtype=np.float32))
    require(set(digests) == set(arrays) and all(digest_array(value) == digests[key] for key, value in arrays.items()),
            'Independent RLE/FP32 factor bytes do not match actual endpoint digests')
    return arrays


def qualify(spec_file, acceptance_file):
    require(Path(spec_file).is_absolute() and Path(acceptance_file).is_absolute() and not Path(acceptance_file).exists(),
            'Absolute SPEC/fresh acceptance required')
    spec_path = Path(spec_file).resolve(strict=True); spec = read(spec_path)
    receipt_path = spec_path.parent / 'finite_point_cache_execution_20261005_v2' / 'runs' / spec['phase'] / 'PARENT_RECEIPT.json'
    receipt = read(receipt_path); terminal(receipt, spec['phase'])
    # No scientific imports or scientific file hashes have occurred above.
    entry, reader, spec_ref, adm_ref, owner = source_gate(spec_path, spec, receipt)
    evaluator = load_source(spec['evaluation_source'], 'passive_CPU_original_E')
    evaluator.qualified_training(spec)  # all3 original terminal guards precede their hashes/science
    old_CPU = load_source(spec['evaluation_CPU_source'], 'passive_CPU_original_state_reader')
    qualifications = old_CPU.qualified_training(spec)
    report_path = Path(spec['output_directory']) / 'REPORT.json'
    report = read(report_path)
    require(report['schema'] == 'native-geometry-paired-passive-replay-report-v1' and
            report['status'] == 'COMPLETE_PASSIVE_PAIRED_REPLAY_PENDING_ROOT_TERMINAL' and
            report['phase'] == spec['phase'] and report['source'] == entry and report['CPU_reader'] == reader and
            report['spec'] == spec_ref and report['GPU_admission'] == adm_ref and report['owner_driver'] == owner and
            report['helper_source'] == spec['helper_source'] and report['expected_epoch'] == spec['expected_epoch'] and
            report['checkpoint'] == spec['checkpointmap']['paired'] and report['qualification'] == spec['qualificationmap']['paired'] and
            report['cases'] == 1 and report['dataset_index'] == 0 and report['feature_extraction_calls'] == 1 and
            report['head_forward_calls'] == 2 and report['both_apply_true'] is True and
            report['full_backbone_independent_replay_claimed'] is False and report['cleanup_allocated_bytes'] == 0 and
            math.isfinite(report['runtime_seconds']) and 0 < report['runtime_seconds'] <= 900,
            'Actual own terminal/report/source/scope linkage differs')
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '2'
    os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'; os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import psutil
    psutil.Process().cpu_affinity([0, 1])
    import numpy as np
    import torch
    from pycocotools import mask as mask_utils
    require(psutil.Process().cpu_affinity() == [0, 1] and not torch.cuda.is_initialized(), 'CPU01/uninitialized CUDA required')
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    qualified = old_CPU.checkpoint_state(torch, spec, qualifications['paired'], 'paired')
    output_names = dict(capture='CAPTURE.json', endpoint_digests='ENDPOINT_DIGESTS.json',
                        official_endpoints='OFFICIAL_ENDPOINTS.json', state_RNG_proof='STATE_RNG_PROOF.json')
    require(set(report['outputs']) == set(output_names) and all(
        Path(report['outputs'][key]['path']).resolve(strict=True) ==
        (Path(spec['output_directory']) / name).resolve(strict=True) for key, name in output_names.items()),
        'Actual evidence paths differ from this owned output directory')
    outputs = {key: read(bound(row)) for key, row in report['outputs'].items()}
    require(set(outputs) == {'capture', 'endpoint_digests', 'official_endpoints', 'state_RNG_proof'}, 'Output evidence roles differ')
    endpoints = outputs['endpoint_digests']; execution = outputs['state_RNG_proof']
    require(set(endpoints) == {'reference', 'observed'} and endpoints['reference'] == endpoints['observed'] and
            set(execution) == {'before', 'middle', 'after'} and execution['before'] == execution['middle'] == execution['after'],
            'Actual exported endpoint/fullstate/RNG summaries disagree')
    private = report['private_RNG']
    require(set(private) == {'reference', 'observed'} and private['reference'] == private['observed'] and
            set(private['reference']) == {'before', 'after'}, 'Actual private RNG before/after replay summaries differ')
    for endpoint_RNG in private.values():
        for generator in endpoint_RNG.values():
            require(set(generator) == {'CPU', 'CUDA'}, 'Both private generators are required')
            for rng in generator.values():
                require(rng['dtype'] == 'uint8' and len(rng['shape']) == 1 and rng['shape'][0] > 0,
                        'Actual private generator byte summary differs')
                digest_contract(rng, rng['shape'], 'uint8')
    for snapshot in execution.values():
        require(snapshot['schema'] == 'passive-replay-execution-snapshot-v1' and snapshot['cuda_device'] == 0 and
                snapshot['state'] == qualified['fullstate'] and set(snapshot['RNG']) == {'CPU', 'CUDA'} and
                all(flag is False for flag in snapshot['parameter_requires_grad'].values()) and
                all(flag is True for flag in snapshot['parameter_grad_is_None'].values()) and
                all(flag is False for flag in snapshot['module_training'].values()),
                'Independent paired612 CP/frozen-mode/current RNG proof differs')
        for rng in snapshot['RNG'].values():
            require(rng['dtype'] == 'uint8' and len(rng['shape']) == 1 and rng['shape'][0] > 0,
                    'Actual CPU/CUDA generator byte summary differs')
            digest_contract(rng, rng['shape'], 'uint8')
    require(report['loading_proof']['trained_fullstate'] == qualified['fullstate'] and
            report['loading_proof']['checkpoint'] == spec['checkpointmap']['paired'], 'Loaded state/actual CP proof differs')
    DEV = read(bound(spec['DEV']))
    require(len(DEV['images']) == 187 and len(DEV['annotations']) == 380, 'Original complete DEV metadata required')
    first = DEV['images'][0]; identity = report['identity']; oh, ow = first['height'], first['width']
    scale = 800. / max(oh, ow); actual = [int(oh * scale + .5), int(ow * scale + .5)]
    require(identity['image_id'] == first['id'] and identity['original_shape'] == [oh, ow] and
            identity['actual_img_shape'] == actual and identity['batch_input_shape'] == [800, 800] and
            identity['ann_ids'] == [row['id'] for row in DEV['annotations'] if row['image_id'] == first['id']] and
            identity['scale_factor'] == [actual[1] / ow, actual[0] / oh] and
            Path(identity['image']['path']).resolve() == (Path(spec['image_root']) / first['file_name']).resolve() and
            report['seed'] == 100000 + first['id'], 'Fixed dataset_index0/source/native geometry/private seed differs')
    bound(identity['image'])
    require(identity['input_identity'] == dict(image=identity['image'], input=identity['input_identity']['input'],
            actual_img_shape=actual, original_shape=[oh, ow]), 'Input identity binding differs')
    digest_contract(identity['input_identity']['input'], (1, 3, 800, 800), 'float32')
    endpoint = endpoints['reference']; metadata = outputs['capture']['metadata']
    require(endpoint['schema'] == 'passive-replay-endpoint-snapshot-v1' and
            endpoint['binding'] == dict(checkpoint=spec['checkpointmap']['paired'], input_identity=identity['input_identity'],
                                        head_seed=report['seed'], mode='paired', metadata=metadata) and
            metadata['img_shape'][:2] == actual and metadata['ori_shape'][:2] == [oh, ow] and
            metadata['batch_input_shape'] == [800, 800] and len(endpoint['classes']) == len(endpoint['masks']) == 10,
            'Actual shared input/metadata/all10 scope differs')
    for row in endpoint['classes']:
        digest_contract(row, (1, 100, 2), 'float32')
    for row in endpoint['masks']:
        digest_contract(row, (1, 100, 200, 200), 'float32')
    capture_proof = check_capture(np, outputs['capture'], metadata)
    official = outputs['official_endpoints']
    require(set(official) == {'reference', 'observed'}, 'Both actual official endpoint populations required')
    decoded = {key: check_official(np, mask_utils, rows, endpoints[key]['official'], [oh, ow])
               for key, rows in official.items()}
    require(all(decoded['reference'][key].tobytes() == decoded['observed'][key].tobytes() for key in decoded['reference']),
            'Independent decoded all100 masks/factors/query IDs differ')
    for hook in (report['reference_hook'], report['observed_hook']):
        require(hook['apply'] is True and hook['mode'] == 'paired' and hook['paired_source'] == spec['adapter_source'] and
                hook['forward_calls'] == hook['completed_forwards'] == hook['pixel_decoder_calls'] == 1 and
                hook['coefficient_calls'] == 10 and hook['unchanged_early_readouts'] == 9 and
                hook['final_coefficient_replacements'] == 1 and hook['hooks_removed'] is True and
                hook['cache_cleared_after_forward'] is True, 'Actual both-apply-true native hook scope differs')
    require(report['result']['status'] == 'PASSIVE_REPLAY_BITWISE_IDENTITY_TECHNICALLY_QUALIFIED' and
            report['result']['CUDA_RNG_observed'] is True and report['result']['full_state_and_CPU_CUDA_RNG_byte_exact'] is True and
            report['no_GT_in_forward'] is report['no_GT_analysis'] is report['no_AP'] is report['no_optimizer'] is True and
            report['full_rho_M0_unknown'] is report['GT_position_is_not_feature_purity'] is True,
            'Runtime byte-check evidence or interpretation scope differs')
    require(not torch.cuda.is_initialized(), 'CPU consumer initialized CUDA unexpectedly')
    acceptance = dict(schema='ROOT-native-geometry-paired-passive-replay-CPU-qualification-v1',
        status='ROOT_PASSIVE_PAIRED_REPLAY_TECHNICALLY_QUALIFIED_NO_METHOD_RESULT',
        phase=spec['phase'], source=entry, reader_source=reader, helper_source=spec['helper_source'], spec=spec_ref,
        GPU_admission=adm_ref, owner_driver=owner, parent_receipt=reference(receipt_path), report=reference(report_path),
        checkpoint=spec['checkpointmap']['paired'], qualification=spec['qualificationmap']['paired'], expected_epoch=spec['expected_epoch'],
        dataset_index=0, image_id=first['id'], official_queries=100, capture_proof=capture_proof,
        independent_CPU_full_CP_and612_before_after_digest_verified=True,
        independent_CPU_decoded_RLE_factors_scores_query_ID_bytes_equal=True,
        GPU_runtime_all10_RAM_bytecheck_source_observed=True, CPU_independently_replayed_ten_outputs=False,
        CPU_independently_verified_omitted_M0_eligibility=False, no_full_backbone_replay_claim=True,
        GPU_RNG_summary_equal_but_no_independent_generator_replay=True,
        no_AP_no_GT_occupancy_no_feature_purity_claim=True, CPU_affinity=[0, 1], CUDA_visible_devices='',
        outputs=report['outputs'], limitations=[
            'All10 dense float fields and complete generator states are omitted: CPU checks bound summaries, not a second model replay.',
            'Actual returned coordinates/E are retained; M0 sign predicates and full rho are not independently reconstructible.',
            'Input and cached backbone features are source-bound digests; CPU does not replay decoding or backbone extraction.',
            'GT position/occupancy labelling is future posthoc analysis, never proof of feature purity or method utility.'])
    with Path(acceptance_file).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(acceptance, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True); parser.add_argument('--acceptance', required=True)
    args = parser.parse_args(); qualify(args.spec, args.acceptance)
