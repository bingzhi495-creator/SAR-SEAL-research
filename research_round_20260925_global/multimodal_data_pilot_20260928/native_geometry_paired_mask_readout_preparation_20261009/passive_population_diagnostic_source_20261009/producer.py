"""Inert ROOT-owned complete-population passive paired sampling diagnostic.

One GT-free backbone/head forward and one posterior final stock matcher per
image. Existing adapter coordinates/eligibility are observed without replacement.
No GT occupancy, AP, promotion or feature-purity statistic is computed here.
All-ten digests are observations, not independent unobserved replay identity.
Scientific imports/hashes follow all3 training and formal-evaluation terminals.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import time

SCHEMA = 'ROOT-native-geometry-paired-passive-population-spec-v1'
REPORT_SCHEMA = 'native-geometry-paired-passive-population-report-v1'
MODES = ('native0', 'paired', 'fg_fine')
OWNER_SHA = 'ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
HELPER_SHA = '54d3b9fb0866fe6e88cea7272b7667fc70997644c2c8732762529968a32f5cb4'
FORMAL_QUAL_SCHEMA = 'ROOT-native-geometry-paired-common-evaluation-CPU-qualification-v1'
FORMAL_QUAL_STATUS = 'ROOT_NATIVE_GEOMETRY_PAIRED_COMMON_DEV_CPU_EVALUATED'

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
    require(isinstance(row, dict) and Path(row['path']).is_absolute(), 'Absolute pinned reference required')
    require(reference(row['path']) == {key: row[key] for key in ('path', 'bytes', 'sha256')},
            'Pinned source or payload differs: ' + row['path'])
    return Path(row['path'])

def load_source(row, name):
    loader = importlib.util.spec_from_file_location(name, bound(row))
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module

def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')

def encode_tree(torch, pref, value, values=False):
    if isinstance(value, torch.Tensor):
        result = pref.digest_tensor(value)
        if values:
            result['values'] = value.tolist()
        return result
    if isinstance(value, dict):
        return {key: encode_tree(torch, pref, item, values) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_tree(torch, pref, item, values) for item in value]
    return value

def official_endpoint(torch, model, classes, masks, samples, identity):
    """Same native render and official fusion as evaluate_common.py; no new ranking."""
    padded = torch.nn.functional.interpolate(masks[-1], size=(800, 800), mode='bilinear', align_corners=False)
    instances = model.panoptic_fusion_head.predict(classes[-1], padded, samples, rescale=True)[0]['ins_results']
    ih, iw = identity['actual_img_shape']; oh, ow = identity['original_shape']
    rendered = torch.nn.functional.interpolate(padded[0, :, :ih, :iw][:, None], size=(oh, ow),
                                               mode='bilinear', align_corners=False)[:, 0]
    probabilities = torch.nn.functional.softmax(classes[-1][0], dim=-1)[:, :-1]
    top_scores, top_indices = probabilities.flatten(0, 1).topk(100, sorted=False)
    query_ids = top_indices // model.panoptic_fusion_head.num_classes
    require(torch.equal(query_ids.sort().values, torch.arange(100, device=query_ids.device)) and
            torch.equal(top_scores, probabilities[query_ids, 0]), 'Actual native query mapping is incomplete')
    logits = rendered[query_ids]; binary = (logits > 0).float()
    quality = (logits.sigmoid() * binary).flatten(1).sum(1) / (binary.flatten(1).sum(1) + 1e-6)
    require(len(instances) == 100 and torch.equal(instances.masks, binary.bool()) and
            torch.equal(instances.scores, probabilities[query_ids, 0] * quality) and
            bool((instances.labels == 0).all()), 'Actual official fusion/query-mask-score identity differs')
    return dict(masks=instances.masks, scores=instances.scores, labels=instances.labels, query_ids=query_ids,
                class_probability=probabilities[query_ids, 0], mask_quality=quality)

def export_official(np, masks, endpoint):
    rows = []
    packet = endpoint['official']
    for index in range(100):
        rle = masks.encode(np.asfortranarray(packet['masks'][index].numpy().astype(np.uint8)))
        counts = rle['counts'].decode('ascii') if isinstance(rle['counts'], bytes) else rle['counts']
        rows.append(dict(query_id=int(packet['query_ids'][index]), label=int(packet['labels'][index]),
                         score=float(packet['scores'][index]), class_probability=float(packet['class_probability'][index]),
                         mask_quality=float(packet['mask_quality'][index]),
                         segmentation=dict(size=list(rle['size']), counts=counts)))
    return rows


def check_spec(spec):
    require(spec['schema'] == SCHEMA and spec['phase'].startswith('native_geometry_paired_passive_population_') and
        spec['modes'] == list(MODES) and set(spec['checkpointmap']) == set(spec['qualificationmap']) == set(MODES) and
        type(spec['expected_epoch']) is int and spec['expected_epoch'] in range(12, 25, 2) and
        spec['evaluation_images'] == 187 and spec['raw_GT'] == 380 and spec['CPU_affinity'] == [0, 1] and
        spec['allocator_limit_MiB'] == 4096 and spec['whole_GPU_owner_limit_MiB'] == 8192 and
        spec['deadline_seconds'] == 900 and spec['automatic_retry'] is False and
        spec['head_seed_base'] == 100000 and spec['matching_seed_base'] == 1000000 and
        spec['matching_num_points'] == 12544 and spec['occupancy_definitions'] ==
            ['direct_nearest800_point', 'original_position_proxy'],
        'Locked all3 even E12..24/full187/380/stock12544/resources/occupancy contract required')


def prerequisite_terminals(spec, evaluator):
    """Open only qualification/receipt metadata, before any scientific hash."""
    for mode in MODES:
        q = read(spec['qualificationmap'][mode]['path'])
        require(q['mode'] == mode, 'Training qualification mode differs')
        evaluator.terminal_guard(read(q['parent_receipt']['path']), q['phase'])
    q = read(spec['formal_evaluation_qualification']['path'])
    require(q['schema'] == FORMAL_QUAL_SCHEMA and q['status'] == FORMAL_QUAL_STATUS,
        'Actual independent formal full-DEV evaluation qualification required')
    evaluator.terminal_guard(read(q['parent_receipt']['path']), q['phase'])
    return q


def admission(spec_path, spec):
    require(Path(spec['GPU_admission_file']).is_absolute(), 'Absolute admission metadata path required')
    adm_ref = reference(spec['GPU_admission_file']); adm = read(spec['GPU_admission_file'])
    entry = reference(__file__); consumer = reference(Path(__file__).with_name('qualify_cpu.py'))
    spec_ref = reference(spec_path)
    owner = reference(spec_path.parent / 'finite_point_cache_execution_20261005_v2' / 'run_owned_gpu_limited.py')
    require(owner['sha256'] == OWNER_SHA and adm['driver_sha256'] == OWNER_SHA and
        adm['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and adm['automatic_retry'] is False and
        adm['phase'] == spec['phase'] and adm['allowed_cpus'] == [0, 1] and adm['cuda_visible_devices'] == '0' and
        adm['qualified_leaf_deadline_seconds'] == 900 and adm['whole_GPU_used_limit_MiB'] == 8192 and
        Path(adm['entry']).resolve(strict=True) == Path(__file__).resolve() and
        adm['entry_args'] == ['--spec', str(spec_path)], 'Own entry/admission/resources differ')
    require(spec['entry_source'] == entry and spec['CPU_reader'] == consumer and
        adm['bound_source_sha256'].get(str(spec_path)) == spec_ref['sha256'], 'Current entry/reader/SPEC absent from admission')
    if 'scientific_spec' in adm:
        require(adm['scientific_spec'] == spec_ref, 'Own admission scientific SPEC differs')
    for row in spec['sources']:
        bound(row)
        require(adm['bound_source_sha256'].get(row['path']) == row['sha256'], 'Source absent from own admission closure')
    for key in ('entry_source', 'CPU_reader', 'helper_source', 'evaluation_source', 'evaluation_CPU_source',
                'training_source', 'initializer_source', 'micro_preflight_source', 'native_helpers', 'adapter_source',
                'hooks_source', 'geometry_source'):
        require(spec[key] in spec['sources'], 'Source role absent from own closure: ' + key)
    require(spec['helper_source']['bytes'] == 16788 and spec['helper_source']['sha256'] == HELPER_SHA,
        'Immutable passive helper differs')
    return adm_ref, owner


def formal_evaluation(spec, q):
    """Bind the qualified official endpoint without reading or selecting its AP."""
    bound(spec['formal_evaluation_qualification'])
    for key in ('source', 'reader_source', 'spec', 'report', 'start', 'parent_receipt', 'GPU_admission', 'owner_driver'):
        bound(q[key])
    require(q['source'] == spec['evaluation_source'] and q['reader_source'] == spec['evaluation_CPU_source'] and
        q['modes'] == list(MODES) and q['expected_epoch'] == spec['expected_epoch'] and
        q['checkpointmap'] == spec['checkpointmap'] and q['qualificationmap'] == spec['qualificationmap'] and
        q['all187_images_all380_original_GT_preserved'] is True and q['all100_predictions_per_image_preserved'] is True,
        'Formal evaluation qualification does not describe these complete same-CP endpoints')
    old_spec = read(q['spec']['path']); adm = read(q['GPU_admission']['path']); receipt = read(q['parent_receipt']['path'])
    require(old_spec['schema'] == 'ROOT-native-geometry-paired-common-evaluation-spec-v1' and
        old_spec['phase'] == q['phase'] == adm['phase'] and old_spec['expected_epoch'] == spec['expected_epoch'] and
        old_spec['checkpointmap'] == spec['checkpointmap'] and old_spec['qualificationmap'] == spec['qualificationmap'] and
        old_spec['DEV'] == spec['DEV'] and old_spec['image_root'] == spec['image_root'] and
        old_spec['evaluation_images'] == 187 and adm['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and
        adm['automatic_retry'] is False and receipt['admission_sha256'] == q['GPU_admission']['sha256'] and
        adm['driver_sha256'] == q['owner_driver']['sha256'] == OWNER_SHA and
        Path(adm['entry']).resolve() == Path(spec['evaluation_source']['path']).resolve() and
        adm['entry_args'] == ['--spec', q['spec']['path']] and adm['allowed_cpus'] == [0, 1] and
        adm['cuda_visible_devices'] == '0' and adm['qualified_leaf_deadline_seconds'] == old_spec['deadline_seconds'] and
        adm['whole_GPU_used_limit_MiB'] == 8192 and
        adm['bound_source_sha256'].get(q['spec']['path']) == q['spec']['sha256'], 'Formal evaluation actual admission chain differs')
    for row in old_spec['sources']:
        bound(row)
        require(row in spec['sources'] and adm['bound_source_sha256'].get(row['path']) == row['sha256'],
            'Formal evaluation source closure differs')
    for key in ('training_source', 'initializer_source', 'micro_preflight_source', 'native_helpers', 'adapter_source', 'hooks_source'):
        require(old_spec[key] == spec[key], 'Formal evaluation model/source role differs: ' + key)
    report = read(q['report']['path'])
    require(report['schema'] == 'native-geometry-paired-common-evaluation-report-v1' and
        report['status'] == 'TECHNICAL_FULL_NATIVE_EVALUATION_EXPORT_COMPLETE_NO_AP' and report['error'] is None and
        report['source'] == q['source'] and report['spec'] == q['spec'] and report['phase'] == q['phase'] and
        report['expected_epoch'] == spec['expected_epoch'] and report['DEV'] == spec['DEV'] and
        report['checkpointmap'] == spec['checkpointmap'] and report['qualificationmap'] == spec['qualificationmap'],
        'Qualified formal evaluation report identity differs')
    record = report['mode_reports']['paired']; result = q['mode_results']['paired']
    require(record['checkpoint'] == result['checkpoint'] == spec['checkpointmap']['paired'] and
        record['qualification'] == result['qualification'] == spec['qualificationmap']['paired'] and
        record['mode'] == record['checkpoint_mode'] == 'paired' and record['completed_epoch'] == spec['expected_epoch'] and
        record['images'] == 187 and record['raw_GT'] == 380 and record['predictions_count'] == 18700 and
        record['query_score_factors_count'] == 18700 and record['cleanup']['allocated_after'] == 0 and
        record['predictions'] == result['predictions'] and record['image_rows'] == result['image_rows'],
        'Formal paired-on complete population/result join differs')
    image_rows = read(bound(record['image_rows'])); predictions = read(bound(record['predictions']))
    factors = read(bound(record['query_score_factors']))
    require(len(image_rows) == 187 and len(predictions) == len(factors) == 18700 and
        sum(row['annotation_count'] for row in image_rows) == 380, 'Formal complete population missing')
    images = {row['image_id']: row for row in image_rows}
    pred = {(row['image_id'], row['query_id']): row for row in predictions}
    fac = {(row['image_id'], row['query_id']): row for row in factors}
    require(len(images) == 187 and len(pred) == len(fac) == 18700 and set(pred) == set(fac) and
        all({qid for iid, qid in pred if iid == image_id} == set(range(100)) for image_id in images),
        'Formal per-image native query IDs are not unique and complete')
    return dict(qualification=spec['formal_evaluation_qualification'], report=q['report'],
        image_rows_ref=record['image_rows'], predictions_ref=record['predictions'], factors_ref=record['query_score_factors'],
        images=images, predictions=pred, factors=fac)


def rng(torch, helper):
    return helper.cpu_tree(torch, dict(CPU=torch.random.get_rng_state(), CUDA=torch.cuda.get_rng_state(0)))


def hook_signature(model):
    return {name: dict(pre=list(module._forward_pre_hooks), post=list(module._forward_hooks))
        for name, module in model.named_modules()}


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def stock_final_match(torch, pref, old, helper, head, final_class, final_mask, data, identity, seed):
    costs = head.assigner.match_costs
    require(head.num_points == 12544 and head.num_queries == 100 and head.num_classes == 1 and
        [type(cost).__name__ for cost in costs] == ['ClassificationCost', 'CrossEntropyLossCost', 'DiceCost'] and
        [float(cost.weight) for cost in costs] == [2., 5., 5.] and costs[1].use_sigmoid is True and
        costs[2].pred_act is True and costs[2].eps == 1. and costs[2].naive_dice is True,
        'Actual native12544/2:5:5/Dice-eps1 matching configuration differs')
    # Ground truth is first used for this posterior matcher, never in forward.
    gt = old.posterior_gt(head, data, identity)
    G = len(identity['ann_ids'])
    require(G <= 100 and gt.masks.dtype == torch.int64 and tuple(gt.masks.shape) == (G, 800, 800),
        'Complete raw GT order/native int64 raster exceeds matching capacity')
    original_costs = head.assigner.match_costs
    external_before = rng(torch, helper)
    with torch.random.fork_rng(devices=[0]):
        torch.random.default_generator.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        private_before = rng(torch, helper)
        with torch.no_grad(), old.observe_native_costs(head) as captured:
            targets = head._get_targets_single(final_class[0], final_mask[0], gt, data['data_samples'][0].metainfo)
        private_after = rng(torch, helper)
    external_after = rng(torch, helper)
    helper.require_byte_identity(torch, external_before, external_after, 'matching_external_CPU_CUDA_RNG')
    require(head.assigner.match_costs is original_costs and len(targets) == 7, 'Native matching observer/7tuple not restored')
    positive = targets[4].detach().cpu(); assigned = targets[6].pos_assigned_gt_inds.detach().cpu()
    require(positive.dtype == assigned.dtype == torch.int64 and positive.numel() == assigned.numel() == G and
        sorted(assigned.tolist()) == list(range(G)) and len(set(positive.tolist())) == G and
        all(0 <= q < 100 for q in positive.tolist()) and torch.equal(targets[2], gt.masks[targets[6].pos_assigned_gt_inds]),
        'Native stock final assignment dropped/reordered a raw GT')
    q_by_GT = [-1] * G; ann_by_query = [-1] * 100
    for query, gt_index in zip(positive.tolist(), assigned.tolist()):
        q_by_GT[gt_index] = query; ann_by_query[query] = identity['ann_ids'][gt_index]
    names = ('classification_weight2', 'BCE_weight5', 'Dice_weight5')
    if G:
        require(len(captured['components']) == 3 and len(captured['totals']) == 1 and
            [row['cost_index'] for row in captured['components']] == list(range(3)), 'Expected 3 actual costs and one actual SciPy call')
        tensors = {name: row['tensor'] for name, row in zip(names, captured['components'])}
        tensors['total'] = captured['totals'][0]
    else:
        require(not captured['components'] and not captured['totals'], 'Empty GT must not claim cost/SciPy observations')
        tensors = {name: torch.empty((100, 0), dtype=torch.float32, device='cpu') for name in names + ('total',)}
    require(all(value.dtype == torch.float32 and tuple(value.shape) == (100, G) and
        bool(torch.isfinite(value).all()) for value in tensors.values()), 'Actual cost matrix shape/dtype/finite differs')
    return dict(image_id=identity['image_id'], ann_ids=list(identity['ann_ids']), GT800=pref.digest_tensor(gt.masks),
        labels=pref.digest_tensor(gt.labels), positive_query_indices=positive.tolist(), assigned_GT_indices=assigned.tolist(),
        q_by_GT=q_by_GT, ann_by_query=ann_by_query, components=encode_tree(torch, pref, tensors, values=True),
        classification_already_weight2=True, BCE_already_weight5=True, Dice_already_weight5=True,
        native_cost_weights=[2., 5., 5.], Dice_eps=1., num_points=12544, seed=seed,
        get_targets_single_calls=1, cost_calls=len(captured['components']), SciPy_calls=len(captured['totals']),
        total_is_actual_SciPy_CPU_argument=bool(G), semantic_empty_not_SciPy_observed=not bool(G),
        private_RNG=dict(before=encode_tree(torch, pref, private_before), after=encode_tree(torch, pref, private_after)),
        external_RNG_before=encode_tree(torch, pref, external_before), external_RNG_after=encode_tree(torch, pref, external_after),
        external_RNG_restored=True, GT_order='raw annotation_mapping/loader order',
        all_raw_GT_retained=True, cost_integrals_independently_recomputed=False, loss_executed=False)


def compact_capture(torch, pref, capture):
    # No 200x200 valid map or dense F/M0/rho is written. The reader rebuilds
    # metadata validity independently; this is not a cell-area GT measure.
    result = {key: value for key, value in capture.items() if key != 'capture'}
    actual = capture['capture']
    require(capture['calls'] == capture['expected_calls'] == 1 and capture['hooks_removed'] is True and
        actual['geometry']['mode'] == 'paired' and actual['geometry']['eligibility_frame'] == 'canonical',
        'Actual paired/canonical passive capture contract differs')
    result['capture'] = dict(geometry=encode_tree(torch, pref, actual['geometry'], values=True),
        coefficient_residual=encode_tree(torch, pref, actual['coefficient_residual']),
        actual_valid_fraction=encode_tree(torch, pref, actual['actual_valid_fraction']),
        F_shape=list(actual['F_shape']), e0_shape=list(actual['e0_shape']), native_query_ids=list(actual['native_query_ids']))
    empty = actual['geometry']['eligible_counts'] == 0
    require(bool((actual['coefficient_residual'][empty] == 0).all()), 'E0 query residual is not zero')
    result['empty_query_residual_exact_zero'] = True
    result['module_has_support'] = encode_tree(torch, pref,
        actual['geometry']['rho_mass'] > actual['geometry']['configuration']['mass_eps'], values=True)
    result['M0_all_nonpositive_supported'] = False
    result['outside_donor_slots_applied'] = False
    return result


def compare_official(formal, identity, final_fields, predictions, factors):
    prior = formal['images'][identity['image_id']]
    require(prior['dataset_index'] == identity['dataset_index'] and prior['image'] == identity['image'] and
        prior['ann_ids'] == identity['ann_ids'] and
        prior['original_shape'] == identity['original_shape'] and prior['actual_img_shape'] == identity['actual_img_shape'] and
        prior['batch_input_shape'] == identity['batch_input_shape'] and prior['scale_factor'] == identity['scale_factor'] and
        prior['head_seed'] == 100000 + identity['image_id'] and prior['final_fields'] == final_fields and
        prior['mode'] == prior['checkpoint_mode'] == 'paired', 'Same-CP formal final dense digest/input regression differs')
    for p, factor in zip(predictions, factors):
        key = (p['image_id'], p['query_id']); original = formal['predictions'][key]; old_factor = formal['factors'][key]
        require(set(p) == set(original) and p['segmentation'] == original['segmentation'] and
            p['category_id'] == original['category_id'] and struct.pack('<f', p['score']) == struct.pack('<f', original['score']),
            'Same-CP official per-query RLE/category/FP32 score regression differs')
        require(set(factor) == set(old_factor) and all(struct.pack('<f', factor[name]) == struct.pack('<f', old_factor[name])
            for name in ('class_probability', 'mask_quality', 'official_score')), 'Same-CP FP32 official factor regression differs')
    return dict(final_dense_digest_exact=True, official_all100_RLE_and_FP32_factors_exact=True,
        joined_by_image_and_query_ID=True, unobserved_all10_identity_claimed=False)


def process_image(torch, np, mask_utils, pref, old, hooks, helper, model, dataset, coco, index, spec, formal):
    data, samples, gray, raw, identity = old.load_image(model, dataset, coco, dataset.get_data_info(index))
    del gray, raw
    require(len(samples) == 1 and all(not any(key in sample for key in ('gt_instances', 'ignored_instances', 'gt_sem_seg'))
        for sample in samples), 'Annotations survived forward sample stripping')
    identity['dataset_index'] = index
    metadata = [sample.metainfo for sample in samples]
    external_before = rng(torch, helper); hooks_before = hook_signature(model)
    head_seed = 100000 + identity['image_id']
    with torch.random.fork_rng(devices=[0]):
        torch.random.default_generator.manual_seed(head_seed); torch.cuda.manual_seed_all(head_seed)
        private_before = rng(torch, helper)
        with torch.no_grad():
            features = model.extract_feat(data['inputs'])
            with helper.passive_adapter_capture(torch, model.panoptic_head.paired_mask_readout, metadata,
                include_coefficients=False) as capture:
                with hooks.paired_readout_context(torch, model.panoptic_head, metadata, apply=True) as hook_record:
                    classes, masks = model.panoptic_head(features, samples)
            require(len(classes) == len(masks) == 10 and all(tuple(value.shape) == (1, 100, 2) and
                value.dtype == torch.float32 and bool(torch.isfinite(value).all()) for value in classes) and
                all(tuple(value.shape) == (1, 100, 200, 200) and value.dtype == torch.float32 and
                bool(torch.isfinite(value).all()) for value in masks), 'Actual all10 native FP32 readouts differ')
            all10 = dict(classes=[pref.digest_tensor(value) for value in classes], masks=[pref.digest_tensor(value) for value in masks])
            final_class, final_mask = classes[-1], masks[-1]
            del classes, masks, features
            official = helper.cpu_tree(torch, official_endpoint(torch, model, [final_class], [final_mask], samples, identity))
        private_after = rng(torch, helper)
    head_external_after = rng(torch, helper)
    helper.require_byte_identity(torch, external_before, head_external_after, 'head_external_CPU_CUDA_RNG')
    require(hooks_before == hook_signature(model) and capture['hooks_removed'] is True,
        'Native/passive head hooks were not removed before posterior matching')
    predictions, factors = [], []
    for p in export_official(np, mask_utils, dict(official=official)):
        predictions.append(dict(image_id=identity['image_id'], query_id=p['query_id'], category_id=1,
            score=p['score'], segmentation=p['segmentation']))
        factors.append(dict(image_id=identity['image_id'], query_id=p['query_id'], class_probability=p['class_probability'],
            mask_quality=p['mask_quality'], official_score=p['score']))
    final_fields = dict(classes=pref.digest_tensor(final_class), masks=pref.digest_tensor(final_mask))
    regression = compare_official(formal, identity, final_fields, predictions, factors)
    matched = stock_final_match(torch, pref, old, helper, model.panoptic_head, final_class, final_mask, data, identity,
        1000000 + identity['image_id'])
    external_after = rng(torch, helper)
    helper.require_byte_identity(torch, external_before, external_after, 'image_external_CPU_CUDA_RNG')
    require(hooks_before == hook_signature(model) and all(parameter.grad is None and not parameter.requires_grad
        for parameter in model.parameters()) and all(not module.training for module in model.modules()),
        'Image changed hooks/frozen flags/parameter gradients/eval modes')
    row = dict(identity=identity, image_id=identity['image_id'], dataset_index=index,
        ann_ids=list(identity['ann_ids']), annotation_count=len(identity['ann_ids']), metadata=helper.metadata_snapshot(metadata),
        GT800=matched['GT800'], labels=matched['labels'], checkpoint=spec['checkpointmap']['paired'], mode='paired', apply=True,
        all10_fields=all10, final_fields=final_fields, all10_is_observed_digest_only=True, formal_regression=regression,
        head_seed=head_seed, matching_seed=matched['seed'], paired_hook=hook_record,
        head_private_RNG=dict(before=encode_tree(torch, pref, private_before), after=encode_tree(torch, pref, private_after)),
        external_RNG_before=encode_tree(torch, pref, external_before), external_RNG_after=encode_tree(torch, pref, external_after),
        external_RNG_restored=True, hook_signature_before=json_digest(hooks_before), hook_signature_after=json_digest(hook_signature(model)),
        native_passive_matching_observers_removed=True, feature_extraction_calls=1, head_forward_calls=1, stock_final_matching_calls=1,
        no_GT_in_forward=True, GT_posthoc_matching=True, loss_executed=False, optimizer_executed=False,
        prediction_query_ids=official['query_ids'].tolist(), predictions=100, query_filter=False, all_raw_GT_retained=True)
    capture_row = dict(image_id=identity['image_id'], dataset_index=index, ann_ids=list(identity['ann_ids']),
        capture=compact_capture(torch, pref, capture))
    return row, capture_row, matched, predictions, factors


def run(spec_file):
    started = time.monotonic()
    require(Path(spec_file).is_absolute(), 'Absolute --spec required')
    spec_path = Path(spec_file).resolve(strict=True); spec = read(spec_path); check_spec(spec)
    output = Path(spec['output_directory'])
    require(output.is_absolute() and not output.exists() and os.environ.get('CUDA_VISIBLE_DEVICES') == '0',
        'Fresh output/ROOT CUDA0 required')
    evaluator = load_source(spec['evaluation_source'], 'population_sealed_evaluator')
    formal_q = prerequisite_terminals(spec, evaluator)
    # All four actual owner terminal gates have passed before these bounds.
    training = evaluator.qualified_training(spec)
    admission_ref, owner_ref = admission(spec_path, spec)
    formal = formal_evaluation(spec, formal_q)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '2'
    os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
    import psutil
    psutil.Process().cpu_affinity([0, 1])
    import numpy as np
    import torch
    from pycocotools import mask as mask_utils
    from mmengine import Config
    from mmengine.model import revert_sync_batchnorm
    from mmengine.registry import init_default_scope
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    from mmdet.datasets import CocoDataset
    from mmdet.datasets.api_wrappers import COCO
    require(psutil.Process().cpu_affinity() == [0, 1] and not torch.cuda.is_initialized(), 'CPU01/cold CUDA required')
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    register_all_modules(init_default_scope=False); init_default_scope('mmdet')
    pref = load_source(spec['micro_preflight_source'], 'population_native_pref')
    initializer = load_source(spec['initializer_source'], 'population_initializer')
    old = load_source(spec['native_helpers'], 'population_native_helpers'); old.torch = torch
    am = load_source(spec['adapter_source'], 'population_adapter')
    hooks = load_source(spec['hooks_source'], 'population_hooks'); helper = load_source(spec['helper_source'], 'population_helper')
    class_names = tuple(CocoDataset.METAINFO['classes'])
    model, state, meta, loading = evaluator.prepare_model(torch, pref, initializer, am, hooks, MODELS,
        revert_sync_batchnorm, Config, class_names, spec, training['paired'], 'paired')
    dataset = CocoDataset(ann_file=str(bound(spec['DEV'])), data_prefix=dict(img=spec['image_root']),
        metainfo=dict(classes=('ship',)), pipeline=[], filter_cfg=None, test_mode=True)
    coco = COCO(str(bound(spec['DEV'])))
    require(len(dataset) == len(coco.imgs) == 187 and len(coco.anns) == 380 and set(coco.imgs) == set(formal['images']),
        'Complete same-CP DEV187/380 population required')
    torch.cuda.init(); capacity = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1., 4096 * 1024 ** 2 / capacity), 0)
    torch.cuda.reset_peak_memory_stats(0)
    output.mkdir(parents=True)
    model.to('cuda').eval()
    try:
        before = helper.execution_snapshot(torch, model, cuda_device=0)
        require(len(before['state']) == 612 and pref.state_proof(before['state']) == loading['trained_fullstate'],
            'Population START does not equal qualified full612 checkpoint')
        write(output / 'START.json', dict(schema='native-geometry-paired-passive-population-start-v1',
            phase=spec['phase'], source=reference(__file__), spec=reference(spec_path), expected_epoch=spec['expected_epoch'],
            checkpoint=spec['checkpointmap']['paired'], formal_evaluation_qualification=spec['formal_evaluation_qualification'],
            images=187, raw_GT=380, no_AP=True, no_retry=True, all_prerequisite_terminal_gates_passed=True))
        image_rows, capture_rows, costs, predictions, factors = [], [], [], [], []
        for index in range(187):
            require(time.monotonic() - started < 900, 'Single-attempt full population deadline exceeded')
            row, capture, cost, image_predictions, image_factors = process_image(torch, np, mask_utils, pref, old, hooks,
                helper, model, dataset, coco, index, spec, formal)
            image_rows.append(row); capture_rows.append(capture); costs.append(cost)
            predictions.extend(image_predictions); factors.extend(image_factors)
            if (index + 1) % 25 == 0 or index == 186:
                print(json.dumps(dict(images=index + 1, elapsed_seconds=time.monotonic() - started)), flush=True)
            del row, capture, cost, image_predictions, image_factors
        after = helper.execution_snapshot(torch, model, cuda_device=0)
        helper.require_byte_identity(torch, before, after, 'population_full612_state_CPU_CUDA_RNG_flags_hooks')
        require(len(image_rows) == len(capture_rows) == len(costs) == 187 and len(predictions) == len(factors) == 18700 and
            sum(row['annotation_count'] for row in image_rows) == 380 and len({row['image_id'] for row in image_rows}) == 187,
            'Complete population/all100/380 GT evidence missing')
        proof = dict(schema='native-geometry-paired-passive-population-state-RNG-proof-v1', loading_proof=loading,
            before=encode_tree(torch, pref, before), after=encode_tree(torch, pref, after),
            full612_state_CPU_CUDA_RNG_flags_hooks_byte_exact=True, CUDA_RNG_observed=True,
            unobserved_all10_population_identity_claimed=False, formal_final_and_official_all100_regression_exact=True)
        for name, value in (('IMAGE_ROWS.json', image_rows), ('CAPTURE_ROWS.json', capture_rows), ('NATIVE_COST_ROWS.json', costs),
                            ('ALL_PREDICTIONS.json', predictions), ('QUERY_SCORE_FACTORS.json', factors), ('STATE_RNG_PROOF.json', proof)):
            write(output / name, value)
        report = dict(schema=REPORT_SCHEMA, status='COMPLETE_PASSIVE_PAIRED_POPULATION_DIAGNOSTIC_PENDING_ROOT_TERMINAL',
            phase=spec['phase'], source=reference(__file__), CPU_reader=spec['CPU_reader'], spec=reference(spec_path),
            GPU_admission=admission_ref, owner_driver=owner_ref, expected_epoch=spec['expected_epoch'],
            checkpointmap=spec['checkpointmap'], qualificationmap=spec['qualificationmap'], mode='paired',
            formal_evaluation_qualification=spec['formal_evaluation_qualification'], formal_evaluation_report=formal['report'],
            formal_image_rows=formal['image_rows_ref'], formal_predictions=formal['predictions_ref'], formal_factors=formal['factors_ref'],
            images=187, raw_GT=380, predictions_count=18700, query_score_factors_count=18700,
            feature_extraction_calls=187, head_forward_calls=187, stock_final_matching_calls=187,
            no_GT_in_forward=True, GT_used_only_in_loader_and_posterior_matching=True, no_optimizer=True, no_loss=True, no_AP=True,
            query_filter=False, outside_donor_slots_applied=False, all_raw_GT_retained=True,
            all10_is_observed_digest_only=True, unobserved_all10_population_identity_claimed=False,
            formal_final_and_official_all100_regression_exact=True, full_rho_M0_unknown=True, M0_all_nonpositive_supported=False,
            occupancy_computed=False, occupancy_definitions=spec['occupancy_definitions'], GT_position_is_not_feature_purity=True,
            actual_total_is_native_SciPy_argument=True, empty_GT_total_semantic_only=True,
            component_sum_CPU_GPU_bitidentity_claimed=False, cost_integrals_independently_recomputed=False,
            outputs={key: reference(output / name) for key, name in
                (('image_rows', 'IMAGE_ROWS.json'), ('capture_rows', 'CAPTURE_ROWS.json'), ('native_cost_rows', 'NATIVE_COST_ROWS.json'),
                 ('official_predictions', 'ALL_PREDICTIONS.json'), ('query_score_factors', 'QUERY_SCORE_FACTORS.json'),
                 ('state_RNG_proof', 'STATE_RNG_PROOF.json'))},
            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(0), error=None)
        del model, state, meta, loading, before, after, dataset, coco, training
        gc.collect(); torch.cuda.synchronize(0)
        torch._C._cuda_clearCublasWorkspaces(); torch.cuda.empty_cache()
        cleanup = torch.cuda.memory_allocated(0)
        require(cleanup == 0, 'Live CUDA tensor storage survived full population cleanup')
        require(time.monotonic() - started < 900, 'Single-attempt full population deadline exceeded')
        report.update(cleanup_allocated_bytes=cleanup, runtime_seconds=time.monotonic() - started)
        write(output / 'REPORT.json', report)
    finally:
        if 'model' in locals():
            model.to('cpu')
        gc.collect(); torch._C._cuda_clearCublasWorkspaces(); torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    run(parser.parse_args().spec)
