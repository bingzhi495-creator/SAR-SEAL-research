"""Inert ROOT-owned one-image passive paired replay; no action runs on import.

The owner executes this only after all three even-epoch training qualifications.
One cached backbone extraction is reused: this is head/fusion replay identity,
not two independent complete backbone forwards. Both endpoints apply the paired
adapter. Full outputs/state stay in RAM for byte comparison and are not exported.
GT is accessed by the existing loader, stripped before forward, never analysed.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time

SCHEMA = 'ROOT-native-geometry-paired-passive-replay-spec-v1'
REPORT_SCHEMA = 'native-geometry-paired-passive-replay-report-v1'
MODES = ('native0', 'paired', 'fg_fine')
OWNER_SHA = 'ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
HELPER_SHA = '54d3b9fb0866fe6e88cea7272b7667fc70997644c2c8732762529968a32f5cb4'


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


def check_spec(spec):
    require(spec['schema'] == SCHEMA and spec['phase'].startswith('native_geometry_paired_passive_replay_') and
            spec['modes'] == list(MODES) and set(spec['checkpointmap']) == set(spec['qualificationmap']) == set(MODES) and
            type(spec['expected_epoch']) is int and spec['expected_epoch'] in range(8, 25, 2) and
            spec['dataset_index'] == 0 and spec['expected_cases'] == 1 and spec['evaluation_images'] == 187 and
            spec['CPU_affinity'] == [0, 1] and spec['allocator_limit_MiB'] == 4096 and
            spec['whole_GPU_owner_limit_MiB'] == 8192 and spec['deadline_seconds'] == 900 and
            spec['automatic_retry'] is False and spec['cached_backbone_replay'] is True,
            'Locked one-case/all3-qualified-even-epoch/900s/4096/8192 contract required')


def admission(spec_path, spec):
    require(Path(spec['GPU_admission_file']).is_absolute(), 'Absolute admission metadata path required')
    adm_ref = reference(spec['GPU_admission_file'])
    adm = read(spec['GPU_admission_file'])
    entry, consumer, spec_ref = reference(__file__), reference(Path(__file__).with_name('qualify_cpu.py')), reference(spec_path)
    owner = reference(spec_path.parent / 'finite_point_cache_execution_20261005_v2' / 'run_owned_gpu_limited.py')
    require(owner['sha256'] == OWNER_SHA and adm['driver_sha256'] == OWNER_SHA and
            adm['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and adm['automatic_retry'] is False and
            adm['phase'] == spec['phase'] and adm['allowed_cpus'] == [0, 1] and adm['cuda_visible_devices'] == '0' and
            adm['qualified_leaf_deadline_seconds'] == 900 and adm['whole_GPU_used_limit_MiB'] == 8192 and
            Path(adm['entry']).resolve(strict=True) == Path(__file__).resolve() and
            adm['entry_args'] == ['--spec', str(spec_path)], 'Actual own admission/source/resources differ')
    require(spec['entry_source'] == entry and spec['CPU_reader'] == consumer and
            adm['bound_source_sha256'].get(str(spec_path)) == spec_ref['sha256'], 'Current entry/reader/SPEC not bound')
    for row in spec['sources']:
        bound(row)
        require(adm['bound_source_sha256'].get(row['path']) == row['sha256'], 'Source absent from own admission closure')
    for key in ('entry_source', 'CPU_reader', 'helper_source', 'evaluation_source', 'evaluation_CPU_source',
                'training_source', 'initializer_source', 'micro_preflight_source', 'native_helpers', 'adapter_source', 'hooks_source'):
        require(spec[key] in spec['sources'], 'Executing source role absent from closure: ' + key)
    require(spec['helper_source']['bytes'] == 16788 and spec['helper_source']['sha256'] == HELPER_SHA,
            'Immutable passive helper differs')
    return adm_ref, owner


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


def replay(torch, model, features, samples, identity, seed, hooks, helper, capture):
    from contextlib import nullcontext
    metadata = [sample.metainfo for sample in samples]
    observer = helper.passive_adapter_capture(torch, model.panoptic_head.paired_mask_readout, metadata,
                                              include_coefficients=True) if capture else nullcontext(None)
    with torch.random.fork_rng(devices=[0]):
        torch.random.default_generator.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        private_before = helper.cpu_tree(torch, dict(CPU=torch.random.get_rng_state(), CUDA=torch.cuda.get_rng_state(0)))
        with torch.no_grad():
            with observer as recorded:
                with hooks.paired_readout_context(torch, model.panoptic_head, metadata, apply=True) as hook_record:
                    classes, masks = model.panoptic_head(features, samples)
                require(all(bool(torch.isfinite(value).all()) for value in classes + masks),
                        'Every actual native readout must be finite')
                official = official_endpoint(torch, model, classes, masks, samples, identity)
                endpoint = helper.endpoint_snapshot(torch, classes, masks, official,
                    dict(checkpoint=identity['checkpoint'], input_identity=identity['input_identity'],
                         head_seed=seed, mode='paired', metadata=helper.metadata_snapshot(metadata)))
        private_after = helper.cpu_tree(torch, dict(CPU=torch.random.get_rng_state(), CUDA=torch.cuda.get_rng_state(0)))
    return endpoint, recorded, hook_record, dict(before=private_before, after=private_after)


def run(spec_file):
    started = time.monotonic()
    require(Path(spec_file).is_absolute(), 'Absolute --spec required')
    spec_path = Path(spec_file).resolve(strict=True); spec = read(spec_path); check_spec(spec)
    output = Path(spec['output_directory'])
    require(output.is_absolute() and not output.exists() and os.environ.get('CUDA_VISIBLE_DEVICES') == '0',
            'Fresh output and ROOT CUDA0 environment required')
    # Only source + qualification/terminal metadata before E's all3 terminal gates.
    evaluator = load_source(spec['evaluation_source'], 'passive_replay_sealed_evaluator')
    training = evaluator.qualified_training(spec)
    adm_ref, owner = admission(spec_path, spec)
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '2'
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
    pref = load_source(spec['micro_preflight_source'], 'passive_replay_native_pref')
    initializer = load_source(spec['initializer_source'], 'passive_replay_initializer')
    old = load_source(spec['native_helpers'], 'passive_replay_native_helpers'); old.torch = torch
    am = load_source(spec['adapter_source'], 'passive_replay_adapter')
    hooks = load_source(spec['hooks_source'], 'passive_replay_hooks')
    helper = load_source(spec['helper_source'], 'passive_replay_helper')
    classes = tuple(CocoDataset.METAINFO['classes'])
    require(len(classes) == 80 and classes[8] == 'boat', 'Official COCO category order differs')
    model, state, meta, loading = evaluator.prepare_model(torch, pref, initializer, am, hooks, MODELS,
        revert_sync_batchnorm, Config, classes, spec, training['paired'], 'paired')
    dataset = CocoDataset(ann_file=str(bound(spec['DEV'])), data_prefix=dict(img=spec['image_root']),
                         metainfo=dict(classes=('ship',)), pipeline=[], filter_cfg=None, test_mode=True)
    coco = COCO(str(bound(spec['DEV'])))
    require(len(dataset) == len(coco.imgs) == 187 and len(coco.anns) == 380, 'Original complete DEV metadata required')
    torch.cuda.init(); capacity = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1., 4096 * 1024 ** 2 / capacity), 0)
    output.mkdir(parents=True)
    model.to('cuda').eval()
    try:
        data, samples, gray, raw, identity = old.load_image(model, dataset, coco, dataset.get_data_info(0))
        del gray, raw
        require(len(samples) == 1 and identity['image_id'] == next(iter(coco.imgs)) and
                all(not any(key in sample for key in evaluator.GT_FIELDS) for sample in samples),
                'Fixed first image/GT-removed model samples required')
        seed = 100000 + identity['image_id']
        identity['checkpoint'] = spec['checkpointmap']['paired']
        identity['input_identity'] = dict(image=identity['image'], input=pref.digest_tensor(data['inputs']),
                                         actual_img_shape=identity['actual_img_shape'], original_shape=identity['original_shape'])
        with torch.no_grad():
            features = model.extract_feat(data['inputs'])
        feature_digests = [pref.digest_tensor(value) for value in features]
        before = helper.execution_snapshot(torch, model, cuda_device=0)
        reference_endpoint, unused, reference_hook, reference_private = replay(torch, model, features, samples, identity, seed, hooks, helper, False)
        middle = helper.execution_snapshot(torch, model, cuda_device=0)
        observed_endpoint, capture, observed_hook, observed_private = replay(torch, model, features, samples, identity, seed, hooks, helper, True)
        after = helper.execution_snapshot(torch, model, cuda_device=0)
        result = helper.qualify_passive_identity(torch, before, middle, middle, after,
                                                reference_endpoint, observed_endpoint, capture)
        helper.require_byte_identity(torch, reference_private, observed_private, 'private_CPU_CUDA_RNG')
        require(feature_digests == [pref.digest_tensor(value) for value in features] and
                result['full_state_and_CPU_CUDA_RNG_byte_exact'] is True and
                all(not value.requires_grad for value in model.parameters()) and
                all(value.grad is None for value in model.parameters()) and
                all(not module.training for module in model.modules()), 'Cached feature/frozen inference identity differs')
        require(time.monotonic() - started < 900, 'Single-attempt replay deadline exceeded')
        write(output / 'CAPTURE.json', encode_tree(torch, pref, capture, values=True))
        write(output / 'ENDPOINT_DIGESTS.json', dict(reference=encode_tree(torch, pref, reference_endpoint),
                                                   observed=encode_tree(torch, pref, observed_endpoint)))
        write(output / 'OFFICIAL_ENDPOINTS.json', dict(reference=export_official(np, mask_utils, reference_endpoint),
                                                    observed=export_official(np, mask_utils, observed_endpoint)))
        write(output / 'STATE_RNG_PROOF.json', dict(before=encode_tree(torch, pref, before),
                                                  middle=encode_tree(torch, pref, middle), after=encode_tree(torch, pref, after)))
        report = dict(schema=REPORT_SCHEMA, status='COMPLETE_PASSIVE_PAIRED_REPLAY_PENDING_ROOT_TERMINAL',
                      phase=spec['phase'], source=reference(__file__), CPU_reader=spec['CPU_reader'],
                      spec=reference(spec_path), GPU_admission=adm_ref, owner_driver=owner,
                      helper_source=spec['helper_source'], expected_epoch=spec['expected_epoch'],
                      checkpoint=spec['checkpointmap']['paired'], qualification=spec['qualificationmap']['paired'],
                      identity=identity, dataset_index=0, cases=1, seed=seed, result=result, loading_proof=loading,
                      reference_hook=reference_hook, observed_hook=observed_hook, cached_feature_digests=feature_digests,
                      private_RNG=dict(reference=encode_tree(torch, pref, reference_private),
                                       observed=encode_tree(torch, pref, observed_private)),
                      feature_extraction_calls=1, head_forward_calls=2, both_apply_true=True,
                      middle_snapshot_shared_boundary=True, full_backbone_independent_replay_claimed=False,
                      outputs={key: reference(output / name) for key, name in
                          (('capture', 'CAPTURE.json'), ('endpoint_digests', 'ENDPOINT_DIGESTS.json'),
                           ('official_endpoints', 'OFFICIAL_ENDPOINTS.json'), ('state_RNG_proof', 'STATE_RNG_PROOF.json'))},
                      no_GT_in_forward=True, no_GT_analysis=True, no_AP=True, no_optimizer=True,
                      full_rho_M0_unknown=True, GT_position_is_not_feature_purity=True)
        del features, data, samples, reference_endpoint, observed_endpoint, capture, before, middle, after
        del reference_private, observed_private
        del classes, model, state, meta, loading, training, dataset, coco, unused
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
        cleanup = torch.cuda.memory_allocated(0)
        require(cleanup == 0, 'Live CUDA tensor storage survived replay cleanup')
        report.update(cleanup_allocated_bytes=cleanup, runtime_seconds=time.monotonic() - started)
        write(output / 'REPORT.json', report)
    finally:
        gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    run(parser.parse_args().spec)
