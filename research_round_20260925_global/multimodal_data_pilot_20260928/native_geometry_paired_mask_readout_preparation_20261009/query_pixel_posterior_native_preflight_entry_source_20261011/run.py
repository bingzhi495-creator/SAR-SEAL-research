"""ROOT-owned one-image native integration entry; inert without --owner-execute.

An immutable fullFIT24 seed0 SPEC supplies construction metadata only. No old
readout checkpoint is loaded. One native teacher forward and two structural
backwards use the immutable injected helper; no optimizer or evaluation runs.
"""
import argparse
from functools import partial
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time
import types

SCHEMA = 'ROOT-query-pixel-posterior-native-preflight-entry-spec-v1'
OWNER_SHA = 'ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
PRIOR = (('teacher_receipt', 'teacher_phase'), ('preflight_receipt', 'preflight_phase'),
    ('short2_receipt', 'short2_phase'), ('prior_DEV_receipt', 'prior_DEV_phase'),
    ('prior_diag_receipt', 'prior_diag_phase'), ('previous_control_receipt', 'previous_control_phase'))
MODULES = ('preflight_source', 'micro_preflight_source', 'initializer_source', 'native_helpers',
    'native_preflight_helpers', 'official_evaluation_source', 'previous_control_entry_source', 'prior_diag_entry_source')


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


def bound(ref):
    require(isinstance(ref, dict) and set(ref) == {'path', 'bytes', 'sha256'} and
        Path(ref['path']).is_absolute() and type(ref['bytes']) is int and ref['bytes'] > 0 and
        isinstance(ref['sha256'], str) and len(ref['sha256']) == 64 and
        set(ref['sha256']) <= set('0123456789abcdef') and reference(ref['path']) == ref, 'Pinned reference differs')
    return Path(ref['path'])


def load_source(ref, name):
    path = bound(ref)
    value = path.read_bytes()
    require(path.suffix == '.py' and len(value) == ref['bytes'] and
        hashlib.sha256(value).hexdigest() == ref['sha256'], 'SOURCE changed before loading')
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(value, str(path), 'exec'), module.__dict__)
    return module


def raw_terminal(raw, phase):
    require(raw['status'] == 'ACTUAL_OWNED_TERMINAL' and raw['original_identity'] is not None and
        type(raw['returncode']) is int and raw['returncode'] == 0 and raw['root_wait_reaped'] is True and
        raw['held_handle_exit'] == dict(wait_state='WAIT_OBJECT_0', actual_exit_code=0) and
        raw['numeric_birth_absence']['original_birth_absent'] is True and raw['abort'] is None and
        raw['error'] is None and raw['wait_errors'] is False and raw['GPU_observation_errors'] == [],
        'STOP before hashes/imports: actual terminal0/reap/birth absence/noerrors required')
    peak = raw['whole_GPU_observed_peak_MiB']
    require(raw.get('phase', phase) == phase and raw['whole_GPU_used_limit_MiB'] == 8192 and
        raw['whole_leaf_VRAM_certified'] is False and
        raw['GPU_scope'] == 'whole GPU observed, not per-process WDDM allocation' and
        type(peak) in (int, float) and math.isfinite(peak) and 0 <= peak <= 8192, 'Prior whole-GPU scope differs')


def gate(spec_path, spec):
    require(spec['schema'] == SCHEMA and Path(spec['context']['path']).is_absolute(), 'Bound context metadata required')
    ctx = read(spec['context']['path'])  # Metadata solely to locate old raw terminals.
    raw = {}
    for receipt, phase in PRIOR:
        require(Path(ctx[receipt]['path']).is_absolute(), 'Absolute original raw receipt required')
        raw[receipt] = read(ctx[receipt]['path'])
        raw_terminal(raw[receipt], ctx[phase])
    require(ctx['schema'] == 'ROOT-shared-readout-objective-control-fullFIT24-spec-v1' and
        ctx['readout_seed']['seed_id'] == 0 and ctx['expected_epoch'] == 24, 'Immutable seed0 construction context required')
    require(spec['CPU_affinity'] == [0, 1] and spec['allocator_limit_MiB'] == 4096 and
        spec['whole_GPU_owner_limit_MiB'] == 8192 and type(spec['deadline_seconds']) is int and
        spec['deadline_seconds'] > 0 and os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'Owned CPU01/CUDA0 resources required')
    require(Path(spec['GPU_admission_file']).is_absolute(), 'Absolute current admission required')
    admission = read(spec['GPU_admission_file'])
    source, spec_ref = reference(__file__), reference(spec_path)
    require(admission['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and
        admission['automatic_retry'] is False and admission['phase'] == spec['phase'] and
        admission['driver_sha256'] == OWNER_SHA and admission['allowed_cpus'] == [0, 1] and
        admission['cuda_visible_devices'] == '0' and admission['whole_GPU_used_limit_MiB'] == 8192 and
        admission['qualified_leaf_deadline_seconds'] == spec['deadline_seconds'] and
        Path(admission['entry']).resolve(strict=True) == Path(source['path']) and
        admission['entry_args'] == ['--spec', str(spec_path), '--owner-execute'], 'Current owner/entry/resources differ')
    require(spec['entry_source'] == source and len({r['path'] for r in spec['sources']}) == len(spec['sources']) and
        all(spec[r] in spec['sources'] for r in ('entry_source', 'context', 'preflight_helper_source',
            'posterior_readout_source', 'native_adapter_source')) and
        all(r in spec['sources'] for r in ctx['sources']) and
        all(ctx[r] in ctx['sources'] for r in MODULES), 'Current complete context/helper closure required')
    hashes = admission['bound_source_sha256']
    require(hashes.get(str(spec_path)) == spec_ref['sha256'] and
        admission.get('scientific_spec', spec_ref) == spec_ref, 'Current SPEC is not owner-bound')
    for ref in spec['sources']:
        require(hashes.get(ref['path']) == ref['sha256'], 'Current closure is not owner-bound')
        bound(ref)
    require(ctx == read(bound(spec['context'])), 'Context changed after metadata read')
    return ctx, raw, source, spec_ref, reference(spec['GPU_admission_file'])


def run(spec_file):
    started = time.monotonic()
    require(Path(spec_file).is_absolute(), 'Absolute --spec required')
    spec_path = Path(spec_file).resolve(strict=True)
    spec = read(spec_path)
    ctx, raw, source, spec_ref, admission = gate(spec_path, spec)
    output = Path(spec['output_directory'])
    require(output.is_absolute() and not output.exists() and output.parent.is_dir(), 'Fresh absolute output directory required')
    expected = spec['expected_row']
    require(type(expected['dataset_index']) is int and expected['dataset_index'] == 0 and
        type(expected['image_id']) is int and expected['image_id'] > 0 and
        isinstance(expected['annotation_ids'], list) and 0 < len(expected['annotation_ids']) <= 100 and
        len(set(expected['annotation_ids'])) == len(expected['annotation_ids']) and
        all(type(i) is int and i > 0 for i in expected['annotation_ids']), 'Prelocked first canonical FIT row/allGT required')
    require(all(type(spec[k]) is int and 0 <= spec[k] <= 2**64 - 1 for k in ('readout_seed', 'point_seed')) and
        spec['matching_seed'] == 2800000 + expected['image_id'], 'Explicit seeds must match the immutable helper protocol')
    modules = {role: load_source(ctx[role], 'native_entry_' + role) for role in MODULES}
    core, pref, initializer, old, native, official, shared, diag = [modules[r] for r in MODULES]
    teacher = core.qualified_teacher(ctx, raw['teacher_receipt'])
    import psutil
    import numpy as np
    import torch
    from mmengine import Config
    from mmengine.model import revert_sync_batchnorm
    from mmengine.registry import init_default_scope
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    from mmdet.datasets import CocoDataset
    from mmdet.datasets.api_wrappers import COCO
    require(psutil.Process().cpu_affinity() == [0, 1] and not torch.cuda.is_initialized(), 'ROOT CPU01/coldCUDA required')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    register_all_modules(init_default_scope=False)
    init_default_scope('mmdet')
    old.torch = torch
    helper = load_source(spec['preflight_helper_source'], 'native_entry_injected_preflight')
    readout = load_source(spec['posterior_readout_source'], 'native_entry_posterior_readout')
    adapter = load_source(spec['native_adapter_source'], 'native_entry_adapter')
    model, state, meta, loading = official.prepare_model(torch, pref, initializer, None, None, None,
        MODELS, revert_sync_batchnorm, Config, tuple(CocoDataset.METAINFO['classes']), ctx, teacher, 'native0')
    model.requires_grad_(False).eval()
    del state, meta
    for obj, role in ((torch.version, 'torch_version_source'), (torch.nn.functional.interpolate, 'interpolation_SOURCE'),
        (type(model.panoptic_head), 'native_mask_head_SOURCE'), (model.panoptic_head.get_targets, 'native_parent_head_SOURCE'),
        (model.panoptic_head.preprocess_gt, 'native_parent_head_SOURCE'), (type(model.panoptic_head.assigner), 'hungarian_SOURCE')):
        shared.source_file(obj, ctx[role])
    require(torch.__version__ == '2.0.1+cu118', 'Qualified Torch version required')
    dataset = CocoDataset(ann_file=str(bound(ctx['FIT_annotation'])), data_prefix=dict(img=ctx['image_root']),
        metainfo=dict(classes=('ship',)), pipeline=[], filter_cfg=None, test_mode=True)
    coco = COCO(str(bound(ctx['FIT_annotation'])))
    require(len(dataset) == len(coco.imgs) == 741 and len(coco.anns) == 1661 and
        int(dataset.get_data_info(0)['img_id']) == expected['image_id'] and
        coco.get_ann_ids(img_ids=[expected['image_id']]) == expected['annotation_ids'] and
        not torch.cuda.is_initialized(), 'All CPU preparation/first FIT row differs')
    require(time.monotonic() - started < spec['deadline_seconds'], 'Owned deadline exceeded before CUDA')
    torch.cuda.init()
    torch.cuda.set_per_process_memory_fraction(min(1., 4096 * 1024**2 / torch.cuda.get_device_properties(0).total_memory), 0)
    torch.cuda.reset_peak_memory_stats(0)
    result, data, samples, gt, processed = None, None, None, None, None
    try:
        model.to('cuda:0')
        data, samples, identity = core.image_data(native, old, model, dataset, coco, expected['image_id'])
        gt, processed = shared.original_targets(torch, np, coco, identity, expected, data, model.panoptic_head)
        result = helper.run_preflight(torch, shared, pref, core, old, readout, adapter,
            partial(diag.render_original, torch), model, data, samples, identity, gt, processed,
            spec['readout_seed'], spec['point_seed'])
    finally:
        model.cpu()
        del model, data, samples, gt, processed
        gc.collect()
        cleanup = pref.clear_gpu(torch)
    require(result is not None and cleanup['allocated_after'] == 0 and
        result['matching']['seed'] == spec['matching_seed'] and result['forward']['seed'] == 100000 + expected['image_id'] and
        time.monotonic() - started < spec['deadline_seconds'], 'Actual preflight/cleanup/deadline differs')
    for ref in spec['sources']:
        bound(ref)
    require(psutil.Process().cpu_affinity() == [0, 1] and torch.get_num_threads() == torch.get_num_interop_threads() == 2,
        'Final CPU resources changed')
    output.mkdir()
    report = dict(schema='query-pixel-posterior-native-preflight-entry-report-v1',
        status='QUERY_PIXEL_POSTERIOR_NATIVE_PREFLIGHT_ENTRY_BACKWARD_PASS', phase=spec['phase'],
        source=source, spec=spec_ref, context=spec['context'], GPU_admission=admission, result=result,
        teacher_loading={k: loading[k] for k in ('checkpoint', 'qualification', 'mode', 'strict_fullstate_loaded',
            'native_state_tensors', 'registered_parameters', 'original_frozen_BN_exact', 'inference_backbone_with_cp', 'no_random_missing_keys')},
        prior_six_raw_gates_before_hashes_imports=True, context_used_for_construction_metadata_only=True,
        old_readout_checkpoint_loaded=False, optimizer_steps=0, AP_computed=False, efficacy_claimed=False,
        cleanup=cleanup, torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(0), elapsed_seconds=time.monotonic()-started)
    with (output / 'REPORT.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=report['status'], output_directory=str(output))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args()
    require(args.owner_execute, 'SOURCE_ONLY: explicit ROOT-owned execution required')
    run(args.spec)


if __name__ == '__main__':
    main()
