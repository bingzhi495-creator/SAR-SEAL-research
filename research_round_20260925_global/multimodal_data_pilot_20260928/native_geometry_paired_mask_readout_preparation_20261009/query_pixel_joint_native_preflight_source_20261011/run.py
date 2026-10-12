"""Inert owned integration entry: each A/B uses E24 weights, four micros, two updates.

A is native30 + original-raster native-logit point BCE; B is native30 + L+u
point BCE. Lambda=1 and accumulation=2. All matching is CURRENT native matching.
This is a PSeg FIT-only integration check, not a performance screen, formal
initializer, HRSID qualification, isolated conditioning or matched compute claim.
Torch/project imports and scientific I/O occur only at explicit future ROOT run.
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
import traceback
import types

PREFIX = 'query-pixel-joint-native-preflight'
SCHEMA = 'ROOT-' + PREFIX + '-spec-v1'
MODES = ('native0', 'global', 'spatial_halves', 'spatial_reversed')
OWNER_SHA = 'ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
MODULES = ('native_loader_source', 'evaluation_source', 'micro_preflight_source', 'initializer_source',
    'training_source', 'native_helpers', 'renderer_source', 'posterior_readout_source',
    'joint_capture_source', 'joint_losses_source', 'joint_adapter_source', 'native_objective_source')
LIBRARIES = ('adamw_SOURCE', 'clip_grad_SOURCE', 'hungarian_SOURCE', 'interpolation_SOURCE',
    'mask_sampler_SOURCE', 'mask_sampling_result_SOURCE', 'match_cost_SOURCE', 'native_fusion_SOURCE',
    'native_mask_head_SOURCE', 'native_parent_head_SOURCE', 'panoptic_GT_SOURCE', 'point_sample_SOURCE',
    'solver_source', 'torch_version_source')
NUMERIC = dict(forward_rtol=1e-4, forward_atol=1e-6, gradient_rtol=1e-3, gradient_atol=1e-7,
    native_projection_rtol=1e-4, native_projection_atol=1e-4)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    require(Path(path).is_absolute(), 'Absolute JSON locator required')
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def reference(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def bound(ref):
    require(set(ref) == {'path', 'bytes', 'sha256'} and Path(ref['path']).is_absolute() and
        type(ref['bytes']) is int and ref['bytes'] > 0 and len(ref['sha256']) == 64 and
        set(ref['sha256']) <= set('0123456789abcdef') and reference(ref['path']) == ref, 'Pinned input differs')
    return Path(ref['path'])


def load(ref, name):
    path = bound(ref)
    source = path.read_bytes()
    require(path.suffix == '.py' and len(source) == ref['bytes'] and
        hashlib.sha256(source).hexdigest() == ref['sha256'], 'SOURCE changed before compilation')
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def terminal(raw, phase):
    require(raw.get('status') == 'ACTUAL_OWNED_TERMINAL' and raw.get('original_identity') is not None and
        type(raw.get('returncode')) is int and raw['returncode'] == 0 and raw.get('root_wait_reaped') is True and
        raw.get('held_handle_exit') == dict(wait_state='WAIT_OBJECT_0', actual_exit_code=0) and
        raw.get('numeric_birth_absence', {}).get('original_birth_absent') is True and
        all(k in raw and raw[k] is None for k in ('abort', 'error')) and raw.get('wait_errors') is False and
        raw.get('GPU_observation_errors') == [], 'STOP before hashes/imports: strict owned terminal0 required')
    peak = raw.get('whole_GPU_observed_peak_MiB')
    require(raw.get('phase', phase) == phase and type(peak) in (int, float) and math.isfinite(peak) and
        0 <= peak <= 8192 and raw.get('whole_GPU_used_limit_MiB') == 8192 and
        raw.get('whole_leaf_VRAM_certified') is False and
        raw.get('GPU_scope') == 'whole GPU observed, not per-process WDDM allocation', 'Raw scope differs')


def source_gate(path, spec):
    require(set(spec['training_receiptmap']) == set(spec['training_phasemap']) == set(MODES), 'Original four raws required')
    for mode in MODES:
        terminal(read(spec['training_receiptmap'][mode]['path']), spec['training_phasemap'][mode])
    for row in spec['prior_receipts']:
        require(set(row) == {'receipt', 'phase'}, 'Explicit extra prior raw identity required')
        terminal(read(row['receipt']['path']), row['phase'])
    owner = read(spec['GPU_admission_file'])
    source, spec_ref = reference(__file__), reference(path)
    require(owner['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and owner['automatic_retry'] is False and
        owner['phase'] == spec['phase'] and owner['allowed_cpus'] == spec['CPU_affinity'] == [0, 1] and
        owner['cuda_visible_devices'] == '0' and owner['qualified_leaf_deadline_seconds'] == spec['deadline_seconds'] and
        owner['whole_GPU_used_limit_MiB'] == spec['whole_GPU_owner_limit_MiB'] == 8192 and
        owner['driver_sha256'] == OWNER_SHA and Path(owner['entry']).resolve() == Path(source['path']) and
        owner['entry_args'] == ['--spec', str(path), '--owner-execute'], 'Current owner/entry/resources differ')
    hashes = owner['bound_source_sha256']
    require(hashes.get(str(path)) == spec_ref['sha256'] and spec['entry_source'] == source and
        len({r['path'] for r in spec['sources']}) == len(spec['sources']), 'Current SPEC/unique sources differ')
    for ref in spec['sources']:
        bound(ref)
        require(hashes.get(ref['path']) == ref['sha256'], 'Whole current closure must be owner-bound')
    for role in MODULES + LIBRARIES + ('entry_source', 'CPU_reader', 'training_CPU_reader', 'reused_CPU_reader',
            'readout_source', 'geometry_source', 'hooks_source', 'FIT'):
        require(spec[role] in spec['sources'], 'Required role absent from closure: ' + role)
    for ref in list(spec['checkpointmap'].values()) + list(spec['qualificationmap'].values()) + list(spec['training_receiptmap'].values()):
        require(ref in spec['sources'], 'Original teacher qualification/checkpoint/raw absent from closure')
    require(all(r['receipt'] in spec['sources'] for r in spec['prior_receipts']), 'Extra prior raw absent from closure')
    if 'scientific_spec' in owner:
        require(owner['scientific_spec'] == spec_ref, 'Current scientific SPEC differs')
    return source, spec_ref, reference(spec['GPU_admission_file'])


def raw_GT_validator(torch, np, cv2, coco, data, identity):
    ids = identity['ann_ids']
    parser_ids = [r['annotation_id'] for r in identity['parser_rows'] if r['native_parser_retained']]
    require(len(ids) == len(set(ids)) and parser_ids == ids and 1 <= len(ids) <= 100 and
        coco.get_ann_ids(img_ids=[identity['image_id']]) == ids, 'Actual parser/raw identities differ')
    mapping = tuple(ids.index(a) for a in parser_ids)  # Derived from actual per-annotation parser, not match order.
    H, W = identity['original_shape']; h, w = identity['actual_img_shape']
    image = coco.load_imgs([identity['image_id']])[0]
    require((image['height'], image['width']) == (H, W), 'Actual original image shape differs')
    raw = np.stack([np.zeros((H, W), np.uint8) if a.get('segmentation') == [] else coco.annToMask(a)
        for a in coco.load_anns(ids)])
    require(raw.shape == (len(ids), H, W) and bool(((raw == 0) | (raw == 1)).all()), 'Complete raw binary GT required')
    padded = np.zeros((len(ids), 800, 800), np.int64)
    for j, mask in enumerate(raw):
        padded[j, :h, :w] = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    expected = torch.from_numpy(padded[list(mapping)]).to(data['inputs'].device)
    sample = data['data_samples'][0]
    require(len(sample.ignored_instances) == 0 and 'gt_sem_seg' not in sample and
        torch.equal(sample.gt_instances.masks.to_tensor(dtype=torch.int64, device=expected.device), expected),
        'Actual resized/padded BitmapMasks disagree with raw rows')
    raw_tensor = torch.from_numpy(raw).to(device=expected.device, dtype=torch.bool)
    def validate(processed, actual_ids, actual_raw, actual_mapping, actual_identity):
        require(tuple(actual_ids) == tuple(ids) and tuple(actual_mapping) == mapping and actual_identity == identity and
            torch.equal(actual_raw, raw_tensor) and processed.labels.dtype == processed.masks.dtype == torch.int64 and
            tuple(processed.labels.shape) == (len(ids),) and bool((processed.labels == 0).all()) and
            torch.equal(processed.masks, expected), 'Actual preprocess_gt/raw correspondence differs')
        return True
    return raw_tensor, mapping, validate


def arm_run(torch, np, cv2, m, prepared, readout, construction, spec, source, spec_ref, output, dataset, coco, started, arm):
    native, legacy, pref, initializer, common, old, diag, posterior, capture, joint, adapter, objective = m
    model, anchor, policy, loading = prepared
    model.to('cuda:0')
    require(dict(pref.trainability(torch, model), kind='mask2former') == policy, 'Native CUDA policy differs')
    native_opt, native_audit = pref.make_optimizer(torch, model, construction['initializers']['mask2former'], policy)
    named = dict(model.named_parameters()); active = [named[n] for n in policy['active_names']]
    heads = {} if readout is None else dict(readout.to('cuda:0').named_parameters())
    head_opt = None if not heads else torch.optim.AdamW([dict(params=[p]) for p in heads.values()], lr=1e-4, weight_decay=.05, betas=(.9,.999), eps=1e-8)
    require(len(native_opt.state) == 0 and len(native_opt.param_groups) == 345 and
        (head_opt is None or len(head_opt.state) == 0 and len(heads) == 8 and sum(p.numel() for p in heads.values()) == 9441), 'Fresh optimizer membership differs')
    fresh_native_optimizer_state_entries = len(native_opt.state)
    fresh_readout_optimizer_state_entries = 0 if head_opt is None else len(head_opt.state)
    before_head = pref.cpu_tree(torch, {} if readout is None else readout.state_dict())
    hooks_before = {n: (tuple(v._forward_pre_hooks), tuple(v._forward_hooks)) for n, v in model.named_modules()}
    rows, audits = [], []
    try:
        for g, image_id in enumerate(spec['image_ids'] * 2):
            require(time.monotonic() - started < spec['deadline_seconds'], 'Integration deadline exceeded')
            index = next(i for i in range(len(dataset)) if int(dataset.get_data_info(i)['img_id']) == image_id)
            data, stripped, gray, raw_gray, identity = old.load_image(model, dataset, coco, dataset.get_data_info(index))
            del stripped, gray, raw_gray
            target_proof = pref.check_targets(torch, data, identity)
            raw_GT, mapping, validator = raw_GT_validator(torch, np, cv2, coco, data, identity)
            rng = torch.Generator(device='cpu').manual_seed(spec['point_seed_base'] + g)
            indices = torch.randint(raw_GT.shape[1] * raw_GT.shape[2], (len(raw_GT), 12544), generator=rng, dtype=torch.int64).to('cuda:0')
            pending = {n: None if p.grad is None else p.grad.detach().cpu().clone() for n, p in heads.items()}
            readout_before = pref.cpu_tree(torch, {} if readout is None else readout.state_dict())
            external = pref.rng_snapshot(torch)
            with torch.random.fork_rng(devices=[0]):
                torch.random.default_generator.manual_seed(spec['loss_seed_base'] + g); torch.cuda.manual_seed(spec['loss_seed_base'] + g)
                r = joint.joint_native_losses(torch, pref, common, old, capture, adapter, objective, model, data, readout,
                    partial(diag.render_original, torch), identity, identity['ann_ids'], raw_GT, mapping, validator, indices, arm, g)
                live = r['live_native_fields']; L = r['auxiliary_fields']['native_logits']; L.retain_grad()
                for name in ('F256', 'e0', 'M200'):
                    live[name].retain_grad()
                audit = dict(schema=PREFIX+'-audit-v1', arm=arm, global_micro=g, identity=identity,
                    raw_GT=raw_GT, query_ids=r['query_ids'], point_indices=indices,
                    F256=live['F256'], e0=live['e0'], M200=live['M200'], readout_before=readout_before,
                    readout_pending_before=pending, auxiliary_loss=float(r['auxiliary'].detach()), native_total=float(r['native_total'].detach()),
                    metadata=r['metadata'], auxiliary_gradient_scale=.5)
                audit = pref.cpu_tree(torch, audit)
                (r['total'] / 2).backward()  # The sole backward for this current native forward.
            require(pref.tree_equal(torch, external, pref.rng_snapshot(torch)), 'External RNG was not restored')
            grad_norms = {n: float(p.grad.norm()) if p.grad is not None else None for n, p in named.items() if n in policy['active_names']}
            head_norms = {n: float(p.grad.norm()) if p.grad is not None else None for n, p in heads.items()}
            require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in active + list(heads.values())), 'All native345/head gradients must be present finite; zero permitted cold')
            require(all(named[n].grad is None for n in policy['frozen_names']), 'Typed BN affine received gradients')
            require(all(live[n].grad is not None and bool(torch.isfinite(live[n].grad).all()) for n in ('F256','e0','M200')), 'Live native total-gradient observation failed')
            live_norms = {n: float(live[n].grad.norm()) for n in ('F256','e0','M200')}
            if arm == 'B' and g >= 2:
                require(all(v > 0 for v in head_norms.values()), 'Actual warm B head gradients must be positive on the selected images')
            require(L.grad is not None and bool(torch.isfinite(L.grad).all()), 'Auxiliary original L local VJP missing/nonfinite')
            audit.update(auxiliary_L_gradient=L.grad[0, r['query_ids']].detach().cpu().clone(),
                readout_accumulated_before_clip={n: p.grad.detach().cpu().clone() for n,p in heads.items()})
            path = output / ('%s_MICRO_%d.pth' % (arm, g))
            with path.open('xb') as stream:
                torch.save(audit, stream)
            audits.append(reference(path))
            stepped = g % 2 == 1; norm = None
            if stepped:
                norm = float(torch.nn.utils.clip_grad_norm_(active + list(heads.values()), .01))
                require(math.isfinite(norm), 'Combined global clip nonfinite')
                native_opt.step()
                if head_opt is not None:
                    head_opt.step()
                native_opt.zero_grad(set_to_none=True)
                if head_opt is not None:
                    head_opt.zero_grad(set_to_none=True)
            require(all(not v.training for n,v in model.named_modules() if n in policy['BN_modules']) and
                all(torch.equal(model.state_dict()[n].detach().cpu(), anchor[n]) for n in policy['frozen_names'] + policy['BN_buffer_names']), 'TypedBN state/mode changed')
            rows.append(dict(global_micro=g, image_id=image_id, identity=identity, target=target_proof, metadata=r['metadata'],
                native_total=audit['native_total'], auxiliary_loss=audit['auxiliary_loss'], native_gradient_norms=grad_norms,
                readout_gradient_norms=head_norms, native_live_total_gradient_norms=live_norms, combined_preclip_norm=norm,
                optimizer_update=stepped, native_updates=(g+1)//2, readout_updates=(g+1)//2 if heads else 0,
                accumulation_divisor=2, actual_backward_calls=1, external_RNG_restored=True))
            del data, raw_GT, validator, indices, pending, readout_before, r, live, L, audit, external
        require(hooks_before == {n: (tuple(v._forward_pre_hooks), tuple(v._forward_hooks)) for n,v in model.named_modules()}, 'Hooks leaked')
        for opt, count in ((native_opt, 345), (head_opt, 8)):
            if opt is not None:
                require(len(opt.state) == count and all(int(v['step']) == 2 and bool(torch.isfinite(v['exp_avg']).all()) and
                    bool(torch.isfinite(v['exp_avg_sq']).all()) for v in opt.state.values()), 'Actual AdamW moments/step2 differ')
        after = pref.cpu_tree(torch, model.state_dict())
        require(len(after) == 610 and all(bool(torch.isfinite(v).all()) for v in after.values()) and
            any(not torch.equal(after[n], anchor[n]) for n in policy['active_names']), 'Native finite endpoint/update proof failed')
        snapshot = dict(schema=PREFIX+'-technical-checkpoint-v1', before_native_state=anchor, after_native_state=after,
            native_optimizer_state=pref.cpu_tree(torch, native_opt.state_dict()), native_optimizer_names=policy['active_names'],
            native_optimizer_audit=native_audit, policy=policy, before_readout_state=before_head,
            after_readout_state=pref.cpu_tree(torch, {} if readout is None else readout.state_dict()),
            readout_optimizer_state=None if head_opt is None else pref.cpu_tree(torch, head_opt.state_dict()), readout_optimizer_names=list(heads),
            meta=dict(source=source, spec=spec_ref, phase=spec['phase'], arm=arm, microsteps=4, native_updates=2,
                readout_updates=2 if heads else 0, teacher_checkpoint=spec['checkpointmap']['native0'],
                fresh_native_optimizer_state_entries=fresh_native_optimizer_state_entries,
                fresh_readout_optimizer_state_entries=fresh_readout_optimizer_state_entries,
                teacher_qualification=spec['qualificationmap']['native0'], technical_only=True, formal_initializer_eligible=False))
        path = output / (arm+'_TECHNICAL_ONLY.pth')
        with path.open('xb') as stream:
            torch.save(snapshot, stream)
        return dict(checkpoint=reference(path), audits=audits, micros=rows, loading=loading, policy=policy,
            native_optimizer_audit=native_audit, readout_optimizer_audit=None if not heads else dict(type='AdamW',lr=1e-4,weight_decay=.05,betas=[.9,.999],eps=1e-8,parameter_names=list(heads)))
    finally:
        native_opt.zero_grad(set_to_none=True)
        if head_opt is not None:
            head_opt.zero_grad(set_to_none=True)
        model.cpu()
        if readout is not None:
            readout.cpu()


def run(spec_file):
    started = time.monotonic(); path = Path(spec_file)
    require(path.is_absolute(), 'Absolute --spec required'); path = path.resolve(strict=True); spec = read(path)
    require(spec['schema'] == SCHEMA and spec['kind'] == 'mask2former' and spec['teacher_epoch'] == 24 and
        spec['CPU_affinity'] == [0,1] and spec['allocator_limit_MiB'] == 4096 and spec['whole_GPU_owner_limit_MiB'] == 8192 and
        type(spec['deadline_seconds']) is int and 0 < spec['deadline_seconds'] <= 10800 and os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and
        spec['CPU_numeric_protocol'] == NUMERIC and spec['kappa'] == 1. and spec['eps'] == 1e-6 and
        len(spec['image_ids']) == 2 and len(set(spec['image_ids'])) == 2 and all(type(i) is int and i > 0 for i in spec['image_ids']) and
        all(type(spec[k]) is int and 0 <= spec[k] < 2**63-4 for k in ('point_seed_base','loss_seed_base','readout_seed')), 'Prelocked tiny native joint spec differs')
    source, spec_ref, admission_ref = source_gate(path, spec)
    legacy = load(spec['evaluation_source'], 'qualified_original_E24_lineage_ONLY'); teachers = legacy.qualified_training(spec)
    for row in teachers.values():
        q = row['qualification']
        for ref in row['training_spec']['sources'] + [q[k] for k in ('source','reader_source','spec','report','start','GPU_admission','owner_driver')] + list(q['model_proof']['proof_refs'].values()):
            require(ref in spec['sources'], 'Complete original teacher/proof lineage absent from current owner closure')
    output = Path(spec['output_directory']); require(output.is_absolute() and not output.exists(), 'Fresh absolute output required'); output.mkdir(parents=True)
    import psutil
    import torch
    import numpy as np
    import cv2
    from mmengine import Config
    from mmengine.model import revert_sync_batchnorm
    from mmengine.registry import init_default_scope
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    from mmdet.datasets import CocoDataset
    from mmdet.datasets.api_wrappers import COCO
    require(psutil.Process().cpu_affinity() == [0,1] and not torch.cuda.is_initialized(), 'CPU01/coldCUDA required')
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    register_all_modules(init_default_scope=False); init_default_scope('mmdet')
    require(CocoDataset.METAINFO['classes'][8] == 'boat' and len(CocoDataset.METAINFO['classes']) == 80, 'boat8/null80 required')
    m = [load(spec[k], '_joint_'+k) for k in MODULES]; native, legacy, pref, initializer, common, old, diag, posterior, capture, joint, adapter, objective = m
    old.torch = torch
    construction = dict(teachers['native0']['training_spec'], training_source=spec['training_source'], checkpointmap=spec['checkpointmap'], qualificationmap=spec['qualificationmap'],
        order_seed=native.ORDER_SEED, phase_loss_seed_base=native.LOSS_SEED_BASE)
    require(construction['FIT'] == spec['FIT'] and construction['image_root'] == spec['image_root'] and construction['backbone_with_cp'] is True, 'Original FIT/GT transform/backbone checkpoint policy required')
    prepared = {arm: native.prepare_training_model(torch,pref,initializer,common,MODELS,revert_sync_batchnorm,Config,tuple(CocoDataset.METAINFO['classes']),construction,teachers['native0']) for arm in ('A','B')}
    heads = {'A':None, 'B':posterior.QueryPixelPosteriorReadout(spec['readout_seed'],spec['kappa'],spec['eps'])}
    for arm in ('A','B'):
        native.imported_library_gate(torch, prepared[arm][0], spec)
    dataset = CocoDataset(ann_file=str(bound(spec['FIT'])),data_prefix=dict(img=spec['image_root']),metainfo=dict(classes=('ship',)),pipeline=[],filter_cfg=None,test_mode=True)
    coco = COCO(str(bound(spec['FIT']))); require(len(dataset)==len(coco.imgs)==741 and len(coco.anns)==1661 and
        spec['image_ids']==[int(dataset.get_data_info(i)['img_id']) for i in range(2)], 'Only original FIT741/1661 first two IDs supported')
    torch.cuda.init(); torch.cuda.set_per_process_memory_fraction(min(1.,4096*1024**2/torch.cuda.get_device_properties(0).total_memory),0); torch.cuda.reset_peak_memory_stats(0)
    result, error, cleanup = {}, None, None
    try:
        for arm in ('A','B'):
            result[arm] = arm_run(torch,np,cv2,m,prepared[arm],heads[arm],construction,spec,source,spec_ref,output,dataset,coco,started,arm)
        del prepared, heads; cleanup = pref.clear_gpu(torch)
        for ref in spec['sources']:
            bound(ref)
        require(time.monotonic()-started < spec['deadline_seconds'], 'Integration deadline exceeded')
    except Exception as exc:
        error = dict(type=type(exc).__name__,message=str(exc),no_retry=True); traceback.clear_frames(exc.__traceback__)
        if 'prepared' in locals():
            for row in prepared.values(): row[0].cpu()
            del prepared
        if 'heads' in locals():
            for head in heads.values():
                if head is not None: head.cpu()
            del heads
        gc.collect(); cleanup = pref.clear_gpu(torch)
    report = dict(schema=PREFIX+'-report-v1',status='QUERY_PIXEL_JOINT_NATIVE_PREFLIGHT_GPU_COMPLETE_PENDING_CPU' if error is None else 'QUERY_PIXEL_JOINT_NATIVE_PREFLIGHT_TECHNICAL_STOP',
        source=source,spec=spec_ref,GPU_admission=admission_ref,phase=spec['phase'],result=result,error=error,cleanup=cleanup,CPU_numeric_protocol=NUMERIC,
        actual_total_backward_calls=8 if error is None else None,extra_backward_or_autograd_grad=False,native30_CPU_replay_claimed=False,
        native_AdamW_groups=345,B_readout_AdamW_groups=8,separate_AdamW_optimizers=True,combined_gradient_clip=True,
        combined_optimizer_bitwise_equivalence_claimed=False,
        isolated_F_e0_M_gradient_replay_claimed=False,formal_training_or_initializer_qualified=False,no_DEV=True,no_TEST=True,no_AP=True,
        projection_or_quality_calibration_claimed=False,efficacy_claimed=False,compute_matched_claimed=False,conditioning_isolated_claimed=False,
        elapsed_seconds=time.monotonic()-started,torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(0))
    write(output/'REPORT.json',report)
    require(error is None, 'TECHNICAL_STOP: '+str(error))


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--spec',required=True); parser.add_argument('--owner-execute',action='store_true')
    args=parser.parse_args(); require(args.owner_execute,'Inert SOURCE: explicit ROOT-owned execution required'); run(args.spec)


if __name__ == '__main__':
    main()
