"""CPU consistency and local auxiliary replay for the tiny native A/B preflight.

Authoring is inert. CPU reconstructs the additional point loss and local VJP from
saved fields. No native backbone, thirty-loss or CUDA optimizer replay is implied.
"""
import argparse
import hashlib
import inspect
import json
import math
import os
import sys
import types
from pathlib import Path

QUAL_SCHEMA = 'ROOT-query-pixel-joint-native-preflight-CPU-qualification-v1'
QUAL_STATUS = 'ROOT_QUERY_PIXEL_JOINT_NATIVE_PREFLIGHT_CPU_QUALIFIED'
READOUT_SHAPES = {
    'feature_projection.weight': (16, 256, 1, 1), 'feature_projection.bias': (16,),
    'query_projection.weight': (16, 256), 'query_projection.bias': (16,),
    'hidden.weight': (32, 36), 'hidden.bias': (32,),
    'final.weight': (1, 32), 'final.bias': (1,)}
PREFIX = 'query-pixel-joint-native-preflight'
MODES = ('native0', 'global', 'spatial_halves', 'spatial_reversed')
NUMERIC = dict(forward_rtol=1e-4, forward_atol=1e-6, gradient_rtol=1e-3,
    gradient_atol=1e-7, native_projection_rtol=1e-4, native_projection_atol=1e-4)


def require(value, message):
    if not value:
        raise AssertionError(message)


def checked_tensor(torch, value, shape, dtype, label):
    require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and
        value.layout == torch.strided and tuple(value.shape) == tuple(shape) and value.dtype == dtype,
        label + ': CPU tensor shape/dtype/layout differs')
    require(bool(torch.isfinite(value).all()), label + ': nonfinite tensor')
    return value


def compare(torch, actual, expected, rtol, atol, label):
    require(actual.shape == expected.shape and actual.dtype == expected.dtype, label + ': shape/dtype')
    require(bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all()), label + ': finite')
    difference = (actual - expected).double()
    norm = float(expected.double().norm())
    summary = dict(max_absolute_error=float(difference.abs().max()) if difference.numel() else 0.,
                   relative_L2_error=float(difference.norm()) / max(norm, 1e-30), rtol=rtol, atol=atol)
    require(torch.allclose(actual, expected, rtol=rtol, atol=atol), label + ': numerical mismatch')
    return summary


def render(torch, tensor, identity):
    current = torch.nn.functional.interpolate(tensor, size=(800, 800), mode='bilinear', align_corners=False)
    height, width = identity['actual_img_shape']
    current = current[:, :, :height, :width]
    return torch.nn.functional.interpolate(current, size=tuple(identity['original_shape']),
        mode='bilinear', align_corners=False)


def replay_auxiliary(torch, arm, F256, e0, original_native_logits, identity,
                     readout_before, query_ids, raw_masks, point_indices):
    """Independent functional loss and local L/2 VJP; F/e0 are fixed CPU values.

    L is independently rendered from saved M200. The compared retained GPU L
    gradient excludes the thirty native atoms; B parameter gradients are aux-only.
    This does not separate native F/e/M total gradients or replay native updates.
    """
    require(arm in ('A', 'B'), 'Unknown auxiliary arm')
    height, width = identity['original_shape']
    checked_tensor(torch, original_native_logits, (1, 100, height, width), torch.float32, 'Actual auxiliary L')
    count = query_ids.numel()
    checked_tensor(torch, query_ids, (count,), torch.int64, 'Matched queries')
    checked_tensor(torch, raw_masks, (count, height, width), torch.bool, 'Original raw masks')
    checked_tensor(torch, point_indices, (count, 12544), torch.int64, 'Original flat indices')
    require(count <= 100 and torch.unique(query_ids).numel() == count and
        not bool(((query_ids < 0) | (query_ids >= 100)).any()) and
        not bool(((point_indices < 0) | (point_indices >= height * width)).any()), 'Query/index range differs')
    L = original_native_logits.detach().clone().requires_grad_(True)
    variables, fields = {}, {}
    if arm == 'B':
        checked_tensor(torch, F256, (1, 256, 200, 200), torch.float32, 'Saved F')
        checked_tensor(torch, e0, (1, 100, 256), torch.float32, 'Saved e0')
        require(set(readout_before) == set(READOUT_SHAPES), 'Eight readout tensor names differ')
        for name, shape in READOUT_SHAPES.items():
            checked_tensor(torch, readout_before[name], shape, torch.float32, name)
            variables[name] = readout_before[name].detach().clone().requires_grad_(True)
        functional = torch.nn.functional
        pixel = render(torch, functional.conv2d(F256.detach(), variables['feature_projection.weight'],
            variables['feature_projection.bias']), identity)
        query = functional.linear(e0.detach(), variables['query_projection.weight'], variables['query_projection.bias'])
        probability = L.sigmoid()
        contrast = probability - functional.avg_pool2d(probability[0, :, None], 3, stride=1, padding=1,
            count_include_pad=False)[:, 0].unsqueeze(0)
        fields = dict(projected_pixel=pixel, projected_query=query, contrast=contrast)
    if count:
        selected = L[0, query_ids].flatten(1).gather(1, point_indices)
        target = raw_masks.flatten(1).gather(1, point_indices).float()
        if arm == 'B':
            pixel_points = pixel[0].flatten(1).t()[point_indices.reshape(-1)]
            query_points = query[0, query_ids][:, None].expand(count, 12544, 16).reshape(-1, 16)
            contrast_points = contrast[0, query_ids].flatten(1).gather(1, point_indices)
            native = selected.reshape(-1)
            inputs = torch.cat((pixel_points, query_points, native[:, None], native.sigmoid()[:, None],
                (native > 0).float()[:, None], contrast_points.reshape(-1, 1)), dim=1)
            hidden = functional.silu(functional.linear(inputs, variables['hidden.weight'], variables['hidden.bias']))
            selected = selected + functional.linear(hidden, variables['final.weight'], variables['final.bias']).reshape(count, 12544)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(selected, target, reduction='none').mean(1).mean()
    else:
        loss = L.reshape(-1)[:0].sum()
        for value in variables.values():
            loss = loss + value.reshape(-1)[:0].sum()
        selected = L.new_empty((0, 12544))
    gradients = torch.autograd.grad(loss / 2, [L] + list(variables.values()), allow_unused=False)
    return dict(loss=float(loss.detach()), point_logits=selected.detach(), native_logits_gradient=gradients[0].detach(),
        readout_gradients={name: gradient.detach() for name, gradient in zip(variables, gradients[1:])},
        fields={name: value.detach() for name, value in fields.items()})


def read(path):
    require(Path(path).is_absolute(), 'Absolute JSON locator required')
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def reference(path):
    path = Path(path).resolve(strict=True); digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def bound(ref):
    require(set(ref) == {'path', 'bytes', 'sha256'} and Path(ref['path']).is_absolute() and
        type(ref['bytes']) is int and ref['bytes'] > 0 and len(ref['sha256']) == 64 and
        set(ref['sha256']) <= set('0123456789abcdef') and reference(ref['path']) == ref, 'Pinned ref differs')
    return Path(ref['path'])


def load(ref, name):
    path = bound(ref); source = path.read_bytes()
    require(path.suffix == '.py' and len(source) == ref['bytes'] and hashlib.sha256(source).hexdigest() == ref['sha256'], 'SOURCE changed')
    module = types.ModuleType(name); module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def expect(row, expected, label):
    require(all(key in row and row[key] == value for key, value in expected.items()), label)


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def terminal(raw, phase):
    expect(raw, dict(status='ACTUAL_OWNED_TERMINAL', returncode=0, root_wait_reaped=True,
        held_handle_exit=dict(wait_state='WAIT_OBJECT_0', actual_exit_code=0), abort=None, error=None,
        wait_errors=False, GPU_observation_errors=[], whole_GPU_used_limit_MiB=8192,
        whole_leaf_VRAM_certified=False, GPU_scope='whole GPU observed, not per-process WDDM allocation'), 'STOP: raw terminal0 required')
    peak = raw.get('whole_GPU_observed_peak_MiB')
    require(type(raw['returncode']) is int and raw.get('original_identity') is not None and
        raw.get('numeric_birth_absence', {}).get('original_birth_absent') is True and raw.get('phase', phase) == phase and
        type(peak) in (int, float) and math.isfinite(peak) and 0 <= peak <= 8192, 'STOP: raw identity/peak differs')


def source_gate(path, spec):
    receipt = path.parent / 'finite_point_cache_execution_20261005_v2' / 'runs' / spec['phase'] / 'PARENT_RECEIPT.json'
    raw = read(receipt); terminal(raw, spec['phase'])  # All raw metadata gates precede the first hash/import.
    require(set(spec['training_receiptmap']) == set(spec['training_phasemap']) == set(MODES), 'Four original raw gates required')
    for mode in MODES:
        terminal(read(spec['training_receiptmap'][mode]['path']), spec['training_phasemap'][mode])
    for row in spec['prior_receipts']:
        require(set(row) == {'receipt', 'phase'}, 'Explicit extra raw identity required')
        terminal(read(row['receipt']['path']), row['phase'])
    reader, source, spec_ref = reference(__file__), reference(Path(__file__).with_name('run.py')), reference(path)
    owner = read(spec['GPU_admission_file']); hashes = owner['bound_source_sha256']
    require(spec['CPU_reader'] == reader and reader in spec['sources'] and spec['entry_source']==source and
        source in spec['sources'] and raw['admission_sha256']==reference(spec['GPU_admission_file'])['sha256'] and
        all(hashes.get(ref['path'])==ref['sha256'] for ref in (reader,source,spec_ref)), 'Exact entry/reader/SPEC raw owner pins required before SOURCE exec')
    producer = load(spec['entry_source'], '_joint_preflight_bound_producer')
    source, spec_ref, admission = producer.source_gate(path, spec)
    require(raw['admission_sha256'] == admission['sha256'], 'Current raw/admission SHA join differs')
    return producer, dict(source=source, reader_source=reader, spec=spec_ref,
        parent_receipt=reference(receipt), GPU_admission=admission, phase=spec['phase'])


def same_state(torch, first, second, label):
    require(set(first) == set(second) and all(first[n].shape == second[n].shape and first[n].dtype == second[n].dtype and
        torch.equal(first[n], second[n]) for n in first), label)


def cold_readout(torch, seed):
    with torch.random.fork_rng(devices=[]):
        generator = torch.Generator(device='cpu').manual_seed(seed); torch.random.set_rng_state(generator.get_state())
        layers = {'feature_projection': torch.nn.Conv2d(256,16,1,device='cpu',dtype=torch.float32),
            'query_projection': torch.nn.Linear(256,16,device='cpu',dtype=torch.float32),
            'hidden': torch.nn.Linear(36,32,device='cpu',dtype=torch.float32),
            'final': torch.nn.Linear(32,1,device='cpu',dtype=torch.float32)}
        torch.nn.init.zeros_(layers['final'].weight); torch.nn.init.zeros_(layers['final'].bias)
    return {prefix+'.'+name: value.detach().clone() for prefix, layer in layers.items() for name, value in layer.state_dict().items()}


def teacher_state(torch, spec, result):
    legacy = load(spec['evaluation_source'], '_joint_original_teacher_lineage')
    teacher = legacy.qualified_training(spec)['native0']; train, q = teacher['training_spec'], teacher['qualification']
    for ref in train['sources'] + [q[k] for k in ('source','reader_source','spec','report','start','GPU_admission','owner_driver')] + list(q['model_proof']['proof_refs'].values()):
        require(ref in spec['sources'], 'Original teacher/proof ref absent from current owner closure')
    old = load(spec['reused_CPU_reader'], '_joint_independent_micro_reader')
    common = load(spec['training_CPU_reader'], '_joint_independent_common_reader')
    initializer = load(spec['initializer_source'], '_joint_independent_initializer')
    from mmengine import Config
    from mmengine.model import revert_sync_batchnorm
    from mmengine.registry import init_default_scope
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    register_all_modules(init_default_scope=False); init_default_scope('mmdet')
    initialized = read(bound(train['initializer_qualification']))
    model, initial = common.build_initial_native(torch, old, initializer, MODELS, revert_sync_batchnorm,
        Config, train, initialized, result['A']['loading']['original_initialization'])
    load(spec['native_loader_source'],'_joint_native_library_binding_only').imported_library_gate(torch,model,spec)
    require(Path(inspect.getsourcefile(torch.nn.functional.interpolate)).resolve()==bound(spec['interpolation_SOURCE']).resolve(),'Actual interpolation SOURCE differs')
    policy = old.independent_policy(torch, model, 'mask2former')
    require(policy['state_entries'] == 610 and policy['parameter_tensors'] == 451 and policy['trainable_tensors'] == 345 and
        policy['frozen_tensors'] == 106 and len(policy['BN_buffer_names']) == 159 and len(policy['BN_modules']) == 53, 'Typed native policy differs')
    cp = torch.load(str(bound(spec['checkpointmap']['native0'])), map_location='cpu'); anchor = cp['state_dict']
    expect(cp['meta'], dict(schema=common.CP_SCHEMA, mode='native0', completed_epoch=24, source=spec['training_source'],
        spec=q['spec'], phase=q['phase'], global_microsteps=17784, global_updates=8892, consumed_raw_GT=39864,
        no_pending_accumulation_gradients=True, eligible_for_automatic_resume=False), 'Qualified E24 native metadata differs')
    require(set(anchor) == set(initial) and len(anchor) == 610, 'Bare native610 required')
    for name, value in anchor.items(): checked_tensor(torch, value, initial[name].shape, initial[name].dtype, name)
    proof = read(bound(q['model_proof']['proof_refs']['state_proof'])); common.cpu_state(torch, old, anchor, proof['after'], 'Qualified teacher610')
    require(all(torch.equal(anchor[n], initial[n]) for n in policy['frozen_names'] + policy['BN_buffer_names']), 'Teacher typedBN initializer exactness differs')
    groups, audit = old.independent_optimizer(torch, model, train['initializers']['mask2former'], policy)
    return old, common, anchor, policy, groups, audit


def checkpoint_check(torch, old, common, spec, refs, row, arm, anchor, policy, groups, audit, cold):
    cp = torch.load(str(bound(row['checkpoint'])), map_location='cpu')
    require(set(cp)=={'schema','before_native_state','after_native_state','native_optimizer_state','native_optimizer_names',
        'native_optimizer_audit','policy','before_readout_state','after_readout_state','readout_optimizer_state','readout_optimizer_names','meta'} and
        cp['schema'] == PREFIX+'-technical-checkpoint-v1' and cp['policy'] == row['policy'] == policy and
        cp['native_optimizer_names'] == policy['active_names'] and canonical(cp['native_optimizer_audit']) == canonical(row['native_optimizer_audit']) == canonical(audit),
        'Technical native policy/optimizer differs')
    expect(cp['meta'], dict(source=refs['source'],spec=refs['spec'],phase=spec['phase'],arm=arm,microsteps=4,native_updates=2,
        readout_updates=2 if arm=='B' else 0,teacher_checkpoint=spec['checkpointmap']['native0'],
        teacher_qualification=spec['qualificationmap']['native0'],technical_only=True,formal_initializer_eligible=False,
        fresh_native_optimizer_state_entries=0,fresh_readout_optimizer_state_entries=0), 'Technical CP identity/fresh state differs')
    require(all(type(cp['meta'][key]) is int for key in ('fresh_native_optimizer_state_entries','fresh_readout_optimizer_state_entries')),'Actual fresh moment entry counts must be integers')
    same_state(torch, cp['before_native_state'], anchor, 'Actual before native610 must equal E24')
    after = cp['after_native_state']; require(set(after) == set(anchor), 'Actual endpoint native610 keys differ')
    for name, value in after.items(): checked_tensor(torch,value,anchor[name].shape,anchor[name].dtype,name)
    require(all(torch.equal(after[n], anchor[n]) for n in policy['frozen_names']+policy['BN_buffer_names']) and
        any(not torch.equal(after[n],anchor[n]) for n in policy['active_names']), 'BN exactness/native update proof differs')
    native = old.validate_optimizer(torch, dict(state_dict=after,optimizer_state_dict=cp['native_optimizer_state'],
        meta=dict(optimizer_param_names=[[n] for n in policy['active_names']])), groups, audit, {n:2 for n in policy['active_names']})
    if arm == 'A':
        require(cp['before_readout_state'] == cp['after_readout_state'] == {} and cp['readout_optimizer_state'] is None and
            cp['readout_optimizer_names'] == [] and row['readout_optimizer_audit'] is None, 'A must have no readout')
    else:
        same_state(torch,cp['before_readout_state'],cold,'B independent private-seed initialization differs')
        require(cp['readout_optimizer_names'] == list(READOUT_SHAPES) and set(cp['after_readout_state'])==set(READOUT_SHAPES), 'B eight ordered names differ')
        expect(row['readout_optimizer_audit'],dict(type='AdamW',lr=1e-4,weight_decay=.05,betas=[.9,.999],eps=1e-8,parameter_names=list(READOUT_SHAPES)),'B reported optimizer recipe differs')
        parameters = [torch.nn.Parameter(cold[n].clone()) for n in READOUT_SHAPES]
        expected = torch.optim.AdamW([dict(params=[p]) for p in parameters],lr=1e-4,weight_decay=.05,betas=(.9,.999),eps=1e-8).state_dict()['param_groups']
        saved = cp['readout_optimizer_state']; require(saved['param_groups'] == expected and len(saved['state']) == 8, 'B AdamW recipe/membership differs')
        for name, group in zip(READOUT_SHAPES, expected):
            checked_tensor(torch,cp['after_readout_state'][name],READOUT_SHAPES[name],torch.float32,name)
            moments = saved['state'][group['params'][0]]; require(set(moments)=={'step','exp_avg','exp_avg_sq'}, 'B moment fields differ')
            checked_tensor(torch,moments['step'],(),torch.float32,'B step'); require(float(moments['step'])==2, 'B actual moment step2 differs')
            for key in ('exp_avg','exp_avg_sq'): checked_tensor(torch,moments[key],READOUT_SHAPES[name],torch.float32,key)
            require(bool((moments['exp_avg_sq']>=0).all()), 'Negative B squared moment')
    expect(row['loading'],dict(schema='native-finetune24-fresh-optimizer-loading-proof-v1',teacher_checkpoint=spec['checkpointmap']['native0'],
        teacher_qualification=spec['qualificationmap']['native0'],teacher_training_source=spec['training_source'],teacher_epoch=24,
        strict_full_native610_loaded=True,typed_BN_initializer_exact=True,native_policy=policy,old_optimizer_restored=False,
        old_RNG_restored=False,old_automatic_resume_eligible=False,fresh_phase_not_formal_resume=True,registered_adapter_tensors=0),'Actual teacher loading flags differ')
    common.cpu_state(torch,old,anchor,row['loading']['teacher_state'],'Reported loading native610')
    return cp, dict(checkpoint=row['checkpoint'],native_optimizer=native, fresh_native_optimizer_state_entries=0,fresh_readout_optimizer_state_entries=0,
        typed_BN106_affine159_buffers_CPU_exact=True,
        initial_native610_CPU_exact=True,endpoint_native610_CPU_finite=True,readout_moment_entries=8 if arm=='B' else 0)


def micro_check(torch, np, cv2, coco, old, spec, row, cp, arm, policy):
    require(len(row['micros']) == len(row['audits']) == 4, 'Four actual micro/audit refs required')
    results, previous_accumulated, previous_weights = [], None, None
    for g, (micro, ref) in enumerate(zip(row['micros'], row['audits'])):
        packet = torch.load(str(bound(ref)), map_location='cpu'); identity = packet['identity']; ids = identity['ann_ids']
        image_id = spec['image_ids'][g % 2]; H, W = identity['original_shape']; h, w = identity['actual_img_shape']
        expect(packet, dict(schema=PREFIX+'-audit-v1',arm=arm,global_micro=g,auxiliary_gradient_scale=.5), 'Actual audit identity differs')
        expect(micro, dict(global_micro=g,image_id=image_id,identity=identity,optimizer_update=bool(g%2),native_updates=(g+1)//2,
            readout_updates=(g+1)//2 if arm=='B' else 0,accumulation_divisor=2,actual_backward_calls=1,external_RNG_restored=True), 'Micro count/RNG proof differs')
        require(identity['image_id']==image_id and ids==coco.get_ann_ids(img_ids=[image_id]) and 1<=len(ids)<=100 and
            [r['annotation_id'] for r in identity['parser_rows'] if r['native_parser_retained']]==ids, 'Complete raw/parser identity differs')
        image = coco.load_imgs([image_id])[0]; require((image['height'],image['width'])==(H,W) and 0<h<=800 and 0<w<=800, 'Image dimensions differ')
        raw = np.stack([np.zeros((H,W),np.uint8) if ann.get('segmentation')==[] else coco.annToMask(ann) for ann in coco.load_anns(ids)])
        masks = torch.from_numpy(raw).bool(); same_state(torch,{'raw':packet['raw_GT']},{'raw':masks},'Independent original raw GT differs')
        padded = np.zeros((len(ids),800,800),np.uint8)
        for j, mask in enumerate(raw): padded[j,:h,:w] = cv2.resize(mask,(w,h),interpolation=cv2.INTER_NEAREST)
        require(old.tensor_digest(torch,torch.from_numpy(padded),'uint8')==micro['target']['masks'] and
            micro['target']['ann_ids']==ids and micro['target']['complete_all_raw_annotations'] is True, 'Actual native padded target proof differs')
        metadata = packet['metadata']; require(metadata==micro['metadata'], 'Metadata audit/report join differs')
        expect(metadata, dict(arm=arm,auxiliary_weight=1.,native_atoms=30,raw_annotation_ids=ids,
            processed_GT_to_raw=list(range(len(ids))),processed_GT_observation=dict(calls=1,correspondence_validator_passed=True),
            native_forward_counts=dict(head_forward_started=1,head_forward_finished=1,pixel_decoder=1,mask_embed=10),
            fixed_teacher_matching_used=False,point_indices_sampled_here=False,backward_or_optimizer_step_performed=False,
            auxiliary_accumulation_scaled=False), 'Current native call/correspondence proof differs')
        require(set(metadata['native_atom_values'])==set(atomic_keys()) and math.isfinite(packet['native_total']) and
            all(type(v) in (int,float) and math.isfinite(v) for v in metadata['native_atom_values'].values()), 'Thirty finite atoms required')
        matches = metadata['current_stage_matches']; require(len(matches)==10, 'Ten CURRENT native stage matches required')
        for match in matches:
            q, assigned = match['positive_query_indices'],match['assigned_GT_indices']
            require(len(q)==len(set(q))==len(ids) and all(type(x)is int and 0<=x<100 for x in q) and
                sorted(assigned)==list(range(len(ids))) and match['positives']==len(ids), 'Current one-to-one all-GT matching differs')
        final = matches[-1]; queries = [final['positive_query_indices'][final['assigned_GT_indices'].index(j)] for j in range(len(ids))]
        require(metadata['final_query_by_raw_GT']==queries and packet['query_ids'].tolist()==queries, 'Final actual query/raw row mapping differs')
        generator = torch.Generator(device='cpu').manual_seed(spec['point_seed_base']+g)
        expected_indices = torch.randint(H*W,(len(ids),12544),generator=generator,dtype=torch.int64)
        require(torch.equal(packet['point_indices'],expected_indices),'Private replacement point indices differ')
        projected = torch.einsum('bqc,bchw->bqhw',packet['e0'],packet['F256'])
        projection = compare(torch,packet['M200'],projected,NUMERIC['native_projection_rtol'],NUMERIC['native_projection_atol'],'Native coefficient projection')
        L = render(torch,packet['M200'],identity)
        replay = replay_auxiliary(torch,arm,packet['F256'],packet['e0'],L,identity,packet['readout_before'],packet['query_ids'],masks,expected_indices)
        require(math.isclose(replay['loss'],packet['auxiliary_loss'],rel_tol=NUMERIC['forward_rtol'],abs_tol=NUMERIC['forward_atol']) and
            micro['auxiliary_loss']==packet['auxiliary_loss'] and micro['native_total']==packet['native_total'], 'Aux BCE/report arithmetic differs')
        vjp = compare(torch,packet['auxiliary_L_gradient'],replay['native_logits_gradient'][0,packet['query_ids']],
            NUMERIC['gradient_rtol'],NUMERIC['gradient_atol'],'Auxiliary original L local VJP/2')
        require(set(micro['native_gradient_norms'])==set(policy['active_names']) and
            all(v is not None and math.isfinite(v) and v>=0 for v in micro['native_gradient_norms'].values()) and
            set(micro['native_live_total_gradient_norms'])=={'F256','e0','M200'} and
            all(math.isfinite(v) and v>=0 for v in micro['native_live_total_gradient_norms'].values()), 'Observed native gradient norms invalid')
        gradients = {}
        if arm=='B':
            require(set(packet['readout_pending_before'])==set(packet['readout_accumulated_before_clip'])==set(READOUT_SHAPES), 'B gradient membership differs')
            if g%2==0: require(all(v is None for v in packet['readout_pending_before'].values()), 'Fresh accumulation window required')
            else:
                same_state(torch,packet['readout_pending_before'],previous_accumulated,'Pending B gradients were flushed/changed')
                same_state(torch,packet['readout_before'],previous_weights,'B weights changed inside accumulation pair')
            if g==0: same_state(torch,packet['readout_before'],cp['before_readout_state'],'First audit/CP init differs')
            for name in READOUT_SHAPES:
                actual = packet['readout_accumulated_before_clip'][name]; checked_tensor(torch,actual,READOUT_SHAPES[name],torch.float32,name)
                pending = packet['readout_pending_before'][name]
                contribution = actual if pending is None else actual-pending
                gradients[name] = compare(torch,contribution,replay['readout_gradients'][name],NUMERIC['gradient_rtol'],NUMERIC['gradient_atol'],name+' aux micro gradient/2')
                require(math.isclose(float(actual.norm()),micro['readout_gradient_norms'][name],rel_tol=1e-6,abs_tol=1e-8),'B reported accumulated norm differs')
            require(g<2 or all(v>0 for v in micro['readout_gradient_norms'].values()), 'Selected warm B activity absent')
            previous_accumulated,previous_weights=packet['readout_accumulated_before_clip'],packet['readout_before']
        else: require(packet['readout_before']==packet['readout_pending_before']==packet['readout_accumulated_before_clip']=={} and micro['readout_gradient_norms']=={},'A has readout state')
        require((g%2==0 and micro['combined_preclip_norm'] is None) or (g%2==1 and math.isfinite(micro['combined_preclip_norm']) and micro['combined_preclip_norm']>=0), 'Combined clip metadata differs')
        results.append(dict(global_micro=g,auxiliary_BCE_CPU=replay['loss'],native_projection=projection,auxiliary_L_VJP=vjp,readout_micro_gradients=gradients))
        del packet,L,projected,replay
    return results


def atomic_keys():
    return ('loss_cls','loss_mask','loss_dice') + tuple('d%d.%s'%(j,k) for j in range(9) for k in ('loss_cls','loss_mask','loss_dice'))


def qualify(spec_file, acceptance_file):
    path=Path(spec_file); output=Path(acceptance_file)
    require(path.is_absolute() and output.is_absolute() and not output.exists() and os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU inert CLI/fresh absolute output required')
    path=path.resolve(strict=True); spec=read(path)
    expect(spec,dict(schema='ROOT-'+PREFIX+'-spec-v1',kind='mask2former',teacher_epoch=24,CPU_affinity=[0,1],
        allocator_limit_MiB=4096,whole_GPU_owner_limit_MiB=8192,CPU_numeric_protocol=NUMERIC,kappa=1.,eps=1e-6),'Prelocked tiny SPEC differs')
    require(type(spec['deadline_seconds']) is int and 0<spec['deadline_seconds']<=10800 and len(spec['image_ids'])==2 and
        len(set(spec['image_ids']))==2 and all(type(i)is int and i>0 for i in spec['image_ids']) and
        all(type(spec[k])is int and 0<=spec[k]<2**63-4 for k in ('point_seed_base','loss_seed_base','readout_seed')),'Prelocked seeds/image/deadline differ')
    producer,refs=source_gate(path,spec); report_ref=reference(Path(spec['output_directory'])/'REPORT.json'); report=read(bound(report_ref))
    expect(report,dict(schema=PREFIX+'-report-v1',status='QUERY_PIXEL_JOINT_NATIVE_PREFLIGHT_GPU_COMPLETE_PENDING_CPU',source=refs['source'],
        spec=refs['spec'],GPU_admission=refs['GPU_admission'],phase=spec['phase'],error=None,CPU_numeric_protocol=NUMERIC,
        actual_total_backward_calls=8,extra_backward_or_autograd_grad=False,native30_CPU_replay_claimed=False,separate_AdamW_optimizers=True,
        combined_gradient_clip=True,isolated_F_e0_M_gradient_replay_claimed=False,formal_training_or_initializer_qualified=False,no_DEV=True,no_TEST=True,no_AP=True,
        efficacy_claimed=False,compute_matched_claimed=False,conditioning_isolated_claimed=False),'Actual report identity/scope differs')
    require(report['cleanup']['allocated_after']==0 and set(report['result'])=={'A','B'},'Actual cleanup/complete two arms required')
    import psutil
    require(psutil.Process().cpu_affinity()==[0,1],'Actual CPU01 affinity required')
    import torch
    import numpy as np
    import cv2
    require(not torch.cuda.is_initialized() and torch.__version__=='2.0.1+cu118','Bound Torch/cold CUDA required')
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    from mmdet.datasets.api_wrappers import COCO
    old,common,anchor,policy,groups,audit=teacher_state(torch,spec,report['result'])
    coco=COCO(str(bound(spec['FIT']))); require(len(coco.imgs)==741 and len(coco.anns)==1661 and list(coco.imgs)[:2]==spec['image_ids'],'Original canonical FIT population differs')
    cold=cold_readout(torch,spec['readout_seed']); summaries={}
    for arm in ('A','B'):
        row=report['result'][arm]; cp,summary=checkpoint_check(torch,old,common,spec,refs,row,arm,anchor,policy,groups,audit,cold)
        summary['audits']=micro_check(torch,np,cv2,coco,old,spec,row,cp,arm,policy); summaries[arm]=summary; del cp
    require(report['result']['A']['loading']==report['result']['B']['loading'],'Independent A/B qualified teacher loading differs')
    for ref in spec['sources']: bound(ref)
    require(reference(path)==refs['spec'] and reference(report_ref['path'])==report_ref and not torch.cuda.is_initialized(),'Final bindings/CUDA isolation differ')
    result=dict(schema=QUAL_SCHEMA,status=QUAL_STATUS,**refs,report=report_ref,arms=summaries,CPU_numeric_protocol=NUMERIC,
        teacher_checkpoint=spec['checkpointmap']['native0'],teacher_qualification=spec['qualificationmap']['native0'],
        components={k:spec[k] for k in ('posterior_readout_source','joint_capture_source','joint_losses_source','joint_adapter_source','native_objective_source','renderer_source')},
        CPU_execution=dict(device='cpu',affinity=[0,1],CUDA_VISIBLE_DEVICES='',torch_num_threads=torch.get_num_threads(),torch_num_interop_threads=torch.get_num_interop_threads(),CUDA_initialized=False),
        technical_microsteps_per_arm=4,native_updates_per_arm=2,B_readout_updates=2,auxiliary_point_BCE_and_local_VJP_CPU_replayed=True,
        B8_auxiliary_micro_gradient_CPU_replayed=True,FP32_accumulated_gradient_subtraction_bitwise_claimed=False,
        local_VJP_uses_CPU_reconstructed_original_L=True,actual_GPU_original_L_tensor_exported=False,
        native30_full_CPU_replay=False,whole_CUDA_optimizer_trajectory_CPU_replay=False,isolated_F_e0_M_gradient_CPU_replay=False,
        raw_processed_correspondence_GPU_observed_and_original_GT_CPU_checked=True,certified_GPU_error_bound=False,
        native_forward_counts_SOURCE_observed=True,formal_training_or_initializer_qualified=False,no_AP=True,no_DEV=True,no_TEST=True,efficacy_claimed=False)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8',newline='\n') as stream: json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False); stream.write('\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--spec',required=True); parser.add_argument('--acceptance',required=True); parser.add_argument('--owner-execute',action='store_true')
    args=parser.parse_args(); require(args.owner_execute,'Inert SOURCE: explicit ROOT-owned CPU execution required'); qualify(args.spec,args.acceptance)


if __name__=='__main__':
    main()
