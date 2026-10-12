"""New UNBOUND N13 frozen-E24 FIT region cache SOURCE; ROOT admits once.
No training, optimizer, DEV, TEST or numerical work at module import.
"""
import argparse
from contextlib import contextmanager
from functools import partial
import gc
import hashlib
import inspect
import json
import math
from pathlib import Path
import time
import types

def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    def reject(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=reject)


def pre_gate(spec, count):
    """Only original RAW and stdlib first; every terminal gate precedes SHA/import."""
    keys = ('receipt', 'phase', 'admission', 'phase_binding', 'bootstrap')
    rows = spec['RAW_inputs']
    require(type(rows) is list and len(rows) == count and
        all(type(row) is dict and set(keys) <= set(row) for row in rows),
        'Complete original normal RAW prefix required')
    failed = [dict(receipt=spec['prior_failed_CPU_receipt'],
        phase=Path(spec['prior_failed_CPU_receipt']['path']).parent.name, returncode=1)]
    require(type(spec['additional_failed_RAW_inputs']) is list and
        len(spec['additional_failed_RAW_inputs']) == 3, 'Four original failures required')
    for row in spec['additional_failed_RAW_inputs']:
        require(type(row) is dict and set(row) == {'receipt', 'phase', 'returncode'} and
            type(row['returncode']) is int and row['returncode'] == 1, 'Explicit failed1 required')
        failed.append(row)
    all_rows = [dict(row, returncode=0) for row in rows] + failed
    locators = [row['receipt']['path'] for row in all_rows]
    pins = spec['terminal_identity_pins']
    require(len(set(locators)) == len(locators) and type(pins) is dict and
        set(pins) == set(locators), 'Unique complete original identity/admission pins required')
    raws = []
    for row in all_rows:
        ref, phase, code = row['receipt'], row['phase'], row['returncode']
        path = Path(ref['path'])
        require(path.is_absolute() and path.name == 'PARENT_RECEIPT.json' and
            path.parent.name == phase, 'Explicit original RAW/phase locator required')
        raw = read(path)
        pin = pins[str(path)]
        require(set(pin) == {'phase', 'returncode', 'original_identity', 'admission_sha256'} and
            pin['phase'] == phase and pin['returncode'] == code and
            type(pin['original_identity']) is dict and pin['original_identity'] and
            raw['original_identity'] == pin['original_identity'] and
            raw['admission_sha256'] == pin['admission_sha256'] and
            raw['status'] == 'ACTUAL_OWNED_TERMINAL' and type(raw['returncode']) is int and
            raw['returncode'] == code and raw['root_wait_reaped'] is True and
            raw['held_handle_exit'] == dict(wait_state='WAIT_OBJECT_0', actual_exit_code=code) and
            type(raw['held_handle_exit']['actual_exit_code']) is int and
            raw['numeric_birth_absence']['original_birth_absent'] is True and
            raw.get('error') is None and raw.get('abort') is None and
            raw['wait_errors'] is False and raw['GPU_observation_errors'] == [],
            'Actual original held/reaped identity-specific terminal required before SHA/import')
        if 'phase' in raw:
            require(raw['phase'] == phase, 'Original RAW phase differs')
        peak = raw['whole_GPU_observed_peak_MiB']
        require(type(peak) in (int, float) and math.isfinite(peak) and 0 <= peak <= 8192 and
            raw['whole_GPU_used_limit_MiB'] == 8192 and raw['whole_leaf_VRAM_certified'] is False and
            raw['GPU_scope'] == 'whole GPU observed, not per-process WDDM allocation',
            'Original RAW resource scope differs')
        raws.append(raw)
    refs = spec['sources']
    require(type(refs) is list and refs, 'Complete source/input closure required')
    closure = {}
    for ref in refs:
        require(type(ref) is dict and set(ref) == {'path', 'bytes', 'sha256'} and
            type(ref['path']) is str and Path(ref['path']).is_absolute() and
            type(ref['bytes']) is int and ref['bytes'] >= 0 and type(ref['sha256']) is str and
            len(ref['sha256']) == 64 and set(ref['sha256']) <= set('0123456789abcdef') and
            ref['path'] not in closure, 'Unique absolute byte/SHA references required')
        closure[ref['path']] = ref
    require(all(row['receipt'] in closure.values() for row in all_rows), 'Original RAW refs absent from closure')
    for ref in refs:
        digest = hashlib.sha256()
        with Path(ref['path']).open('rb') as stream:
            for block in iter(lambda: stream.read(1048576), b''):
                digest.update(block)
        require(Path(ref['path']).stat().st_size == ref['bytes'] and digest.hexdigest() == ref['sha256'],
            'Bound source/input bytes changed: ' + ref['path'])
    common_ref = spec['region_common_source']
    require(common_ref in closure.values(), 'Actually bound common SOURCE required')
    module = types.ModuleType('_N13_region_common')
    module.__file__ = common_ref['path']
    exec(compile(Path(common_ref['path']).read_bytes(), common_ref['path'], 'exec'), module.__dict__)
    return module, raws[:count], closure

@contextmanager
def observed_fields(torch, head):
    before = {name: (tuple(module._forward_pre_hooks.items()), tuple(module._forward_hooks.items())) for name, module in head.named_modules()}
    cache = dict(F=[], e=[], output=[]); handles = []
    def feature(module, args, result):
        require(isinstance(result, (tuple, list)) and len(result) == 2, 'Native pixel-decoder two outputs required')
        cache['F'].append(result[0]); return None
    def embedding(module, args, result): cache['e'].append(result); return None
    def output(module, args, result): cache['output'].append(result); return None
    try:
        handles = [head.pixel_decoder.register_forward_hook(feature), head.mask_embed.register_forward_hook(embedding),
                   head.register_forward_hook(output)]
        yield cache
        require(len(cache['F']) == len(cache['output']) == 1 and len(cache['e']) == 10, 'One original loss forward/ten embeddings required')
        classes, masks = cache['output'][0]
        require(len(classes) == len(masks) == 10, 'All ten native stages required')
        F, e, M, C = cache['F'][0], cache['e'][-1], masks[-1], classes[-1]
        for value, shape in ((F, (1,256,200,200)), (e, (1,100,256)), (M, (1,100,200,200)), (C, (1,100,2))):
            require(isinstance(value, torch.Tensor) and value.dtype == torch.float32 and tuple(value.shape) == shape and
                    not value.requires_grad and bool(torch.isfinite(value).all()), 'Finite frozen native field differs')
        require(torch.equal(torch.einsum('bqc,bchw->bqhw', e, F), M), 'Native final F/e projection differs')
    finally:
        for handle in handles: handle.remove()
        require(before == {name: (tuple(module._forward_pre_hooks.items()), tuple(module._forward_hooks.items()))
                           for name, module in head.named_modules()}, 'Passive hook inventory not restored')


def matched_queries(matches, mapping, G):
    require(len(matches) == 10 and 1 <= G <= 100 and sorted(mapping) == list(range(G)), 'Positive complete original FIT GT/mapping required')
    for row in matches:
        queries, targets = row['positive_query_indices'], row['assigned_GT_indices']
        require(row['positives'] == len(queries) == len(set(queries)) == G and all(type(q) is int and 0 <= q < 100 for q in queries) and
                all(type(i) is int for i in targets) and sorted(targets) == list(range(G)), 'Original one-to-one matching must retain every GT')
    output = [None]*G
    for query, target in zip(matches[-1]['positive_query_indices'], matches[-1]['assigned_GT_indices']): output[mapping[target]] = query
    require(None not in output and len(set(output)) == G, 'Final query-to-raw annotation assignment differs')
    return output


def sampled_features(torch, renderer, F, identity, indices):
    G, P = indices.shape; on_device = indices.to(F.device).reshape(-1); parts = []
    for first in range(0, 256, 16):
        original = renderer(F[:,first:first+16], identity)
        require(tuple(original.shape) == (16,)+tuple(identity['original_shape']), 'Actual two-stage original F projection differs')
        values = original.flatten(1)[:,on_device].transpose(0,1).reshape(G,P,16)
        parts.append(values.cpu().clone()); del original, values
    result = torch.cat(parts, dim=2).contiguous()
    require(tuple(result.shape) == (G,1024,256) and bool(torch.isfinite(result).all()), 'Sampled original coarse features differ')
    return result


def cache_image(torch, np, cv2, ndimage, masks_api, modules, model, dataset, coco, spec, gate, index):
    common=gate['common']; native,legacy,pref,initializer,native_common,old,diag,joint,preflight=modules
    context,closure=gate['context'],gate['closure']
    old_row=gate['old_report']['rows'][index]; old_ref=gate['old_report']['packet_refs'][index]
    old_packet=torch.load(str(common.bound(old_ref)),map_location='cpu')
    common.require(old_packet['schema']==gate['old'].PACKET_SCHEMA and old_packet['identity']==old_row['identity'] and
                   old_row['packet']==old_ref and old_packet['dataset_index']==old_row['dataset_index']==index and
                   old_packet['source']==gate['old_report']['source'] and old_packet['spec']==gate['old_report']['spec'] and
                   old_packet['teacher_checkpoint']==context['checkpointmap']['native0'], 'Old qualified FIT carrier identity differs')
    info=dataset.get_data_info(index); image=common.reference(info['img_path'])
    common.require(image in closure.values(), 'Each actual FIT JPEG must be bound before loading')
    data,stripped,gray800,gray,identity=old.load_image(model,dataset,coco,info)
    del stripped,gray800,gray
    identity['dataset_index']=index
    common.require(identity==old_packet['identity'] and identity['image']==image, 'Original FIT image/GT/parser identity differs')
    processed_proof=pref.check_targets(torch,data,identity)
    raw_GT,mapping,validator=preflight.raw_GT_validator(torch,np,cv2,coco,data,identity)
    external=pref.rng_snapshot(torch); head=model.panoptic_head
    matching_seed=spec['matching_loss_seed_base']+index
    rank_seed=spec['region_rank_seed_base']+index; uniform_seed=spec['uniform_seed_base']+index
    with torch.no_grad(),torch.random.fork_rng(devices=[0]):
        torch.random.default_generator.manual_seed(matching_seed); torch.cuda.manual_seed(matching_seed)
        with joint._observe_processed_GT(torch,head,identity['ann_ids'],raw_GT,mapping,identity,validator) as gt_observation:
            with old.observe_targets(head) as matches:
                with observed_fields(torch,head) as fields:
                    losses=model.loss(data['inputs'],data['data_samples'])
        common.require(set(losses)==set(old.ATOMIC_KEYS) and len(losses)==30, 'Exactly thirty original native loss atoms required')
        loss_values={}
        for name,value in losses.items():
            items=list(value) if isinstance(value,(tuple,list)) else [value]
            common.require(items and all(isinstance(v,torch.Tensor) and not v.requires_grad and bool(torch.isfinite(v).all())
                           for v in items), 'Original frozen loss nonfinite/live')
            loss_values[name]=float(sum(v.mean() for v in items).cpu())
        queries=matched_queries(matches,mapping,len(raw_GT))
        common.require(queries==old_packet['query_ids'].tolist()==old_row['query_ids'] and
                       matches==old_packet['native_stage_matches'], 'Exact original ten-stage matching/GT query map required')
        F,e=fields['F'][0],fields['e'][-1]; classes,mask_stages=fields['output'][0]
        M,C=mask_stages[-1],classes[-1]
        query_tensor=torch.tensor(queries,device=M.device,dtype=torch.int64)
        embeddings=e[0,query_tensor].cpu().clone().contiguous(); class_cpu=C.cpu().clone().contiguous()
        common.require(torch.equal(embeddings,old_packet['query_embeddings']) and torch.equal(class_cpu,old_packet['native_class_logits']),
                       'Original exact frozen native e/C differ from qualified FIT cache')
        renderer=partial(diag.render_original,torch)
        natural_L=renderer(M,identity)
        common.require(natural_L.device.type=='cuda' and natural_L.dtype==torch.float32 and
                       tuple(natural_L.shape)==(100,)+tuple(identity['original_shape']) and bool(torch.isfinite(natural_L).all()),
                       'Original natural all100 GPU renderer required')
        L=natural_L[query_tensor].cpu().clone().contiguous(); del natural_L
        old_at_points=L.flatten(1).gather(1,old_packet['point_indices'])
        common.require(torch.allclose(old_at_points,old_packet['native_logits'],rtol=common.NUMERIC['native_L_rtol'],
                       atol=common.NUMERIC['native_L_atol']), 'Natural100 versus legacy sampled-query replay outside practical tolerance')
        Ps=[]; region_rows={arm:[] for arm in common.ARMS}; Ks=[]; fallback=[]
        for g in range(len(queries)):
            P,regions,K,no_opposite=common.gpu_regions(np,ndimage,L[g].numpy())
            Ps.append(P); Ks.append(K); fallback.append(bool(no_opposite))
            for arm in common.ARMS: region_rows[arm].append(regions[arm])
        indices=common.gpu_sample_points(torch,np,region_rows,Ks,rank_seed,uniform_seed)
        arms={}; raw_cpu=raw_GT.cpu().contiguous()
        for arm in common.ARMS:
            idx=indices[arm]
            memberships=torch.from_numpy(np.stack(region_rows[arm])).flatten(1).gather(1,idx).contiguous()
            arms[arm]=dict(F_samples=sampled_features(torch,renderer,F,identity,idx),
                           native_logits=L.flatten(1).gather(1,idx).contiguous(),
                           targets=raw_cpu.flatten(1).gather(1,idx).contiguous(),
                           point_indices=idx,point_in_region=memberships)
        packet=dict(schema=common.PACKET_SCHEMA,dataset_index=index,identity=identity,
                    query_ids=torch.tensor(queries,dtype=torch.int64),query_embeddings=embeddings,
                    native_class_logits=class_cpu,matched_native_L=L,
                    native_P_rles=[common.rle(np,masks_api,P) for P in Ps],
                    region_rles={arm:[common.rle(np,masks_api,R) for R in region_rows[arm]] for arm in common.ARMS},
                    K=Ks,no_opposite_fallback=fallback,arms=arms,
                    original_cache_packet=old_ref,raw_GT_digest=pref.digest_tensor(raw_GT),
                    raw_GT_area=raw_cpu.flatten(1).sum(1).tolist(),raw_target_mapping=mapping,
                    region_rank_seed=rank_seed,uniform_seed=uniform_seed,matching_loss_seed=matching_seed,
                    native_stage_matches=matches,native_loss_atoms=loss_values,
                    actual_raw_processed_GT_correspondence=gt_observation,processed_target_proof=processed_proof,
                    GT_selected_regions=False,replacement=True,duplicates_preserved=True,
                    actual_GPU_L_used_for_selection=True,CPU_approximate_L_substituted_for_selection=False,
                    JPEG_patch_used_by_PR=False,native_natural100_rendered_before_matching_query_selection=True)
        carrier=dict(schema=common.CARRIER_SCHEMA,dataset_index=index,identity=identity,
                     query_ids=packet['query_ids'].clone(),native_F256=F.cpu().clone().contiguous(),
                     matched_M200=M[:,query_tensor].cpu().clone().contiguous())
    common.require(pref.tree_equal(torch,external,pref.rng_snapshot(torch)), 'External RNG changed during frozen cache generation')
    common.require(not any(p.requires_grad or p.grad is not None for p in model.parameters()) and not model.training,
                   'Frozen teacher has trainable/live gradient or training mode')
    for arm in common.ARMS:
        common.require(arms[arm]['targets'].dtype==arms[arm]['point_in_region'].dtype==torch.bool and
                       tuple(arms[arm]['targets'].shape)==(len(raw_GT),1024), 'Hard original labels/memberships differ')
    row=dict(dataset_index=index,image_id=identity['image_id'],identity=identity,raw_GT=len(raw_GT),query_ids=queries,
             points_per_arm=len(raw_GT)*1024,K=Ks,no_opposite_fallback=fallback,
             region_rank_seed=rank_seed,uniform_seed=uniform_seed,matching_loss_seed=matching_seed,
             original_cache_packet=old_ref,raw_GT_ids=identity['ann_ids'],
             duplicate_points_by_arm={arm:[1024-int(torch.unique(v).numel()) for v in arms[arm]['point_indices']] for arm in common.ARMS},
             sampled_foreground_by_arm={arm:arms[arm]['targets'].sum(1).tolist() for arm in common.ARMS},
             sampled_editable_by_arm={arm:arms[arm]['point_in_region'].sum(1).tolist() for arm in common.ARMS},
             carrier_digests={k:common.digest_tensor(carrier[k]) for k in ('query_ids','native_F256','matched_M200')},
             one_teacher_forward=True,all_ten_matching_stages=10,native_loss_atoms=30,
             backward_calls=0,optimizer_created=False,weights_updated=False,external_RNG_restored=True,
             old_query_matching_exact=True,old_native_e_C_exact=True,old_sampled_L_practical_check=True)
    return row,packet,carrier


def run(spec_path):
    started=time.monotonic(); path=Path(spec_path)
    require(path.is_absolute(), 'Absolute ROOT cache SPEC required'); path=path.resolve(strict=True)
    spec=read(path); common,raws,closure=pre_gate(spec,37)
    gate=common.source_gate(path,spec,__file__,raws,closure); gate['common']=common
    context=gate['context']; output=Path(spec['output_directory'])
    common.require(output.is_absolute() and not output.exists(), 'Fresh absolute cache output required')
    import psutil
    common.require(psutil.Process().cpu_affinity()==[0,1], 'Actual CPU01 required before numerical imports')
    import torch
    common.require(torch.__version__=='2.0.1+cu118' and not torch.cuda.is_initialized(), 'Original Torch/cold CUDA required')
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    legacy=common.load(context['evaluation_source'],'_N13_original_E24_qualification')
    teachers=legacy.qualified_training(context)
    import numpy as np
    import cv2
    from scipy import ndimage
    from pycocotools import mask as masks_api
    from mmengine import Config
    from mmengine.model import revert_sync_batchnorm
    from mmengine.registry import init_default_scope
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    from mmdet.datasets import CocoDataset
    from mmdet.datasets.api_wrappers import COCO
    common.require(np.__version__=='1.23.5' and cv2.__version__=='4.8.0', 'Original NumPy/OpenCV required')
    libraries=common.imported_files(torch,np,cv2,masks_api,COCO,closure,gate['admission'],ndimage)
    modules=[common.load(context[k],'_N13_'+k) for k in common.MODULES]
    native,legacy,pref,initializer,native_common,old,diag,joint,preflight=modules; old.torch=torch
    register_all_modules(init_default_scope=False); init_default_scope('mmdet')
    construction=dict(teachers['native0']['training_spec'],training_source=context['training_source'],
                      checkpointmap=context['checkpointmap'],qualificationmap=context['qualificationmap'],
                      order_seed=native.ORDER_SEED,phase_loss_seed_base=native.LOSS_SEED_BASE)
    model,anchor,policy,loading=native.prepare_training_model(torch,pref,initializer,native_common,MODELS,
                          revert_sync_batchnorm,Config,tuple(CocoDataset.METAINFO['classes']),construction,teachers['native0'])
    native.imported_library_gate(torch,model,context)
    common.require(Path(inspect.getsourcefile(torch.nn.functional.interpolate)).resolve()==common.bound(context['interpolation_SOURCE']).resolve(),
                   'Original interpolation SOURCE differs')
    model.requires_grad_(False); model.eval(); model.backbone.with_cp=False
    common.require(len(anchor)==len(model.state_dict())==610 and len(tuple(model.parameters()))==451 and
                   not any(p.requires_grad or p.grad is not None for p in model.parameters()), 'Strict frozen native610/451 required')
    dataset=CocoDataset(ann_file=str(common.bound(spec['FIT'])),data_prefix=dict(img=spec['image_root']),
                        metainfo=dict(classes=('ship',)),pipeline=[],filter_cfg=None,test_mode=True)
    coco=COCO(str(common.bound(spec['FIT']))); image_ids=[dataset.get_data_info(i)['img_id'] for i in range(len(dataset))]
    common.require(len(dataset)==len(coco.imgs)==741 and len(coco.anns)==1661 and set(coco.cats)=={1} and
                   image_ids==list(coco.imgs) and len(set(image_ids))==741, 'Canonical full FIT741/1661 required')
    common.require(all(str(Path(dataset.get_data_info(i)['img_path']).resolve(strict=True)) in closure for i in range(741)),
                   'Every original FIT JPEG must be input-bound')
    torch.cuda.init(); torch.cuda.set_per_process_memory_fraction(min(1.,4096*1024**2/torch.cuda.get_device_properties(0).total_memory),0)
    torch.cuda.reset_peak_memory_stats(0); model.to('cuda:0'); output.mkdir(); (output/'VALIDATION_CARRIERS').mkdir()
    common.write(output/'START.json',dict(schema='N13-sameK-region-cache-start-v1',source=gate['source'],spec=gate['spec_ref'],
        GPU_admission=gate['admission_ref'],phase=spec['phase'],teacher_checkpoint=context['checkpointmap']['native0'],
        teacher_loading=loading,recipe=common.RECIPE,regions_recipe=common.REGIONS_RECIPE,numeric_protocol=common.NUMERIC,
        prior_N12_CPU_qualification=spec['prior_N12_CPU_qualification'],original_cache_CPU_qualification=spec['original_cache_CPU_qualification'],
        frozen_state_entries=610,frozen_parameter_tensors=451,heads_loaded=False,optimizer_created=False,no_DEV=True,no_TEST=True))
    packet_refs=[]; carrier_refs=[]; rows=[]; total_GT=0
    with (output/'TRACE.jsonl').open('x',encoding='utf-8',newline='\n') as trace:
        for index in range(741):
            common.require(time.monotonic()-started<spec['deadline_seconds'], 'Prelocked N13 cache deadline exceeded; no retry/fallback')
            row,packet,carrier=cache_image(torch,np,cv2,ndimage,masks_api,modules,model,dataset,coco,spec,gate,index)
            common.require(row['image_id']==image_ids[index], 'Canonical FIT row order changed')
            metadata=dict(source=gate['source'],spec=gate['spec_ref'],phase=spec['phase'],teacher_checkpoint=context['checkpointmap']['native0'])
            packet.update(metadata); carrier.update(metadata)
            target=output/'VALIDATION_CARRIERS'/('CARRIER_%04d.pth'%index)
            with target.open('xb') as stream: torch.save(carrier,stream)
            carrier_ref=common.reference(target); packet['validation_carrier']=carrier_ref; row['validation_carrier']=carrier_ref
            row['tensor_digests']=common.packet_digests(packet)
            target=output/('PACKET_%04d.pth'%index)
            with target.open('xb') as stream: torch.save(packet,stream)
            row['packet']=common.reference(target); packet_refs.append(row['packet']); carrier_refs.append(carrier_ref); rows.append(row)
            total_GT+=row['raw_GT']; trace.write(json.dumps(row,allow_nan=False)+'\n'); trace.flush()
            del packet,carrier
            print(json.dumps(dict(cached_FIT_images=index+1,raw_GT=total_GT,points_per_arm=total_GT*1024)),flush=True)
    final=pref.cpu_tree(torch,model.state_dict())
    common.require(total_GT==1661 and final.keys()==anchor.keys() and all(torch.equal(final[n],anchor[n]) for n in anchor) and
                   not any(p.grad is not None for p in model.parameters()), 'Full FIT population/unchanged native610 differs')
    peak=torch.cuda.max_memory_allocated(0); model.cpu(); del model,anchor,final; gc.collect(); cleanup=pref.clear_gpu(torch)
    common.require(cleanup['allocated_after']==0 and psutil.Process().cpu_affinity()==[0,1] and
                   torch.get_num_threads()==torch.get_num_interop_threads()==2, 'Final cache cleanup/resources differ')
    libraries=common.imported_files(torch,np,cv2,masks_api,COCO,closure,gate['admission'],ndimage)
    for ref in spec['sources']: common.bound(ref)
    common.require(common.reference(path)==gate['spec_ref'] and common.reference(__file__)==gate['source'] and
                   common.reference(spec['GPU_admission_file'])==gate['admission_ref'] and time.monotonic()-started<spec['deadline_seconds'],
                   'Final cache SOURCE/SPEC/admission/deadline changed')
    report=dict(schema=common.REPORT_SCHEMA,status=common.REPORT_STATUS,source=gate['source'],spec=gate['spec_ref'],
        GPU_admission=gate['admission_ref'],phase=spec['phase'],start=common.reference(output/'START.json'),trace=common.reference(output/'TRACE.jsonl'),
        packet_refs=packet_refs,validation_carrier_refs=carrier_refs,rows=rows,images=741,raw_GT=1661,point_count_per_arm=1661*1024,
        recipe=common.RECIPE,regions_recipe=common.REGIONS_RECIPE,numeric_protocol=common.NUMERIC,
        teacher_checkpoint=context['checkpointmap']['native0'],prior_N12_CPU_qualification=spec['prior_N12_CPU_qualification'],
        original_cache_CPU_qualification=spec['original_cache_CPU_qualification'],actual_imported_files=libraries,
        all_raw_GT_retained=True,one_teacher_forward_per_image=True,teacher_forwards=741,
        native610_unchanged_SOURCE_observed=True,native_gradient_tensors=0,optimizer_created=False,weights_updated=False,backward_calls=0,
        heads_loaded=False,heads_trained=False,head_epochs=0,AP_computed=False,no_DEV=True,no_TEST=True,
        independent_CPU_cache_qualification=False,saved_original_GPU_L_is_region_and_sampling_input=True,
        CPU_approximate_L_substituted_for_selection=False,validation_carriers_separate_from_training_packets=True,
        current_owned_GPU_terminal_self_certified=False,whole_leaf_VRAM_certified=False,automatic_retry=False,
        torch_peak_allocated_bytes=peak,cleanup=cleanup,elapsed_seconds=time.monotonic()-started,deadline_seconds=spec['deadline_seconds'],
        evidence_limits=['Frozen FIT cache generation is not whole-model or head training.',
            'Geometry/original-band arms share rank draws and uniform points but have different region-supervision locations.',
            'All GT/query matching remains original training supervision; GT never selects regions.',
            'No zero-GT FIT carrier, TEST, model efficacy or innovation is qualified here.',
            'Independent CPU qualification and ROOT original held/reaped terminal are separate prerequisites.'])
    common.write(output/'REPORT.json',report); print(json.dumps(dict(status=common.REPORT_STATUS,report=common.reference(output/'REPORT.json'))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--spec',required=True)
    parser.add_argument('--owner-execute',action='store_true'); args=parser.parse_args()
    require(args.owner_execute,'Explicit ROOT-owned single cache execution required'); run(args.spec)
