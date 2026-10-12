"""Independent UNBOUND v2 CPU qualifier for N13 same-K region FIT cache.
OpenCV selection, exact saved-GPU-L sampling and GT are independently checked;
CPU render replays are practical only and never replace metric inputs.
"""
import argparse
import gc
import hashlib
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
        len(spec['additional_failed_RAW_inputs']) == 5, 'Six original failures required')
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
    adapter_ref=spec['cache_contract_source']; require(adapter_ref in closure.values(), 'New adapter SOURCE must be closure-bound')
    adapter=module.load(adapter_ref,'_N13_v2_cache_contract')
    require(module.reference(adapter.__file__)==adapter_ref,'Actually loaded new adapter SOURCE differs')
    return module, adapter, raws[:count], closure


def independent_regions(np, cv2, L):
    """OpenCV domain distances; never call the producer's SciPy/region helper."""
    require(L.dtype==np.float32 and L.ndim==2 and bool(np.isfinite(L).all()), 'Exact saved original GPU L required')
    P=L>np.float32(0); log_region=np.abs(L)<=np.float32(2); K=int(np.count_nonzero(log_region))
    flat=np.arange(L.size,dtype=np.int64); absolute=np.abs(L).reshape(-1)
    no_opposite=not bool(P.any()) or bool(P.all())
    if no_opposite:
        distances=np.full(L.shape,np.inf,dtype=np.float64)
    else:
        to_background=cv2.distanceTransform(P.astype(np.uint8),cv2.DIST_C,3)
        to_foreground=cv2.distanceTransform((~P).astype(np.uint8),cv2.DIST_C,3)
        distances=np.where(P,to_background,to_foreground)
        require(bool((distances>=1).all()) and bool(np.isfinite(distances).all()) and
                bool((distances==np.floor(distances)).all()), 'Independent integer opposing-class distances differ')
    rank=np.lexsort((flat,absolute,distances.reshape(-1)))
    geometry=np.zeros(L.size,dtype=np.bool_); geometry[rank[:K]]=True; geometry=geometry.reshape(L.shape)
    require(int(geometry.sum())==K, 'Independent same-K geometry count differs')
    uncertainty=np.zeros(L.size,dtype=np.bool_); uncertainty[np.lexsort((flat,absolute))[:K]]=True
    require(np.array_equal(uncertainty.reshape(L.shape),log_region), 'Independent oldB/topK uncertainty identity differs')
    return P,{'LOG_PR':log_region,'GEO_PR':geometry},K,no_opposite


def independent_sample_points(torch, np, region_rows, Ks, rank_seed, uniform_seed):
    """Independent private RNG replay, shared ranks/common uniform draws."""
    private_rank=torch.Generator(device='cpu'); private_rank.manual_seed(rank_seed)
    private_uniform=torch.Generator(device='cpu'); private_uniform.manual_seed(uniform_seed)
    outputs={'LOG_PR':[],'GEO_PR':[]}
    for g,K in enumerate(Ks):
        area=region_rows['LOG_PR'][g].size
        if K==0:
            draws=torch.randint(area,(1024,),generator=private_uniform)
            outputs['LOG_PR'].append(draws.clone()); outputs['GEO_PR'].append(draws.clone())
        else:
            ranks=torch.randint(K,(768,),generator=private_rank)
            uniform=torch.randint(area,(256,),generator=private_uniform)
            for name in ('LOG_PR','GEO_PR'):
                ordered=np.flatnonzero(region_rows[name][g].reshape(-1)).astype(np.int64)
                require(len(ordered)==K, 'Independent exact-K ordered ROI required')
                chosen=torch.from_numpy(ordered)[ranks]
                outputs[name].append(torch.cat((chosen,uniform)))
    return {name:torch.stack(outputs[name]).contiguous() for name in outputs}


def render_original_cpu(torch, value, identity):
    """Independent N07-compatible practical two-stage bilinear replay."""
    h,w=identity['actual_img_shape']; H,W=identity['original_shape']
    padded=torch.nn.functional.interpolate(value,size=(800,800),mode='bilinear',align_corners=False)
    crop=padded[0,:,:h,:w]
    return torch.nn.functional.interpolate(crop[:,None],size=(H,W),mode='bilinear',align_corners=False)[:,0]


def sampled_features_cpu(torch, F, identity, indices):
    G,P=indices.shape; parts=[]; flat=indices.reshape(-1)
    for first in range(0,256,16):
        projected=render_original_cpu(torch,F[:,first:first+16],identity)
        gathered=projected.flatten(1)[:,flat].transpose(0,1).reshape(G,P,16)
        parts.append(gathered.clone()); del projected,gathered
    return torch.cat(parts,dim=2).contiguous()


def packet_check(torch,np,cv2,masks,coco,packet,carrier,row,index,image_id,spec,gate):
    common=gate['common']; report,gpu=gate['report'],gate['gpu']; closure=gate['closure']
    G=common.packet_validate(torch,packet,row,report); identity=packet['identity']; H,W=identity['original_shape']
    annotations=coco.getAnnIds(imgIds=[image_id]); image=coco.imgs[image_id]
    require(identity['image_id']==row['image_id']==image_id and row['dataset_index']==packet['dataset_index']==index and
            identity['ann_ids']==row['raw_GT_ids']==annotations and len(annotations)==G and
            [p['annotation_id'] for p in identity['parser_rows'] if p['native_parser_retained']]==annotations and
            [H,W]==[image['height'],image['width']] and identity['batch_input_shape']==[800,800] and
            len(identity['actual_img_shape'])==2 and all(type(x) is int and 0<x<=800 for x in identity['actual_img_shape']),
            'Independent canonical FIT annotation/raster identity differs')
    require(Path(identity['image']['path'])==(Path(spec['image_root'])/image['file_name']).resolve(strict=True) and
            identity['image'] in closure.values(), 'Original input JPEG/annotation lineage differs')
    old_ref=gate['old_report']['packet_refs'][index]
    require(packet['original_cache_packet']==row['original_cache_packet']==old_ref and
            common.reference(old_ref['path'])==old_ref, 'Old qualified matching carrier ref differs')
    old=torch.load(str(common.bound(old_ref)),map_location='cpu')
    require(identity==old['identity'] and torch.equal(packet['query_ids'],old['query_ids']) and
            torch.equal(packet['query_embeddings'],old['query_embeddings']) and
            torch.equal(packet['native_class_logits'],old['native_class_logits']) and
            packet['native_stage_matches']==old['native_stage_matches'] and
            packet['matching_loss_seed']==old['matching_loss_seed']==gpu['matching_loss_seed_base']+index and
            packet['actual_raw_processed_GT_correspondence']==old['actual_raw_processed_GT_correspondence'] and
            type(packet['processed_target_proof']) is dict and bool(packet['processed_target_proof']),
            'Exact original GT/query matching, e/C and processed-target proof differ')
    require(packet['region_rank_seed']==row['region_rank_seed']==gpu['region_rank_seed_base']+index and
            packet['uniform_seed']==row['uniform_seed']==gpu['uniform_seed_base']+index,
            'Independent per-image seed identities differ')
    stages=packet['native_stage_matches']; require(len(stages)==10 and
            sorted(packet['raw_target_mapping'])==list(range(G)), 'Complete raw target mapping required')
    expected_queries=[None]*G
    for stage in stages:
        queries,targets=stage['positive_query_indices'],stage['assigned_GT_indices']
        require(stage['positives']==len(queries)==len(set(queries))==G and sorted(targets)==list(range(G)),
                'Original one-to-one native matching population differs')
    for q,t in zip(stages[-1]['positive_query_indices'],stages[-1]['assigned_GT_indices']):
        expected_queries[packet['raw_target_mapping'][t]]=q
    require(expected_queries==packet['query_ids'].tolist(), 'Independent raw-GT/query correspondence differs')
    require(carrier['schema']==common.CARRIER_SCHEMA and carrier['identity']==identity and carrier['dataset_index']==index and
            carrier['source']==report['source'] and carrier['spec']==report['spec'] and carrier['phase']==report['phase'] and
            carrier['teacher_checkpoint']==report['teacher_checkpoint'] and torch.equal(carrier['query_ids'],packet['query_ids']),
            'Separate validation carrier lineage differs')
    for k,shape in (('native_F256',(1,256,200,200)),('matched_M200',(1,G,200,200))):
        value=carrier[k]; require(value.device.type=='cpu' and value.dtype==torch.float32 and tuple(value.shape)==shape and
            not value.requires_grad and bool(torch.isfinite(value).all()), 'Finite separate CPU validation carrier required')
    require({k:common.digest_tensor(carrier[k]) for k in ('query_ids','native_F256','matched_M200')}==row['carrier_digests'],
            'Separate carrier exact tensor digests differ')
    raw=np.stack([np.zeros((H,W),np.uint8) if a.get('segmentation')==[] else coco.annToMask(a) for a in coco.loadAnns(annotations)])
    require(raw.shape==(G,H,W) and bool(((raw==0)|(raw==1)).all()), 'Independent original binary GT rasters required')
    raw_tensor=torch.from_numpy(raw.astype(np.bool_))
    require(common.digest_tensor(raw_tensor)==packet['raw_GT_digest']==old['raw_GT_digest'] and
            raw_tensor.flatten(1).sum(1).tolist()==packet['raw_GT_area']==old['raw_GT_area'], 'Independent GT area/digest differs')
    region_rows={arm:[] for arm in common.ARMS}; Ks=[]; no_opposite=[]
    for g in range(G):
        P,regions,K,fallback=independent_regions(np,cv2,packet['matched_native_L'][g].numpy())
        require(np.array_equal(P,common.rle_decode(np,masks,packet['native_P_rles'][g])), 'Exact saved GPU native support differs')
        Ks.append(K); no_opposite.append(bool(fallback))
        for arm in common.ARMS:
            require(np.array_equal(regions[arm],common.rle_decode(np,masks,packet['region_rles'][arm][g])),
                    'Independent OpenCV region raster differs: '+arm)
            region_rows[arm].append(regions[arm])
    require(Ks==packet['K']==row['K'] and no_opposite==packet['no_opposite_fallback']==row['no_opposite_fallback'],
            'Exact K/fallback population differs')
    replay=independent_sample_points(torch,np,region_rows,Ks,packet['region_rank_seed'],packet['uniform_seed'])
    for arm in common.ARMS:
        fields=packet['arms'][arm]; idx=fields['point_indices']; arrays=idx.numpy()
        require(torch.equal(idx,replay[arm]), 'Independent private region/uniform draw replay differs')
        expected_y=np.take_along_axis(raw.astype(np.bool_).reshape(G,-1),arrays,axis=1)
        expected_region=np.take_along_axis(np.stack(region_rows[arm]).reshape(G,-1),arrays,axis=1)
        require(torch.equal(fields['targets'],torch.from_numpy(expected_y)) and
                torch.equal(fields['point_in_region'],torch.from_numpy(expected_region)) and
                torch.equal(fields['native_logits'],packet['matched_native_L'].flatten(1).gather(1,idx)),
                'Exact saved L, independently rasterized labels or region flags differ')
        require(row['sampled_foreground_by_arm'][arm]==fields['targets'].sum(1).tolist() and
                row['sampled_editable_by_arm'][arm]==fields['point_in_region'].sum(1).tolist() and
                row['duplicate_points_by_arm'][arm]==[1024-int(torch.unique(v).numel()) for v in idx],
                'Exact sampled foreground/duplicates/editable counts differ')
        for g,K in enumerate(Ks):
            require(bool(fields['point_in_region'][g,:768].all()) if K else not bool(fields['point_in_region'][g].any()),
                    'ROI points outside independent region')
    log_idx,geo_idx=(packet['arms'][arm]['point_indices'] for arm in common.ARMS)
    for g,K in enumerate(Ks):
        require(torch.equal(log_idx[g,768:],geo_idx[g,768:]) if K else torch.equal(log_idx[g],geo_idx[g]),
                'Shared uniform draws/K0 points differ')
    combined=torch.cat((log_idx,geo_idx,old['point_indices']),dim=1)
    expected_F=sampled_features_cpu(torch,carrier['native_F256'],identity,combined)
    feature_errors={}
    for offset,arm in ((0,'LOG_PR'),(1024,'GEO_PR')):
        expected=expected_F[:,offset:offset+1024]; actual=packet['arms'][arm]['F_samples']
        require(torch.allclose(actual,expected,rtol=common.NUMERIC['sampled_F_rtol'],atol=common.NUMERIC['sampled_F_atol']),
                'Independent practical original sampled F differs: '+arm)
        feature_errors[arm]=float((actual-expected).abs().max())
    require(torch.allclose(old['F_samples'],expected_F[:,2048:],rtol=common.NUMERIC['sampled_F_rtol'],
            atol=common.NUMERIC['sampled_F_atol']), 'Practical F carrier relation to original qualified FIT samples differs')
    expected_L=render_original_cpu(torch,carrier['matched_M200'],identity); actual_L=packet['matched_native_L']
    require(bool(torch.isfinite(expected_L).all()),'CPU L diagnostic must remain finite')
    native_L_allclose=bool(torch.allclose(actual_L,expected_L,rtol=common.NUMERIC['native_L_rtol'],atol=common.NUMERIC['native_L_atol']))
    native_L_error=(actual_L-expected_L).abs()
    native_L_failed_pixels=int((native_L_error>(common.NUMERIC['native_L_atol']+common.NUMERIC['native_L_rtol']*expected_L.abs())).sum())
    # Original v1 thresholds are recorded unchanged; they are not a v2 admission gate.
    require(torch.allclose(actual_L.flatten(1).gather(1,old['point_indices']),old['native_logits'],
            rtol=common.NUMERIC['native_L_rtol'],atol=common.NUMERIC['native_L_atol']), 'New natural100/old sampled L practical relation differs')
    # Exact alias checks do not use practical replay tolerances.
    all_indices=torch.cat((log_idx,geo_idx),dim=1).reshape(-1); order=torch.argsort(all_indices)
    all_F=torch.cat((packet['arms']['LOG_PR']['F_samples'],packet['arms']['GEO_PR']['F_samples']),dim=1).reshape(-1,256)[order]
    equal=all_indices[order][1:]==all_indices[order][:-1]
    require(torch.equal(all_F[1:][equal],all_F[:-1][equal]), 'Same image/pixel exact native F differs across arms/GT/duplicates')
    require(all(packet[k] is True for k in ('replacement','duplicates_preserved','actual_GPU_L_used_for_selection',
        'native_natural100_rendered_before_matching_query_selection')) and
        packet['GT_selected_regions'] is False and packet['JPEG_patch_used_by_PR'] is False and
        packet['CPU_approximate_L_substituted_for_selection'] is False and row['old_query_matching_exact'] is True and
        row['one_teacher_forward'] is True and row['external_RNG_restored'] is True,
        'Cache scientific scope flags differ')
    return dict(dataset_index=index,image_id=image_id,raw_GT=G,K=Ks,
                exact_regions_and_private_draws=True,hard_GT_and_region_flags=True,
                practical_sampled_F_max_abs_error=feature_errors,
                practical_native_L_max_abs_error=float(native_L_error.max()),
                native_L_CPU_full_allclose_passed=native_L_allclose,native_L_CPU_full_allclose_failed_pixels=native_L_failed_pixels,
                native_L_CPU_full_allclose_is_qualification_gate=False,
                CPU_approximate_L_substituted_for_selection=False,validation_carrier_loaded_only_for_qualification=True)


def run(spec_path):
    started=time.monotonic(); path=Path(spec_path)
    require(path.is_absolute(), 'Absolute ROOT CPU SPEC required'); path=path.resolve(strict=True)
    spec=read(path); original_common,adapter,raws,closure=pre_gate(spec,39)
    gate=adapter.source_gate(path,spec,__file__,raws,closure,original_common)
    common=adapter.bind_common(original_common); gate['common']=common
    target=Path(spec['Q_path']); require(target.is_absolute() and target.parent.is_dir() and not target.exists(), 'Fresh absolute CPU Q required')
    import psutil
    require(psutil.Process().cpu_affinity()==[0,1], 'Actual CPU01 required before numerical imports')
    import torch
    require(torch.__version__=='2.0.1+cu118' and not torch.cuda.is_initialized(), 'Original Torch/cold CPU leaf required')
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    import numpy as np
    import cv2
    from pycocotools.coco import COCO
    from pycocotools import mask as masks
    require(np.__version__=='1.23.5' and cv2.__version__=='4.8.0', 'Original NumPy/OpenCV required')
    initial_RNG=torch.get_rng_state().clone()
    common.imported_files(torch,np,cv2,masks,COCO,closure,gate['admission'])
    coco=COCO(str(common.bound(spec['FIT']))); ids=list(coco.imgs)
    require(len(ids)==len(set(ids))==741 and len(coco.anns)==1661 and set(coco.cats)=={1}, 'Independent full FIT741/1661 required')
    report,gpu=gate['report'],gate['gpu']; checks=[]; total_GT=0
    for index,(image_id,row,ref,carrier_ref) in enumerate(zip(ids,report['rows'],report['packet_refs'],report['validation_carrier_refs'])):
        require(time.monotonic()-started<spec['deadline_seconds'], 'Prelocked N13 CPU cache deadline exceeded')
        require(row['packet']==ref and row['validation_carrier']==carrier_ref and
                Path(ref['path'])==Path(gpu['output_directory'])/('PACKET_%04d.pth'%index) and
                Path(carrier_ref['path'])==Path(gpu['output_directory'])/'VALIDATION_CARRIERS'/('CARRIER_%04d.pth'%index),
                'Actual packet/sidecar path and canonical order differ')
        packet=torch.load(str(common.bound(ref)),map_location='cpu'); carrier=torch.load(str(common.bound(carrier_ref)),map_location='cpu')
        require(packet['validation_carrier']==carrier_ref, 'Packet sidecar binding differs')
        checked=packet_check(torch,np,cv2,masks,coco,packet,carrier,row,index,image_id,spec,gate)
        checks.append(checked); total_GT+=checked['raw_GT']; del packet,carrier
        print(json.dumps(dict(N13_CPU_cache_images_checked=index+1,raw_GT=total_GT)),flush=True)
    require(total_GT==1661 and torch.equal(initial_RNG,torch.get_rng_state()) and not torch.cuda.is_initialized(),
            'Full cache population/CPU RNG/CUDA scope differs')
    libraries=common.imported_files(torch,np,cv2,masks,COCO,closure,gate['admission'])
    for ref in spec['sources']: common.bound(ref)
    require(common.reference(path)==gate['spec_ref'] and common.reference(__file__)==gate['source'] and
            common.reference(spec['CPU_admission_file'])==gate['admission_ref'] and psutil.Process().cpu_affinity()==[0,1] and
            torch.get_num_threads()==torch.get_num_interop_threads()==2 and time.monotonic()-started<spec['deadline_seconds'],
            'Final source/SPEC/admission/resources/deadline changed')
    q=dict(schema=common.Q_SCHEMA,status=common.Q_STATUS,phase=spec['phase'],source=spec['cache_entry_source'],
        reader_source=gate['source'],spec=gate['spec_ref'],CPU_admission=gate['admission_ref'],
        GPU_spec=spec['cache_spec'],GPU_report=spec['cache_report'],GPU_RAW=spec['cache_GPU_RAW'],GPU_admission=spec['cache_GPU_admission'],
        RAW_inputs=spec['RAW_inputs'],prior_failed_CPU_receipt=spec['prior_failed_CPU_receipt'],
        additional_failed_RAW_inputs=spec['additional_failed_RAW_inputs'],terminal_identity_pins=spec['terminal_identity_pins'],
        FIT=spec['FIT'],image_root=spec['image_root'],teacher_checkpoint=report['teacher_checkpoint'],
        prior_N12_CPU_qualification=spec['prior_N12_CPU_qualification'],original_cache_CPU_qualification=spec['original_cache_CPU_qualification'],
        packet_refs=report['packet_refs'],validation_carrier_refs=report['validation_carrier_refs'],
        recipe=common.RECIPE,regions_recipe=common.REGIONS_RECIPE,numeric_protocol=common.NUMERIC,
        validation_protocol=adapter.VALIDATION_PROTOCOL,region_common_source=spec['region_common_source'],
        cache_contract_source=spec['cache_contract_source'],
        **{k:spec[k] for k in adapter.RELATION_REF_KEYS},
        images=741,raw_GT=1661,point_count_per_arm=1700864,arms=list(common.ARMS),per_image_checks=checks,
        canonical_GPU_M_to_L_exact_bytes_ROOT_qualified=True,saved_GPU_L_geometry_and_sampling_CPU_exact_replayed=True,
        hard_targets_CPU_checked=True,
        F_e_L_shape_finiteness_digest_and_duplicate_consistency_CPU_checked=True,sampled_F_original_two_stage_CPU_replayed=True,
        native_L_CPU_full_allclose_passed=all(row['native_L_CPU_full_allclose_passed'] for row in checks),
        native_L_CPU_full_allclose_failed_images=sum(not row['native_L_CPU_full_allclose_passed'] for row in checks),
        native_L_CPU_full_allclose_is_qualification_gate=False,
        exact_GPU_L_P_RLE_K_used=True,CPU_approximate_L_substituted_for_selection=False,
        original_matching_query_GT_identity_CPU_checked=True,native_matching_replayed=False,processed_GT_transform_replayed=False,
        native610_unchanged_SOURCE_observed=True,native610_unchanged_independent_CPU_verified=False,
        validation_carriers_separate_from_training_packets=True,heads_loaded=False,heads_trained=False,head_epochs=0,
        optimizer_created=False,weights_updated=False,backward_calls=0,no_DEV=True,no_TEST=True,no_AP=True,
        actual_efficacy_established=False,innovation_established=False,certified_GPU_error_bound=False,
        current_owned_CPU_terminal_self_certified=False,whole_leaf_VRAM_certified=False,automatic_retry=False,
        CPU_affinity=[0,1],threads=2,CUDA_initialized=False,actual_imported_files=libraries,
        elapsed_seconds=time.monotonic()-started,deadline_seconds=spec['deadline_seconds'],
        evidence_limits=['The original full native L CPU allclose thresholds remain diagnostic and can fail; no full1e-4 pass is asserted.',
            'A separate already-ROOT-qualified canonical100 GPU matched-M-to-L exact-byte replay is required; dummy unmatched slots do not recover the discarded full100 M.',
            'Sampled F CPU1e-4 remains a strong gate; saved GPU L is the sole selector and training logit input.',
            'Training matching is reused and identities checked; the CPU does not re-run native assignment/model forward or all610 weight equality.',
            'Region and supervision sampling are changed together; this cache alone proves no efficacy or innovation.',
            'ROOT must independently close the current actual CPU terminal before cache admission to training.'])
    common.write(target,q); print(json.dumps(dict(status=common.Q_STATUS,qualification=common.reference(target))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--spec',required=True)
    parser.add_argument('--owner-execute',action='store_true'); args=parser.parse_args()
    require(args.owner_execute,'Explicit ROOT-owned CPU cache qualification required'); run(args.spec)
