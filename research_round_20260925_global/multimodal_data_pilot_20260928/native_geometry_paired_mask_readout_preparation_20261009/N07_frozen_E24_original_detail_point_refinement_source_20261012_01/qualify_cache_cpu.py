"""Independent N07 CPU cache/head qualification; not a native-model replay.

Only future ROOT-owned execution loads scientific inputs or libraries. The
cache contains sampled F/e/L, so their native provenance and native610 state
are SOURCE-observed only; no CPU backbone/full-field equality is certified.
"""
import argparse
import ast
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import types

SCHEMA = 'ROOT-N07-frozen-E24-original-detail-point-cache-CPU-qualification-spec-v1'
Q_SCHEMA = 'ROOT-N07-frozen-E24-original-detail-point-cache-CPU-qualification-v1'
STATUS = 'ROOT_N07_FROZEN_E24_ORIGINAL_DETAIL_POINT_CACHE_CPU_QUALIFIED'
GPU_SCHEMA = 'ROOT-N07-frozen-E24-original-detail-point-cache-spec-v1'
GPU_STATUS = 'N07_FROZEN_E24_FIT_POINT_CACHE_GPU_COMPLETE_PENDING_CPU'
PACKET_SCHEMA = 'N07-frozen-E24-original-detail-point-cache-packet-v1'
OWNER_SHA = 'ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
RAW_KEYS = ('receipt', 'phase', 'admission', 'phase_binding', 'bootstrap')
TENSOR_KEYS = ('query_ids', 'F_samples', 'query_embeddings', 'native_logits', 'patches', 'targets', 'point_indices', 'native_class_logits')
RECIPE = dict(images=741, raw_GT=1661, native_queries=100, points_per_GT=1024,
              band_points=768, uniform_points=256, band_abs_logit=2., patch_size=7,
              patch_padding='replicate', feature_channel_chunk=16, no_augmentation=True,
              train_epochs=8, train_seeds=[202610120701, 202610120702], arms=['C', 'R'],
              head_parameters=40257, head_parameter_tensors=6, AdamW_lr=3e-4, AdamW_weight_decay=1e-4)


def require(ok, message):
    if not ok: raise RuntimeError(message)

def read(path):
    def reject(value): raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=reject)

def reference(path):
    path = Path(path).resolve(strict=True); digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())

def ref_shape(ref):
    require(type(ref) is dict and set(ref) == {'path', 'bytes', 'sha256'} and
            type(ref['path']) is str and Path(ref['path']).is_absolute() and
            type(ref['bytes']) is int and ref['bytes'] >= 0 and type(ref['sha256']) is str and
            len(ref['sha256']) == 64 and set(ref['sha256']) <= set('0123456789abcdef'), 'Malformed reference')

def bound(ref):
    ref_shape(ref); require(reference(ref['path']) == ref, 'Pinned input changed: ' + ref['path'])
    return Path(ref['path'])

def terminal(ref, phase, exit_code=0):
    ref_shape(ref); path = Path(ref['path'])
    require(type(phase) is str and phase and path.name == 'PARENT_RECEIPT.json' and
            path.parent.name == phase, 'Absolute original receipt/phase locator required')
    raw = read(path)
    require(raw['status'] == 'ACTUAL_OWNED_TERMINAL' and type(raw['returncode']) is int and raw['returncode'] == exit_code and
            type(raw['original_identity']) is dict and raw['original_identity'] and raw['root_wait_reaped'] is True and
            raw['held_handle_exit'] == dict(wait_state='WAIT_OBJECT_0', actual_exit_code=exit_code) and
            type(raw['held_handle_exit']['actual_exit_code']) is int and raw['numeric_birth_absence']['original_birth_absent'] is True and
            raw.get('error') is None and raw.get('abort') is None and raw['wait_errors'] is False and
            raw['GPU_observation_errors'] == [], 'Actual held/reaped terminal%d required' % exit_code)
    if 'phase' in raw: require(type(raw['phase']) is str and raw['phase'] == phase, 'Original RAW phase differs')
    sha, peak = raw['admission_sha256'], raw['whole_GPU_observed_peak_MiB']
    require(type(sha) is str and len(sha) == 64 and set(sha) <= set('0123456789abcdef') and
            type(peak) in (int, float) and math.isfinite(peak) and 0 <= peak <= 8192 and
            type(raw['whole_GPU_used_limit_MiB']) is int and raw['whole_GPU_used_limit_MiB'] == 8192 and
            raw['whole_leaf_VRAM_certified'] is False and raw['GPU_scope'] ==
            'whole GPU observed, not per-process WDDM allocation', 'Original RAW resource scope differs')
    return raw

def owner_join(row, raw, closure):
    owner, pb = read(bound(row['admission'])), read(bound(row['phase_binding']))
    bootstrap = bound(row['bootstrap']); runtime = Path(row['receipt']['path']).parent
    require(Path(row['phase_binding']['path']) == runtime/'PHASE_BINDING.json' and bootstrap == runtime/'bootstrap.py' and
            owner['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and owner['phase'] == row['phase'] and
            owner['driver_sha256'] == OWNER_SHA and owner['automatic_retry'] is False and owner['allowed_cpus'] == [0, 1] and
            owner['cuda_visible_devices'] in ('', '0') and owner['whole_GPU_used_limit_MiB'] == 8192 and
            raw['admission_sha256'] == row['admission']['sha256'] == pb['admission_sha256'] and
            pb['bootstrap_sha256'] == row['bootstrap']['sha256'] and pb['argv'][-1] == str(bootstrap) and
            pb['CPU_affinity'] == owner['allowed_cpus'] and pb['CUDA_visible_devices'] == owner['cuda_visible_devices'] and
            pb['automatic_retry'] is False, 'Original owner/admission/PB/bootstrap differs')
    entry, args = owner['entry'], owner['entry_args']
    require(entry in closure and owner['bound_source_sha256'].get(entry) == closure[entry]['sha256'], 'Original entry absent/unbound')
    tree = ast.parse(bootstrap.read_text(encoding='utf-8'))
    argv = [ast.literal_eval(node.value) for node in ast.walk(tree) if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'sys' and target.attr == 'argv'
        for target in node.targets)]
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and
             isinstance(node.func.value, ast.Name) and node.func.value.id == 'runpy' and node.func.attr == 'run_path']
    require(argv == [[entry]+args] and len(calls) == 1 and ast.literal_eval(calls[0].args[0]) == entry and
            {item.arg: ast.literal_eval(item.value) for item in calls[0].keywords} == {'run_name': '__main__'}, 'Original bootstrap differs')
    if '--spec' in args:
        require(args.count('--spec') == 1, 'Ambiguous original SPEC')
        path = args[args.index('--spec')+1]
        require(path in closure and owner['bound_source_sha256'].get(path) == closure[path]['sha256'], 'Original SPEC absent/unbound')
    if 'scientific_spec' in owner:
        ref = owner['scientific_spec']
        require(ref in closure.values() and owner['bound_source_sha256'].get(ref['path']) == ref['sha256'], 'Scientific SPEC absent/unbound')
    return owner

def load(ref, name):
    path = bound(ref); source = path.read_bytes()
    require(len(source) == ref['bytes'] and hashlib.sha256(source).hexdigest() == ref['sha256'], 'SOURCE bytes changed')
    module = types.ModuleType(name); module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module

def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def source_gate(path, spec):
    require(spec['schema'] == SCHEMA and type(spec['phase']) is str and spec['phase'] and
            type(spec['RAW_inputs']) is list and len(spec['RAW_inputs']) == 18 and
            all(set(RAW_KEYS) <= set(row) for row in spec['RAW_inputs']) and
            len({row['receipt']['path'] for row in spec['RAW_inputs']}) == 18, 'Exactly18 normal original RAWs required')
    raws = [terminal(row['receipt'], row['phase']) for row in spec['RAW_inputs']]
    failed = terminal(spec['prior_failed_CPU_receipt'], Path(spec['prior_failed_CPU_receipt']['path']).parent.name, 1)
    require(spec['prior_failed_CPU_receipt']['path'] not in {row['receipt']['path'] for row in spec['RAW_inputs']}, 'Failed attempt cannot qualify an ancestor')
    # Every19 terminal state above precedes the first SHA/import/payload read.
    for ref in spec['sources']: ref_shape(ref)
    closure = {ref['path']: ref for ref in spec['sources']}
    require(len(closure) == len(spec['sources']), 'Unique complete CPU closure required')
    roles = ('entry_source','cache_entry_source','point_heads_source','fixture_source','cache_spec','cache_report',
             'cache_GPU_admission','cache_GPU_RAW','FIT','prior_failed_CPU_receipt')
    required = [spec[key] for key in roles]+[row[key] for row in spec['RAW_inputs'] for key in RAW_KEYS if key != 'phase']
    require(all(ref in closure.values() for ref in required), 'CPU inputs must be sources-bound')
    for ref in spec['sources']: bound(ref)
    source, spec_ref = reference(__file__), reference(path)
    require(source == spec['entry_source'], 'Actual CPU reader SOURCE differs')
    owners = [owner_join(row, raw, closure) for row, raw in zip(spec['RAW_inputs'], raws)]
    gpu = read(bound(spec['cache_spec'])); report = read(bound(spec['cache_report']))
    require(gpu['schema'] == GPU_SCHEMA and gpu['entry_source'] == spec['cache_entry_source'] and gpu['FIT'] == spec['FIT'] and
            gpu['image_root'] == spec['image_root'] and gpu['point_heads_source'] == spec['point_heads_source'] and gpu['recipe'] == RECIPE and
            gpu['prior_failed_CPU_receipt'] == spec['prior_failed_CPU_receipt'], 'Actual N07 cache recipe/lineage differs')
    require(all(ref in closure.values() for ref in gpu['sources']) and len(gpu['RAW_inputs']) == 17 and
            all(all(spec['RAW_inputs'][i][key] == row[key] for key in RAW_KEYS) for i, row in enumerate(gpu['RAW_inputs'])), 'Entire cache closure/17 ancestors must be preserved')
    require(gpu['context_GPU_spec'] in closure.values() and gpu['context_CPU_qualification'] in closure.values(), 'Original native context refs absent')
    ownrow, gpu_owner = spec['RAW_inputs'][-1], owners[-1]
    require(ownrow['receipt'] == spec['cache_GPU_RAW'] and ownrow['admission'] == spec['cache_GPU_admission'] and
            ownrow['phase'] == gpu['phase'] and Path(gpu['GPU_admission_file']) == Path(ownrow['admission']['path']) and
            gpu_owner['entry'] == spec['cache_entry_source']['path'] and
            gpu_owner['entry_args'] == ['--spec', spec['cache_spec']['path'], '--owner-execute'] and gpu_owner['scientific_spec'] == spec['cache_spec'] and
            all(gpu_owner['bound_source_sha256'].get(ref['path']) == ref['sha256'] for ref in gpu['sources']), 'Own cache GPU actual owner differs')
    require(Path(spec['cache_report']['path']) == Path(gpu['output_directory'])/'REPORT.json' and
            report['schema'] == 'N07-frozen-E24-point-cache-report-v1' and report['status'] == GPU_STATUS and
            report['source'] == spec['cache_entry_source'] and report['spec'] == spec['cache_spec'] and
            report['GPU_admission'] == spec['cache_GPU_admission'] and report['phase'] == gpu['phase'] and report['recipe'] == RECIPE and
            report['images'] == 741 and report['raw_GT'] == 1661 and report['point_count'] == 1700864 and
            report['teacher_forwards'] == 741 and report['one_teacher_forward_per_image'] is True and
            report['native610_unchanged_SOURCE_observed'] is True and report['native_gradient_tensors'] == 0 and
            report['optimizer_created'] is False and report['weights_updated'] is False and report['backward_calls'] == 0 and
            report['independent_CPU_cache_qualification'] is False and report['heads_trained'] is False and report['head_epochs'] == 0 and
            report['AP_computed'] is False and report['no_DEV'] is True and report['no_TEST'] is True and
            report['all_raw_GT_retained'] is True and report['cleanup']['allocated_after'] == 0, 'Cache actual report fields differ')
    require(len(report['rows']) == len(report['packet_refs']) == 741 and
            all(ref in closure.values() for ref in [report['start'], report['trace']]+report['packet_refs']), 'All cache artifacts must be current CPU inputs')
    require(Path(report['start']['path']) == Path(gpu['output_directory'])/'START.json' and
            Path(report['trace']['path']) == Path(gpu['output_directory'])/'TRACE.jsonl', 'Actual cache start/trace locator differs')
    admission_path = Path(spec['CPU_admission_file']); require(admission_path.is_absolute(), 'Absolute own CPU admission required')
    admit_ref = reference(admission_path); admission = read(admission_path)
    require(admission['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and admission['phase'] == spec['phase'] and
            admission['driver_sha256'] == OWNER_SHA and admission['entry'] == source['path'] and admission['scientific_spec'] == spec_ref and
            admission['entry_args'] == ['--spec',str(path),'--owner-execute'] and admission['allowed_cpus'] == [0,1] and
            admission['cuda_visible_devices'] == '' and admission['automatic_retry'] is False and admission['whole_GPU_used_limit_MiB'] == 8192 and
            type(admission['qualified_leaf_deadline_seconds']) is int and admission['qualified_leaf_deadline_seconds'] > 0 and
            admission['bound_source_sha256'].get(str(path)) == spec_ref['sha256'] and
            all(admission['bound_source_sha256'].get(ref['path']) == ref['sha256'] for ref in spec['sources']), 'Own CPU admission/full closure differs')
    return gpu, report, closure, source, spec_ref, admit_ref


def digest_tensor(value):
    array = value.detach().cpu().contiguous().numpy()
    return dict(shape=list(array.shape), dtype=str(array.dtype), sha256=hashlib.sha256(array.tobytes()).hexdigest())


def patch_reference(np, gray, indices):
    """Independent clip-coordinate implementation; no padding/unfold helper."""
    height, width = gray.shape; y, x = indices//width, indices % width
    return np.stack([gray[np.clip(y+dy,0,height-1),np.clip(x+dx,0,width-1)]
                     for dy in range(-3,4) for dx in range(-3,4)], axis=-1).astype(np.float32)/np.float32(255.)


def uniform_draws_check(torch, indices, band_counts, seed, area):
    """Replay uniform draws; full band membership/order is unavailable."""
    generator = torch.Generator(device='cpu'); generator.manual_seed(seed)
    for row, band_count in zip(indices, band_counts):
        require(type(band_count) is int and 0 <= band_count <= area, 'Source-observed full band count invalid')
        if band_count:
            torch.randint(band_count,(768,),generator=generator)  # Advance exact near-draw RNG count.
            expected = torch.randint(area,(256,),generator=generator)
            require(torch.equal(row[768:],expected), 'Independent uniform256 draw differs')
        else:
            expected = torch.randint(area,(1024,),generator=generator)
            require(torch.equal(row,expected), 'Empty-band independent uniform1024 draw differs')
    return len(band_counts)


def imported_files(torch, np, cv2, COCO, masks, closure, owner):
    modules = dict(torch=torch, torch_binary=torch._C, numpy=np, numpy_binary=np.core._multiarray_umath,
                   cv2=cv2, cv2_binary=cv2._native, COCO=sys.modules[COCO.__module__], masks=masks, mask_binary=masks._mask)
    refs = {}
    for key, module in modules.items():
        ref = reference(module.__file__)
        require(closure.get(ref['path']) == ref and owner['bound_source_sha256'].get(ref['path']) == ref['sha256'], 'Actual imported file absent/unbound: '+key)
        refs[key] = ref
    return refs


def packet_check(torch, np, cv2, coco, packet, row, index, image_id, gpu, spec, closure, teacher_checkpoint):
    identity = packet['identity']; annotations = coco.getAnnIds(imgIds=[image_id]); G = len(annotations)
    require(packet['schema'] == PACKET_SCHEMA and packet['dataset_index'] == identity['dataset_index'] == row['dataset_index'] == index and
            identity == row['identity'] and identity['image_id'] == row['image_id'] == image_id and
            packet['source'] == spec['cache_entry_source'] and packet['spec'] == spec['cache_spec'] and packet['phase'] == gpu['phase'], 'Packet lineage/image order differs')
    require(packet['teacher_checkpoint'] == teacher_checkpoint, 'Packet native0 E24 anchor ref differs')
    image = coco.imgs[image_id]; H,W = image['height'],image['width']
    require(1 <= G <= 100 and identity['ann_ids'] == row['raw_GT_ids'] == annotations and
            [a['annotation_id'] for a in identity['parser_rows'] if a['native_parser_retained']] == annotations and
            identity['original_shape'] == [H,W] and identity['batch_input_shape'] == [800,800] and
            len(identity['actual_img_shape']) == 2 and all(type(v) is int and 0 < v <= 800 for v in identity['actual_img_shape']) and
            identity['gray_decode'] == 'cv2.IMREAD_GRAYSCALE', 'Original raw annotation/raster identity differs')
    image_ref = identity['image']; expected_image = (Path(spec['image_root'])/image['file_name']).resolve(strict=True)
    require(image_ref in closure.values() and Path(image_ref['path']) == expected_image, 'Every actual original image must be input-bound')
    gray = cv2.imread(str(bound(image_ref)), cv2.IMREAD_GRAYSCALE)
    require(gray is not None and gray.dtype == np.uint8 and gray.shape == (H,W), 'Original uint8 gray decode differs')
    require(packet['original_JPEG_gray_digest'] == dict(shape=[H,W],dtype=str(gray.dtype),sha256=hashlib.sha256(gray.tobytes()).hexdigest()), 'Original JPEG decoder identity differs')
    shapes = dict(query_ids=(G,),F_samples=(G,1024,256),query_embeddings=(G,256),native_logits=(G,1024),
                  patches=(G,1024,49),targets=(G,1024),point_indices=(G,1024),native_class_logits=(1,100,2))
    for name, shape in shapes.items():
        value = packet[name]; dtype = torch.int64 if name in ('query_ids','point_indices') else (torch.bool if name=='targets' else torch.float32)
        require(isinstance(value,torch.Tensor) and value.device.type=='cpu' and value.dtype==dtype and tuple(value.shape)==shape and
                not value.requires_grad and (dtype in (torch.int64,torch.bool) or bool(torch.isfinite(value).all())), 'Detached CPU cache field differs: '+name)
    queries = packet['query_ids'].tolist(); indices=packet['point_indices']; logits=packet['native_logits']
    require(queries == row['query_ids'] and len(set(queries))==G and all(type(q) is int and 0<=q<100 for q in queries) and
            bool(((indices>=0)&(indices<H*W)).all()) and bool(((packet['patches']>=0)&(packet['patches']<=1)).all()), 'Query/point/intensity domains differ')
    raw = np.stack([np.zeros((H,W),np.uint8) if a.get('segmentation')==[] else coco.annToMask(a) for a in coco.loadAnns(annotations)])
    require(raw.shape==(G,H,W) and bool(((raw==0)|(raw==1)).all()), 'All original binary GT rasters required')
    raw_tensor=torch.from_numpy(raw.astype(np.bool_)); point_array=indices.numpy()
    sampled=np.take_along_axis(raw.astype(np.bool_).reshape(G,-1),point_array,axis=1)
    require(torch.equal(packet['targets'],torch.from_numpy(sampled)) and packet['raw_GT_digest']==digest_tensor(raw_tensor) and
            packet['raw_GT_area']==raw_tensor.flatten(1).sum(1).tolist(), 'Independent hard GT/area/digest differs')
    expected_patch=patch_reference(np,gray,point_array)
    require(torch.equal(packet['patches'],torch.from_numpy(expected_patch)), 'Independent actual JPEG49 patch arithmetic differs')
    require(packet['point_seed']==row['point_seed']==gpu['point_seed_base']+index and
            packet['matching_loss_seed']==row['matching_loss_seed']==gpu['matching_loss_seed_base']+index and
            packet['band_counts']==row['band_counts'] and len(packet['band_counts'])==G, 'Fixed private point/matching seed differs')
    uniform_draws_check(torch,indices,packet['band_counts'],packet['point_seed'],H*W)
    for values, band_count in zip(logits,packet['band_counts']):
        require(bool((values[:768].abs()<=2).all()) if band_count else bool((values.abs()>2).all()), 'Sampled native band eligibility differs')
    for query_indices, query_logits in zip(indices,logits):
        order=torch.argsort(query_indices); equal=query_indices[order][1:]==query_indices[order][:-1]
        require(torch.equal(query_logits[order][1:][equal],query_logits[order][:-1][equal]), 'Same query/pixel native L differs')
    flat=indices.flatten(); order=torch.argsort(flat); equal=flat[order][1:]==flat[order][:-1]
    features=packet['F_samples'].reshape(-1,256)[order]
    require(torch.equal(features[1:][equal],features[:-1][equal]), 'Same image/pixel coarse F differs across cache rows')
    stages=packet['native_stage_matches']; require(type(stages) is list and len(stages)==10, 'Ten actual native matching stage records required')
    expected_queries=[None]*G
    for stage in stages:
        qids,gtids=stage['positive_query_indices'],stage['assigned_GT_indices']
        require(stage['positives']==G==len(qids)==len(set(qids)) and all(type(q) is int and 0<=q<100 for q in qids) and
                all(type(g) is int for g in gtids) and sorted(gtids)==list(range(G)), 'Complete native matching metadata differs')
    for q,g in zip(stages[-1]['positive_query_indices'],stages[-1]['assigned_GT_indices']): expected_queries[g]=q
    require(queries==expected_queries and packet['actual_raw_processed_GT_correspondence']==dict(calls=1,correspondence_validator_passed=True) and
            type(packet['processed_target_proof']) is dict and bool(packet['processed_target_proof']), 'Source-observed native query/processed GT proof differs')
    losses=packet['native_loss_atoms']; require(type(losses) is dict and len(losses)==30 and all(type(v) in(int,float) and math.isfinite(v) for v in losses.values()), 'Source-observed native30 atom values invalid')
    for key in ('replacement','duplicates_preserved','features_are_coarse_native_bilinear','patch_is_original_uint8_JPEG_intensity'):
        require(packet[key] is True, 'Cache protocol differs: '+key)
    require(packet['GT_selected_points'] is False and packet['patch_has_SAR_complex_measurements'] is False and row['raw_GT']==G and
            row['points']==G*1024 and row['all_ten_matching_stages']==10 and row['native_loss_atoms']==30 and row['one_teacher_forward'] is True and
            row['backward_calls']==0 and row['optimizer_created'] is False and row['weights_updated'] is False and row['external_RNG_restored'] is True and
            row['sampled_foreground_by_GT']==packet['targets'].sum(1).tolist() and
            row['duplicate_points_by_GT']==[1024-int(torch.unique(v).numel()) for v in indices] and
            row['empty_band_instances']==sum(v==0 for v in packet['band_counts']) and
            row['cache_digests']=={name:digest_tensor(packet[name]) for name in TENSOR_KEYS}, 'Actual cache row/digests/protocol differ')
    return G


def run(spec_path):
    started=time.monotonic(); path=Path(spec_path); require(path.is_absolute(),'Absolute CPU SPEC required'); path=path.resolve(strict=True)
    spec=read(path); output=Path(spec['Q_path'])
    require(output.is_absolute() and output.parent.is_dir() and not output.exists() and not Path(str(output)+'.tmp').exists(),'Fresh absolute CPU Q path required')
    gpu,report,closure,source,spec_ref,admit_ref=source_gate(path,spec)
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(key)=='2' for key in
            ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),'CPU-only thread environment required')
    import psutil
    require(psutil.Process().cpu_affinity()==[0,1],'CPU01 before scientific imports required')
    import torch
    require(torch.__version__=='2.0.1+cu118' and not torch.cuda.is_initialized(),'Original Torch/cold CUDA required')
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    import numpy as np
    import cv2
    from pycocotools.coco import COCO
    from pycocotools import mask as masks
    require(np.__version__=='1.23.5','Original NumPy1.23.5 required')
    owner=read(bound(admit_ref)); libraries=imported_files(torch,np,cv2,COCO,masks,closure,owner)
    cache=load(spec['cache_entry_source'],'_N07_bound_cache_source'); heads=load(spec['point_heads_source'],'_N07_bound_point_heads')
    fixture=load(spec['fixture_source'],'_N07_bound_synthetic_fixture')
    self_module=types.SimpleNamespace(patch_reference=patch_reference,uniform_draws_check=uniform_draws_check)
    external=torch.get_rng_state().clone(); synthetic=fixture.run_fixture(torch,np,cache,heads,self_module)
    require(synthetic['status']=='N07_POINT_HEAD_AND_CACHE_COMPONENTS_SYNTHETIC_CPU_PASS' and all(synthetic['checks'].values()),'Actual synthetic head/cache fixture failed')
    start=read(bound(report['start']))
    require(start['schema']=='N07-point-cache-start-v1' and start['source']==spec['cache_entry_source'] and start['spec']==spec['cache_spec'] and
            start['GPU_admission']==spec['cache_GPU_admission'] and start['phase']==gpu['phase'] and start['recipe']==RECIPE and
            start['point_heads_source']==spec['point_heads_source'] and start['frozen_state_entries']==610 and start['frozen_parameter_tensors']==451 and
            start['native_weights_epoch']==24 and start['optimizer_created'] is False and start['weights_updated'] is False and
            start['all_GT_training_images_only'] is True and start['zero_GT_carrier_qualification_claimed'] is False and
            start['no_DEV'] is True and start['no_TEST'] is True and start['no_AP'] is True,'Actual cache START differs')
    with bound(report['trace']).open(encoding='utf-8') as stream:
        trace=[json.loads(line,parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value))) for line in stream if line.strip()]
    require(trace==report['rows'],'Actual741 cache trace/report rows differ')
    coco=COCO(str(bound(spec['FIT']))); image_ids=list(coco.imgs)
    require(len(image_ids)==len(set(image_ids))==741 and len(coco.anns)==1661 and set(coco.cats)=={1},'Full FIT741/1661 population required')
    context=read(bound(gpu['context_GPU_spec'])); teacher_checkpoint=context['checkpointmap']['native0']
    require(teacher_checkpoint in closure.values(),'Actual native0 E24 anchor must remain input-bound')
    total_GT=0
    for index,(image_id,row,ref) in enumerate(zip(image_ids,report['rows'],report['packet_refs'])):
        require(row['packet']==ref and Path(ref['path'])==Path(gpu['output_directory'])/('PACKET_%04d.pth'%index),'Actual cache packet path/order differs')
        packet=torch.load(str(bound(ref)),map_location='cpu')
        total_GT+=packet_check(torch,np,cv2,coco,packet,row,index,image_id,gpu,spec,closure,teacher_checkpoint)
        del packet
        print(json.dumps(dict(N07_CPU_cache_images_checked=index+1,raw_GT=total_GT)),flush=True)
    require(total_GT==1661 and torch.equal(external,torch.get_rng_state()) and not torch.cuda.is_initialized(),'Full cache/CPU RNG/CUDA scope differs')
    for ref in spec['sources']: bound(ref)
    require(reference(path)==spec_ref and reference(__file__)==source and reference(spec['CPU_admission_file'])==admit_ref and
            psutil.Process().cpu_affinity()==[0,1] and torch.get_num_threads()==torch.get_num_interop_threads()==2 and
            libraries==imported_files(torch,np,cv2,COCO,masks,closure,owner),'Final inputs/imports/resources changed')
    q=dict(schema=Q_SCHEMA,status=STATUS,phase=spec['phase'],source=spec['cache_entry_source'],reader_source=source,spec=spec_ref,
           cache_spec=spec['cache_spec'],cache_report=spec['cache_report'],cache_GPU_admission=spec['cache_GPU_admission'],cache_GPU_RAW=spec['cache_GPU_RAW'],
           CPU_admission=admit_ref,cache_phase=gpu['phase'],point_heads_source=spec['point_heads_source'],fixture_source=spec['fixture_source'],
           FIT=spec['FIT'],image_root=spec['image_root'],RAW_inputs=spec['RAW_inputs'],prior_failed_CPU_receipt=spec['prior_failed_CPU_receipt'],
           packet_refs=report['packet_refs'],images=741,raw_GT=1661,point_count=1700864,recipe=RECIPE,
           hard_targets_CPU_checked=True,indices_JPEG_patches_CPU_checked=True,point_head_synthetic_CPU_qualified=True,synthetic_fixture=synthetic,
           F_e_L_shape_finiteness_digest_and_duplicate_consistency_CPU_checked=True,F_e_L_backbone_CPU_replayed=False,
           full_native_L_band_population_and_near_draws_CPU_replayed=False,uniform_draws_CPU_replayed=True,
           native610_unchanged_SOURCE_observed=True,native610_unchanged_independent_CPU_verified=False,
           native_matching_replayed=False,processed_GT_transform_replayed=False,heads_trained=False,head_epochs=0,
           optimizer_created=False,no_DEV=True,no_TEST=True,no_AP=True,CPU_execution=dict(affinity=[0,1],threads=2,CUDA_VISIBLE_DEVICES='',CUDA_initialized=False),
           actual_imported_files=libraries,elapsed_seconds=time.monotonic()-started,
           evidence_limits=['Sampled F/e/L alone cannot independently reconstruct full native fields, model matching, or all610 state equality.',
               'Near-point eligibility and private uniform draws are checked; full band population and near-index membership/order remain GPU SOURCE-observed.',
               'JPEG patches and sampled hard raw GT are independently decoded for all741 images; no GT-based point selection is introduced.',
               'Synthetic CPU head math is not C/R training, optimizer qualification, all100 DEV evaluation, or a measured deployment resource result.',
               'ROOT must separately observe/close this own CPU tool terminal; no final CPU RAW is self-claimed.'])
    write(output,q); print(json.dumps(dict(status=STATUS,qualification=reference(output))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--spec',required=True)
    parser.add_argument('--owner-execute',action='store_true'); args=parser.parse_args()
    require(args.owner_execute,'Explicit ROOT-owned CPU execution required'); run(args.spec)
