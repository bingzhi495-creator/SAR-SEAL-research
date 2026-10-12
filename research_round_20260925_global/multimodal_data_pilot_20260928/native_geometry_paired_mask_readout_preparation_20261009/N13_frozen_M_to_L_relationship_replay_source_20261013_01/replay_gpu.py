"""Independent frozen matched M200-to-L GPU relationship replay, no cache Q.
Only ROOT may execute this finite SOURCE after all38 normal/all6 failed RAW gates.
Canonical100 unmatched slots are dummy zeros, not reconstructed original all100 M.
Authoring/import uses stdlib only; scientific imports occur strictly after gates.
"""
import argparse
import ast
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

OWNER_SHA='ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
SCHEMA='ROOT-N13-frozen-M-to-L-relationship-GPU-replay-spec-v1'
REPORT_SCHEMA='N13-frozen-M-to-L-relationship-GPU-replay-report-v1'
PASS_STATUS='N13_FROZEN_M_TO_L_CANONICAL100_EXACT_BYTES_REPLAY_COMPLETE_NO_CACHE_Q'
FAIL_STATUS='N13_FROZEN_M_TO_L_CANONICAL100_EXACT_BYTES_REPLAY_FAILED_NO_CACHE_Q'
RAW_KEYS=('receipt','phase','admission','phase_binding','bootstrap')
REF_KEYS=('entry_source','cache_GPU_entry_source','cache_GPU_spec','cache_GPU_report',
          'cache_GPU_RAW','cache_GPU_admission')
RECIPE=dict(images=741,matched_queries=1661,normal_RAW_count=38,failed_RAW_count=6,
 canonical_input_shape=[1,100,200,200],canonical_input_dtype='float32',
 canonical_input_layout='contiguous_NCHW',unmatched_slots='dummy_zeros_not_original_all100_M',
 insert_saved_matched_M_by_original_query_index=True,
 interpolation=dict(first_size=[800,800],mode='bilinear',align_corners=False,
                    crop='padded[0,:,:actual_h,:actual_w]',second_input='crop[:,None]',
                    second_size='original_shape',second_output='[:,0]'),
 decisive_check='all_saved_matched_native_L_FP32_bytes_equal_canonical100_matched_outputs',
 subset_G_replay='GPU_exact_bytes_diagnostic_only_not_fallback',
 fail_after_full_report_if_any_canonical_mismatch=True,
 output_cache_qualification=False,no_GT=True,no_model=True,no_teacher=True,no_AP=True,
 automatic_retry=False)

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


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


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
    return raws[:count], closure


def imported_files(torch,np,psutil,closure,admission):
    result={}
    for name,module in list(sys.modules.items()):
        if module is None or not any(name==p or name.startswith(p+'.') for p in ('torch','numpy','psutil')): continue
        file=getattr(module,'__file__',None)
        if file is None or not Path(file).is_file(): continue
        ref=reference(file)
        require(closure.get(ref['path'])==ref and admission['bound_source_sha256'].get(ref['path'])==ref['sha256'],
                'Actual diagnostic numerical module unbound: '+ref['path'])
        result[name]=ref
    for name,module in (('torch_C',torch._C),('numpy_binary',np.core._multiarray_umath)):
        ref=reference(module.__file__); require(closure.get(ref['path'])==ref and
            admission['bound_source_sha256'].get(ref['path'])==ref['sha256'], 'Actual diagnostic binary unbound')
        result[name]=ref
    return result


def digest_tensor(value):
    array=value.detach().cpu().contiguous().numpy()
    return dict(shape=list(array.shape),dtype=str(array.dtype),sha256=hashlib.sha256(array.tobytes()).hexdigest())


def source_gate(path,spec,raws,closure):
    require(spec['schema']==SCHEMA and spec['recipe']==RECIPE and len(raws)==38 and
            spec['normal_RAW_count']==38 and spec['deadline_seconds']==1800 and
            spec['allocator_limit_MiB']==4096 and spec['whole_GPU_limit_MiB']==8192 and
            all(spec[k] in closure.values() for k in REF_KEYS), 'Exact relationship replay SPEC/closure required')
    source,spec_ref=reference(__file__),reference(path)
    require(source==spec['entry_source'], 'Actual replay entry SOURCE differs')
    owners=[owner_join(row,raw,closure) for row,raw in zip(spec['RAW_inputs'],raws)]
    admission_path=Path(spec['GPU_admission_file']); require(admission_path.is_absolute(), 'Absolute own admission required')
    admission_ref,admission=reference(admission_path),read(admission_path)
    require(admission['status']=='ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and
            admission['driver_sha256']==OWNER_SHA and admission['phase']==spec['phase'] and
            admission['scientific_spec']==spec_ref and admission['entry']==source['path'] and
            admission['entry_args']==['--spec',str(path),'--owner-execute'] and admission['allowed_cpus']==[0,1] and
            admission['cuda_visible_devices']=='0' and admission['automatic_retry'] is False and
            admission['whole_GPU_used_limit_MiB']==8192 and admission['qualified_leaf_deadline_seconds']==1800 and
            admission['bound_source_sha256'].get(str(path))==spec_ref['sha256'] and
            all(admission['bound_source_sha256'].get(ref['path'])==ref['sha256'] for ref in spec['sources']) and
            os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and all(os.environ.get(k)=='2' for k in
                ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),
            'Actual replay admission/environment differs')
    gpu,report=(read(bound(spec[k])) for k in ('cache_GPU_spec','cache_GPU_report'))
    require(gpu['schema']=='ROOT-N13-sameK-region-FIT-cache-GPU-spec-v1' and
            gpu['entry_source']==spec['cache_GPU_entry_source'] and gpu['RAW_inputs']==spec['RAW_inputs'][:37] and
            gpu['prior_failed_CPU_receipt']==spec['prior_failed_CPU_receipt'] and
            gpu['additional_failed_RAW_inputs']==spec['additional_failed_RAW_inputs'][:3] and
            all(spec['terminal_identity_pins'].get(k)==v for k,v in gpu['terminal_identity_pins'].items()) and
            report['schema']=='N13-sameK-region-FIT-cache-report-v1' and
            report['status']=='N13_SAMEK_REGION_FIT_CACHE_GPU_COMPLETE_PENDING_CPU' and
            report['source']==spec['cache_GPU_entry_source'] and report['spec']==spec['cache_GPU_spec'] and
            report['GPU_admission']==spec['cache_GPU_admission'] and report['phase']==gpu['phase'] and
            report['images']==741 and report['raw_GT']==1661 and len(report['rows'])==len(report['packet_refs'])==741 and
            len(report['validation_carrier_refs'])==741 and report['cleanup']['allocated_after']==0,
            'Exact original cache GPU report/SPEC/RAW history required')
    original,owner=spec['RAW_inputs'][37],owners[37]
    require(original['receipt']==spec['cache_GPU_RAW'] and original['admission']==spec['cache_GPU_admission'] and
            original['phase']==gpu['phase'] and owner['entry']==spec['cache_GPU_entry_source']['path'] and
            owner['entry_args']==['--spec',spec['cache_GPU_spec']['path'],'--owner-execute'] and
            owner['scientific_spec']==spec['cache_GPU_spec'], 'Original cache GPU actual terminal join differs')
    paths=[]; total=0
    for index,row in enumerate(report['rows']):
        packet,carrier=report['packet_refs'][index],report['validation_carrier_refs'][index]
        require(row['dataset_index']==index and type(row['raw_GT']) is int and 1<=row['raw_GT']<=100 and
                row['packet']==packet and row['validation_carrier']==carrier and
                packet in closure.values() and carrier in closure.values() and
                Path(packet['path'])==Path(gpu['output_directory'])/('PACKET_%04d.pth'%index) and
                Path(carrier['path'])==Path(gpu['output_directory'])/'VALIDATION_CARRIERS'/('CARRIER_%04d.pth'%index),
                'Full canonical packet/carrier row and SOURCE closure required')
        paths.extend((packet['path'],carrier['path'])); total+=row['raw_GT']
    require(len(set(paths))==1482 and total==1661, 'Unique full741/1661 original carrier cohort required')
    return dict(source=source,spec_ref=spec_ref,admission_ref=admission_ref,admission=admission,report=report)


def render_original(torch,masks200,identity):
    """Independent renderer: preserve original crop[:,None] dispatch/layout."""
    ih,iw=identity['actual_img_shape']; oh,ow=identity['original_shape']
    padded=torch.nn.functional.interpolate(masks200,size=(800,800),mode='bilinear',align_corners=False)
    crop=padded[0,:,:ih,:iw]
    return torch.nn.functional.interpolate(crop[:,None],size=(oh,ow),mode='bilinear',align_corners=False)[:,0]


def saved_inputs(torch,packet,carrier,row,spec,report):
    identity=row['identity']; H,W=identity['original_shape']; ih,iw=identity['actual_img_shape']
    require(all(type(n) is int and n>0 for n in (H,W,ih,iw)) and ih<=800 and iw<=800,
            'Positive original/effective dimensions within padded800 required')
    require(packet['schema']=='N13-sameK-region-FIT-cache-packet-v1' and
            carrier['schema']=='N13-sameK-region-FIT-validation-carrier-v1' and
            packet['dataset_index']==carrier['dataset_index']==row['dataset_index'] and
            packet['identity']==carrier['identity']==identity and row['image_id']==identity['image_id'] and
            packet['source']==carrier['source']==spec['cache_GPU_entry_source'] and
            packet['spec']==carrier['spec']==spec['cache_GPU_spec'] and
            packet['phase']==carrier['phase']==report['phase'] and
            packet['teacher_checkpoint']==carrier['teacher_checkpoint']==report['teacher_checkpoint'] and
            packet['validation_carrier']==row['validation_carrier'], 'Saved packet/carrier exact lineage differs')
    qids=packet['query_ids']; M=carrier['matched_M200']; actual=packet['matched_native_L']; G=len(qids)
    require(1<=G<=100 and G==row['raw_GT'] and qids.device.type=='cpu' and qids.dtype==torch.int64 and
            tuple(qids.shape)==(G,) and qids.tolist()==row['query_ids'] and torch.equal(qids,carrier['query_ids']) and
            len(set(qids.tolist()))==G and all(0<=q<100 for q in qids.tolist()) and
            tuple(M.shape)==(1,G,200,200) and tuple(actual.shape)==(G,H,W) and
            M.dtype==actual.dtype==torch.float32 and M.device.type==actual.device.type=='cpu' and
            M.is_contiguous() and not M.requires_grad and not actual.requires_grad and
            bool(torch.isfinite(M).all()) and bool(torch.isfinite(actual).all()), 'Saved finite FP32 M/L/query dimensions differ')
    require(digest_tensor(qids)==row['tensor_digests']['native']['query_ids']==row['carrier_digests']['query_ids'] and
            digest_tensor(M)==row['carrier_digests']['matched_M200'] and
            digest_tensor(actual)==row['tensor_digests']['native']['matched_native_L'], 'Saved original tensor digests differ')
    return qids,M,actual


def exact_compare(torch,np,actual,expected):
    require(actual.dtype==expected.dtype==torch.float32 and actual.shape==expected.shape and
            actual.device.type==expected.device.type=='cpu' and bool(torch.isfinite(actual).all()) and
            bool(torch.isfinite(expected).all()), 'Exact finite FP32 same-raster operands required')
    # Byte comparison retains signed-zero and every FP32 bit; no allclose/tolerance.
    a=actual.contiguous().numpy(); e=expected.contiguous().numpy()
    bits_different=a.view(np.uint32)!=e.view(np.uint32)
    per_query=bits_different.reshape(len(a),-1).sum(axis=1,dtype=np.int64).tolist()
    equal=a.tobytes()==e.tobytes()
    require(equal==(sum(per_query)==0), 'Byte and FP32 word comparisons disagree')
    return dict(exact_bytes=equal,per_query_exact_bytes=[n==0 for n in per_query],
                per_query_mismatched_pixels=per_query,mismatched_pixels=sum(per_query),
                compared_bytes=int(a.nbytes),max_abs=float(np.abs(a-e).max()),
                saved_shape=list(actual.shape),replay_shape=list(expected.shape),
                saved_stride=list(actual.stride()),replay_stride=list(expected.stride()))


def replay_image(torch,np,spec,gate,index):
    report=gate['report']; row=report['rows'][index]
    packet=torch.load(str(bound(row['packet'])),map_location='cpu')
    carrier=torch.load(str(bound(row['validation_carrier'])),map_location='cpu')
    qids,M,actual=saved_inputs(torch,packet,carrier,row,spec,report)
    with torch.no_grad():
        query_gpu=qids.to('cuda:0'); matched=M.to('cuda:0')
        canonical=torch.zeros((1,100,200,200),dtype=torch.float32,device='cuda:0')
        require(canonical.is_contiguous() and canonical.stride()==(4000000,40000,200,1),
                'Prelocked canonical100 contiguous NCHW shape/stride differs')
        canonical.index_copy_(1,query_gpu,matched)
        selected_M=canonical.index_select(1,query_gpu).cpu()
        require(selected_M.contiguous().numpy().tobytes()==M.contiguous().numpy().tobytes(),
                'Exact original query-index insertion changed saved matched M bytes')
        # Unmatched slots remain zeros; no teacher/all100-M recovery is asserted.
        native100=render_original(torch,canonical,row['identity'])
        selected=native100.index_select(0,query_gpu).cpu()
        main=exact_compare(torch,np,actual,selected)
        del native100,canonical,selected,selected_M
        subset=render_original(torch,matched,row['identity']).cpu()
        diagnostic=exact_compare(torch,np,actual,subset)
        result=dict(dataset_index=index,image_id=row['image_id'],identity=row['identity'],query_ids=qids.tolist(),
            matched_queries=len(qids),packet=row['packet'],validation_carrier=row['validation_carrier'],
            input_tensor_digests=dict(query_ids=digest_tensor(qids),matched_M200=digest_tensor(M),matched_native_L=digest_tensor(actual)),
            saved_M_shape=list(M.shape),saved_M_stride=list(M.stride()),
            canonical_input_shape=[1,100,200,200],canonical_input_stride=[4000000,40000,200,1],
            original_full100_M_stride_archived=False,unmatched_slots_are_dummy=True,
            canonical100_exact_bytes=main['exact_bytes'],per_query_canonical100_exact_bytes=main['per_query_exact_bytes'],
            canonical100=main,subset_G=diagnostic,subset_G_is_diagnostic_only=True)
        del subset,matched,query_gpu
    del packet,carrier,qids,M,actual
    torch.cuda.synchronize(0)
    return result


def run(spec_path):
    started=time.monotonic(); path=Path(spec_path)
    require(path.is_absolute(),'Absolute ROOT replay SPEC required'); path=path.resolve(strict=True)
    spec=read(path); raws,closure=pre_gate(spec,38); gate=source_gate(path,spec,raws,closure)
    output=Path(spec['output_directory']); require(output.is_absolute() and not output.exists(), 'Fresh absolute replay output required')
    import psutil
    require(psutil.Process().cpu_affinity()==[0,1], 'Actual CPU01 required before numerical imports')
    import torch
    require(torch.__version__=='2.0.1+cu118' and not torch.cuda.is_initialized(), 'Original Torch/cold CUDA required')
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    import numpy as np
    require(np.__version__=='1.23.5', 'Original NumPy required')
    imported_files(torch,np,psutil,closure,gate['admission'])
    torch.cuda.init(); torch.cuda.set_per_process_memory_fraction(min(1.,4096*1024**2/torch.cuda.get_device_properties(0).total_memory),0)
    torch.cuda.reset_peak_memory_stats(0)
    output.mkdir()
    write(output/'START.json',dict(schema='N13-frozen-M-to-L-relationship-GPU-replay-start-v1',source=gate['source'],
        spec=gate['spec_ref'],GPU_admission=gate['admission_ref'],phase=spec['phase'],recipe=RECIPE,
        cache_GPU_report=spec['cache_GPU_report'],cache_GPU_RAW=spec['cache_GPU_RAW'],cache_qualified=False,Q_written=False))
    rows=[]
    with (output/'TRACE.jsonl').open('x',encoding='utf-8',newline='\n') as trace:
        for index in range(741):
            require(time.monotonic()-started<1800, 'Prelocked replay deadline exceeded; no retry/fallback')
            row=replay_image(torch,np,spec,gate,index); rows.append(row)
            trace.write(json.dumps(row,allow_nan=False)+'\n'); trace.flush()
            require(torch.cuda.max_memory_allocated(0)<=4096*1024**2, 'Replay allocated GPU limit exceeded')
            print(json.dumps(dict(replayed_images=index+1,canonical100_exact_bytes=row['canonical100_exact_bytes'])),flush=True)
    require(len(rows)==741 and sum(row['matched_queries'] for row in rows)==1661, 'Full replay741/1661 differs')
    peak=torch.cuda.max_memory_allocated(0); gc.collect(); torch.cuda.synchronize(0); torch.cuda.empty_cache()
    cleanup=dict(allocated_after=int(torch.cuda.memory_allocated(0)),reserved_after=int(torch.cuda.memory_reserved(0)))
    require(cleanup['allocated_after']==0 and peak<=4096*1024**2 and psutil.Process().cpu_affinity()==[0,1] and
            torch.get_num_threads()==torch.get_num_interop_threads()==2, 'Replay cleanup/resources differ')
    libraries=imported_files(torch,np,psutil,closure,gate['admission'])
    for ref in spec['sources']: bound(ref)
    require(reference(path)==gate['spec_ref'] and reference(__file__)==gate['source'] and
            reference(spec['GPU_admission_file'])==gate['admission_ref'] and time.monotonic()-started<1800,
            'Final replay SOURCE/SPEC/admission/deadline changed')
    failed_images=sum(not row['canonical100_exact_bytes'] for row in rows)
    failed_queries=sum(sum(not ok for ok in row['per_query_canonical100_exact_bytes']) for row in rows)
    failed_pixels=sum(row['canonical100']['mismatched_pixels'] for row in rows)
    passed=failed_images==failed_queries==failed_pixels==0
    result=dict(schema=REPORT_SCHEMA,status=PASS_STATUS if passed else FAIL_STATUS,
        source=gate['source'],spec=gate['spec_ref'],GPU_admission=gate['admission_ref'],phase=spec['phase'],recipe=RECIPE,
        RAW_inputs=spec['RAW_inputs'],prior_failed_CPU_receipt=spec['prior_failed_CPU_receipt'],
        additional_failed_RAW_inputs=spec['additional_failed_RAW_inputs'],terminal_identity_pins=spec['terminal_identity_pins'],
        cache_GPU_source=spec['cache_GPU_entry_source'],cache_GPU_spec=spec['cache_GPU_spec'],cache_GPU_report=spec['cache_GPU_report'],
        cache_GPU_RAW=spec['cache_GPU_RAW'],cache_GPU_admission=spec['cache_GPU_admission'],
        start=reference(output/'START.json'),trace=reference(output/'TRACE.jsonl'),rows=rows,
        images=741,matched_queries=1661,canonical100_all_exact_bytes=passed,
        canonical100_mismatched_images=failed_images,canonical100_mismatched_queries=failed_queries,
        canonical100_mismatched_pixels=failed_pixels,canonical100_compared_bytes=sum(row['canonical100']['compared_bytes'] for row in rows),
        subset_G_all_exact_bytes=all(row['subset_G']['exact_bytes'] for row in rows),
        subset_G_mismatched_images=sum(not row['subset_G']['exact_bytes'] for row in rows),
        subset_G_mismatched_queries=sum(sum(not ok for ok in row['subset_G']['per_query_exact_bytes']) for row in rows),
        subset_G_mismatched_pixels=sum(row['subset_G']['mismatched_pixels'] for row in rows),
        subset_G_is_diagnostic_only=True,actual_imported_files=libraries,cleanup=cleanup,torch_peak_allocated_bytes=int(peak),
        cache_qualified=False,Q_written=False,tolerance_used=False,GT_annotations_loaded=False,GT_labels_accessed=False,
        native_model_loaded=False,teacher_loaded=False,native_forward_calls=0,matching_calls=0,AP_computed=False,
        original_full100_M_reconstructed=False,original_full100_M_stride_archived=False,unmatched_slots_are_dummy=True,
        canonical_input_layout='contiguous_NCHW',current_owned_GPU_terminal_self_certified=False,
        whole_leaf_VRAM_certified=False,CPU_affinity=[0,1],threads=2,allocator_limit_MiB=4096,whole_GPU_limit_MiB=8192,
        automatic_retry=False,elapsed_seconds=time.monotonic()-started,deadline_seconds=1800,
        evidence_limits=['Only the frozen saved matched M200/L relation at the canonical contiguous100 dispatch is checked.',
            'Dummy unmatched slots are not original all100 M; original full-M stride was not archived and is not certified.',
            'Full exact bytes include signed zeros; the smallerG GPU replay is diagnostic and cannot replace the main check.',
            'Packet/carrier deserialization includes unused point supervision/F buffers; no GT file or label field is accessed.',
            'No backbone, matching, cache Q, efficacy, AP or innovation is established; ROOT held/reaped exit is a separate requirement.'])
    write(output/'REPORT.json',result)
    print(json.dumps(dict(status=result['status'],report=reference(output/'REPORT.json'),canonical100_all_exact_bytes=passed)),flush=True)
    require(passed,'Prelocked canonical100 FP32 exact-byte relationship replay failed; no cache qualification or fallback')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--spec',required=True)
    parser.add_argument('--owner-execute',action='store_true'); args=parser.parse_args()
    require(args.owner_execute,'Explicit ROOT-owned single GPU relationship leaf required'); run(args.spec)
