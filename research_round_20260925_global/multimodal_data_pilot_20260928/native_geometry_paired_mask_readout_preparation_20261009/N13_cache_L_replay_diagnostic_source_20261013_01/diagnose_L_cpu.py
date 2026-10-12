"""Independent UNBOUND CPU diagnostic of fixed N13 cache image3 L replay.
No cache Q is produced. Original FP32 rtol/atol1e-4 remains decisive: even after
the diagnostic report is saved, a failure causes this owned leaf to exit1.
Authoring/import performs no scientific imports, tensor work or file access.
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

OWNER_SHA='ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
SCHEMA='ROOT-N13-cache-L-replay-CPU-diagnostic-spec-v1'
REPORT_SCHEMA='N13-cache-L-replay-CPU-diagnostic-report-v1'
RAW_KEYS=('receipt','phase','admission','phase_binding','bootstrap')
REF_KEYS=('entry_source','cache_GPU_entry_source','cache_GPU_spec','cache_GPU_report',
 'cache_GPU_RAW','cache_GPU_admission','failed_CPU_source','failed_CPU_spec',
 'failed_CPU_RAW','failed_CPU_admission','packet','carrier')
RECIPE=dict(dataset_index=3,normal_RAW_count=38,failed_RAW_count=5,
 variants=['CPU_FP32_original_two_stage','CPU_FP64_two_stage_cast_FP32'],
 rtol=1e-4,atol=1e-4,top_errors=10,quantile=0.99,
 preserve_original_decisive_FP32_check=True,fail_after_report_if_original_check_fails=True,
 output_cache_qualification=False,no_GT=True,no_model=True,no_CUDA=True)

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
        len(spec['additional_failed_RAW_inputs']) == 4, 'Five original failures required')
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


def source_gate(path,spec,raws,closure):
    require(spec['schema']==SCHEMA and spec['recipe']==RECIPE and len(raws)==38 and
            spec['deadline_seconds']==1800 and spec['allocator_limit_MiB']==4096 and spec['whole_GPU_limit_MiB']==8192 and
            all(spec[k] in closure.values() for k in REF_KEYS), 'Strict fixed-image diagnostic SPEC/closure required')
    source,spec_ref=reference(__file__),reference(path)
    require(source==spec['entry_source'], 'Actual diagnostic SOURCE differs')
    owners=[owner_join(row,raw,closure) for row,raw in zip(spec['RAW_inputs'],raws)]
    admission_path=Path(spec['CPU_admission_file']); require(admission_path.is_absolute(), 'Absolute own admission required')
    admission_ref,admission=reference(admission_path),read(admission_path)
    require(admission['status']=='ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and
            admission['driver_sha256']==OWNER_SHA and admission['phase']==spec['phase'] and
            admission['scientific_spec']==spec_ref and admission['entry']==source['path'] and
            admission['entry_args']==['--spec',str(path),'--owner-execute'] and admission['allowed_cpus']==[0,1] and
            admission['cuda_visible_devices']=='' and admission['automatic_retry'] is False and
            admission['whole_GPU_used_limit_MiB']==8192 and admission['qualified_leaf_deadline_seconds']==1800 and
            admission['bound_source_sha256'].get(str(path))==spec_ref['sha256'] and
            all(admission['bound_source_sha256'].get(ref['path'])==ref['sha256'] for ref in spec['sources']) and
            os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='2' for k in
                ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')),
            'Actual own diagnostic admission/environment differs')
    gpu,report,cpu=(read(bound(spec[k])) for k in ('cache_GPU_spec','cache_GPU_report','failed_CPU_spec'))
    require(gpu['schema']=='ROOT-N13-sameK-region-FIT-cache-GPU-spec-v1' and
            gpu['entry_source']==spec['cache_GPU_entry_source'] and gpu['RAW_inputs']==spec['RAW_inputs'][:37] and
            report['schema']=='N13-sameK-region-FIT-cache-report-v1' and
            report['status']=='N13_SAMEK_REGION_FIT_CACHE_GPU_COMPLETE_PENDING_CPU' and
            report['source']==spec['cache_GPU_entry_source'] and report['spec']==spec['cache_GPU_spec'] and
            report['GPU_admission']==spec['cache_GPU_admission'] and report['phase']==gpu['phase'] and
            report['images']==741 and report['raw_GT']==1661 and len(report['rows'])==len(report['packet_refs'])==741 and
            len(report['validation_carrier_refs'])==741 and report['cleanup']['allocated_after']==0 and
            cpu['schema']=='ROOT-N13-sameK-region-FIT-cache-CPU-spec-v1' and
            cpu['entry_source']==spec['failed_CPU_source'] and cpu['cache_spec']==spec['cache_GPU_spec'] and
            cpu['cache_report']==spec['cache_GPU_report'] and cpu['cache_GPU_RAW']==spec['cache_GPU_RAW'] and
            cpu['cache_GPU_admission']==spec['cache_GPU_admission'] and cpu['RAW_inputs']==spec['RAW_inputs'] and
            cpu['prior_failed_CPU_receipt']==spec['prior_failed_CPU_receipt'] and
            cpu['additional_failed_RAW_inputs']==spec['additional_failed_RAW_inputs'][:3] and
            cpu['numeric_protocol']['native_L_rtol']==cpu['numeric_protocol']['native_L_atol']==1e-4,
            'Exact original GPU/failed CPU report/SPEC lineage required')
    row,owner=spec['RAW_inputs'][37],owners[37]
    require(row['receipt']==spec['cache_GPU_RAW'] and row['admission']==spec['cache_GPU_admission'] and row['phase']==gpu['phase'] and
            owner['entry']==spec['cache_GPU_entry_source']['path'] and
            owner['entry_args']==['--spec',spec['cache_GPU_spec']['path'],'--owner-execute'] and
            owner['scientific_spec']==spec['cache_GPU_spec'], 'Actual original cache GPU terminal join differs')
    fail=spec['additional_failed_RAW_inputs'][-1]
    failed_raw=read(bound(spec['failed_CPU_RAW'])); failed_admission=read(bound(spec['failed_CPU_admission']))
    require(fail['receipt']==spec['failed_CPU_RAW'] and fail['phase']==cpu['phase'] and fail['returncode']==1 and
            failed_raw['admission_sha256']==spec['failed_CPU_admission']['sha256'] and
            failed_admission['phase']==cpu['phase'] and failed_admission['driver_sha256']==OWNER_SHA and
            failed_admission['entry']==spec['failed_CPU_source']['path'] and
            failed_admission['entry_args']==['--spec',spec['failed_CPU_spec']['path'],'--owner-execute'] and
            failed_admission['scientific_spec']==spec['failed_CPU_spec'] and
            failed_admission['bound_source_sha256'].get(spec['failed_CPU_source']['path'])==spec['failed_CPU_source']['sha256'] and
            failed_admission['bound_source_sha256'].get(spec['failed_CPU_spec']['path'])==spec['failed_CPU_spec']['sha256'],
            'Original actual CPU failure/source/SPEC/admission identity differs')
    selected=report['rows'][3]
    require(selected['dataset_index']==3 and selected['packet']==spec['packet']==report['packet_refs'][3] and
            selected['validation_carrier']==spec['carrier']==report['validation_carrier_refs'][3] and
            Path(spec['packet']['path'])==Path(gpu['output_directory'])/'PACKET_0003.pth' and
            Path(spec['carrier']['path'])==Path(gpu['output_directory'])/'VALIDATION_CARRIERS'/'CARRIER_0003.pth',
            'Fixed original index3 packet/carrier relation differs')
    return dict(source=source,spec_ref=spec_ref,admission_ref=admission_ref,admission=admission,row=selected,report=report)


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


def render(torch,value,identity):
    h,w=identity['actual_img_shape']; H,W=identity['original_shape']
    padded=torch.nn.functional.interpolate(value,size=(800,800),mode='bilinear',align_corners=False)
    crop=padded[0,:,:h,:w]
    return torch.nn.functional.interpolate(crop[:,None],size=(H,W),mode='bilinear',align_corners=False)[:,0]


def compare(torch,np,actual,expected,query_ids):
    require(actual.dtype==expected.dtype==torch.float32 and actual.shape==expected.shape and
            actual.device.type==expected.device.type=='cpu' and bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all()),
            'Finite unchanged FP32 same-raster comparison operands required')
    difference=(actual-expected).abs(); allowed=RECIPE['atol']+RECIPE['rtol']*expected.abs()
    violations=difference>allowed; ratio=difference/allowed
    a,e,d,t,r,v=(x.contiguous().numpy().reshape(-1) for x in (actual,expected,difference,allowed,ratio,violations))
    flat=np.arange(d.size,dtype=np.int64); ordered=np.lexsort((flat,-r))[:min(10,len(flat))]
    G,H,W=actual.shape; top=[]
    for pos in ordered.tolist():
        g,pixel=divmod(pos,H*W); y,x=divmod(pixel,W)
        top.append(dict(query_slot=g,query_id=query_ids[g],flat_pixel_index=pixel,y=y,x=x,
            actual=float(a[pos]),expected=float(e[pos]),absolute_error=float(d[pos]),allowed=float(t[pos]),
            error_to_allowance_ratio=float(r[pos]),violates_original_tolerance=bool(v[pos])))
    sign_flips=(actual>0)!=(expected>0); near_zero=expected.abs()<=allowed
    band_flips=(actual.abs()<=2)!=(expected.abs()<=2); near_band=(expected.abs()-2).abs()<=allowed
    passed=bool(torch.allclose(actual,expected,rtol=RECIPE['rtol'],atol=RECIPE['atol']))
    require(passed==(not bool(violations.any())), 'Allclose and original elementwise tolerance criterion disagree')
    return dict(original_allclose=passed,rtol=RECIPE['rtol'],atol=RECIPE['atol'],
        pixels=int(d.size),violations=int(violations.sum()),violations_per_query=violations.flatten(1).sum(1).tolist(),
        max_abs=float(d.max()),mean_abs=float(d.astype(np.float64).mean()),p99=float(np.quantile(d,0.99)),
        max_error_to_allowance_ratio=float(r.max()),top10_by_error_ratio=top,
        actual_min=float(actual.min()),actual_max=float(actual.max()),expected_min=float(expected.min()),expected_max=float(expected.max()),
        actual_shape=list(actual.shape),expected_shape=list(expected.shape),actual_stride=list(actual.stride()),expected_stride=list(expected.stride()),
        actual_contiguous=actual.is_contiguous(),expected_contiguous=expected.is_contiguous(),
        sign_flips=int(sign_flips.sum()),expected_near_zero_pixels=int(near_zero.sum()),
        sign_flips_outside_expected_practical_near_zero=int((sign_flips&~near_zero).sum()),
        band_flips=int(band_flips.sum()),expected_near_band_pixels=int(near_band.sum()),
        band_flips_outside_expected_practical_near_band=int((band_flips&~near_band).sum()))


def run(spec_path):
    started=time.monotonic(); path=Path(spec_path)
    require(path.is_absolute(),'Absolute ROOT diagnostic SPEC required'); path=path.resolve(strict=True)
    spec=read(path); raws,closure=pre_gate(spec,38); gate=source_gate(path,spec,raws,closure)
    output=Path(spec['output_directory']); require(output.is_absolute() and not output.exists(), 'Fresh absolute diagnostic output required')
    import psutil
    require(psutil.Process().cpu_affinity()==[0,1], 'Actual CPU01 required before numerical imports')
    import torch
    require(torch.__version__=='2.0.1+cu118' and not torch.cuda.is_initialized(), 'Original Torch/cold CUDA required')
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    import numpy as np
    require(np.__version__=='1.23.5','Original NumPy required')
    imported_files(torch,np,psutil,closure,gate['admission'])
    packet=torch.load(str(bound(spec['packet'])),map_location='cpu'); carrier=torch.load(str(bound(spec['carrier'])),map_location='cpu')
    row=gate['row']; report=gate['report']; identity=row['identity']; H,W=identity['original_shape']
    require(packet['schema']=='N13-sameK-region-FIT-cache-packet-v1' and
            carrier['schema']=='N13-sameK-region-FIT-validation-carrier-v1' and
            packet['dataset_index']==carrier['dataset_index']==row['dataset_index']==3 and
            packet['identity']==carrier['identity']==identity and
            packet['source']==carrier['source']==spec['cache_GPU_entry_source'] and
            packet['spec']==carrier['spec']==spec['cache_GPU_spec'] and
            packet['phase']==carrier['phase']==report['phase'] and
            packet['teacher_checkpoint']==carrier['teacher_checkpoint']==report['teacher_checkpoint'] and
            packet['validation_carrier']==spec['carrier'], 'Fixed actual packet/carrier lineage differs')
    qids=packet['query_ids']; M=carrier['matched_M200']; actual=packet['matched_native_L']; G=len(qids)
    require(1<=G<=100 and G==row['raw_GT'] and qids.device.type=='cpu' and qids.dtype==torch.int64 and qids.tolist()==row['query_ids'] and
            torch.equal(qids,carrier['query_ids']) and len(set(qids.tolist()))==G and all(0<=q<100 for q in qids.tolist()) and
            tuple(M.shape)==(1,G,200,200) and tuple(actual.shape)==(G,H,W) and
            M.dtype==actual.dtype==torch.float32 and M.device.type==actual.device.type=='cpu' and
            not M.requires_grad and not actual.requires_grad and bool(torch.isfinite(M).all()) and bool(torch.isfinite(actual).all()),
            'Exact saved FP32 M/L/query dimensions differ')
    require(digest_tensor(qids)==row['tensor_digests']['native']['query_ids']==row['carrier_digests']['query_ids'] and
            digest_tensor(M)==row['carrier_digests']['matched_M200'] and
            digest_tensor(actual)==row['tensor_digests']['native']['matched_native_L'], 'Exact native input tensor digests differ')
    original=render(torch,M,identity)
    fp64_cast=render(torch,M.to(torch.float64),identity).to(torch.float32)
    variants={'CPU_FP32_original_two_stage':compare(torch,np,actual,original,qids.tolist()),
              'CPU_FP64_two_stage_cast_FP32':compare(torch,np,actual,fp64_cast,qids.tolist())}
    libraries=imported_files(torch,np,psutil,closure,gate['admission'])
    for ref in spec['sources']: bound(ref)
    require(reference(path)==gate['spec_ref'] and reference(__file__)==gate['source'] and
            reference(spec['CPU_admission_file'])==gate['admission_ref'] and not torch.cuda.is_initialized() and
            psutil.Process().cpu_affinity()==[0,1] and torch.get_num_threads()==torch.get_num_interop_threads()==2 and
            time.monotonic()-started<1800, 'Final diagnostic SOURCE/SPEC/resources/deadline changed')
    output.mkdir(); original_pass=variants['CPU_FP32_original_two_stage']['original_allclose']
    result=dict(schema=REPORT_SCHEMA,
        status=('N13_CACHE_L_REPLAY_ORIGINAL_CHECK_PASSED_NOT_CACHE_QUALIFIED' if original_pass else
                'N13_CACHE_L_REPLAY_ORIGINAL_CHECK_FAILED_NOT_CACHE_QUALIFIED'),
        source=gate['source'],spec=gate['spec_ref'],CPU_admission=gate['admission_ref'],phase=spec['phase'],recipe=RECIPE,
        RAW_inputs=spec['RAW_inputs'],prior_failed_CPU_receipt=spec['prior_failed_CPU_receipt'],
        additional_failed_RAW_inputs=spec['additional_failed_RAW_inputs'],terminal_identity_pins=spec['terminal_identity_pins'],
        input_refs={k:spec[k] for k in REF_KEYS if k!='entry_source'},dataset_index=3,image_id=identity['image_id'],query_ids=qids.tolist(),
        actual_img_shape=identity['actual_img_shape'],original_shape=[H,W],M_shape=list(M.shape),M_stride=list(M.stride()),
        M_min=float(M.min()),M_max=float(M.max()),variants=variants,
        FP32_vs_FP64_expected_max_abs=float((original-fp64_cast).abs().max()),actual_imported_files=libraries,
        original_FP32_check_remains_decisive=True,terminal_exit1_if_original_check_fails=True,
        cache_qualified=False,Q_written=False,tolerances_loosened=False,GT_annotations_loaded=False,GT_labels_accessed=False,
        native_model_loaded=False,native_forward_calls=0,matching_calls=0,CUDA_initialized=False,automatic_retry=False,
        current_owned_CPU_terminal_self_certified=False,whole_leaf_VRAM_certified=False,CPU_affinity=[0,1],threads=2,
        elapsed_seconds=time.monotonic()-started,deadline_seconds=1800,
        evidence_limits=['Only fixed original index3 is diagnosed; no general cache qualification or mechanism efficacy is established.',
            'GPU saved L stays the reference; FP64 is a diagnostic variant and never replaces inputs or the decisive FP32 check.',
            'Point packet deserialization also contains unused supervision buffers; no GT file is opened or label field accessed.',
            'Two-stage equations are equal; actual finite-precision cause requires this numerical evidence, not static assumptions.'])
    write(output/'REPORT.json',result)
    print(json.dumps(dict(status=result['status'],report=reference(output/'REPORT.json'),original_allclose=original_pass)),flush=True)
    require(original_pass,'Prelocked original FP32 L replay still fails; diagnostic report is not cache qualification')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--spec',required=True)
    parser.add_argument('--owner-execute',action='store_true'); args=parser.parse_args()
    require(args.owner_execute,'Explicit ROOT-owned single diagnostic leaf required'); run(args.spec)

