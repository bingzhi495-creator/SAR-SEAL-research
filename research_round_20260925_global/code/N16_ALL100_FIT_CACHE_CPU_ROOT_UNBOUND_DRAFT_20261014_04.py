"""UNBOUND_DRAFT: complete N16 CPU cache qualification wrapper; NOT SCIENCE-EXECUTED.
ROOT binds actual refs/ABI and terminal evidence; no automatic retry or self-certified exit.
"""
import argparse
import copy
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import types

OWNER_SHA="ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924"
RAW_KEYS=("receipt","phase","admission","phase_binding","bootstrap")
THREAD_ENV=("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS")

def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def api(module, name):
    value = getattr(module, name, None)
    require(callable(value), 'ROOT must adapt/review missing external ABI: ' + name)
    return value


def read(path):
    def reject(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=reject)


def ref_shape(ref):
    require(type(ref) is dict and set(ref) == {'path', 'bytes', 'sha256'} and
            type(ref['path']) is str and Path(ref['path']).is_absolute() and
            type(ref['bytes']) is int and ref['bytes'] >= 0 and type(ref['sha256']) is str and
            len(ref['sha256']) == 64 and set(ref['sha256']) <= set('0123456789abcdef'),
            'Malformed absolute pinned reference')


def reference(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def bound(ref):
    ref_shape(ref)
    require(reference(ref['path']) == ref, 'Pinned input changed: ' + ref['path'])
    return Path(ref['path'])


def closed(closure, ref):
    ref_shape(ref)
    require(closure.get(ref['path']) == ref, 'Missing exact complete SOURCE reference: ' + ref['path'])
    return bound(ref)


def load(ref, name):
    path = bound(ref)
    source = path.read_bytes()
    require(len(source) == ref['bytes'] and hashlib.sha256(source).hexdigest() == ref['sha256'], 'SOURCE changed before import')
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def terminal_raw_only(ref, phase, code):
    # Reads only original RAW JSON; no SOURCE hash/import or process probe.
    ref_shape(ref)
    path = Path(ref['path'])
    require(path.name == 'PARENT_RECEIPT.json' and type(phase) is str and phase and path.parent.name == phase,
            'Original RAW locator required')
    raw = read(path)
    require(raw['status'] == 'ACTUAL_OWNED_TERMINAL' and type(raw['returncode']) is int and raw['returncode'] == code and
            type(raw['original_identity']) is dict and raw['original_identity'] and raw['root_wait_reaped'] is True and
            raw['held_handle_exit'] == dict(wait_state='WAIT_OBJECT_0', actual_exit_code=code) and
            type(raw['held_handle_exit']['actual_exit_code']) is int and
            raw['numeric_birth_absence']['original_birth_absent'] is True and raw.get('error') is None and
            raw.get('abort') is None and raw['wait_errors'] is False and raw['GPU_observation_errors'] == [],
            'Actual held exit/reap/original birth absence required before any SHA/import')
    if 'phase' in raw:
        require(raw['phase'] == phase, 'Original RAW phase differs')
    sha, peak = raw['admission_sha256'], raw['whole_GPU_observed_peak_MiB']
    require(type(sha) is str and len(sha) == 64 and set(sha) <= set('0123456789abcdef') and
            type(peak) in (int, float) and math.isfinite(peak) and 0 <= peak <= 8192 and
            raw['whole_GPU_used_limit_MiB'] == 8192 and raw['whole_leaf_VRAM_certified'] is False and
            raw['GPU_scope'] == 'whole GPU observed, not per-process WDDM allocation',
            'Preserved RAW resource/admission scope differs')
    return raw


def check_pin(pin, raw, phase, code):
    require(type(pin) is dict and set(pin) == {'phase', 'returncode', 'original_identity', 'admission_sha256'} and
            pin['phase'] == phase and type(pin['returncode']) is int and pin['returncode'] == code and
            pin['original_identity'] == raw['original_identity'] and pin['admission_sha256'] == raw['admission_sha256'],
            'Original actual owner differs from its immutable identity/admission pin')


def ninth_failed_raw_only(spec):
    extra = spec['additional_preserved_independent_CPU_failure']
    require(type(extra) is dict and set(extra) == {'RAW_ref', 'phase', 'pin', 'returncode'} and
            type(extra['returncode']) is int and extra['returncode'] == 1,
            'Independent failed9 needs exactly RAW_ref/phase/pin/returncode1')
    raw = terminal_raw_only(extra['RAW_ref'], extra['phase'], 1)
    check_pin(extra['pin'], raw, extra['phase'], 1)
    expected = spec['normal_RAW_count']
    require(type(expected) is int and expected >= 97 and
            type(spec['total_failed_RAW_count']) is int and spec['total_failed_RAW_count'] == 12 and
            type(spec['total_terminal_identity_pin_count']) is int and
            spec['total_terminal_identity_pin_count'] == expected + 12 and
            extra['RAW_ref']['path'] not in spec['terminal_identity_pins'],
            'Failed9 is separate; total terminal pins must equal normal+9')
    return raw


def pre_gate(spec, minimum):
    # All normal and preserved failed RAW checks precede the first SOURCE SHA.
    # Separate failed9 has ALREADY passed its RAW-only identity check.
    rows = spec['RAW_inputs']
    expected = spec['normal_RAW_count']
    require(type(expected) is int and expected >= minimum and type(rows) is list and len(rows) == expected and
            len({row['receipt']['path'] for row in rows}) == expected, 'All original normal RAW ancestors required')
    normals = [terminal_raw_only(row['receipt'], row['phase'], 0) for row in rows]
    failed_ref = spec['prior_failed_CPU_receipt']
    failed_phase = Path(failed_ref['path']).parent.name
    failed = terminal_raw_only(failed_ref, failed_phase, 1)
    extra = spec['additional_failed_RAW_inputs']
    require(type(extra) is list and len(extra) == 7 and
            all(type(row) is dict and set(row) == {'receipt', 'phase', 'returncode'} and
                type(row['returncode']) is int and row['returncode'] == 1 for row in extra),
            'Do not alter the original extra failed7 list or preserved failed8')
    failures = [failed] + [terminal_raw_only(row['receipt'], row['phase'], 1) for row in extra]
    failed_paths = [failed_ref['path']] + [row['receipt']['path'] for row in extra]
    require(len(set(failed_paths)) == 8 and not {row['receipt']['path'] for row in rows}.intersection(failed_paths),
            'Normal owners and preserved failed8 owners must remain distinct')
    all_rows = [dict(row, returncode=0) for row in rows] + [dict(receipt=failed_ref, phase=failed_phase, returncode=1)] + extra
    pins = spec['terminal_identity_pins']
    require(type(pins) is dict and set(pins) == {row['receipt']['path'] for row in all_rows},
            'Original normal-plus-failed8 pins must remain unchanged')
    for row, raw in zip(all_rows, normals + failures):
        check_pin(pins[row['receipt']['path']], raw, row['phase'], row['returncode'])
    require(len(pins) == expected + 8 and
            spec['additional_preserved_independent_CPU_failure']['RAW_ref']['path'] not in
            {row['receipt']['path'] for row in all_rows},
            'Legacy normal+failed8 pins required; independent failed9 owner distinct')
    # FIRST HASH: failed9 and every normal/legacy failed8 RAW owner passed.
    closure = {ref['path']: ref for ref in spec['sources']}
    require(len(closure) == len(spec['sources']), 'Unique SOURCE closure required')
    for ref in spec['sources']:
        bound(ref)
    require(all(row[key] in closure.values() for row in rows for key in RAW_KEYS if key != 'phase') and
            failed_ref in closure.values() and all(row['receipt'] in closure.values() for row in extra),
            'Original owner metadata and failed8 receipts must be sources-bound')
    return normals, closure




def tenth_failed_raw_only(spec):
    extra=spec['additional_preserved_N16_cache_failure']
    require(extra['returncode']==1 and extra['RAW_ref']['path'] not in spec['terminal_identity_pins'] and
        extra['RAW_ref']['path']!=spec['additional_preserved_independent_CPU_failure']['RAW_ref']['path'],
        'Original N16 failed owner10 must remain separate from normal and old failure owners')
    raw=terminal_raw_only(extra['RAW_ref'],extra['phase'],1)
    check_pin(extra['pin'],raw,extra['phase'],1)
    return raw


def eleventh_failed_raw_only(spec):
    extra=spec['additional_preserved_N16_cache_target_dtype_failure']
    require(extra['returncode']==1 and extra['RAW_ref']['path'] not in spec['terminal_identity_pins'] and
        extra['RAW_ref']['path'] not in (spec['additional_preserved_independent_CPU_failure']['RAW_ref']['path'],
            spec['additional_preserved_N16_cache_failure']['RAW_ref']['path']),
        'Original N16 target-dtype failed owner11 must remain separate')
    raw=terminal_raw_only(extra['RAW_ref'],extra['phase'],1)
    check_pin(extra['pin'],raw,extra['phase'],1)
    return raw


ABORT = 'QUALIFIED_ORIGINAL_LEAF_RESOURCE_OR_DEADLINE'
GPU_SCOPE = 'whole GPU observed, not per-process WDDM allocation'
MIN_RAM_BYTES = 16 * 1024**3
GPU_LIMIT_MIB = 8192


class RawGateError(ValueError):
    pass


def helpercheck(raw, pin):
    """Check canonical owned controlled-failure structure; allow extra fields.

    pin.phase is a record-locator obligation of the ROOT wrapper: the canonical
    actual RAW has no phase field. No synthetic field may be added to obtain a
    pass. This function never opens a record, computes SHA, or reads Source.
    """
    def need(condition, field):
        if not condition:
            raise RawGateError('Controlled resource RAW gate failed: '+field)

    def field(obj,key):
        need(type(obj) is dict and key in obj,key+' missing')
        return obj[key]

    def finite_number(value):
        return type(value) is int or (type(value) is float and math.isfinite(value))

    def same_json(left,right):
        if type(left) is not type(right):
            return False
        if type(left) is dict:
            return (all(type(k) is str for k in left) and set(left)==set(right)
                    and all(same_json(left[k],right[k]) for k in left))
        if type(left) is list:
            return len(left)==len(right) and all(same_json(a,b) for a,b in zip(left,right))
        if type(left) is float:
            return math.isfinite(left) and math.isfinite(right) and left==right
        return type(left) in (str,int,bool,type(None)) and left==right

    need(type(raw) is dict and type(pin) is dict,'raw/pin dictionaries')
    need(field(raw,'status')=='ACTUAL_OWNED_TERMINAL','status')
    rc=field(raw,'returncode')
    need(type(rc) is int and rc==1,'returncode must be exact failure1')
    need(field(raw,'root_wait_reaped') is True,'root_wait_reaped')
    held=field(raw,'held_handle_exit')
    need(field(held,'wait_state')=='WAIT_OBJECT_0','held wait_state')
    actual_exit=field(held,'actual_exit_code')
    need(type(actual_exit) is int and actual_exit==1,'held actual_exit_code must be failure1')
    birth=field(raw,'numeric_birth_absence')
    need(field(birth,'original_birth_absent') is True,'original birth absence')
    need(field(raw,'error') is None,'error must be None')
    need(field(raw,'abort')==ABORT,'exact controlled resource/deadline abort')
    need(field(raw,'wait_errors') is False,'wait_errors must be False')
    gpu_errors=field(raw,'GPU_observation_errors')
    need(type(gpu_errors) is list and gpu_errors==[],'GPU_observation_errors must be empty list')
    identity,pinned_identity=field(raw,'original_identity'),field(pin,'original_identity')
    need(type(identity) is dict and bool(identity) and type(pinned_identity) is dict
         and bool(pinned_identity) and same_json(identity,pinned_identity),'original_identity pin exact')
    sha,pinned_sha=field(raw,'admission_sha256'),field(pin,'admission_sha256')
    need(type(sha) is str and len(sha)==64 and all(c in '0123456789abcdef' for c in sha)
         and type(pinned_sha) is str and sha==pinned_sha,'admission SHA pin exact; no SHA computed')
    if 'returncode' in pin:
        need(type(pin['returncode']) is int and pin['returncode']==1,'optional pin returncode must be failure1')
    if 'phase' in raw and 'phase' in pin:
        need(same_json(raw['phase'],pin['phase']),'optional RAW phase matches pin')
    minimum_ram=field(raw,'minimum_ram_bytes')
    need(finite_number(minimum_ram) and minimum_ram>=MIN_RAM_BYTES,'minimum_ram_bytes >=16GiB')
    peak=field(raw,'whole_GPU_observed_peak_MiB')
    need(finite_number(peak) and 0<=peak<=GPU_LIMIT_MIB,'whole GPU observed peak <=8192MiB')
    limit=field(raw,'whole_GPU_used_limit_MiB')
    need(type(limit) is int and limit==GPU_LIMIT_MIB,'whole GPU limit must remain8192MiB')
    need(field(raw,'whole_leaf_VRAM_certified') is False,'per-leaf VRAM certification must remain False')
    need(field(raw,'GPU_scope')==GPU_SCOPE,'whole GPU observation scope')
    elapsed=field(raw,'elapsed_seconds')
    need(finite_number(elapsed) and 1800<=elapsed<=1805,'controlled resource/deadline elapsed1800..1805')
    return dict(passed=True,normal_success=False,resource_or_deadline_failure=True,
                numeric_or_method_failure_proven=False,partial_caches_qualified=False,
                returncode=1,held_actual_exit_code=1,abort=ABORT,
                minimum_ram_bytes=minimum_ram,whole_GPU_observed_peak_MiB=peak,
                elapsed_seconds=elapsed,phase_locator_proof_required_from_ROOT_wrapper=True,
                record_IO_performed=False,Source_read_or_hash_performed=False)




def twelfth_resource_failure_raw_only(spec):
    extra=spec['additional_preserved_N16_controlled_resource_failure']
    ref_shape(extra['RAW_ref']);p=Path(extra['RAW_ref']['path'])
    require(p.name=='PARENT_RECEIPT.json' and p.parent.name==extra['phase']==extra['pin']['phase'] and
        extra['returncode']==1 and extra['RAW_ref']['path'] not in spec['terminal_identity_pins'] and
        extra['RAW_ref']['path'] not in (spec['additional_preserved_independent_CPU_failure']['RAW_ref']['path'],
            spec['additional_preserved_N16_cache_failure']['RAW_ref']['path'],
            spec['additional_preserved_N16_cache_target_dtype_failure']['RAW_ref']['path']),
        'Original resource-failed12 locator/pin must remain distinct')
    raw=read(p)
    decision=helpercheck(raw,extra['pin'])
    require(decision['passed'] and decision['normal_success'] is False and decision['partial_caches_qualified'] is False,
        'Controlled failure is preserved, never promoted to scientific cache success')
    return raw

CPU_SCHEMA='ROOT-N16-all100-FIT-action-cache-CPU-spec-v1'
GPU_REPORT_SCHEMA='N16-all100-FIT-action-cache-GPU-report-v1'
GPU_STATUS='N16_FULLFIT_ALL100_ACTION_CACHE_COMPLETE_PENDING_INDEPENDENT_CPU'
STATUS='N16_FULLFIT_ALL100_ACTION_CACHE_CPU_CHECKS_COMPLETE_PENDING_ROOT_ACTUAL0'
EXTERNAL_FIELDS=('identity','raw_target_mapping','native_stage_matches','raw_GT_digest',
                 'processed_target_proof','actual_raw_processed_GT_correspondence')


def same_tree(torch,a,b):
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.device.type==b.device.type=='cpu' and a.shape==b.shape and torch.equal(a,b)
    if type(a) is not type(b):
        return False
    if isinstance(a,dict):
        return a.keys()==b.keys() and all(same_tree(torch,a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):
        return len(a)==len(b) and all(same_tree(torch,x,y) for x,y in zip(a,b))
    return a==b


def write_new(path,value):
    with Path(path).open('x',encoding='utf8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')
    return reference(path)


def raw_first(spec):
    require(spec['schema']==CPU_SCHEMA and spec['expected_normal_RAW_count']==spec['normal_RAW_count'] and
            type(spec['normal_RAW_count']) is int and spec['normal_RAW_count']>=106 and
            spec['total_failed_RAW_count']==12 and spec['total_terminal_identity_pin_count']==spec['normal_RAW_count']+12,
            'Full original105 plus completedGPU normal owner and preserved failed12 required')
    ninth_failed_raw_only(spec)
    tenth_failed_raw_only(spec)
    eleventh_failed_raw_only(spec)
    twelfth_resource_failure_raw_only(spec)
    return pre_gate(spec,spec['expected_normal_RAW_count'])


def completed_GPU_gate(gpu,gate,spec):
    closure=gate['closure']
    refs=dict(source=spec['gate_GPU_source'],spec=spec['cache_GPU_spec'],report=spec['cache_GPU_report'],
              RAW=spec['cache_GPU_RAW'],admission=spec['cache_GPU_admission'])
    for ref in refs.values():
        closed(closure,ref)
    parent=read(closed(closure,refs['spec']))
    report=read(closed(closure,refs['report']))
    require(parent['schema']==gpu.SCHEMA and parent['entry_source']==refs['source'] and
            report['schema']==GPU_REPORT_SCHEMA and report['status']==GPU_STATUS and
            report['source']==refs['source'] and report['spec']==refs['spec'] and report['GPU_admission']==refs['admission'] and
            report['phase']==parent['phase'] and parent['RAW_inputs']==spec['RAW_inputs'][:-1] and
            parent['normal_RAW_count']==spec['normal_RAW_count']-1,
            'Exactly completed cache GPU/source/spec/report/admission and chronology required')
    require(Path(parent['GPU_admission_file']).resolve(strict=True)==Path(refs['admission']['path']).resolve(strict=True),
            'Completed GPU admission locator must name same actual bound file')
    require(all(closure.get(r['path'])==r for r in parent['sources']) and
            all(spec['terminal_identity_pins'].get(k)==v for k,v in parent['terminal_identity_pins'].items()),
            'Entire completed GPU Source/pins must be inherited unchanged')
    inherited=gpu.CARRIER_KEYS+('locked_protocol','cache_body_source','labels_source','FIT','original_fit_cache_report',
        'native_context_spec','region_common_source','original_fit_cache_entry_source','action_diagnostic_source',
        'opportunity_ROOT_result','opportunity_CPU_qualification','additional_preserved_independent_CPU_failure',
        'additional_preserved_N16_cache_failure','additional_preserved_N16_cache_target_dtype_failure',
        'C_engine_source','additional_preserved_N16_controlled_resource_failure')
    require(all(spec[k]==parent[k] for k in inherited),'Inherited scientific recipe/carrier/failure fields differ')
    index=gate['lineage'].owner_ref_join(gate['context'],gate['owners'],refs['RAW'],refs['admission'],
                                       refs['source'],refs['spec'],'0')
    require(index==spec['normal_RAW_count']-1 and spec['RAW_inputs'][index]['receipt']==refs['RAW'],
            'Latest actual GPU0 owner must occupy last normal inherited row')
    require(report['images']==741 and report['raw_GT']==1661 and report['natural_queries']==74100 and
            len(report['packet_refs'])==len(report['native_L_refs'])==len(report['rows'])==741 and
            report['fixed_PR_checkpoint']==gate['context']['endpoint']['checkpoints']['LOG_PR'] and
            report['original_native610_unchanged'] is True and report['original_PR8_unchanged'] is True and
            report['new_background_empty_set_supervision'] is True and report['native_negative_mask_supervision_is_not_claimed'] is True and
            report['one_teacher_forward_per_image'] is True and report['teacher_forwards']==741 and
            report['weights_updated'] is False and report['optimizer_created'] is False and report['backward_calls']==0 and
            report['scouts_trained'] is False and report['AP_evaluated'] is False and report['no_TEST'] is True and
            report['recipe']==gate['protocol'],'Full all100 cache terminal scientific scope differs')
    for ref in report['packet_refs']+report['native_L_refs']+[report['start'],report['trace']]:
        closed(closure,ref)
    start=read(closed(closure,report['start']))
    require(start['source']==refs['source'] and start['spec']==refs['spec'] and
            start['GPU_admission']==refs['admission'] and start['phase']==parent['phase'] and
            start['protocol']==gate['protocol'],'Completed GPU START binding differs')
    return dict(refs=refs,parent=parent,report=report,start=start)


def resources(spec,gate,psutil,torch):
    process=psutil.Process()
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and process.cpu_affinity()==[0,1] and
            all(os.environ.get(name)=='2' for name in THREAD_ENV) and not torch.cuda.is_initialized() and
            spec['deadline_seconds']==1800 and spec['automatic_retry'] is False,
            'ROOT CPUempty/CPU01/two-thread/no-retry/deadline resource contract differs')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    require(torch.get_num_threads()==torch.get_num_interop_threads()==2,'CPU2 runtime thread limits differ')
    return process


def imported_files(gate,spec,torch,np,cv2,psutil,masks,COCO,ndimage):
    common=load(spec['region_common_source'],'_N16_CPU_original_region_common')
    libraries=api(common,'imported_files')(torch,np,cv2,masks,COCO,gate['closure'],gate['admission'],ndimage)
    modules=dict(torch=torch,torch_binary=torch._C,numpy=np,numpy_binary=np.core._multiarray_umath,
                 cv2=cv2,psutil=psutil,psutil_platform=psutil._psplatform,psutil_binary=psutil._psplatform.cext)
    refs={key:reference(value.__file__) for key,value in modules.items()}
    for name,obj in dict(Tensor=torch.Tensor,Module=torch.nn.Module,Linear=torch.nn.Linear,
                         layer_norm=torch.nn.functional.layer_norm,torch_load=torch.load).items():
        source=inspect.getsourcefile(obj)
        if source is not None:
            refs[name]=reference(source)
    require(all(gate['closure'].get(r['path'])==r and
                gate['admission']['bound_source_sha256'].get(r['path'])==r['sha256'] for r in refs.values()),
            'Actual independent CPU libraries missing from complete admission/SOURCE')
    return dict(original_libraries=libraries,CPU_libraries=refs)



def cache_lineage_bridge(closure,spec,gate,evidence):
    """Bridge distinct qualified N13 region cache and original N07 FIT cache."""
    old_report=read(closed(closure,spec['original_fit_cache_report']))
    region_spec=read(closed(closure,gate['cache_report']['spec']))
    require(region_spec['original_cache_report']==spec['original_fit_cache_report'] and
            region_spec['original_cache_spec']==old_report['spec'] and
            region_spec['original_cache_entry_source']==old_report['source'],
            'Qualified N13-to-original-N07 report/spec/source bridge differs')
    closed(closure,old_report['source'])
    original_spec=read(closed(closure,old_report['spec']))
    context_ref=original_spec['context_GPU_spec']
    context=read(closed(closure,context_ref))
    teacher=context['checkpointmap']['native0']
    require(teacher==gate['cache_report']['teacher_checkpoint']==evidence['report']['teacher_checkpoint'],
            'N07 context/N13 qualified cache/completed N16 teacher checkpoint differs')
    return dict(old_report=old_report,original_spec=original_spec,context=context,
                qualified_region_cache_spec=region_spec,teacher_checkpoint=teacher)

def run(spec_path):
    started=time.monotonic()
    path=Path(spec_path).resolve(strict=True)
    require(Path(spec_path).is_absolute(),'Absolute ROOT CPU spec required')
    spec=read(path)
    raws,closure=raw_first(spec)  # ALL RAW-only checks precede first SHA and any external import.
    for key in ('entry_source','gate_GPU_source','qualifier_body_source','PR_formula_source'):
        closed(closure,spec[key])
    require(Path(spec['entry_source']['path']).resolve(strict=True)==Path(__file__).resolve(strict=True),
            'Independent CPU must name its own wrapper entry')
    gpu=load(spec['gate_GPU_source'],'_N16_CPU_completed_GPU_gate')
    require(gpu.CPU_SCHEMA==CPU_SCHEMA,'Closed GPU CPU schema ABI differs')
    gate=gpu.source_gate(path,spec,cpu=True)
    require(gate['closure']==closure and gate['source']==spec['entry_source'] and gate['spec_ref']==reference(path) and
            Path(gate['admission_ref']['path']).resolve(strict=True)==Path(spec['CPU_admission_file']).resolve(strict=True),
            'Current CPU own source/spec/admission differs')
    evidence=completed_GPU_gate(gpu,gate,spec)
    output=Path(spec['output_directory'])
    require(output.is_absolute() and output.parent.is_dir() and not output.exists(),'Fresh absolute CPU output required')
    require(all(str(output/name) not in closure for name in ('START.json','TRACE.jsonl','DETAILS.json','QUALIFICATION.json','FAILURE.json')),
            'New runtime CPU output cannot be an input SOURCE')
    recipe=spec['qualification_recipe']
    require(recipe==dict(point_chunk=2048,points_two_groups_each=2048,points_single_group=4096,num_classes=1,
        PR_atol=1e-4,PR_rtol=0,background_empty_set_extension=True,carrier_epoch=24,carrier_seed=202610120701,
        point_seed_base=202610140160000,matching_loss_seed_base=202610122001),
        'Original locked cache qualification recipe must be exact, no new allowance')
    import psutil
    import torch
    require(torch.__version__=='2.0.1+cu118','Original qualified Torch runtime required')
    process=resources(spec,gate,psutil,torch)
    import numpy as np,cv2
    from scipy import ndimage
    from pycocotools import mask as masks
    from mmdet.datasets.api_wrappers import COCO
    require(np.__version__=='1.23.5' and cv2.__version__=='4.8.0','Original numeric versions required')
    libraries=imported_files(gate,spec,torch,np,cv2,psutil,masks,COCO,ndimage)
    body=load(spec['qualifier_body_source'],'_N16_CPU_independent_qualifier_body')
    formula=load(spec['PR_formula_source'],'_N16_CPU_closed_original_N15_formulas')
    require(callable(getattr(body,'validate_packet',None)) and callable(getattr(formula,'independent_PR',None)),
            'ROOT must bind exact CPU validator and original independent PR callable')
    heads,states,proofs=gate['evaluation'].load_heads(torch,gate['context'],gate)
    head,state=heads['LOG_PR'],states['LOG_PR']
    require(len(head.state_dict())==8 and sum(p.numel() for p in head.parameters())==40497 and not head.training and
            proofs==evidence['start']['head_state_proofs'] and not torch.cuda.is_initialized(),
            'CPU must load exact qualified frozen strongest E24 head/proofs without CUDA')
    head.requires_grad_(False)
    anchor={k:v.detach().clone() for k,v in head.state_dict().items()}
    bridge=cache_lineage_bridge(closure,spec,gate,evidence)
    old_report=bridge['old_report']
    require(old_report['images']==741 and old_report['raw_GT']==1661 and
            len(old_report['packet_refs'])==len(old_report['rows'])==741,
            'Original qualified N07 FIT741/1661 cache report required')
    native_context=read(closed(closure,spec['native_context_spec']))
    require(native_context['FIT']==spec['FIT']==bridge['context']['FIT'] and
            native_context['checkpointmap']['native0']==bridge['teacher_checkpoint'] and
            all(closure.get(r['path'])==r for r in native_context['sources']),
            'Exact original FIT context/teacher/source proof differs')
    coco=COCO(str(closed(closure,spec['FIT'])))
    ids=list(coco.imgs)
    require(len(ids)==741 and len(coco.anns)==1661 and set(coco.cats)=={1},'Canonical FIT741/1661 only')
    output.mkdir()
    start_ref=write_new(output/'START.json',dict(schema='N16-all100-FIT-action-cache-CPU-start-v1',
        source=gate['source'],spec=gate['spec_ref'],CPU_admission=gate['admission_ref'],phase=spec['phase'],
        cache_GPU=evidence['refs'],qualifier_body_source=spec['qualifier_body_source'],PR_formula_source=spec['PR_formula_source'],
        locked_protocol=spec['locked_protocol'],qualification_recipe=recipe,libraries=libraries,head_state_proofs=proofs,
        CPU_actual_terminal_self_certified=False,original_teacher_CPU_loaded=False,no_AP=True,no_training=True))
    rows=[]
    points=total_GT=0
    maxima={}
    trace_path=output/'TRACE.jsonl'
    try:
        with trace_path.open('x',encoding='utf8',newline='\n') as trace:
            for index in range(741):
                require(time.monotonic()-started<1800,'Prelocked bounded CPU deadline; no retry')
                ref=evidence['report']['packet_refs'][index]
                L_ref=evidence['report']['native_L_refs'][index]
                gpurow=evidence['report']['rows'][index]
                require(gpurow['packet']==ref and gpurow['natural_native_L_carrier']==L_ref and
                        gpurow['dataset_index']==index and gpurow['image_id']==ids[index],
                        'GPU row packet/nativeL/index/identity refs differ')
                packet=torch.load(str(closed(closure,ref)),map_location='cpu')
                carrier=torch.load(str(closed(closure,L_ref)),map_location='cpu')
                for payload in (packet,carrier):
                    require(payload['source']==evidence['refs']['source'] and payload['spec']==evidence['refs']['spec'] and
                            payload['GPU_admission']==evidence['refs']['admission'] and payload['phase']==evidence['parent']['phase'] and
                            payload['dataset_index']==index and payload['identity']==gpurow['identity'],
                            'Real packet/nativeL source/spec/admission/phase/identity differs')
                require(carrier['schema']=='N16-actual-GPU-natural100-L-carrier-v1' and
                        packet['natural_native_L_carrier']==L_ref and 'natural_native_L' not in packet and
                        packet['protocol']==spec['locked_protocol'] and
                        packet['fixed_PR_checkpoint']==evidence['report']['fixed_PR_checkpoint'] and
                        packet['teacher_checkpoint']==evidence['report']['teacher_checkpoint'],
                        'Actual thin packet/carrier/head/teacher/protocol binding differs')
                packet['natural_native_L']=carrier['native_L']
                old_ref=old_report['packet_refs'][index]
                old_row=old_report['rows'][index]
                require(old_row['packet']==old_ref and packet['original_cache_packet']==old_ref,
                        'Original actual qualified cache ref differs')
                old=torch.load(str(closed(closure,old_ref)),map_location='cpu')
                require(old['source']==old_report['source'] and old['spec']==old_report['spec'] and
                        old['teacher_checkpoint']==bridge['teacher_checkpoint'] and
                        all(k in old and k in packet and same_tree(torch,old[k],packet[k]) for k in EXTERNAL_FIELDS),
                        'Original qualified identity/mapping/matches/GTdigest/processed target correspondence differs; ROOT must adapt explicit field ABI')
                info=coco.imgs[ids[index]]
                require(packet['identity']['image_id']==ids[index] and packet['identity']['original_shape']==[info['height'],info['width']],
                        'Original COCO image/shape differs')
                annids=packet['raw_GT_ids']
                require(annids==packet['identity']['ann_ids'] and len(set(annids))==len(annids) and
                        set(annids)==set(coco.getAnnIds(imgIds=[ids[index]])) and
                        all(coco.anns[a]['image_id']==ids[index] and coco.anns[a]['category_id']==1 for a in annids),
                        'Actual rawGT canonical annotation set differs')
                arrays=[coco.annToMask(coco.anns[a]).astype(np.bool_) for a in annids]
                rawGT=torch.from_numpy(np.stack(arrays) if arrays else np.empty((0,info['height'],info['width']),dtype=np.bool_))
                require(body.digest(rawGT)==packet['raw_GT_digest']==old['raw_GT_digest'],
                        'Actual COCO rawGT independent digest must match qualified old/new packets')
                row=body.validate_packet(torch,np,cv2,packet,recipe,state,
                    dict(original_head=head,independent_PR=formula.independent_PR),rawGT)
                require(row['dataset_index']==index and row['image_id']==ids[index] and row['points']==gpurow['points'] and
                        row['raw_GT']==gpurow['raw_GT'],'Independent CPU/GPU perimage point/GT identities differ')
                for key in ('natural_queries','active_queries','active_matched_queries','active_background_queries',
                    'C_counts_all100','K_counts_all100','utility_class_counts','error_class_counts',
                    'weighted_class_counts_by_stratum','importance_weight_min','importance_weight_max','importance_weight_sum',
                    'native_background_queries','native_background_max_IoU_ge_point5'):
                    require(row[key]==gpurow[key],'Actual independent CPU/GPU exact row summary differs: '+key)
                row.update(packet=ref,natural_native_L_carrier=L_ref,original_cache_packet=old_ref,
                           ROOT_external_proofs_checked=True,wrapper_checked_external_fields=list(EXTERNAL_FIELDS))
                rows.append(row)
                points+=row['points']
                total_GT+=row['raw_GT']
                for key,value in row['PR_errors'].items():
                    maxima[key]=max(maxima.get(key,0.),value)
                trace.write(json.dumps(row,allow_nan=False)+'\n')
                trace.flush()
                require(not torch.cuda.is_initialized() and process.cpu_affinity()==[0,1] and
                        torch.get_num_threads()==torch.get_num_interop_threads()==2,'CPU resource contract changed')
                del old,packet,carrier,rawGT,arrays
                gc.collect()
                print(json.dumps(dict(CPU_checked_images=index+1,sampled_points=points)),flush=True)
        require(total_GT==1661 and points==evidence['report']['sampled_points'] and
                all(torch.equal(anchor[k],v) for k,v in head.state_dict().items()) and
                not any(p.grad is not None or p.requires_grad for p in head.parameters()) and not torch.cuda.is_initialized(),
                'Complete FIT count/frozen original PR/coldCUDA must remain unchanged')
        for ref in spec['sources']:
            bound(ref)
        require(reference(path)==gate['spec_ref'] and reference(__file__)==gate['source'] and
                reference(spec['CPU_admission_file'])==gate['admission_ref'],'Current own source/spec/ad changed')
        details_ref=write_new(output/'DETAILS.json',dict(schema='N16-all100-FIT-action-cache-CPU-details-v1',
            source=gate['source'],spec=gate['spec_ref'],CPU_admission=gate['admission_ref'],phase=spec['phase'],
            cache_GPU=evidence['refs'],start=start_ref,trace=reference(trace_path),rows=rows,
            images=741,raw_GT=total_GT,natural_queries=74100,sampled_points=points,max_PR_errors=maxima,
            qualification_recipe=recipe,locked_protocol=spec['locked_protocol'],ROOT_external_proofs_checked=True,
            CPU_actual_terminal_self_certified=False,no_native_teacher_forward_replay=True,
            same_actual_GPU_sample_inputs=True,background_empty_set_is_new_shared_supervision=True))
        q=dict(schema='N16-all100-FIT-action-cache-CPU-qualification-v1',status=STATUS,
            source=gate['source'],spec=gate['spec_ref'],CPU_admission=gate['admission_ref'],phase=spec['phase'],
            cache_GPU_source=evidence['refs']['source'],cache_GPU_spec=evidence['refs']['spec'],
            cache_GPU_report=evidence['refs']['report'],cache_GPU_RAW=evidence['refs']['RAW'],cache_GPU_admission=evidence['refs']['admission'],
            details=details_ref,images=741,raw_GT=total_GT,natural_queries=74100,sampled_points=points,
            locked_protocol=spec['locked_protocol'],qualifier_body_source=spec['qualifier_body_source'],
            PR_formula_source=spec['PR_formula_source'],qualification_recipe=recipe,
            all100_C_K_private_sampling_importance_exact=True,actual_rawGT_targets_both_label_classes_exact=True,
            two_same_actual_GPU_input_PR_formulas_absolute1e4_strictsign=True,ROOT_external_proofs_checked=True,
            new_background_empty_set_supervision=True,original_native_negative_mask_supervision_not_claimed=True,
            numerical_semantic_checks_complete=True,qualified=False,ROOT_actual_CPU0_verified=False,
            checkpoint_or_cache_training_eligibility_pending_ROOT_actualCPU0=True,CPU_actual_terminal_self_certified=False,
            original_PR8_unchanged=True,teacher_CPU_loaded=False,optimizer_created=False,weights_updated=False,
            AP_evaluated=False,no_TEST=True,elapsed_seconds=time.monotonic()-started)
        qref=write_new(output/'QUALIFICATION.json',q)
        print(json.dumps(dict(status=STATUS,qualification=qref)),flush=True)
    except BaseException as error:
        write_new(output/'FAILURE.json',dict(schema='N16-all100-FIT-action-cache-CPU-failure-v1',
            source=gate['source'],spec=gate['spec_ref'],CPU_admission=gate['admission_ref'],phase=spec['phase'],
            checked_images=len(rows),elapsed_seconds=time.monotonic()-started,error=type(error).__name__+': '+str(error),
            qualified=False,no_retry=True,CPU_actual_terminal_self_certified=False))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--spec',required=True)
    parser.add_argument('--owner-execute',action='store_true')
    args=parser.parse_args()
    require(args.owner_execute,'Single actual ROOT CPU owner required')
    run(args.spec)
