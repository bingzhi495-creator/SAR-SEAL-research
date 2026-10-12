"""UNBOUND v3 SOURCE: CPU adoption checks for the one failed seed701 E01--08 GPU attempt.
Original actual GPU exit1 and missing original REPORT remain preserved. This
entry does not rerun training, create an optimizer, replay updates or certify
its own terminal. ROOT must qualify its actual CPU0 before future adoption.
"""
import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import types

SCHEMA='ROOT-N13-completed-epoch-artifact-recovery-CPU-spec-v3'
REPORT_SCHEMA='N13-completed-epoch-artifact-recovery-CPU-report-v3'
Q_SCHEMA='ROOT-N13-completed-epoch-artifact-recovery-CPU-qualification-v3'
Q_STATUS='ROOT_N13_COMPLETED8_FAILED_GPU_ENDPOINT_CPU_CHECKED_PENDING_ROOT_TERMINAL'
OWNER_SHA='ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
ARMS=('LOG_PR','GEO_PR')
NAMES=('hidden1.weight','hidden1.bias','hidden2.weight','hidden2.bias','hidden3.weight','hidden3.bias','final.weight','final.bias')
SEEDS=(202610120701,202610120702,202610120703)
SHAPES=((91,257),(91,),(91,92),(91,),(91,92),(91,),(1,92),(1,))
THREAD_ENV=('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')
REF_KEYS=('entry_source','training_entry_source','training_spec','training_GPU_admission','training_GPU_RAW','training_GPU_phase_binding','training_GPU_bootstrap','training_failure_ROOT_record','original_CPU_validator_source','region_common_source','cache_contract_source','point_heads_source','cache_CPU_qualification','relationship_entry_source','relationship_spec','relationship_report','relationship_GPU_admission','relationship_GPU_RAW','relationship_ROOT_qualification',)
NUMERIC=dict(endpoint_BCE_rtol=1e-4,endpoint_BCE_atol=1e-6,point_BCE_arithmetic_rtol=1e-4,point_BCE_arithmetic_atol=1e-6)
ARTIFACT_NAMES=('START.json','TRACE.jsonl','UPDATE_TRACE.jsonl','DECISION.json','INITIAL_LOADED_STATE.pth',
    'HEAD_LOG_PR_SEED202610120701_EPOCH08.pth','HEAD_LOG_PR_SEED202610120701_EPOCH08.PROOF.json',
    'HEAD_GEO_PR_SEED202610120701_EPOCH08.pth','HEAD_GEO_PR_SEED202610120701_EPOCH08.PROOF.json')
FAILURE_SCHEMA='ROOT-N13-training-attempt02-failure-preservation-v1'
FAILURE_STATUS='ORIGINAL_GPU_EXIT1_E01_08_ARTIFACTS_PRESERVED_NOT_QUALIFIED'

def require(ok,message):
    if not ok: raise RuntimeError(message)

def read(path):
    def reject(value): raise ValueError('Nonfinite JSON constant: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),parse_constant=reject)

def ref_shape(ref):
    require(type(ref) is dict and set(ref)=={'path','bytes','sha256'} and type(ref['path']) is str and Path(ref['path']).is_absolute() and
        type(ref['bytes']) is int and ref['bytes']>=0 and type(ref['sha256']) is str and len(ref['sha256'])==64 and
        set(ref['sha256'])<=set('0123456789abcdef'),'Malformed pinned reference')

def reference(path):
    path=Path(path).resolve(strict=True); h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''): h.update(block)
    return dict(path=str(path),bytes=path.stat().st_size,sha256=h.hexdigest())

def bound(ref):
    ref_shape(ref); require(reference(ref['path'])==ref,'Pinned input changed: '+ref['path']); return Path(ref['path'])

def terminal(ref,phase,code=0):
    ref_shape(ref); path=Path(ref['path'])
    require(path.name=='PARENT_RECEIPT.json' and type(phase) is str and phase and path.parent.name==phase,'Original RAW locator required')
    raw=read(path)
    require(raw['status']=='ACTUAL_OWNED_TERMINAL' and type(raw['returncode']) is int and raw['returncode']==code and
        type(raw['original_identity']) is dict and raw['original_identity'] and raw['root_wait_reaped'] is True and
        raw['held_handle_exit']==dict(wait_state='WAIT_OBJECT_0',actual_exit_code=code) and
        type(raw['held_handle_exit']['actual_exit_code']) is int and raw['numeric_birth_absence']['original_birth_absent'] is True and
        raw.get('error') is None and raw.get('abort') is None and raw['wait_errors'] is False and raw['GPU_observation_errors']==[],
        'Actual held exit/reap/original birth absence required before every SHA/import')
    if 'phase' in raw: require(raw['phase']==phase,'RAW phase differs')
    peak,sha=raw['whole_GPU_observed_peak_MiB'],raw['admission_sha256']
    require(type(sha) is str and len(sha)==64 and set(sha)<=set('0123456789abcdef') and
        type(peak) in (int,float) and math.isfinite(peak) and 0<=peak<=8192 and raw['whole_GPU_used_limit_MiB']==8192 and
        raw['whole_leaf_VRAM_certified'] is False and raw['GPU_scope']=='whole GPU observed, not per-process WDDM allocation','RAW resource differs')
    return raw

def load(ref,name):
    path=bound(ref); source=path.read_bytes(); require(len(source)==ref['bytes'] and hashlib.sha256(source).hexdigest()==ref['sha256'],'SOURCE changed')
    module=types.ModuleType(name); module.__file__=str(path); exec(compile(source,str(path),'exec'),module.__dict__); return module

def deadline(started,spec):
    require(time.monotonic()-started<spec['deadline_seconds'],'Prelocked CPU leaf endpoint deadline exceeded; no fallback/retry')

def resources(psutil,torch):
    require(psutil.Process().cpu_affinity()==[0,1] and torch.get_num_threads()==torch.get_num_interop_threads()==2 and
        not torch.cuda.is_initialized(),'Actual CPU01/threads2/coldCUDA required')

def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False); stream.write('\n')

def pre_gate(spec):
    # No reference(), SOURCE load, numerical import or artifact read before all47 RAWs.
    rows=spec['RAW_inputs']
    require(type(spec['normal_RAW_count']) is int and spec['normal_RAW_count']==40 and
        type(rows) is list and len(rows)==40 and len({r['receipt']['path'] for r in rows})==40,
        'This recovery requires exact40 actual normal ancestors')
    for row in rows:
        require(type(row) is dict and set(row)=={'receipt','phase','admission','phase_binding','bootstrap'},'Exact normal owner row required')
    normals=[terminal(row['receipt'],row['phase'],0) for row in rows]
    primary=spec['prior_failed_CPU_receipt']; primary_phase=Path(primary['path']).parent.name
    failure0=terminal(primary,primary_phase,1)
    extra=spec['additional_failed_RAW_inputs']
    require(type(extra) is list and len(extra)==6 and all(type(r) is dict and set(r)=={'receipt','phase','returncode'} and
        type(r['returncode']) is int and r['returncode']==1 for r in extra),'Exact additional6 failed rows required')
    failures=[failure0]+[terminal(row['receipt'],row['phase'],1) for row in extra]
    failed_paths=[primary['path']]+[row['receipt']['path'] for row in extra]
    require(len(set(failed_paths))==7 and not set(failed_paths).intersection(row['receipt']['path'] for row in rows),
        'Seven failures must remain separate from40 normal ancestors')
    all_rows=[dict(row,returncode=0) for row in rows]+[dict(receipt=primary,phase=primary_phase,returncode=1)]+extra
    pins=spec['terminal_identity_pins']
    require(type(pins) is dict and len(pins)==47 and set(pins)=={row['receipt']['path'] for row in all_rows},'Exact47 identity pins required')
    for row,raw in zip(all_rows,normals+failures):
        pin=pins[row['receipt']['path']]
        require(type(pin) is dict and set(pin)=={'phase','returncode','original_identity','admission_sha256'} and
            pin['phase']==row['phase'] and type(pin['returncode']) is int and pin['returncode']==row['returncode'] and
            pin['original_identity']==raw['original_identity'] and pin['admission_sha256']==raw['admission_sha256'],
            'All47 original identities/admissions must match before every SHA/import')
    refs=spec['sources']; require(type(refs) is list,'Explicit closure required')
    for ref in refs: ref_shape(ref)
    closure={ref['path']:ref for ref in refs}
    require(len(closure)==len(refs),'Unique complete recovery closure required')
    # FIRST HASH only here, after all47 actual terminal/identity checks.
    for ref in refs: bound(ref)
    require(all(row[k] in closure.values() for row in rows for k in ('receipt','admission','phase_binding','bootstrap')) and
        primary in closure.values() and all(row['receipt'] in closure.values() for row in extra),'Complete RAW metadata closure required')
    return normals,failures,closure


def missing_original_report(training):
    path=Path(training['output_directory'])/'REPORT.json'
    require(not path.exists() and not path.is_symlink(),'Original GPU REPORT must remain absent; no backfill allowed')
    return str(path)


def failure_record_join(spec,training,record,raw,closure):
    require(record['schema']==FAILURE_SCHEMA and record['status']==FAILURE_STATUS and record['phase']==training['phase'] and
        record['actual_returncode']==1 and type(record['actual_returncode']) is int and record['normal_RAW_count']==40 and
        record['preserved_failed_RAW_count']==7 and record['full_source_checked_count']==len(training['sources'])==8719 and
        record['GPU_spec']==spec['training_spec'] and record['GPU_admission']==spec['training_GPU_admission'] and
        record['GPU_RAW']==spec['training_GPU_RAW'] and record['phase_binding']==spec['training_GPU_phase_binding'] and
        record['bootstrap']==spec['training_GPU_bootstrap'] and record['artifacts']==spec['artifacts'] and
        record['report_absent'] is True and record['eight_epochs_logged_and_E08_checkpoints_saved'] is True and
        record['training_endpoint_qualified'] is False and record['AP_evaluated'] is False and
        record['original_exit_is_not_relabelled'] is True,'Actual failure preservation record differs')
    require(record['stdout'] in closure.values() and record['stderr'] in closure.values(),'Actual original stdout/stderr refs absent')
    # RAW1 itself, owner_join and the47 pins remain the authority; the record never substitutes for them.
    require(raw['returncode']==1 and raw['admission_sha256']==spec['training_GPU_admission']['sha256'],'Original GPU actual1 changed')
    resource=record['leaf_resources']
    require(type(resource['elapsed_seconds']) in (int,float) and math.isfinite(resource['elapsed_seconds']) and
        resource['elapsed_seconds']==raw['elapsed_seconds'] and 0<resource['elapsed_seconds']<training['deadline_seconds'] and
        resource['whole_GPU_observed_peak_MiB']==raw['whole_GPU_observed_peak_MiB'] and
        type(resource['minimum_ram_bytes']) in (int,float) and math.isfinite(resource['minimum_ram_bytes']) and
        resource['minimum_ram_bytes']==raw['minimum_ram_bytes'] and resource['minimum_ram_bytes']>=16*1024**3,
        'Preserved actual resource record differs or original owner exceeded locked deadline')


def source_gate(path,spec):
    require(spec['schema']==SCHEMA and spec['numeric_protocol']==NUMERIC and spec['seed']==202610120701 and
        type(spec['seed']) is int and spec['first_epoch']==1 and spec['last_epoch']==8 and spec['resume'] is None and
        type(spec['deadline_seconds']) is int and spec['deadline_seconds']==1800 and spec['allocator_limit_MiB']==4096 and
        spec['whole_GPU_limit_MiB']==8192,'Exact seed701 complete8 recovery/schema/resources required')
    raws,failures,closure=pre_gate(spec)
    require(all(spec[k] in closure.values() for k in REF_KEYS) and spec['recovery_protocol'] in closure.values() and
        spec['preregistered_protocol'] in closure.values(),'Recovery/frozen source/protocol refs absent')
    training=read(bound(spec['training_spec']))
    require(training['schema']=='ROOT-N13-sameK-region-point-training-spec-v2' and training['entry_source']==spec['training_entry_source'] and
        training['normal_RAW_count']==40 and training['RAW_inputs']==spec['RAW_inputs'] and
        training['prior_failed_CPU_receipt']==spec['prior_failed_CPU_receipt'] and
        training['additional_failed_RAW_inputs']==spec['additional_failed_RAW_inputs'][:5] and
        all(spec['terminal_identity_pins'].get(k)==v for k,v in training['terminal_identity_pins'].items()) and
        all(ref in closure.values() for ref in training['sources']) and spec['preregistered_protocol'] in training['sources'] and
        spec['original_CPU_validator_source'] in training['sources'] and
        Path(spec['original_CPU_validator_source']['path']).parent==Path(spec['training_entry_source']['path']).parent and
        Path(spec['original_CPU_validator_source']['path']).name=='qualify_train_cpu.py' and
        training['seed']==spec['seed'] and training['first_epoch']==1 and training['last_epoch']==8 and training['resume'] is None and
        training['recipe']==spec['recipe'] and training['validation_protocol']==spec['validation_protocol'] and
        all(training[k]==spec[k] for k in ('region_common_source','cache_contract_source','point_heads_source','cache_CPU_qualification')),
        'Original40+six-failure training spec/controls/source closure must be inherited exactly')
    last=spec['additional_failed_RAW_inputs'][-1]
    require(last==dict(receipt=spec['training_GPU_RAW'],phase=training['phase'],returncode=1),'Last preserved failure must be original GPU701 actual1')
    original_missing=missing_original_report(training)
    root_record=read(bound(spec['training_failure_ROOT_record']))
    failure_record_join(spec,training,root_record,failures[-1],closure)
    require(type(spec['artifacts']) is dict and set(spec['artifacts'])==set(ARTIFACT_NAMES), 'Exact nine original artifacts required')
    for name,ref in spec['artifacts'].items():
        require(ref in closure.values() and Path(ref['path'])==Path(training['output_directory'])/name,'Every artifact must retain its original path')
    train=load(spec['training_entry_source'],'_N13_v2_original_failed_training_ABI')
    old=load(spec['original_CPU_validator_source'],'_N13_v2_original_CPU_numerical_helpers')
    require(train.SCHEMA==training['schema'] and train.ARMS==old.ARMS==ARMS and train.SEEDS==old.SEEDS==SEEDS and
        train.NAMES==old.NAMES==NAMES and train.SHAPES==old.SHAPES==SHAPES and train.PARAMETERS==40497 and
        old.SCHEMA=='ROOT-N13-sameK-region-point-training-CPU-spec-v2' and old.NUMERIC==NUMERIC and
        old.CP_SCHEMA==train.CP_SCHEMA and old.START_SCHEMA==train.START_SCHEMA and old.LOADED_SCHEMA==train.LOADED_SCHEMA,
        'Exact frozen training/independent CPU math ABI required')
    # Historical gate ONLY on the untouched original spec, not the recovery spec.
    # Never call original run(), original CPU source_gate() or original qualify().
    historical=train.source_gate(Path(spec['training_spec']['path']),training,owner_environment=False)
    common=historical['common']
    require(train.RECIPE==spec['recipe'] and spec['validation_protocol']==common.VALIDATION_PROTOCOL and
        all(spec[k]==training[k] for k in common.RELATION_REF_KEYS),'Original recipe/qualified cache relation refs differ')
    owners=[common.owner_join(row,raw,closure) for row,raw in zip(spec['RAW_inputs'],raws)]
    failed_row=dict(receipt=spec['training_GPU_RAW'],phase=training['phase'],admission=spec['training_GPU_admission'],
        phase_binding=spec['training_GPU_phase_binding'],bootstrap=spec['training_GPU_bootstrap'])
    failed_owner=common.owner_join(failed_row,failures[-1],closure)
    require(failed_owner['scientific_spec']==spec['training_spec'] and failed_owner['entry']==spec['training_entry_source']['path'] and
        failed_owner['entry_args']==['--spec',spec['training_spec']['path'],'--owner-execute'] and failed_owner['cuda_visible_devices']=='0' and
        historical['admission_ref']==spec['training_GPU_admission'] and training['GPU_admission_file']==spec['training_GPU_admission']['path'] and
        all(failed_owner['bound_source_sha256'].get(ref['path'])==ref['sha256'] for ref in training['sources']),
        'Original failed owner/source/spec/admission/PB/bootstrap join differs')
    source,spec_ref=reference(__file__),reference(path)
    require(source==spec['entry_source'],'Actual recovery entry differs')
    adpath=Path(spec['CPU_admission_file']); require(adpath.is_absolute(),'Absolute CPU admission required')
    ad_ref=reference(adpath); admission=read(adpath)
    require(admission['status']=='ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and admission['driver_sha256']==OWNER_SHA and
        admission['phase']==spec['phase'] and admission['scientific_spec']==spec_ref and admission['entry']==source['path'] and
        admission['entry_args']==['--spec',str(path),'--owner-execute'] and admission['allowed_cpus']==[0,1] and
        admission['cuda_visible_devices']=='' and admission['automatic_retry'] is False and admission['whole_GPU_used_limit_MiB']==8192 and
        admission['qualified_leaf_deadline_seconds']==1800 and admission['bound_source_sha256'].get(str(path))==spec_ref['sha256'] and
        all(admission['bound_source_sha256'].get(ref['path'])==ref['sha256'] for ref in spec['sources']) and
        os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='2' for k in THREAD_ENV),'Actual owned CPU admission differs')
    output=Path(spec['output_directory'])
    require(output.is_absolute() and not output.exists() and not output.is_symlink() and
        Path(spec['report_path'])==output/'REPORT.json' and Path(spec['Q_path'])==output/'QUALIFICATION.json' and
        output!=Path(training['output_directory']) and Path(training['output_directory']) not in output.parents,
        'Fresh separate absolute recovery output and exact report/Q paths required')
    return dict(train=train,old=old,common=common,training=training,GPU_gate=historical,closure=closure,owners=owners,
        failed_owner=failed_owner,failed_RAW=failures[-1],failure_record=root_record,missing_GPU_report=original_missing,
        source=source,spec_ref=spec_ref,admission_ref=ad_ref,admission=admission)

def artifact_derived_view(torch,gate,spec):
    artifacts=spec['artifacts']; original=gate['training']; old=gate['old']
    # pre_gate already hashed every artifact before any torch.load. Repeat own pins at read.
    temporary={arm:torch.load(str(bound(artifacts['HEAD_'+arm+'_SEED202610120701_EPOCH08.pth'])),map_location='cpu') for arm in ARMS}
    start=read(bound(artifacts['START.json']))
    checkpoints={}; histories={}; activities={}
    for arm,cp in temporary.items():
        require(type(cp) is dict and set(cp)=={'state_dict','optimizer_state_dict','private_order_state','meta'},'Exact checkpoint container required')
        old.finite_state(torch,cp['state_dict']); meta=cp['meta']
        require(type(meta) is dict and meta['schema']==old.CP_SCHEMA and meta['completed_head_epochs']==8 and
            meta['seed']==202610120701 and meta['arm']==arm and len(meta['endpoint_BCE_history'])==8,'Only actual complete8 seed701 endpoint allowed')
        histories[arm]=meta['endpoint_BCE_history']; activities[arm]=meta['gradient_activity']
        checkpoints[arm]=dict(checkpoint=artifacts['HEAD_'+arm+'_SEED202610120701_EPOCH08.pth'],
            proof=artifacts['HEAD_'+arm+'_SEED202610120701_EPOCH08.PROOF.json'],updates=meta['updates'],
            completed_head_epochs=meta['completed_head_epochs'],endpoint=meta['endpoint_BCE_history'][-1],stability=meta['stability'])
    view=dict(view_type='ARTIFACT_DERIVED_COMPATIBILITY_VIEW_NOT_ORIGINAL_GPU_REPORT',
        source=spec['training_entry_source'],spec=spec['training_spec'],GPU_admission=spec['training_GPU_admission'],phase=original['phase'],
        seed=202610120701,first_epoch=1,last_epoch=8,recipe=spec['recipe'],start=artifacts['START.json'],trace=artifacts['TRACE.jsonl'],
        update_trace=artifacts['UPDATE_TRACE.jsonl'],decision=artifacts['DECISION.json'],initial_loaded_state=artifacts['INITIAL_LOADED_STATE.pth'],
        checkpoints=checkpoints,endpoint_histories=histories,gradient_activity=activities,cold_state_proofs=start['cold_state_proofs'],
        resume_parent_CPU_qualification=start['resume_parent_CPU_qualification'],
        unknown_original_GPU_report_fields=['torch_peak_allocated_bytes','cleanup.allocated_before','cleanup.allocated_after','imported_library_refs','elapsed_seconds'],
        original_GPU_report_written=False,original_GPU_actual_returncode=1)
    del temporary
    return view

def records_check(torch,gate,spec,cps,loaded):
    report=gate['report']; first,last=spec['first_epoch'],spec['last_epoch']; seed=spec['seed']; orders=cps[ARMS[0]]['meta']['epoch_orders']
    require(len(orders)==last and orders[:first-1]==loaded['epoch_orders'],'Cumulative paired complete image orders required')
    generator=torch.Generator(device='cpu'); generator.manual_seed(seed)
    for epoch in range(1,last+1):
        indices=torch.randperm(741,generator=generator).tolist(); sha=hashlib.sha256(json.dumps(indices,separators=(',',':')).encode('ascii')).hexdigest()
        require(orders[epoch-1]==dict(epoch=epoch,indices=indices,sha256=sha),'Independent private image permutation differs')
        if epoch==first-1: require(torch.equal(generator.get_state(),loaded['private_order_state']),'Actual resume private RNG is not uninterrupted')
    require(torch.equal(generator.get_state(),cps[ARMS[0]]['private_order_state']),'Independent final private image RNG differs')
    trace=[json.loads(line) for line in bound(report['trace']).read_text(encoding='utf-8').splitlines() if line]
    require(len(trace)==last-first+1 and [row['epoch'] for row in trace]==list(range(first,last+1)),'One complete trace per new epoch required')
    records_path=bound(report['update_trace']); normal_rows=gate['GPU_gate']['cache_report']['rows']
    expected_steps=1661*(last-first+1); activity={arm:{n:dict(present=0,nonzero=0) for n in NAMES} for arm in ARMS}
    with records_path.open('r',encoding='utf-8') as stream:
        counted=0
        for epoch,row in zip(range(first,last+1),trace):
            require(row['seed']==seed and row['images']==741 and row['GT_instances']==1661 and row['updates_per_model']==1661 and
                row['cumulative_updates_per_model']==1661*epoch and row['point_visits_per_model']==1700864 and
                row['image_order_sha256']==orders[epoch-1]['sha256'] and row['paired_initialization'] is True and row['paired_image_order'] is True and
                row['native_forward_calls']==0 and row['native_weights_updated'] is False and set(row['online_BCE'])==set(ARMS),
                'Complete per-epoch paired visit/recipe trace required')
            online={arm:[] for arm in ARMS}; epoch_count=0
            for index in orders[epoch-1]['indices']:
                identity=normal_rows[index]['identity']; ann_ids=identity['ann_ids']
                # query IDs are verified against qualified packets by endpoint_check;
                # a tiny row query list is preferred, otherwise load the core packet.
                if 'query_ids' in normal_rows[index]: qids=normal_rows[index]['query_ids']
                else:
                    value=gate['train'].cache_packet(torch,gate['GPU_gate'],index); qids=value['query_ids'].tolist(); del value
                require(len(qids)==len(ann_ids)==normal_rows[index]['raw_GT'],'Qualified update identity population required')
                for g,(ann,qid) in enumerate(zip(ann_ids,qids)):
                    line=stream.readline(); require(bool(line),'Incomplete paired update records'); actual=json.loads(line)
                    require(actual['seed']==seed and actual['epoch']==epoch and actual['image_index']==index and actual['GT_index']==g and
                        actual['annotation_id']==ann and actual['query_id']==qid and actual['global_step']==1661*(epoch-1)+epoch_count+1 and
                        set(actual['BCE'])==set(actual['gradient_present'])==set(actual['gradient_nonzero'])==set(ARMS),'Exact ordered complete paired update identity required')
                    for arm in ARMS:
                        loss=actual['BCE'][arm]; present=actual['gradient_present'][arm]; nonzero=actual['gradient_nonzero'][arm]
                        require(type(loss) in (int,float) and math.isfinite(loss) and loss>=0 and present==[1]*8 and len(nonzero)==8 and
                            all(type(v) is int and v in (0,1) for v in nonzero),'Finite recorded BCE/8 connected gradient flags required')
                        online[arm].append(loss)
                        for n,nz in zip(NAMES,nonzero): activity[arm][n]['present']+=1; activity[arm][n]['nonzero']+=nz
                    epoch_count+=1; counted+=1
            require(epoch_count==1661,'Each epoch requires all1661 instances')
            for arm in ARMS:
                require(math.fsum(online[arm])/1661==row['online_BCE'][arm] and row['full_FIT_endpoint'][arm]==
                    {k:v for k,v in report['endpoint_histories'][arm][epoch-1].items() if k!='epoch'},'Record-derived online means and bound endpoint history differ')
        require(not stream.read().strip() and counted==expected_steps,'No missing/extra update visits allowed')
    for arm in ARMS:
        for n in NAMES:
            incoming=loaded['arms'][arm]['gradient_activity'][n]; expected={k:incoming[k]+activity[arm][n][k] for k in ('present','nonzero')}
            require(report['gradient_activity'][arm][n]==cps[arm]['meta']['gradient_activity'][n]==expected and
                expected['present']==1661*last and 0<=expected['nonzero']<=expected['present'],'Cumulative recorded gradient presence/counts differ')
    # The original GPU REPORT was never written. Do not supply fabricated cleanup/scope fields.
    # START and TRACE were independently bound above and checked by the old state helper.
    require(first==1 and last==8 and counted==13288 and 1661*last==13288,
        'Recovery adopts exactly the already recorded complete8 paired sequence')
    return dict(epochs_checked=8,paired_update_records_checked=counted,updates_per_model=13288,
        per_step_gradient_or_optimizer_replayed=False,original_GPU_report_scope_assertions_reused=False)

def recover(spec_path):
    started=time.monotonic(); path=Path(spec_path)
    require(path.is_absolute(),'Absolute ROOT recovery spec required'); path=path.resolve(strict=True)
    spec=read(path); gate=source_gate(path,spec); old=gate['old']; output=Path(spec['output_directory'])
    import psutil
    require(psutil.Process().cpu_affinity()==[0,1],'CPU01 required before numerical imports')
    import torch
    import numpy as np
    require(torch.__version__=='2.0.1+cu118' and np.__version__=='1.23.5' and not torch.cuda.is_initialized(),
        'Pinned original Torch/NumPy and coldCUDA required')
    torch.set_num_threads(2); torch.set_num_interop_threads(2); resources(psutil,torch)
    imported=old.imported_files(psutil,torch,np,gate); external=torch.random.get_rng_state().clone()
    deadline(started,spec)
    gate['report']=artifact_derived_view(torch,gate,spec)
    states,cps,loaded,start,decision,cold=old.load_and_check_states(torch,gate,spec)
    deadline(started,spec)
    records=records_check(torch,gate,spec,cps,loaded)
    endpoints,identities,arithmetic=old.endpoint_check(torch,states,gate,spec,started)
    resources(psutil,torch)
    require(torch.equal(external,torch.random.get_rng_state()),'Recovery changed external CPU RNG')
    # Recheck all actual terminal identities before final integrity SHA/imported-file checks.
    pre_gate(spec)
    require(old.imported_files(psutil,torch,np,gate)==imported,'Actual CPU imported files changed')
    require(reference(path)==gate['spec_ref'] and reference(__file__)==gate['source'] and
        reference(spec['CPU_admission_file'])==gate['admission_ref'],'Recovery source/spec/admission changed')
    require(missing_original_report(gate['training'])==gate['missing_GPU_report'],'Original missing GPU REPORT path changed')
    deadline(started,spec)
    # Only these two fresh metadata outputs are written; no original artifact is altered.
    output.mkdir()
    report=dict(schema=REPORT_SCHEMA,status=Q_STATUS,source=gate['source'],spec=gate['spec_ref'],CPU_admission=gate['admission_ref'],
        phase=spec['phase'],adoption_type='COMPLETE_E08_ENDPOINT_FROM_ORIGINAL_FAILED_GPU_ATTEMPT',
        **{k:spec[k] for k in REF_KEYS if k!='entry_source'},
        recovery_protocol=spec['recovery_protocol'],preregistered_protocol=spec['preregistered_protocol'],
        original_GPU_actual_returncode=1,original_GPU_report_path=gate['missing_GPU_report'],original_GPU_report_written=False,
        original_exit_is_not_relabelled=True,original_GPU_artifacts_unchanged=True,artifacts=spec['artifacts'],
        artifact_derived_training_view=gate['report'],recipe=spec['recipe'],seed=202610120701,first_epoch=1,last_epoch=8,resume=None,
        fixed_final_epoch=48,completed_head_epochs=8,updates_per_model=13288,checkpoints=gate['report']['checkpoints'],
        normal_RAW_count=40,RAW_inputs=spec['RAW_inputs'],prior_failed_CPU_receipt=spec['prior_failed_CPU_receipt'],
        additional_failed_RAW_inputs=spec['additional_failed_RAW_inputs'],terminal_identity_pins=spec['terminal_identity_pins'],
        validation_protocol=spec['validation_protocol'],numeric_protocol=NUMERIC,cold_state_proofs=cold,records=records,
        final_endpoints=endpoints,endpoint_instances=identities,stable_BCE_vs_functional_max_absolute_difference=arithmetic,
        preserved_original_owner_resources=gate['failure_record']['leaf_resources'],
        unknown_original_GPU_report_fields=gate['report']['unknown_original_GPU_report_fields'],
        cold_paired_weights_CPU_checked=True,initial_loaded_state_CPU_checked=True,
        private_image_order_and_endpoint_RNG_CPU_reconstructed=True,optimizer_endpoint_structure_and_stepcounts_CPU_checked=True,
        full_update_visit_sequence_CPU_checked=True,final_fullFIT_endpoint_BCE_CPU_recomputed=True,final_editable_BCE_CPU_recomputed=True,
        intermediate_epoch_losses_independently_recomputed=False,per_step_gradient_or_AdamW_replayed=False,
        optimizer_constructed=False,optimizer_steps_executed=0,training_rerun=False,native_model_loaded=False,native_forward_calls=0,
        no_DEV=True,no_TEST=True,AP_computed=False,CPU_affinity=[0,1],threads=2,CUDA_initialized=False,
        resume_eligible_only_after_ROOT_actual_CPU0=True,current_endpoint_adoption_self_certified=False,
        current_owned_CPU_terminal_self_certified=False,imported_library_refs=imported,elapsed_seconds=time.monotonic()-started,
        evidence_limits=[
            'The original GPU attempt ended with actual exit1 after all eight epochs and artifacts; no original GPU report is fabricated.',
            'The artifact-derived view is a new explicitly labelled reconstruction, not an original report or original terminal0.',
            'CPU independently checks final cached-point losses, paired initialization, recorded order and optimizer endpoint structure/steps.',
            'Gradient and AdamW arithmetic are not replayed; intermediate epoch losses remain GPU records.',
            'Unpublished original allocator/cleanup/imported-ref fields remain unknown; whole-GPU owner resources retain their original scope.',
            'This endpoint check does not prove efficacy, convergence, novelty or TGRS readiness.',
            'Future resume requires a distinct adoption-aware admission and ROOT actual CPU0 qualification after this entry exits.'])
    deadline(started,spec)
    write(spec['report_path'],report); report_ref=reference(spec['report_path'])
    require(missing_original_report(gate['training'])==gate['missing_GPU_report'],'Original GPU REPORT appeared during recovery')
    resources(psutil,torch); deadline(started,spec)
    q=dict(schema=Q_SCHEMA,status=Q_STATUS,source=gate['source'],reader_source=gate['source'],spec=gate['spec_ref'],
        CPU_admission=gate['admission_ref'],phase=spec['phase'],recovery_report=report_ref,
        **{k:spec[k] for k in REF_KEYS if k!='entry_source'},
        recovery_protocol=spec['recovery_protocol'],preregistered_protocol=spec['preregistered_protocol'],
        original_GPU_actual_returncode=1,original_GPU_report_written=False,original_exit_is_not_relabelled=True,
        original_GPU_artifacts_unchanged=True,artifacts=spec['artifacts'],checkpoints=gate['report']['checkpoints'],
        recipe=spec['recipe'],seed=202610120701,first_epoch=1,last_epoch=8,resume=None,fixed_final_epoch=48,
        completed_head_epochs=8,updates_per_model=13288,normal_RAW_count=40,RAW_inputs=spec['RAW_inputs'],
        prior_failed_CPU_receipt=spec['prior_failed_CPU_receipt'],additional_failed_RAW_inputs=spec['additional_failed_RAW_inputs'],
        terminal_identity_pins=spec['terminal_identity_pins'],numeric_protocol=NUMERIC,validation_protocol=spec['validation_protocol'],
        cold_paired_weights_CPU_checked=True,initial_loaded_state_CPU_checked=True,
        private_image_order_and_endpoint_RNG_CPU_reconstructed=True,optimizer_endpoint_structure_and_stepcounts_CPU_checked=True,
        full_update_visit_sequence_CPU_checked=True,final_fullFIT_endpoint_BCE_CPU_recomputed=True,final_editable_BCE_CPU_recomputed=True,
        final_endpoints=endpoints,records=records,intermediate_epoch_losses_independently_recomputed=False,
        per_step_gradient_or_AdamW_replayed=False,optimizer_constructed=False,optimizer_steps_executed=0,training_rerun=False,
        native_model_loaded=False,native_forward_calls=0,no_DEV=True,no_TEST=True,AP_computed=False,
        CPU_affinity=[0,1],threads=2,CUDA_initialized=False,resume_eligible_only_after_ROOT_actual_CPU0=True,
        current_endpoint_adoption_self_certified=False,current_owned_CPU_terminal_self_certified=False)
    write(spec['Q_path'],q)
    require(missing_original_report(gate['training'])==gate['missing_GPU_report'],'Original GPU REPORT appeared after new Q write')
    resources(psutil,torch); deadline(started,spec)
    print(json.dumps(dict(status=Q_STATUS,report=report_ref,Q=reference(spec['Q_path']),original_GPU_actual_returncode=1)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--spec',required=True)
    parser.add_argument('--owner-execute',action='store_true'); args=parser.parse_args()
    require(args.owner_execute,'Explicit single ROOT-owned CPU recovery execution required'); recover(args.spec)
