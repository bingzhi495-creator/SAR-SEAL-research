"""Synthetic external facts only; no process/scientific library or real input."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
from pathlib import PurePosixPath
import types
import unittest
from unittest.mock import patch

import acceptance


def ref(path,data=b'x'):
    return {'path':path,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}


def terminal_fixture():
    spec=ref('/synthetic/attempt/SPEC.json')
    owner=ref('/synthetic/sources/linux_owner.py')
    command=['exec','/synthetic/python','-u',owner['path'],'--spec',spec['path'],'--spec-sha',spec['sha256']]
    terminal=dict(root_observed_tool_exit_code=0,root_observed_tool_chunk_id='SYNTHETIC_NOT_ACTUAL',
        SSH_returncode=0,owner_process_exit_code=0,original_exec_wait_observed=True,command=command,
        stdout_ref=ref('/synthetic/attempt/original.stdout',b''),stderr_ref=ref('/synthetic/attempt/original.stderr',b''),
        attempt_id='SYNTHETIC_ATTEMPT',phase='SYNTHETIC_PHASE',spec_ref=spec)
    return terminal,spec,owner


def post_fixture(terminal):
    owner_id=dict(pid=200,state='S',parent_pid=100,process_group=100,session=100,start_ticks=1000)
    child_id=dict(pid=201,state='R',parent_pid=200,process_group=201,session=201,start_ticks=1001)
    owner=dict(owner_pid=200,owner_identity=owner_id,child_identity=child_id,boot_id='11111111-2222-3333-4444-555555555555')
    post=dict(root_observed_tool_exit_code=0,root_observed_tool_chunk_id='SYNTHETIC_POST_NOT_ACTUAL',
        spec_ref=terminal['spec_ref'],phase=terminal['phase'],attempt_id=terminal['attempt_id'],original_terminal=deepcopy(terminal),
        boot_id=owner['boot_id'],owner_identity=deepcopy(owner_id),child_identity=deepcopy(child_id),
        owner_current_identity=None,child_current_identity=None,owner_original_birth_absent=True,
        child_original_birth_absent=True,owned_group_scan_resolved=True,remaining_owned_group=[],all_owned_processes_absent=True,
        evidence_refs={name:ref('/synthetic/attempt/'+name+'.json') for name in
            ('owner_receipt','owner_run_state','entry_report','worker_report','prediction')})
    return post,owner


class MemoryFile:
    def __init__(self,path,data):self.path,self.data=path,data
    def __str__(self):return self.path
    def open(self,mode):
        if mode!='rb':raise AssertionError('read-only synthetic bytes')
        return io.BytesIO(self.data)


class MetadataPath(PurePosixPath):
    def resolve(self,strict=False):return self
    def is_symlink(self):return False
    def is_dir(self):return True


def synthetic_inputs(shard_index=0):
    """Invented facts/raw JSON; real ordinary stdlib definitions, never real runs."""
    root=Path(__file__).parent.parent
    objects={}
    def add(path,data):
        if type(data) is str:data=data.encode()
        objects[path]=data;return ref(path,data)
    def save(path,value):return add(path,(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n').encode())
    ordinary={
        'entry':root/'U74_DGX_FIXED50_SINGLE_SHARD_SUPERVISION_DRAFT_20261021_01/entry.py',
        'owner':root/'U74_DGX_FIXED50_SINGLE_SHARD_SUPERVISION_DRAFT_20261021_01/linux_owner.py',
        'worker':root/'U73_DGX_FIXED50_PREDICTION_SOURCE_CLOSURE_DRAFT_20261021_01/worker.py',
        'core':root/'U66_DGX_FIXED50_PREDICTION_CORE_DRAFT_20261021_01/prediction_core.py'}
    code={name:add('/synthetic/code/'+name+'.py',path.read_bytes()) for name,path in ordinary.items()}
    entry=types.ModuleType('_SYNTHETIC_ENTRY_DEFS');exec(compile(objects[code['entry']['path']],code['entry']['path'],'exec'),entry.__dict__)
    worker=types.ModuleType('_SYNTHETIC_WORKER_DEFS');exec(compile(objects[code['worker']['path']],code['worker']['path'],'exec'),worker.__dict__)
    sources={role:add('/synthetic/components/'+role+'.py',
        ('# '+role+'\n').ljust(19618 if role=='fit_portable_reader' else 61452 if role=='encoder' else 40,'x'))
        for role in worker.SOURCE_ROLES}
    libs={role:add('/synthetic/library/'+role+'.py',b'' if role=='pycocotools' else b'SYNTHETIC_LIBRARY') for role in worker.LIBRARIES}
    cps={arm:add('/synthetic/formal/'+arm+'.pth',b'NOT_A_CHECKPOINT') for arm in ('PLAIN','FULL')}
    cursor=dict(global_updates=37050,completed_epochs=50,within_epoch=0)
    original12={role:sources[role] for role in worker.RESTORE_ROLES if role not in ('training_state','training_entry')}
    original12.update(image_geometry=sources['image_geometry'],portable_reader=sources['fit_portable_reader'])
    original14=dict(training_state=sources['training_state'],training_entry=sources['training_entry'],**original12)
    original_spec=dict(body_source=sources['training_state'],entry_source=sources['training_entry'],source_refs=original12,
        prelock={'mode':'FORMAL'},segment={'start':36864,'stop':37050})
    endpoint_refs={}
    endpoint_refs['spec']=save('/synthetic/formal/SPEC.json',original_spec)
    endpoint_refs['order']=save('/synthetic/formal/ORDER.json',{'synthetic_original_order':True})
    pair=dict(schema='U57-DGX-paired-checkpoint-commit-v1',cursor=cursor,checkpoint_refs=cps,order_ref=endpoint_refs['order'],span=original_spec['segment'])
    endpoint_refs['pair_commit']=save('/synthetic/formal/PAIR_COMMIT.json',pair)
    q=dict(schema='Root-U57-DGX-original-update-training-accepted-v1',root_accepted=True,mode='FORMAL',accepted_span=original_spec['segment'],
        evidence_refs={'spec_ref':endpoint_refs['spec']},result={'paired_checkpoint_prefix':cursor,'checkpoint_refs':cps,
            'pair_commit_ref':endpoint_refs['pair_commit'],'order_ref':endpoint_refs['order']})
    endpoint_refs['qualified']=save('/synthetic/formal/QUALIFIED.json',q)
    endpoint=dict(schema='U58-U57-DGX-fixed-endpoint-metadata-v1',mode='FORMAL',formal_endpoint=True,run_id='SYNTHETIC_ORIGINAL_RUN',
        cursor=cursor,checkpoint_refs=cps,pair_commit_ref=endpoint_refs['pair_commit'],order_ref=endpoint_refs['order'],
        training_qualified=endpoint_refs['qualified'],endpoint_refs=endpoint_refs,restore_source_refs=original14,metadata_only=True,numerical_runtime_qualified=False)
    entries=[dict(dataset_index=i,image_id=1000+i,query_ids=list(range(99,-1,-1)),
        package_ref=add('/synthetic/packages/DEV_'+str(i)+'.pt',b'NOT_A_TENSOR')) for i in range(187)]
    manifest=save('/synthetic/DEV_MANIFEST.json',{'entries':entries})
    span=list(worker.RANGES[shard_index]);n=span[1]-span[0];phase='SYNTHETIC_PHASE_'+str(shard_index)
    output='/synthetic/attempt_'+str(shard_index)
    request=dict(worker_source=code['worker'],prediction_core_source=code['core'],source_refs=sources,library_sources=libs,
        endpoint_refs=endpoint_refs,checkpoint_refs=cps,manifest_ref=manifest,shard_index=shard_index,dataset_range=span,output_directory=output+'/WORKER')
    spec=dict(schema=entry.SCHEMA,operation=entry.OPERATION,phase=phase,entry_source=code['entry'],owner_source=code['owner'],
        request=request,limits=deepcopy(entry.LIMITS),output_directory=output+'/ENTRY',owner_output_directory=output+'/OWNER')
    refs=[code['entry'],code['owner'],code['worker'],code['core'],manifest]+list(sources.values())+list(libs.values())+list(endpoint_refs.values())+list(cps.values())
    refs += [row['package_ref'] for row in entries[span[0]:span[1]]]
    spec['sources']=sorted(refs,key=lambda x:x['path']);spec_ref=save(output+'/SPEC.json',spec)
    predictions=[dict(image_id=row['image_id'],query_id=query,category_id=1,score=0.25,
        segmentation={'size':[1,1],'counts':'1'}) for row in entries[span[0]:span[1]] for query in row['query_ids']]
    proofs=[dict(dataset_index=row['dataset_index'],image_id=row['image_id'],queries=100,
        zero_residual_native_L_exact=True,native_RLE_and_prediction_dictionaries_exact=True,original_full_S0_class_order_preserved=True)
        for row in entries[span[0]:span[1]]]
    cost=dict(encode_calls=n,readout_calls=25*n,queries=100*n)
    support=[dict(dataset_index=i//100+span[0],image_id=value['image_id'],query_id=value['query_id']) for i,value in enumerate(predictions)]
    arms={arm:dict(status='PREDICTIONS_COMPLETE_PENDING_ROOT_ACCEPTANCE',predictions=deepcopy(predictions),identity_predictions=deepcopy(predictions),
        Omega_support=deepcopy(support) if arm=='PLAIN' else [],proofs=deepcopy(proofs),cost=deepcopy(cost),
        cleanup={'allocated_after':0,'reserved_after':0},resources={'CUDA_peak_allocated_bytes':1024,'sampled_RSS_peak_bytes':1024,'RSS_sampling_errors':[]})
        for arm in ('PLAIN','FULL')}
    prediction=dict(schema='U66-DGX-fixed50-DEV-shard-pending-v1',status='PREDICTIONS_COMPLETE_PENDING_ROOT_ACCEPTANCE',
        shard_index=shard_index,dataset_range=span,images=n,queries=100*n,fixed_epoch=50,updates_per_arm=37050,query_batch=4,
        endpoint=endpoint,source_refs={k:v for k,v in sources.items() if k!='fit_portable_reader'},arms=arms,
        predictions={a:arms[a]['predictions'] for a in arms},identity_predictions=predictions,Omega_support=support,cost={a:cost for a in arms},
        qualified=False,AP=False,DEV=True,DEV_technical_input=True,TEST=False,new_training=False,formal_training_budget_credit=0,
        whole_population_predicted=False,DEV_metrics_computed=False,GT_used_for_prediction=False,GT_rasterization_executed=False,
        remaining_model_objects=0,cleanup={'allocated_after':0,'reserved_after':0},model_BN_Torch_RNG_unchanged=True,
        original_full_S0_class_order_preserved=True,source_bytes_rechecked_after=True)
    prediction_ref=save(request['output_directory']+'/U66_SHARD.json',prediction)
    body=dict(schema='U73-single-shard-worker-report-v1',status='COMPLETE_PENDING_ROOT_ACCEPTANCE',qualified=False,AP=False,DEV=True,
        DEV_technical_input=True,TEST=False,new_training=False,formal_training_budget_credit=0,ROOT_actual_terminal_self_certified=False,
        automatic_retry=False,cleanup_errors=[],shard_index=shard_index,dataset_range=span,worker_source=code['worker'],prediction_core_source=code['core'],
        source_refs=sources,original_restore_source_closure_verified=True,prediction_ref=prediction_ref,
        direct_input_refs=[r for r in spec['sources'] if r not in (code['entry'],code['owner'])],source_aliases_restored=True,
        source_and_metadata_bytes_rechecked=True,outer_runtime_states_restored=True,elapsed_seconds=1.)
    body_ref=save(request['output_directory']+'/REPORT.json',body)
    observed=dict(schema='U74-Linux-child-control-observations-v1',samples=2,violations=[],peak_RSS_bytes=1024,minimum_available_RAM_bytes=32*1024**3,elapsed_seconds=1.)
    entry_report=dict(schema='U74-DGX-fixed50-single-shard-entry-report-v1',status='COMPLETE_PENDING_ROOT_ACCEPTANCE',operation=entry.OPERATION,
        phase=phase,spec_ref=spec_ref,**{k:deepcopy(spec[k]) for k in ('entry_source','owner_source','request','limits','sources','output_directory','owner_output_directory')},
        flags={k:True for k in entry.SUCCESS_FLAGS},cleanup_errors=[],DEV=True,DEV_technical_input=True,formal_training_budget_credit=0,
        worker_report=body,worker_report_ref=body_ref,prediction_ref=prediction_ref,resource_observations=observed,cost={'elapsed_seconds':1.})
    entry_report.update({k:False for k in entry.FALSE_FIELDS})
    entry_report['prediction_metadata']=entry.prediction_metadata(prediction,request)
    entry_ref=save(spec['output_directory']+'/REPORT.json',entry_report)
    terminal,_,_=terminal_fixture();terminal.update(spec_ref=spec_ref,phase=phase,attempt_id='SYNTHETIC_ATTEMPT_'+str(shard_index),
        command=['exec','/synthetic/python','-u',code['owner']['path'],'--spec',spec_ref['path'],'--spec-sha',spec_ref['sha256']],
        stdout_ref=add(output+'/original.stdout',b''),stderr_ref=add(output+'/original.stderr',b''))
    post,owner=post_fixture(terminal)
    logs={name:add(spec['owner_output_directory']+'/'+name,b'') for name in ('stdout.log','stderr.log')}
    owner.update(schema='U74-Linux-owned-fixed50-single-shard-prediction-receipt-v1',status='OWNED_CHILD_COMPLETE_PENDING_ROOT_REVIEW',
        operation=entry.OPERATION,phase=phase,spec_ref=spec_ref,returncode=0,reaped=True,original_birth_absent=True,child_launched=True,
        child_alive=False,remaining_owned_group=[],final_group_scan_resolved=True,all_owned_processes_absent=True,
        termination_requested=False,signals=[],violations=[],owner_termination_signal=None,controller_returning=True,
        single_shard_prediction_complete=True,entry_report_status='COMPLETE_PENDING_ROOT_ACCEPTANCE',worker_report_status='COMPLETE_PENDING_ROOT_ACCEPTANCE',
        entry_report_ref=entry_ref,worker_report_ref=body_ref,prediction_ref=prediction_ref,child_resource_observations=observed,
        qualified=False,ROOT_actual_terminal_self_certified=False,automatic_retry=False,formal_training_budget_credit=0,AP=False,DEV=True,
        DEV_technical_input=True,TEST=False,new_training=False,full_binary_closure_claimed=False,CUDA_visible_devices='0',
        GPU_allocator_contract_applicable=True,GPU_allocator_limit_bytes=4*1024**3,whole_GPU_observed_bytes=None,
        whole_GPU_observation_status='NOT_AVAILABLE_GB10_UNIFIED_MEMORY',whole_GPU_WDDM8GiB_contract_certified=False,
        source_union_verified_before_payload_hash=True,source_bytes_verified_before_launch=True,
        peak_group_RSS_bytes=1024,minimum_available_RAM_bytes=32*1024**3,elapsed_seconds=1.,
        observations=[dict(elapsed_seconds=1.,group_RSS_bytes=1024,available_RAM_bytes=32*1024**3,member_identities=[],child_returncode=0)],
        child_CPU_affinity=[0,1],parent_death_signal=dict(signal='SIGKILL',prctl='PR_SET_PDEATHSIG',expected_parent_pid=200,preexec_succeeded=True),
        sys_executable='/synthetic/python',python_version='3.12.14',psutil_version='6.1.1',logs=logs)
    owner_ref=save(spec['owner_output_directory']+'/OWNER_RECEIPT.json',owner)
    state_ref=save(spec['owner_output_directory']+'/RUN_STATE.json',owner)
    post['evidence_refs']=dict(owner_receipt=owner_ref,owner_run_state=state_ref,entry_report=entry_ref,worker_report=body_ref,prediction=prediction_ref)
    post_ref=save(output+'/POST.json',post)
    visits=[]
    def closed(value):
        visits.append(value['path']);return MemoryFile(value['path'],objects[value['path']])
    return dict(spec_ref=spec_ref,terminal=terminal,post_ref=post_ref,closed=closed,objects=objects,
        visits=visits,prediction=prediction,synthetic_source_pins={k:v['sha256'] for k,v in sources.items() if k!='encoder'},encoder=sources['encoder'])


def accept_inputs(data):
    """Only test fixtures may replace source-pin globals with invented byte identities."""
    original=acceptance.definitions
    def definitions(value,raw):
        module=original(value,raw)
        if hasattr(module,'ENCODER_SHA256'):module.ENCODER_SHA256=data['encoder']['sha256']
        if hasattr(module,'SOURCE_PINS'):module.SOURCE_PINS=data['synthetic_source_pins'];module.Path=MetadataPath
        return module
    with patch.object(acceptance,'definitions',side_effect=definitions):
        proof=acceptance.accept_shard(data['spec_ref'],data['terminal'],data['post_ref'],closed=data['closed'])
    return proof


def accept_synthetic(shard_index=0):
    data=synthetic_inputs(shard_index)
    return accept_inputs(data),data


class Acceptance(unittest.TestCase):
    def test_original_terminal_requires_true_integer_exits_and_exact_exec_identity(self):
        terminal,spec,owner=terminal_fixture()
        kwargs=dict(phase='SYNTHETIC_PHASE',spec_ref=spec,owner_source=owner,sys_executable='/synthetic/python')
        self.assertTrue(acceptance.validate_original_terminal(terminal,**kwargs))
        for field in ('root_observed_tool_exit_code','SSH_returncode','owner_process_exit_code'):
            for value in (False,None,1,255):
                bad=deepcopy(terminal);bad[field]=value
                with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                    acceptance.validate_original_terminal(bad,**kwargs)
        for mutation in ('no_exec','other_spec','other_phase','no_chunk','no_wait'):
            bad=deepcopy(terminal)
            if mutation=='no_exec':bad['command']=bad['command'][1:]
            elif mutation=='other_spec':bad['command'][5]='/synthetic/other/SPEC.json'
            elif mutation=='other_phase':bad['phase']='OTHER_PHASE'
            elif mutation=='no_chunk':bad['root_observed_tool_chunk_id']=''
            else:bad['original_exec_wait_observed']=False
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                acceptance.validate_original_terminal(bad,**kwargs)

    def test_external_post_is_independent_and_bound_to_same_attempt_births_and_group(self):
        terminal,_,_=terminal_fixture();post,owner=post_fixture(terminal)
        self.assertTrue(acceptance.validate_postcheck(post,terminal,owner))
        for mutation in ('other_attempt','other_phase','other_terminal','same_chunk','unknown_group','live_owner','bad_birth','post_nonzero'):
            bad=deepcopy(post)
            if mutation=='other_attempt':bad['attempt_id']='OTHER_ATTEMPT'
            elif mutation=='other_phase':bad['phase']='OTHER_PHASE'
            elif mutation=='other_terminal':bad['original_terminal']['command'][1]='/synthetic/other/python'
            elif mutation=='same_chunk':bad['root_observed_tool_chunk_id']=terminal['root_observed_tool_chunk_id']
            elif mutation=='unknown_group':bad['owned_group_scan_resolved']=False
            elif mutation=='live_owner':bad['owner_current_identity']=deepcopy(owner['owner_identity'])
            elif mutation=='bad_birth':bad['child_identity']['start_ticks']=True
            else:bad['root_observed_tool_exit_code']=255
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                acceptance.validate_postcheck(bad,terminal,owner)

    def test_actual_callback_bytes_are_hashed_not_a_self_reported_closure_flag(self):
        data=b'{"synthetic":true}\n';value=ref('/synthetic/report.json',data)
        self.assertEqual(acceptance.close_reference(value,lambda item:MemoryFile(item['path'],data),retain_bytes=True),data)
        with self.assertRaises(ValueError):
            acceptance.close_reference(value,lambda item:MemoryFile(item['path'],data+b'changed'))
        with self.assertRaises(ValueError):
            acceptance.close_reference(value,lambda item:MemoryFile('/synthetic/other.json',data))

    def test_full_raw_records_and_original_terminal_produce_only_scoped_shard_acceptance(self):
        proof,data=accept_synthetic()
        self.assertTrue(acceptance.validate_accepted_proof(proof,data['prediction']))
        self.assertTrue(proof['root_accepted']);self.assertFalse(proof['qualified']);self.assertFalse(proof['AP'])
        self.assertEqual(len(proof['source_refs']),21);self.assertEqual(len(proof['outer_source_refs']),22)
        self.assertEqual(proof['images'],47);self.assertEqual(proof['queries'],4700)
        bad=deepcopy(proof);bad['original_terminal']['owner_process_exit_code']=None
        with self.assertRaises(ValueError):acceptance.validate_accepted_proof(bad,data['prediction'])

    def test_invalid_original_terminal_is_rejected_before_scientific_payload_byte_access(self):
        data=synthetic_inputs();data['terminal']['SSH_returncode']=255
        with self.assertRaises(ValueError):accept_inputs(data)
        self.assertFalse(any(path.endswith(('.pth','.pt')) for path in data['visits']))

    def test_full_body_rejects_partial_candidates_bad_aliases_and_boolean_counters(self):
        proof,data=accept_synthetic()
        for mutation in ('cleanup_bool','missing_candidate','changed_S0','wrong_support','proof_bool'):
            bad=deepcopy(data['prediction'])
            if mutation=='cleanup_bool':bad['arms']['FULL']['cleanup']['allocated_after']=False
            elif mutation=='missing_candidate':bad['arms']['FULL']['predictions'].pop()
            elif mutation=='changed_S0':bad['arms']['FULL']['predictions'][0]['score']=0.9
            elif mutation=='wrong_support':bad['Omega_support'][0]['dataset_index']=1
            else:bad['arms']['PLAIN']['proofs'][0]['dataset_index']=False
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                acceptance.validate_accepted_proof(proof,bad)

    def test_proof_rejects_unknown_fields_and_preserves_exact_raw_JSON_hashes(self):
        proof,data=accept_synthetic()
        extra=deepcopy(proof);extra['self_certified']=True
        with self.assertRaises(ValueError):acceptance.validate_accepted_proof(extra,data['prediction'])
        changed=deepcopy(proof);changed['raw_JSON']['owner_receipt']='e30='
        with self.assertRaises(ValueError):acceptance.validate_accepted_proof(changed,data['prediction'])
        empty=ref('/synthetic/log',b'')
        self.assertEqual(acceptance.close_reference(empty,lambda item:MemoryFile(item['path'],b''),allow_empty=True),empty)
        with self.assertRaises(ValueError):acceptance.close_reference(empty,lambda item:MemoryFile(item['path'],b''))


if __name__=='__main__':unittest.main()
