"""Invented JSON only. These fixtures are never real U65 execution evidence."""
from copy import deepcopy
import hashlib
import json
import unittest
import acceptance as subject


def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def ref(path,raw=b'invented metadata fixture'):
    return dict(path=path,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def fixture():
    limits=dict(allowed_cpus=[0,1],threads=2,maximum_RSS_bytes=16*1024**3,
        minimum_available_RAM_bytes=16*1024**3,deadline_seconds=1800)
    environment=dict(python='3.12.14',torch='2.10.0+cu130',numpy='1.26.4',system='Linux',machine='aarch64')
    pre=dict(schema='U65-DEV187-CPU-reader-validation-prelock-v1',environment=environment,
        dataset_indices=list(range(187)),images=187,queries=18700,CUDA_visible_devices='',limits=limits,
        no_new_training=True,no_forward=True,no_matching=True,no_AP=True,no_DEV=True,no_TEST=True,
        no_GT_rasterization=True,DEV_technical_input=True)
    packages=[ref('/invented/data/DEV_FULL100_%d.pt'%i) for i in range(187)]
    entries=[dict(dataset_index=i,image_id=1000+i,query_ids=list(reversed(range(100))),package_ref=p)
        for i,p in enumerate(packages)]
    manifest=dict(schema='INVENTED-NOT-ACTUAL-MANIFEST',entries=entries)
    spec=dict(schema='ROOT-U65-DGX-full-DEV-CPU-reader-validation-spec-v1',
        operation='CPU_U65_DGX_DEV_FULL187_PORTABLE_READER_VALIDATION',phase='INVENTED_ONLY',
        entry_source=ref('/invented/source/entry.py'),body_source=ref('/invented/source/body.py'),
        owner_source=ref('/invented/source/owner.py'),
        source_refs={role:ref('/invented/source/'+role+'.py')
            for role in ('portable_reader','dev_reader','dev_rgb_reader','image_geometry')},
        library_sources={role:ref('/invented/library/'+role+'.py') for role in ('psutil','torch','numpy')},
        manifest_ref=ref('/invented/MANIFEST.json',encoded(manifest)),package_refs=packages,prelock=pre,
        output_directory='/invented/worker',owner_output_directory='/invented/owner')
    direct=[spec[k] for k in ('entry_source','body_source','owner_source','manifest_ref')]
    direct+=list(spec['source_refs'].values())+list(spec['library_sources'].values())+packages
    spec['sources']=sorted(direct,key=lambda r:r['path'])
    records=[dict(dataset_index=e['dataset_index'],image_id=e['image_id'],qids=e['query_ids']) for e in entries]
    audits=[dict(dataset_index=i,image_id=e['image_id'],queries=100,status='VALIDATED',
        source_package=e['package_ref'],original_record_sha256=digest(records[i]),
        native_predictions_metadata_sha256=hashlib.sha256(('invented-native-%d'%i).encode()).hexdigest(),
        query_ids_metadata_sha256=digest(e['query_ids']),CUDA_initialized=False,
        returned_tensor_objects_released=True) for i,e in enumerate(entries)]
    audit_ref=ref('/invented/worker/IMAGE_CHECKS.jsonl',b''.join(encoded(row)+b'\n' for row in audits))
    body_flags=('CUDA_uninitialized_before','CUDA_uninitialized_after','metadata_population_verified',
        'all_field_proofs_verified','all_returned_tensor_objects_released')
    body_false=('qualified','new_training','AP','DEV','TEST','main_training','model_constructed',
        'model_forward_executed','optimizer_created','GT_matching_recomputed','GT_rasterization_executed')
    body=dict(schema='U65-DEV187-CPU-reader-validation-body-report-v1',status='COMPLETE_PENDING_ROOT',
        flags=dict.fromkeys(body_flags,True),attempted_images=187,validated_images=187,released_images=187,
        recorded_images=187,queries=18700,reader_visits=list(range(187)),reader_failed_loads=0,
        remaining_returned_tensor_objects=0,
        original_metadata_population=dict(images=187,queries=18700,tensor_population_revalidated=False),
        original_annotation_records_parsed=False,raw_provenance_inputs_revalidated=False,
        manifest_sha256=digest(manifest),record_metadata_sha256=digest(records),private_scalar_audit_only=True,
        DEV_technical_input=True,formal_training_budget_credit=0,cost=dict(elapsed_seconds=9),
        **dict.fromkeys(body_false,False))
    false_fields=body_false+('pilot_only','U40_frontier_modified','innovation_established','speedup_established')
    worker_flags=('source_bytes_verified','CUDA_uninitialized_before','CUDA_uninitialized_after',
        'source_unchanged_after','module_aliases_restored','caller_CPU_RNG_preserved','caller_NumPy_RNG_preserved',
        'caller_Python_RNG_preserved','dtype_grad_threads_preserved','child_resources_within_limits')
    observations=dict(schema='U65-Linux-child-control-observations-v1',samples=30,violations=[],
        elapsed_seconds=10,peak_RSS_bytes=1024**3,minimum_available_RAM_bytes=20*1024**3)
    spec_ref=ref('/invented/SPEC.json',encoded(spec))
    worker=dict(schema='U65-DEV187-CPU-reader-validation-worker-report-v1',status='COMPLETE',
        operation=spec['operation'],phase=spec['phase'],spec_ref=spec_ref,actual_environment=environment,
        flags=dict.fromkeys(worker_flags,True),body_report=body,cleanup_errors=[],formal_training_budget_credit=0,
        validated_images=187,queries=18700,audit_records_written=187,private_audit_ref=audit_ref,
        DEV_technical_input=True,full_binary_closure_claimed=False,Linux_original_terminal_qualification_claimed=False,
        resource_observations=observations,cost=dict(elapsed_seconds=11),
        actual_library_paths={k:v['path'] for k,v in spec['library_sources'].items()},**dict.fromkeys(false_fields,False))
    for key in ('entry_source','body_source','owner_source','source_refs','library_sources','manifest_ref','package_refs','prelock'):
        worker[key]=spec[key]
    owner_identity=dict(pid=910,state='R',parent_pid=900,process_group=900,session=900,start_ticks=100)
    child_identity=dict(pid=911,state='R',parent_pid=910,process_group=911,session=911,start_ticks=101)
    logs={name:ref('/invented/owner/'+name,b'') for name in ('stdout.log','stderr.log')}
    worker_ref=ref('/invented/worker/REPORT.json',encoded(worker))
    owner=dict(schema='U65-Linux-owned-full-DEV-CPU-reader-validation-receipt-v1',
        status='OWNED_CHILD_COMPLETE_PENDING_ROOT_REVIEW',operation=spec['operation'],phase=spec['phase'],
        spec_ref=spec_ref,worker_report_ref=worker_ref,private_audit_ref=audit_ref,logs=logs,
        owner_pid=910,owner_identity=owner_identity,child_identity=child_identity,
        returncode=0,reaped=True,original_birth_absent=True,final_group_scan_resolved=True,
        all_owned_processes_absent=True,CPU_only=True,CPU_validation_complete=True,child_launched=True,
        child_alive=False,termination_requested=False,remaining_owned_group=[],violations=[],signals=[],
        CUDA_visible_devices='',GPU_allocator_contract_applicable=False,DEV_technical_input=True,
        validated_images=187,queries=18700,audit_records_written=187,formal_training_budget_credit=0,
        child_resource_observations=observations,worker_cost=worker['cost'],elapsed_seconds=12,
        peak_group_RSS_bytes=1024**3,minimum_available_RAM_bytes=20*1024**3,
        controller_returning=True,automatic_retry=False,**dict.fromkeys(false_fields,False))
    owner_ref=ref('/invented/owner/OWNER_RECEIPT.json',encoded(owner))
    terminal=dict(root_observed_tool_exit_code=0,SSH_returncode=0,worker_returncode=0,reaped=True,
        owner_original_birth_absent=True,child_original_birth_absent=True,all_owned_processes_absent=True,
        source_and_logs_byte_verified=True)
    post=dict(schema='Root-U65-explicit-byte-postcheck-v1',spec_ref=spec_ref,worker_report_ref=worker_ref,
        owner_receipt_ref=owner_ref,private_audit_ref=audit_ref,logs=logs,
        verified_source_refs=spec['sources'],owner_identity=owner_identity,child_identity=child_identity,
        source_and_logs_byte_verified=True,manifest=manifest,private_audit_records=audits)
    post['verified_output_refs']=sorted([spec_ref,worker_ref,owner_ref,audit_ref]+list(logs.values()),key=lambda r:r['path'])
    return spec,terminal,owner,worker,post


class MetadataTests(unittest.TestCase):
    def test_full187_technical_input_metadata_accepts_without_changing_inputs(self):
        args=fixture();before=deepcopy(args)
        result=subject.accept_validation(*args)
        self.assertEqual(result['status'],'ROOT_ACCEPTED_ALL187_CPU_READER_VALIDATION')
        self.assertEqual((result['images'],result['queries'],result['failed_loads']),(187,18700,0))
        self.assertIs(result['DEV_technical_input'],True)
        self.assertIs(result['AP'],False)
        self.assertEqual(len(args[0]['sources']),198)
        self.assertEqual(args,before)

    def test_old_target_read_claim_cannot_enter_new_technical_input_acceptance(self):
        args=fixture()
        args[3]['body_report']['FIT_GT_tensors_read']=True
        with self.assertRaises(ValueError):subject.accept_validation(*args)

    def test_partial_query_audit_or_release_is_rejected_even_with_complete_label(self):
        def set_body(args,key,value):args[3]['body_report'][key]=value
        changes=[lambda a:set_body(a,'validated_images',186),lambda a:set_body(a,'queries',18699),
            lambda a:set_body(a,'reader_failed_loads',1),lambda a:set_body(a,'recorded_images',186),
            lambda a:set_body(a,'remaining_returned_tensor_objects',1),
            lambda a:set_body(a,'raw_provenance_inputs_revalidated',True),
            lambda a:set_body(a,'original_annotation_records_parsed',True),
            lambda a:a[4]['private_audit_records'].pop(),
            lambda a:a[4]['private_audit_records'][1].update(dataset_index=0),
            lambda a:a[4]['private_audit_records'][0].update(returned_tensor_objects_released=False),
            lambda a:a[4]['private_audit_records'][0].update(native_predictions_metadata_sha256='f'*64),
            lambda a:a[4]['manifest']['entries'][0]['query_ids'].reverse()]
        for change in changes:
            args=fixture();change(args)
            with self.subTest(change=changes.index(change)):
                with self.assertRaises(ValueError):subject.accept_validation(*args)

    def test_real_external_terminal_and_root_byte_postcheck_cannot_be_replaced_by_self_claims(self):
        for field,value in (('root_observed_tool_exit_code',None),('SSH_returncode',False),
            ('worker_returncode',1),('reaped',False),('owner_original_birth_absent',False),
            ('child_original_birth_absent',False),('all_owned_processes_absent',False),
            ('source_and_logs_byte_verified',False)):
            args=fixture();args[1][field]=value
            with self.subTest(field=field):
                with self.assertRaises(ValueError):subject.accept_validation(*args)
        args=fixture();args[4]['source_and_logs_byte_verified']=False
        with self.assertRaises(ValueError):subject.accept_validation(*args)
        args=fixture();args[2]['child_identity']['start_ticks']+=1
        args[4]['child_identity']=deepcopy(args[4]['child_identity'])
        args[4]['child_identity']['start_ticks']-=1
        with self.assertRaises(ValueError):subject.accept_validation(*args)

    def test_direct_source_closure_and_technical_scope_have_no_alias_or_model_credit(self):
        changes=[lambda a:a[0]['sources'].pop(),
            lambda a:a[0]['source_refs'].update(dev_reader=a[0]['source_refs']['dev_rgb_reader']),
            lambda a:a[3].update(queries=18701),
            lambda a:a[3].update(DEV_technical_input=False),
            lambda a:a[3].update(AP=True),
            lambda a:a[3]['body_report'].update(GT_rasterization_executed=True),
            lambda a:a[3]['body_report'].update(formal_training_budget_credit=False),
            lambda a:a[3]['body_report']['flags'].update(all_field_proofs_verified=False)]
        for change in changes:
            args=fixture();change(args)
            with self.subTest(change=changes.index(change)):
                with self.assertRaises(ValueError):subject.accept_validation(*args)


if __name__=='__main__':unittest.main()
