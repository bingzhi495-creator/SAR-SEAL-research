"""Finite U65 stdlib metadata delta tests; no main/owner/ML/data execution."""
import ast
from copy import deepcopy
import hashlib
from pathlib import Path
import types
import unittest


HERE=Path(__file__).resolve().parent
OLD=Path('E:/SAR/seal/research_round_20260925_global/code/U54_FULL_FIT_VALIDATION_DRAFT_20261018_01')
PINS={
    'portable_reader':'8fabc800e4d36e2da3988e42acf88058c80790f9c8b3d664a1a4fa9232233473',
    'dev_reader':'daabec8388e23d9ff646ae78eba69ba2168099a72971578a8d34cef04f1268c0',
    'dev_rgb_reader':'b91375e5366da3f74272ce811f453be1379b46a4bb96103f9d8039a4998a1204',
    'image_geometry':'65db107de80c5f1d4dade7cb2ffd3f241fd6bf7f9b233cb1aa9c90cbfd00abf7',
}


def module(name):
    path=HERE/(name+'.py');value=types.ModuleType('_u65_metadata_'+name)
    value.__file__=str(path)
    exec(compile(path.read_bytes(),str(path),'exec'),value.__dict__)
    return value


def ref(name,sha='a'*64,bytes=1):
    return dict(path='/u65-unexecuted/'+name,bytes=bytes,sha256=sha)


def spec_fixture(entry):
    sources={key:ref(key+'.py',pin) for key,pin in PINS.items()}
    libs={key:ref(key+'.py') for key in ('psutil','torch','numpy')}
    packages=[ref('DEV_FULL100_'+str(i)+'.pt') for i in range(187)]
    pre=dict(schema='U65-DEV187-CPU-reader-validation-prelock-v1',
        environment=dict(python='3.12.14',torch='2.10.0+cu130',numpy='1.26.4',system='Linux',machine='aarch64'),
        dataset_indices=list(range(187)),images=187,queries=18700,CUDA_visible_devices='',
        limits=dict(allowed_cpus=[0,1],threads=2,maximum_RSS_bytes=16*1024**3,
                    minimum_available_RAM_bytes=16*1024**3,deadline_seconds=1800),
        no_new_training=True,no_forward=True,no_matching=True,no_AP=True,no_DEV=True,no_TEST=True,
        no_GT_rasterization=True,DEV_technical_input=True)
    spec=dict(schema='ROOT-U65-DGX-full-DEV-CPU-reader-validation-spec-v1',
        operation='CPU_U65_DGX_DEV_FULL187_PORTABLE_READER_VALIDATION',phase='UNEXECUTED_METADATA_ONLY',
        entry_source=ref('entry.py'),body_source=ref('validation_body.py',entry.BODY_SHA256,entry.BODY_BYTES),
        owner_source=ref('linux_owner.py'),source_refs=sources,library_sources=libs,
        manifest_ref=ref('MANIFEST.json'),package_refs=packages,prelock=pre,
        output_directory='/u65-worker-unexecuted',owner_output_directory='/u65-owner-unexecuted')
    spec['sources']=entry.canonical_sources([spec[k] for k in ('entry_source','body_source','owner_source','manifest_ref')]
        +list(sources.values())+list(libs.values())+packages)
    return spec


def report_fixture(entry,spec,spec_ref):
    body=dict(schema='U65-DEV187-CPU-reader-validation-body-report-v1',status='COMPLETE_PENDING_ROOT',
        flags={k:True for k in ('CUDA_uninitialized_before','CUDA_uninitialized_after',
              'metadata_population_verified','all_field_proofs_verified','all_returned_tensor_objects_released')},
        attempted_images=187,validated_images=187,released_images=187,recorded_images=187,queries=18700,
        reader_visits=list(range(187)),reader_failed_loads=0,remaining_returned_tensor_objects=0,
        formal_training_budget_credit=0,DEV_technical_input=True,
        original_metadata_population=dict(images=187,queries=18700,tensor_population_revalidated=False),
        original_annotation_records_parsed=False,raw_provenance_inputs_revalidated=False)
    for k in ('qualified','new_training','AP','DEV','TEST','main_training','model_constructed',
              'model_forward_executed','optimizer_created','GT_matching_recomputed','GT_rasterization_executed'):
        body[k]=False
    report=dict(schema='U65-DEV187-CPU-reader-validation-worker-report-v1',status='COMPLETE',
        operation=spec['operation'],phase=spec['phase'],spec_ref=spec_ref,
        flags={k:True for k in entry.SUCCESS_FLAGS},body_report=body,
        formal_training_budget_credit=0,DEV_technical_input=True,validated_images=187,queries=18700,
        audit_records_written=187,private_audit_ref=ref('dummy-audit'))
    for k in entry.FALSE_FIELDS:report[k]=False
    for k in ('entry_source','body_source','owner_source','source_refs','library_sources','manifest_ref','package_refs','prelock'):
        report[k]=deepcopy(spec[k])
    report['private_audit_ref']['path']=spec['output_directory']+'/IMAGE_CHECKS.jsonl'
    report['resource_observations']=dict(schema='U65-Linux-child-control-observations-v1',samples=1,
        violations=[],elapsed_seconds=1.0,peak_RSS_bytes=1,minimum_available_RAM_bytes=16*1024**3)
    report['cost']=dict(elapsed_seconds=1.0)
    return report


class MetadataDelta(unittest.TestCase):
    def test_only_full187_dev_reader_scope_is_admitted(self):
        entry,owner=module('entry'),module('linux_owner');spec=spec_fixture(entry)
        raw=(HERE/'validation_body.py').read_bytes()
        self.assertEqual((len(raw),hashlib.sha256(raw).hexdigest()),(entry.BODY_BYTES,entry.BODY_SHA256))
        self.assertEqual(set(entry.source_table(spec)),{r['path'] for r in spec['sources']})
        owner.validate_spec(spec)
        for change in ('missing_package','extra_role','hidden_source','old_prelock'):
            bad=deepcopy(spec)
            if change=='missing_package':bad['package_refs'].pop()
            if change=='extra_role':bad['source_refs']['teacher']=ref('teacher.py')
            if change=='hidden_source':bad['sources'].append(ref('hidden.pt'))
            if change=='old_prelock':bad['prelock']['GT']=1661
            with self.subTest(change=change):
                with self.assertRaises(ValueError):owner.validate_spec(bad)
        bad=deepcopy(spec);bad['source_refs']['portable_reader']['sha256']='b'*64
        with self.assertRaises(ValueError):entry.source_table(bad)

    def test_manifest_joins_original_entries_without_reading_historical_inputs(self):
        entry=module('entry');spec=spec_fixture(entry)
        manifest=dict(entries=[dict(dataset_index=i,image_id=1000+i,query_ids=list(range(100)),
                                    package_ref=deepcopy(r)) for i,r in enumerate(spec['package_refs'])],
                      historical_metadata={'unopened_input':'C:/historical/provenance-only'})
        original=deepcopy(manifest)
        entry.validate_manifest(manifest,spec['package_refs'])
        self.assertEqual(manifest,original)
        for change in ('old_rows','reordered_refs','partial'):
            bad=deepcopy(manifest)
            if change=='old_rows':bad['rows']=bad.pop('entries')
            if change=='reordered_refs':bad['entries'][0]['package_ref']=bad['entries'][1]['package_ref']
            if change=='partial':bad['entries'].pop()
            with self.subTest(change=change):
                with self.assertRaises(ValueError):entry.validate_manifest(bad,spec['package_refs'])

    def test_completion_requires_18700_queries_and_technical_only_scope(self):
        entry,owner=module('entry'),module('linux_owner');spec=spec_fixture(entry);sr=ref('SPEC.json')
        report=report_fixture(entry,spec,sr)
        self.assertIs(owner.completed_validation(report,spec,sr),report['body_report'])
        for change in ('partial_queries','missing_visit','failed_load','evaluation','not_technical',
                       'body_not_technical','GT_decode','raw_revalidation'):
            bad=deepcopy(report);body=bad['body_report']
            if change=='partial_queries':body['queries']=18699
            if change=='missing_visit':body['reader_visits'].pop()
            if change=='failed_load':body['reader_failed_loads']=1
            if change=='evaluation':bad['DEV']=True
            if change=='not_technical':bad['DEV_technical_input']=False
            if change=='body_not_technical':body['DEV_technical_input']=False
            if change=='GT_decode':body['original_annotation_records_parsed']=True
            if change=='raw_revalidation':body['raw_provenance_inputs_revalidated']=True
            with self.subTest(change=change):
                with self.assertRaises(ValueError):owner.completed_validation(bad,spec,sr)

    def test_original_owner_process_guards_and_only_two_executed_reader_sources(self):
        original=ast.parse((OLD/'linux_owner.py').read_text(encoding='utf-8'))
        adapted=ast.parse((HERE/'linux_owner.py').read_text(encoding='utf-8').replace('U65','U54'))
        old={n.name:n for n in original.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        new={n.name:n for n in adapted.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        unchanged=('FileAllowlist','ChildControl','resource_violations','make_control','parse_proc_stat',
                   'proc_identity','birth_absent','terminal_success','_json_write','_group_members',
                   '_kill_owned_group','_prepare_child','main')
        for name in unchanged:
            with self.subTest(name=name):
                self.assertEqual(ast.dump(old[name],include_attributes=False),ast.dump(new[name],include_attributes=False))
        entry=ast.parse((HERE/'entry.py').read_text(encoding='utf-8'))
        main=next(n for n in entry.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        loaders=[n for n in ast.walk(main) if isinstance(n,ast.DictComp)
                 and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name)
                 and n.value.func.id=='load_closed']
        self.assertEqual(len(loaders),1)
        self.assertEqual(ast.literal_eval(loaders[0].generators[0].iter),('portable_reader','image_geometry'))


if __name__=='__main__':unittest.main(verbosity=2)
