"""New U69 metadata/entry wiring only. No scientific imports or real inputs."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import types
import unittest
from unittest.mock import patch

import entry


def ref(path,sha='a'*64,size=1):return dict(path=path,bytes=size,sha256=sha)


def fixture():
    roles=('training_state','training_entry','u36_helpers','u39_training_ops','shared_initialization',
        'full_legacy','full_dsconv','full_head','full_feature','plain_legacy','plain_feature','instance_adapter',
        'restore_state','portable_reader','image_geometry','prediction_core','encoder','mask_api',
        'population_contract','dev_reader','dev_rgb_reader')
    request=dict(worker_source=ref('/source/U68.py',entry.U68_SHA256),prediction_core_source=ref('/source/U66.py'),
        source_refs={k:ref('/source/'+k+'.py') for k in roles},
        library_sources={k:ref('/library/'+k+'.py',size=0 if k=='pycocotools' else 1) for k in
            ('torch','numpy','psutil','pycocotools','mask_extension')},
        endpoint_refs={k:ref('/endpoint/'+k+'.json') for k in ('qualified','spec','pair_commit','order')},
        checkpoint_refs={k:ref('/endpoint/'+k+'.pth') for k in ('PLAIN','FULL')},
        manifest_ref=ref('/input/MANIFEST.json'),shard_index=1,dataset_range=[47,94],output_directory='/output/WORKER')
    request['source_refs']['encoder']=ref('/source/encoder.py',
        '49c6dbc37434be372d8bb5b7e841a519491db8122f1eacef314a9e87f29a94e7',61452)
    entries=[dict(dataset_index=i,image_id=i+1000,query_ids=list(range(99,-1,-1)),
        package_ref=ref('/input/DEV_FULL100_'+str(i)+'.pt')) for i in range(187)]
    spec=dict(schema=entry.SCHEMA,operation=entry.OPERATION,phase='U69_synthetic_only',
        entry_source=ref('/source/entry.py'),owner_source=ref('/source/linux_owner.py'),request=request,
        limits=deepcopy(entry.LIMITS),output_directory='/output/ENTRY',owner_output_directory='/output/OWNER')
    sources=[spec['entry_source'],spec['owner_source'],request['worker_source'],request['prediction_core_source'],request['manifest_ref']]
    sources+=list(request['source_refs'].values())+list(request['library_sources'].values())
    sources+=list(request['endpoint_refs'].values())+list(request['checkpoint_refs'].values())
    sources += [row['package_ref'] for row in entries[47:94]]
    spec['sources']=sorted(sources,key=lambda r:r['path'])
    return spec,entries


class MemoryPath:
    def __init__(self,path,raw):self.path,self.raw=path,raw
    def __str__(self):return self.path
    def read_bytes(self):return self.raw


class MetadataCloser:
    def __init__(self):self.data={};self.reads=[]
    def add(self,path,text):
        raw=text.encode('utf-8');self.data[path]=raw
        return ref(path,hashlib.sha256(raw).hexdigest(),len(raw))
    def __call__(self,value):
        self.reads.append(value['path'])
        if value['path'] not in self.data:raise AssertionError('Unapproved payload read: '+value['path'])
        return MemoryPath(value['path'],self.data[value['path']])


class VirtualPath:
    def __init__(self,fs,path):self.fs,self.logical=fs,PurePosixPath(str(path))
    def __str__(self):return str(self.logical)
    def __eq__(self,other):return str(self)==str(other)
    def __truediv__(self,other):return self.fs.path(str(self.logical/other))
    @property
    def local(self):return self.fs.root.joinpath(*self.logical.parts[1:])
    @property
    def parent(self):return self.fs.path(str(self.logical.parent))
    def resolve(self,strict=False):
        if strict and not self.local.exists():raise FileNotFoundError(str(self))
        return self
    def is_absolute(self):return self.logical.is_absolute()
    def is_symlink(self):return False
    def is_dir(self):return self.local.is_dir()
    def exists(self):return self.local.exists()
    def mkdir(self):self.local.mkdir()
    def stat(self):return self.local.stat()
    def read_bytes(self):return self.local.read_bytes()
    def open(self,*args,**kwargs):return self.local.open(*args,**kwargs)


class VirtualFS:
    def __init__(self,root):self.root=Path(root).resolve();(self.root/'output').mkdir()
    def path(self,value):return VirtualPath(self,value)
    def write_json(self,path,value):
        obj=self.path(path);raw=(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n').encode()
        obj.local.write_bytes(raw)
        return ref(path,hashlib.sha256(raw).hexdigest(),len(raw))


class SyntheticOwnedControl(MetadataCloser):
    def __init__(self):super().__init__();self.checks=0
    def closed(self,value):
        if value['path'] in self.data:return self(value)
        self.reads.append(value['path']);return MemoryPath(value['path'],b'synthetic-input-not-decoded')
    def check(self):self.checks+=1
    def summary(self):return dict(schema='U69-Linux-child-control-observations-v1',samples=self.checks,
        peak_RSS_bytes=1024,minimum_available_RAM_bytes=32*1024**3,violations=[],elapsed_seconds=0.1)


SYNTHETIC_U68='''
import builtins
def direct_references(request,entries):
    builtins.U69_STUB_EVENTS.append('DIRECT_REFS')
    assert len(entries)==187 and entries[0]['query_ids']==list(range(99,-1,-1))
    result=[request[k] for k in ('worker_source','prediction_core_source','manifest_ref')]
    for key in ('source_refs','library_sources','endpoint_refs','checkpoint_refs'):result+=list(request[key].values())
    result += [row['package_ref'] for row in entries[request['dataset_range'][0]:request['dataset_range'][1]]]
    values={r['path']:r for r in result}
    return [values[p] for p in sorted(values)]
'''


class EntryWiring(unittest.TestCase):
    def owned_inputs(self,fs,fail=False):
        spec,entries=fixture();c=SyntheticOwnedControl()
        worker_ref=c.add('/source/U68.py',SYNTHETIC_U68)
        manifest_ref=c.add('/input/MANIFEST.json',json.dumps({'entries':entries}))
        spec['request']['worker_source'].update(worker_ref);spec['request']['manifest_ref'].update(manifest_ref)
        worker=types.ModuleType('_synthetic_worker');worker.__file__=worker_ref['path']
        def run_worker(control,request,*,report):
            report['synthetic_worker_was_called']=True
            control.worker_entered=True
            if fail:raise RuntimeError('synthetic U68 prediction failed')
            fs.path(request['output_directory']).mkdir();n=47
            prediction=dict(schema='U66-DGX-fixed50-DEV-shard-pending-v1',status='PREDICTIONS_COMPLETE_PENDING_ROOT_ACCEPTANCE',
                shard_index=1,dataset_range=[47,94],images=n,queries=100*n,fixed_epoch=50,updates_per_arm=37050,query_batch=4,
                endpoint=dict(schema='U58-U57-DGX-fixed-endpoint-metadata-v1',mode='FORMAL',formal_endpoint=True,
                    cursor=dict(global_updates=37050,completed_epochs=50,within_epoch=0),checkpoint_refs=request['checkpoint_refs']),
                source_refs=request['source_refs'],arms={a:{'status':'PREDICTIONS_COMPLETE_PENDING_ROOT_ACCEPTANCE'} for a in ('PLAIN','FULL')},
                qualified=False,AP=False,DEV=True,DEV_technical_input=True,TEST=False,new_training=False,
                formal_training_budget_credit=0,whole_population_predicted=False,DEV_metrics_computed=False,
                GT_used_for_prediction=False,GT_rasterization_executed=False,remaining_model_objects=0,
                cleanup={'allocated_after':0,'reserved_after':0},model_BN_Torch_RNG_unchanged=True,
                original_full_S0_class_order_preserved=True,source_bytes_rechecked_after=True)
            prediction_ref=fs.write_json(request['output_directory']+'/U66_SHARD.json',prediction)
            report.update(schema='U68-single-shard-worker-report-v1',status='COMPLETE_PENDING_ROOT_ACCEPTANCE',
                qualified=False,AP=False,DEV=True,DEV_technical_input=True,TEST=False,new_training=False,
                formal_training_budget_credit=0,ROOT_actual_terminal_self_certified=False,automatic_retry=False,
                cleanup_errors=[],shard_index=1,dataset_range=[47,94],worker_source=request['worker_source'],
                prediction_core_source=request['prediction_core_source'],prediction_ref=prediction_ref,
                direct_input_refs=[r for r in spec['sources'] if r not in (spec['entry_source'],spec['owner_source'])],
                source_aliases_restored=True,source_and_metadata_bytes_rechecked=True,
                outer_runtime_states_restored=True,elapsed_seconds=0.1)
            fs.write_json(request['output_directory']+'/REPORT.json',report)
            return report
        worker.run_worker=run_worker
        return spec,c,worker
    def test_metadata_gate_is_pure_and_keeps_original_limits_and_three_outputs(self):
        spec,entries=fixture()
        with patch('builtins.open',side_effect=AssertionError('pure metadata must not open files')):
            admitted=entry.validate_envelope(spec)
        self.assertEqual(admitted,{r['path']:r for r in spec['sources']})
        for change in ('limit','overlap','unknown'):
            bad=deepcopy(spec)
            if change=='limit':bad['limits']['threads']=3
            elif change=='overlap':bad['request']['output_directory']='/output/ENTRY/subdir'
            else:bad['unasked_owner_field']=True
            with self.assertRaises(ValueError):entry.validate_envelope(bad)

    def test_exact_union_uses_worker_metadata_before_any_other_payload_is_closed(self):
        spec,entries=fixture();close=MetadataCloser()
        worker_ref=close.add('/source/U68.py',SYNTHETIC_U68)
        manifest_ref=close.add('/input/MANIFEST.json',json.dumps({'entries':entries}))
        for key,value in (('worker_source',worker_ref),('manifest_ref',manifest_ref)):
            spec['request'][key].update(value)
        events=[]
        with patch.object(entry,'U68_SHA256',worker_ref['sha256']),patch('builtins.U69_STUB_EVENTS',events,create=True):
            admitted=entry.close_source_union(spec,close)
            self.assertEqual(admitted,{r['path']:r for r in spec['sources']})
            bad=deepcopy(spec);bad['sources'].append(ref('/unadmitted/extra_payload.pt'))
            bad['sources'].sort(key=lambda r:r['path'])
            with self.assertRaises(ValueError):entry.close_source_union(bad,close)
        self.assertEqual(set(close.reads),{'/source/U68.py','/input/MANIFEST.json'})
        self.assertEqual(events,['DIRECT_REFS','DIRECT_REFS'])

    def test_only_root_traced_original_encoder_bytes_are_admitted(self):
        spec,entries=fixture()
        for field,value in (('sha256','b'*64),('bytes',61451)):
            bad=deepcopy(spec);bad['request']['source_refs']['encoder'][field]=value
            with self.assertRaises(ValueError):entry.validate_envelope(bad)

    def test_owned_entry_calls_worker_and_verifies_its_real_json_outputs(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as root:
            fs=VirtualFS(root);spec,c,worker=self.owned_inputs(fs);report={};events=[]
            with patch.object(entry,'U68_SHA256',spec['request']['worker_source']['sha256']),\
                 patch.object(entry,'Path',fs.path,create=True),patch('builtins.U69_STUB_EVENTS',events,create=True):
                result=entry.execute_owned(spec,ref('/metadata/SPEC.json'),c,worker,report=report)
            self.assertIs(result,report)
            self.assertEqual(report['status'],'COMPLETE_PENDING_ROOT_ACCEPTANCE')
            self.assertTrue(report['worker_report']['synthetic_worker_was_called'])
            self.assertEqual(json.loads(fs.path('/output/ENTRY/REPORT.json').read_bytes()),report)
            self.assertEqual(report['prediction_ref'],report['worker_report']['prediction_ref'])
            self.assertFalse(report['qualified']);self.assertFalse(report['AP'])
            self.assertEqual(report['formal_training_budget_credit'],0)

    def test_cli_bootstrap_loads_explicit_owner_and_worker_without_starting_an_owner(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as root:
            fs=VirtualFS(root);spec,c,worker=self.owned_inputs(fs)
            raw=SYNTHETIC_U68+"\ndef run_worker(c,request,*,report):\n    return builtins.U69_STUB_CALLBACK(c,request,report=report)\n"
            spec['request']['worker_source'].update(c.add('/source/U68.py',raw))
            owner_code='import builtins\nSCHEMA='+repr(entry.SCHEMA)+'\nOPERATION='+repr(entry.OPERATION)+\
                '\nLIMITS='+repr(entry.LIMITS)+'\ndef make_control(spec,psutil):\n    return builtins.U69_STUB_CONTROL\n'
            spec['owner_source'].update(c.add('/source/linux_owner.py',owner_code))
            spec['entry_source'].update(c.add('/source/entry.py','# synthetic entry source identity'))
            spec['request']['library_sources']['psutil'].update(c.add('/library/psutil.py','# synthetic selected psutil'))
            for path,raw in c.data.items():
                target=fs.path(path).local;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
            (fs.root/'metadata').mkdir();spec_ref=fs.write_json('/metadata/SPEC.json',spec)
            psutil=types.ModuleType('psutil');psutil.__file__='/library/psutil.py';psutil.__version__='6.1.1';c.psutil=psutil
            importer=__import__('importlib').import_module
            with patch.object(entry,'U68_SHA256',spec['request']['worker_source']['sha256']),patch.object(entry,'Path',fs.path),\
                 patch.object(entry,'__file__',spec['entry_source']['path']),\
                 patch('platform.system',return_value='Linux'),patch('platform.machine',return_value='aarch64'),\
                 patch('platform.python_version',return_value='3.12.14'),\
                 patch.dict('os.environ',dict(CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8',
                    OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')),\
                 patch('builtins.U69_STUB_EVENTS',[],create=True),patch('builtins.U69_STUB_CONTROL',c,create=True),\
                 patch('builtins.U69_STUB_CALLBACK',worker.run_worker,create=True),\
                 patch('importlib.import_module',side_effect=lambda name:psutil if name=='psutil' else importer(name)):
                with self.assertRaises(ValueError):entry.main(['--spec','/metadata/SPEC.json','--spec-sha','0'*64])
                self.assertFalse(fs.path('/output/ENTRY').exists())
                result=entry.main(['--spec','/metadata/SPEC.json','--spec-sha',spec_ref['sha256']])
            self.assertEqual(result['status'],'COMPLETE_PENDING_ROOT_ACCEPTANCE')
            self.assertFalse(fs.path('/output/OWNER').exists())

    def test_worker_exception_keeps_failed_entry_and_visible_trace_without_qualification(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as root:
            fs=VirtualFS(root);spec,c,worker=self.owned_inputs(fs,fail=True);report={}
            original_check=c.check;original_summary=c.summary
            def resource_failure():
                original_check()
                if getattr(c,'worker_entered',False):raise ValueError('synthetic resource violation')
            def failed_summary():
                value=original_summary();value['violations']=['RSS'];value['peak_RSS_bytes']=17*1024**3
                return value
            c.check=resource_failure;c.summary=failed_summary
            with patch.object(entry,'U68_SHA256',spec['request']['worker_source']['sha256']),\
                 patch.object(entry,'Path',fs.path),patch('builtins.U69_STUB_EVENTS',[],create=True):
                with self.assertRaisesRegex(RuntimeError,'synthetic U68 prediction failed'):
                    entry.execute_owned(spec,ref('/metadata/SPEC.json'),c,worker,report=report)
                with self.assertRaises(ValueError):entry.completed_entry(report,spec,ref('/metadata/SPEC.json'))
            self.assertEqual(report['status'],'FAILED')
            self.assertTrue(report['worker_report']['synthetic_worker_was_called'])
            self.assertIn('synthetic U68 prediction failed',report['error']['traceback'])
            self.assertEqual(report['resource_observations']['violations'],['RSS'])
            self.assertFalse(report['flags']['child_resources_within_limits'])
            self.assertEqual(json.loads(fs.path('/output/ENTRY/REPORT.json').read_bytes()),report)
            self.assertFalse(report['qualified']);self.assertFalse(report['AP'])


if __name__=='__main__':unittest.main()
