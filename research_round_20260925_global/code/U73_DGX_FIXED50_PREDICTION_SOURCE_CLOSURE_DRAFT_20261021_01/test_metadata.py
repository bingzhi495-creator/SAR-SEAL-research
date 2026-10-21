"""New closure regressions only: synthetic refs, JSON and stdlib source stubs."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import worker

FIT_SHA='b65ed9ecb7a9a704899f3c75983b2cb9676e26b42d87f65336cfbcb16bd94557'
OLD_ROLES=tuple(role for role,_ in worker.LOAD_ORDER)+('mask_api','dev_reader','dev_rgb_reader')


def ref(path,sha='a'*64,size=1):
    return dict(path=path,bytes=size,sha256=sha)


def fixture(target):
    sources={role:ref('/synthetic/sources/'+role+'.py',worker.SOURCE_PINS.get(role,'e'*64)) for role in OLD_ROLES}
    sources['fit_portable_reader']=ref('/synthetic/original_training/portable_reader.py',FIT_SHA,19618)
    entries=[dict(dataset_index=i,image_id=1000+i,query_ids=list(range(99,-1,-1)),
        package_ref=ref('/synthetic/packages/DEV_FULL100_'+str(i)+'.pt')) for i in range(187)]
    request=dict(worker_source=ref('/synthetic/U73/worker.py'),
        prediction_core_source=ref('/synthetic/U66/prediction_core.py',worker.CORE_SHA256),
        source_refs=sources,
        library_sources={role:ref('/synthetic/lib/'+role+'.py') for role in worker.LIBRARIES},
        endpoint_refs={role:ref('/synthetic/endpoint/'+role+'.json') for role in ('qualified','spec','pair_commit','order')},
        checkpoint_refs={arm:ref('/synthetic/endpoint/'+arm+'.pth') for arm in ('PLAIN','FULL')},
        manifest_ref=ref('/synthetic/DEV_MANIFEST.json'),shard_index=0,dataset_range=[0,47],output_directory=str(target))
    original={role:deepcopy(sources[role]) for role in worker.RESTORE_ROLES}
    original.update(image_geometry=deepcopy(sources['image_geometry']),portable_reader=deepcopy(sources['fit_portable_reader']))
    return request,entries,dict(restore_source_refs=original,checkpoint_refs=deepcopy(request['checkpoint_refs']))


class MemoryPath:
    def __init__(self,path,raw):self.path,self.raw=path,raw
    def __str__(self):return self.path
    def read_bytes(self):return self.raw


class SyntheticControl:
    def __init__(self):self.objects={};self.closed_paths=[]
    def add(self,path,text):
        data=text.encode('utf-8');self.objects[path]=data
        return ref(path,hashlib.sha256(data).hexdigest(),len(data))
    def closed(self,value):
        self.closed_paths.append(value['path']);data=self.objects[value['path']]
        if len(data)!=value['bytes'] or hashlib.sha256(data).hexdigest()!=value['sha256']:raise ValueError('synthetic bytes mismatch')
        return MemoryPath(value['path'],data)
    def check(self):pass


def source_fixture(c):
    text={role:'# harmless synthetic '+role for role in OLD_ROLES}
    text.update(full_dsconv='class DSConv:pass',full_head='class CoordConv:pass',
        full_legacy='from dsconv import DSConv\nfrom head import CoordConv\n',
        dev_reader="raise AssertionError('historical reader must not execute')",
        dev_rgb_reader="raise AssertionError('historical RGB reader must not execute')",
        fit_portable_reader="raise AssertionError('FIT must stay byte-only')\n")
    text['fit_portable_reader']+='\n#'+'x'*(19618-len(text['fit_portable_reader'])-2)
    sources={role:c.add('/synthetic/sources/'+role+'.py',data) for role,data in text.items()}
    core=c.add('/synthetic/u66.py','def predict_shard(*args,**kwargs):pass\n')
    mask=types.ModuleType('pycocotools.mask');mask.__file__=sources['mask_api']['path']
    return sources,core,mask


def run_fixture(c,target):
    request,entries,_=fixture(target);sources,core,mask=source_fixture(c)
    restore='''
ORIGINAL_12_GATE_CALLED=False
def validate_final_endpoint(training,entry,metadata):
    assert metadata['qualified']['synthetic_only'] is True
    return {'restore_source_refs':metadata['spec']['original14'],
        'checkpoint_refs':metadata['pair_commit']['checkpoint_refs']}
def validate_restore_sources(endpoint,refs):
    global ORIGINAL_12_GATE_CALLED
    assert len(refs)==12
    assert all(value==endpoint['restore_source_refs'][role] for role,value in refs.items())
    ORIGINAL_12_GATE_CALLED=True
'''
    sources['restore_state']=c.add('/synthetic/sources/restore_state.py',restore)
    sources['portable_reader']=c.add('/synthetic/sources/portable_reader.py',
        'def validate_entries(entries):\n    assert len(entries)==187\n')
    core=c.add('/synthetic/u66.py','''
def predict_shard(torch,np,c,*,endpoint_metadata,entries,modules,source_refs,shard_index,dataset_range,report):
    assert len(modules)==19 and len(source_refs)==21
    assert 'fit_portable_reader' not in source_refs and 'fit_portable_reader' not in modules
    assert modules['restore_state'].ORIGINAL_12_GATE_CALLED is True
    report.update(schema='U66-DGX-fixed50-DEV-shard-pending-v1',
        status='PREDICTIONS_COMPLETE_PENDING_ROOT_ACCEPTANCE',qualified=False,AP=False,DEV=True,
        TEST=False,new_training=False,formal_training_budget_credit=0,shard_index=shard_index,
        dataset_range=dataset_range,source_refs=source_refs,synthetic_exact21_core_called=True)
    return report
''')
    original={role:deepcopy(sources[role]) for role in worker.RESTORE_ROLES}
    original.update(image_geometry=deepcopy(sources['image_geometry']),portable_reader=deepcopy(sources['fit_portable_reader']))
    docs=dict(qualified={'synthetic_only':True},spec={'original14':original},
        pair_commit={'checkpoint_refs':request['checkpoint_refs']},order={'synthetic_order':True})
    request.update(source_refs=sources,prediction_core_source=core,
        endpoint_refs={role:c.add('/synthetic/endpoint/'+role+'.json',json.dumps(value)) for role,value in docs.items()},
        manifest_ref=c.add('/synthetic/DEV_MANIFEST.json',json.dumps({'entries':entries})),
        library_sources={role:c.add('/synthetic/lib/'+role+'.py','# synthetic '+role) for role in worker.LIBRARIES},
        worker_source=c.add('/synthetic/U73/worker.py','# synthetic worker identity'))
    state=types.SimpleNamespace();state.clone=lambda:state
    backend=types.SimpleNamespace(cudnn=types.SimpleNamespace(deterministic=False,benchmark=True,allow_tf32=True),
        cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=True)))
    torch=types.SimpleNamespace(get_rng_state=lambda:state,set_rng_state=lambda value:None,equal=lambda a,b:a is b,
        get_default_dtype=lambda:'float32',set_default_dtype=lambda value:None,is_grad_enabled=lambda:True,
        set_grad_enabled=lambda value:None,get_num_threads=lambda:2,get_num_interop_threads=lambda:2,
        backends=backend,cuda=types.SimpleNamespace(is_initialized=lambda:False))
    rng=('synthetic',[1,2],0,0,0.)
    np=types.SimpleNamespace(random=types.SimpleNamespace(get_state=lambda:rng,set_state=lambda value:None),array_equal=lambda a,b:a==b)
    return request,torch,np,mask


class SourceClosure(unittest.TestCase):
    def test_exact22_union_covers_original14_and_rejects_bad_fit_or_role_set(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as root:
            request,entries,endpoint=fixture((Path(root)/'fresh').resolve())
            direct=worker.direct_references(request,entries)
            self.assertEqual(len(request['source_refs']),22)
            self.assertTrue(all(value in direct for value in endpoint['restore_source_refs'].values()))
            old21=[request['source_refs'][role] for role in OLD_ROLES]
            self.assertNotIn(endpoint['restore_source_refs']['portable_reader'],old21)
            self.assertIn(endpoint['restore_source_refs']['image_geometry'],old21)
            for mutation in ('missing','extra','bad_hash','bad_bytes'):
                bad=deepcopy(request)
                if mutation=='missing':del bad['source_refs']['fit_portable_reader']
                elif mutation=='extra':bad['source_refs']['invented_source']=ref('/synthetic/invented.py')
                elif mutation=='bad_hash':bad['source_refs']['fit_portable_reader']['sha256']='f'*64
                else:bad['source_refs']['fit_portable_reader']['bytes']=19617
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                    worker.direct_references(bad,entries)

    def test_geometry_and_fit_require_original_entire_refs_not_same_content_at_new_paths(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as root:
            request,_,endpoint=fixture((Path(root)/'fresh').resolve())
            self.assertTrue(worker.validate_endpoint_source_closure(endpoint,request['source_refs']))
            for role in ('image_geometry','fit_portable_reader'):
                moved=deepcopy(request['source_refs'])
                moved[role]['path']='/synthetic/new_copy/'+role+'.py'
                self.assertEqual(moved[role]['sha256'],request['source_refs'][role]['sha256'])
                with self.subTest(role=role),self.assertRaisesRegex(ValueError,'original accepted'):
                    worker.validate_endpoint_source_closure(endpoint,moved)
            missing=deepcopy(endpoint);del missing['restore_source_refs']['portable_reader']
            with self.assertRaises(ValueError):worker.validate_endpoint_source_closure(missing,request['source_refs'])

    def test_fit_is_byte_closed_without_execution_or_an_extra_module(self):
        c=SyntheticControl();sources,core,mask=source_fixture(c)
        with worker.source_modules(c,sources,core,mask) as (_,modules):
            self.assertEqual(len(modules),19)
            self.assertNotIn('fit_portable_reader',modules)
            self.assertNotIn('fit_portable_reader',dict(worker.LOAD_ORDER))
            self.assertIn(sources['fit_portable_reader']['path'],c.closed_paths)

    def test_worker_projects_original21_into_core_but_persists_honest_outer22_evidence(self):
        c=SyntheticControl()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as root:
            target=(Path(root)/'fresh').resolve();request,torch,np,mask=run_fixture(c,target)
            synthetic_pins={role:value['sha256'] for role,value in request['source_refs'].items() if role!='encoder'}
            report={}
            with patch.object(worker,'SOURCE_PINS',synthetic_pins),patch.object(worker,'CORE_SHA256',request['prediction_core_source']['sha256']),\
                 patch.object(worker,'__file__',request['worker_source']['path']),patch.object(worker,'import_runtime',return_value=(torch,np,mask)):
                returned=worker.run_worker(c,request,report=report)
            self.assertIs(returned,report)
            self.assertEqual(report['schema'],'U73-single-shard-worker-report-v1')
            self.assertEqual(report['source_refs'],request['source_refs'])
            self.assertEqual(len(report['source_refs']),22)
            self.assertTrue(report['original_restore_source_closure_verified'])
            fit=request['source_refs']['fit_portable_reader']
            self.assertIn(fit,report['direct_input_refs'])
            self.assertGreaterEqual(c.closed_paths.count(fit['path']),3)
            saved=json.loads((target/'U66_SHARD.json').read_text())
            self.assertTrue(saved['synthetic_exact21_core_called'])
            self.assertEqual(set(saved['source_refs']),set(OLD_ROLES))
            self.assertEqual(json.loads((target/'REPORT.json').read_text()),report)
            self.assertFalse(any(path.endswith(('.pt','.pth')) for path in c.closed_paths))

    def test_original_ref_mismatch_fails_before_core_and_retains_failed_evidence(self):
        for role in ('image_geometry','portable_reader','training_state'):
            with self.subTest(original_role=role),tempfile.TemporaryDirectory(dir=Path(__file__).parent) as root:
                c=SyntheticControl();target=(Path(root)/'failed').resolve()
                request,torch,np,mask=run_fixture(c,target)
                spec=json.loads(c.objects[request['endpoint_refs']['spec']['path']])
                spec['original14'][role]['path']='/synthetic/other_original_path/'+role+'.py'
                request['endpoint_refs']['spec']=c.add('/synthetic/endpoint/spec.json',json.dumps(spec))
                synthetic_pins={key:value['sha256'] for key,value in request['source_refs'].items() if key!='encoder'}
                report={}
                with patch.object(worker,'SOURCE_PINS',synthetic_pins),patch.object(worker,'CORE_SHA256',request['prediction_core_source']['sha256']),\
                     patch.object(worker,'__file__',request['worker_source']['path']),patch.object(worker,'import_runtime',return_value=(torch,np,mask)):
                    with self.assertRaises((AssertionError,ValueError)):worker.run_worker(c,request,report=report)
                self.assertEqual(report['status'],'FAILED')
                self.assertFalse(report['original_restore_source_closure_verified'])
                self.assertEqual(report['source_refs'],request['source_refs'])
                self.assertNotIn('synthetic_exact21_core_called',report['partial_prediction_report'])
                self.assertFalse((target/'U66_SHARD.json').exists())
                self.assertEqual(json.loads((target/'REPORT.json').read_text()),report)
                expected='validate_restore_sources' if role=='training_state' else 'validate_endpoint_source_closure'
                self.assertIn(expected,report['error']['traceback'])


if __name__=='__main__':unittest.main()
