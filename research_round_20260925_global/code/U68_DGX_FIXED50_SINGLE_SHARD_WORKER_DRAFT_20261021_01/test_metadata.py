"""Only synthetic JSON/source I/O; never import a scientific library."""
import hashlib
import json
from pathlib import Path
import tempfile
import sys
import types
import unittest
from unittest.mock import patch

import worker


class MemoryPath:
    def __init__(self,path,raw):self.path,self.raw=path,raw
    def __str__(self):return self.path
    def read_bytes(self):return self.raw


class SyntheticControl:
    def __init__(self):self.objects={};self.closed_paths=[];self.checks=0
    def add(self,path,text):
        raw=text.encode('utf-8');self.objects[path]=raw
        return dict(path=path,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    def closed(self,ref):
        self.closed_paths.append(ref['path']);raw=self.objects[ref['path']]
        if len(raw)!=ref['bytes'] or hashlib.sha256(raw).hexdigest()!=ref['sha256']:raise ValueError('bad synthetic bytes')
        return MemoryPath(ref['path'],raw)
    def check(self):self.checks+=1


class WorkerWiring(unittest.TestCase):
    def sources(self,c):
        roles=('training_state','training_entry','u36_helpers','u39_training_ops',
            'shared_initialization','full_legacy','full_dsconv','full_head','full_feature',
            'plain_legacy','plain_feature','instance_adapter','restore_state','portable_reader',
            'image_geometry','prediction_core','encoder','mask_api','population_contract',
            'dev_reader','dev_rgb_reader')
        raw={role:'# harmless synthetic '+role for role in roles}
        raw.update(full_dsconv='class DSConv:pass',full_head='class CoordConv:pass',
            full_legacy='from dsconv import DSConv\nfrom head import CoordConv\n',
            dev_reader="raise RuntimeError('must never execute historical reader')",
            dev_rgb_reader="raise RuntimeError('must never execute historical RGB reader')")
        refs={role:c.add('/synthetic/'+role+'.py',raw[role]) for role in roles}
        core=c.add('/synthetic/u66.py','def predict_shard(*args,**kwargs):pass\n')
        mask=types.ModuleType('pycocotools.mask');mask.__file__=refs['mask_api']['path']
        return refs,core,mask

    def test_pending_shard_is_saved_without_replacing_its_u66_shape(self):
        prediction = dict(schema='U66-DGX-fixed50-DEV-shard-pending-v1',
            status='PREDICTIONS_COMPLETE_PENDING_ROOT_ACCEPTANCE', qualified=False,
            AP=False, DEV=True, TEST=False, new_training=False,
            formal_training_budget_credit=0, shard_index=0, dataset_range=[0,47],
            predictions={'PLAIN':[{'synthetic':True}], 'FULL':[]})
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as root:
            target=Path(root)/'fresh'
            result=worker.persist_prediction(target,prediction)
            self.assertEqual(json.loads((target/'U66_SHARD.json').read_text()),prediction)
            raw=(target/'U66_SHARD.json').read_bytes()
            self.assertEqual(result['bytes'],len(raw))
            self.assertEqual(result['sha256'],hashlib.sha256(raw).hexdigest())
            with self.assertRaises(FileExistsError):worker.persist_prediction(target,prediction)
            self.assertEqual(json.loads((target/'U66_SHARD.json').read_text()),prediction)

    def test_source_loader_resolves_original_dependencies_and_restores_aliases(self):
        c=SyntheticControl();refs,core,mask=self.sources(c)
        prior=types.ModuleType('prior_dsconv');existed='dsconv' in sys.modules;old=sys.modules.get('dsconv')
        sys.modules['dsconv']=prior
        try:
            with worker.source_modules(c,refs,core,mask) as (prediction,modules):
                self.assertTrue(callable(prediction.predict_shard))
                self.assertEqual(len(modules),19)
                self.assertIs(modules['full_legacy'].DSConv,modules['full_dsconv'].DSConv)
                self.assertIs(modules['full_legacy'].CoordConv,modules['full_head'].CoordConv)
                self.assertIs(modules['mask_api'],mask)
                self.assertNotIn('dev_reader',modules)
                self.assertIn(refs['dev_reader']['path'],c.closed_paths)
                self.assertIn(refs['dev_rgb_reader']['path'],c.closed_paths)
            self.assertIs(sys.modules['dsconv'],prior)
        finally:
            if existed:sys.modules['dsconv']=old
            else:sys.modules.pop('dsconv',None)

    def worker_inputs(self,c,target,fail=False):
        refs,core,mask=self.sources(c)
        restore='''
def validate_final_endpoint(training,entry,metadata):
    if metadata['qualified']['mode']!='FORMAL' or metadata['pair_commit']['cursor']!={'global_updates':37050,'completed_epochs':50,'within_epoch':0}:
        raise ValueError('synthetic nonfinal endpoint rejected')
    return {'checkpoint_refs':metadata['pair_commit']['checkpoint_refs'],'restore_source_refs':metadata['spec']['restore_sources']}
def validate_restore_sources(endpoint,refs):
    if endpoint['restore_source_refs']!=refs:raise ValueError('synthetic restore source mismatch')
'''
        refs['restore_state']=c.add('/synthetic/restore_state.py',restore)
        refs['portable_reader']=c.add('/synthetic/portable_reader.py',
            "def validate_entries(entries):\n    assert len(entries)==187\n    assert [r['dataset_index'] for r in entries]==list(range(187))\n")
        body="""
def predict_shard(torch,np,c,*,endpoint_metadata,entries,modules,source_refs,shard_index,dataset_range,report):
    assert set(endpoint_metadata)=={'refs','qualified','spec','pair_commit','order'}
    assert len(entries)==187 and len(modules)==19 and len(source_refs)==21
    assert shard_index==1 and dataset_range==[47,94]
    report.update(schema='U66-DGX-fixed50-DEV-shard-pending-v1',qualified=False,AP=False,
        DEV=True,TEST=False,new_training=False,formal_training_budget_credit=0,
        shard_index=1,dataset_range=[47,94],predictions={'PLAIN':[],'FULL':[]})
    report['synthetic_core_was_called']=True
    if FAIL:
        report['status']='FAILED'
        raise RuntimeError('synthetic actual-call failure')
    report['status']='PREDICTIONS_COMPLETE_PENDING_ROOT_ACCEPTANCE'
    return report
""".replace('if FAIL:','if '+repr(fail)+':')
        core=c.add('/synthetic/u66.py',body)
        checkpoints={arm:dict(path='/synthetic/'+arm+'.pth',bytes=1,sha256='a'*64) for arm in ('PLAIN','FULL')}
        restore_roles=('training_state','training_entry','u36_helpers','u39_training_ops',
            'shared_initialization','full_legacy','full_dsconv','full_head','full_feature',
            'plain_legacy','plain_feature','instance_adapter')
        docs=dict(qualified={'mode':'FORMAL'},spec={'restore_sources':{r:refs[r] for r in restore_roles}},
            pair_commit={'cursor':{'global_updates':37050,'completed_epochs':50,'within_epoch':0},'checkpoint_refs':checkpoints},
            order={'synthetic_original_order':True})
        endpoint={k:c.add('/synthetic/'+k+'.json',json.dumps(v)) for k,v in docs.items()}
        entries=[dict(dataset_index=i,image_id=1000+i,query_ids=list(range(99,-1,-1)),
            package_ref=dict(path='/synthetic/data/DEV_FULL100_'+str(i)+'.pt',bytes=1,sha256='b'*64)) for i in range(187)]
        manifest=c.add('/synthetic/MANIFEST.json',json.dumps({'entries':entries,'synthetic_historical_metadata':True}))
        libs={r:c.add('/synthetic/lib/'+r+'.py','# synthetic '+r) for r in
            ('torch','numpy','psutil','pycocotools','mask_extension')}
        own=c.add('/synthetic/worker.py','# synthetic worker identity')
        request=dict(worker_source=own,prediction_core_source=core,source_refs=refs,library_sources=libs,
            endpoint_refs=endpoint,checkpoint_refs=checkpoints,manifest_ref=manifest,
            shard_index=1,dataset_range=[47,94],output_directory=str(target))
        class State:
            def clone(self):return self
        state=State()
        cuda=types.SimpleNamespace(is_initialized=lambda:False)
        backend=types.SimpleNamespace(cudnn=types.SimpleNamespace(deterministic=False,benchmark=True,allow_tf32=True),
            cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=True)))
        torch=types.SimpleNamespace(get_rng_state=lambda:state,set_rng_state=lambda s:None,equal=lambda a,b:a is b,
            get_default_dtype=lambda:'float32',set_default_dtype=lambda d:None,is_grad_enabled=lambda:True,
            set_grad_enabled=lambda x:None,get_num_threads=lambda:2,get_num_interop_threads=lambda:2,
            backends=backend,cuda=cuda)
        rng=('synthetic',[1,2],0,0,0.0)
        np=types.SimpleNamespace(random=types.SimpleNamespace(get_state=lambda:rng,set_state=lambda s:None),array_equal=lambda a,b:a==b)
        return request,torch,np,mask

    def test_worker_reads_explicit_metadata_calls_core_and_keeps_pending_output(self):
        c=SyntheticControl()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as root:
            target=(Path(root)/'attempt').resolve();request,torch,np,mask=self.worker_inputs(c,target)
            pins={role:ref['sha256'] for role,ref in request['source_refs'].items() if role!='encoder'}
            report={}
            with patch.object(worker,'SOURCE_PINS',pins),patch.object(worker,'CORE_SHA256',request['prediction_core_source']['sha256']),\
                 patch.object(worker,'import_runtime',return_value=(torch,np,mask)),patch.object(worker,'__file__',request['worker_source']['path']):
                result=worker.run_worker(c,request,report=report)
            self.assertIs(result,report)
            saved=json.loads((target/'U66_SHARD.json').read_text())
            self.assertTrue(saved['synthetic_core_was_called'])
            self.assertEqual(saved['dataset_range'],[47,94])
            self.assertEqual(report['status'],'COMPLETE_PENDING_ROOT_ACCEPTANCE')
            self.assertFalse(report['qualified']);self.assertFalse(report['AP'])
            self.assertEqual(json.loads((target/'REPORT.json').read_text()),report)
            self.assertFalse(any(p.endswith('.pt') or p.endswith('.pth') for p in c.closed_paths))
            self.assertFalse(torch.backends.cudnn.deterministic)
            self.assertTrue(torch.backends.cudnn.benchmark)

    def test_nonfinal_endpoint_never_calls_core_or_saves_prediction(self):
        c=SyntheticControl()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as root:
            target=(Path(root)/'rejected').resolve();request,torch,np,mask=self.worker_inputs(c,target)
            request['endpoint_refs']['qualified']=c.add('/synthetic/qualified.json',json.dumps({'mode':'SMOKE'}))
            pins={r:v['sha256'] for r,v in request['source_refs'].items() if r!='encoder'};report={}
            with patch.object(worker,'SOURCE_PINS',pins),patch.object(worker,'CORE_SHA256',request['prediction_core_source']['sha256']),\
                 patch.object(worker,'import_runtime',return_value=(torch,np,mask)),patch.object(worker,'__file__',request['worker_source']['path']):
                with self.assertRaisesRegex(ValueError,'nonfinal endpoint'):worker.run_worker(c,request,report=report)
            self.assertEqual(report['status'],'FAILED')
            self.assertNotIn('synthetic_core_was_called',report['partial_prediction_report'])
            self.assertFalse((target/'U66_SHARD.json').exists())
            self.assertEqual(json.loads((target/'REPORT.json').read_text())['status'],'FAILED')

    def test_empty_package_initializer_is_closed_but_other_empty_libraries_are_rejected(self):
        c=SyntheticControl()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as root:
            target=(Path(root)/'emptyinit').resolve();request,torch,np,mask=self.worker_inputs(c,target)
            request['library_sources']['pycocotools']=c.add('/synthetic/lib/pycocotools.py','')
            pins={r:v['sha256'] for r,v in request['source_refs'].items() if r!='encoder'}
            with patch.object(worker,'SOURCE_PINS',pins),patch.object(worker,'CORE_SHA256',request['prediction_core_source']['sha256']):
                worker.validate_request(request)
                entries=json.loads(c.objects[request['manifest_ref']['path']])['entries']
                refs=worker.direct_references(request,entries)
                self.assertIn(request['library_sources']['pycocotools'],refs)
                request['library_sources']['torch']=c.add('/synthetic/lib/torch.py','')
                with self.assertRaises(ValueError):worker.validate_request(request)

    def test_core_failure_survives_outer_cleanup_failure_and_retains_partial(self):
        c=SyntheticControl()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as root:
            target=(Path(root)/'failed').resolve();request,torch,np,mask=self.worker_inputs(c,target,fail=True)
            def bad_cleanup(state):raise ValueError('synthetic cleanup failure')
            torch.set_rng_state=bad_cleanup
            pins={r:v['sha256'] for r,v in request['source_refs'].items() if r!='encoder'};report={}
            with patch.object(worker,'SOURCE_PINS',pins),patch.object(worker,'CORE_SHA256',request['prediction_core_source']['sha256']),\
                 patch.object(worker,'import_runtime',return_value=(torch,np,mask)),patch.object(worker,'__file__',request['worker_source']['path']):
                with self.assertRaisesRegex(RuntimeError,'synthetic actual-call failure'):worker.run_worker(c,request,report=report)
            self.assertEqual(report['status'],'FAILED')
            self.assertTrue(report['partial_prediction_report']['synthetic_core_was_called'])
            self.assertIn('Traceback',report['error']['traceback'])
            self.assertIn('predict_shard',report['error']['traceback'])
            self.assertIn('synthetic actual-call failure',report['error']['traceback'])
            self.assertIn('cleanup failure',report['cleanup_errors'][0]['message'])
            self.assertFalse((target/'U66_SHARD.json').exists())
            self.assertEqual(json.loads((target/'REPORT.json').read_text())['status'],'FAILED')
            self.assertFalse(torch.backends.cudnn.deterministic)
            self.assertTrue(torch.backends.cudnn.benchmark)


if __name__=='__main__':unittest.main()
