"""Synthetic injected ABI/control-flow checks only; never Torch/NumPy or ML."""
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace as NS
import unittest
import weakref
import prediction_core as subject


class Token:
    def __init__(self,name):self.name=name
    def to(self,*args,**kwargs):raise AssertionError('Caller must not move/reorder reader tensors')


class RNG:
    def clone(self):return RNG()
    def cpu(self):return self


class Node:
    def __init__(self):self.training=False;self.hooks=[];self.fail_remove=False
    def register_forward_hook(self,hook):
        self.hooks.append(hook)
        def remove():
            self.hooks.remove(hook)
            if self.fail_remove:raise RuntimeError('SYNTHETIC_HOOK_CLEANUP_FAILURE')
        return NS(remove=remove)
    def call(self):
        for hook in self.hooks:hook(self,(),None)


class Adapter(Node):
    def __init__(self,arm,events):
        super().__init__();self.arm=arm;self.events=events
        self.backbone=Node();self.backbone.legacy=NS(CONV0_4=Node(),OLD_HEAD=Node())
        self.param=NS(device=NS(type='cpu'),dtype='FP32',requires_grad=True,grad=None,label=arm)
    def eval(self):
        for node in self.modules():node.training=False
        return self
    def requires_grad_(self,value):self.param.requires_grad=value;return self
    def to(self,**kwargs):self.events.append(('MODEL_TO',self.arm));self.param.device.type='cuda';return self
    def state_dict(self):return {'trained-head':self.param}
    def parameters(self):return iter([self.param])
    def modules(self):return iter([self,self.backbone,self.backbone.legacy.CONV0_4,self.backbone.legacy.OLD_HEAD])


def fixture(*,fail_restore=None,fail_predict=False,fail_hook_cleanup=False):
    events=[];models=[];seen=[]
    class Cuda:
        active=False
        def is_initialized(self):return self.active
        def init(self):events.append(('CUDA_INIT',));self.active=True
        def set_device(self,index):pass
        def get_device_properties(self,index):return NS(total_memory=16*1024**3)
        def set_per_process_memory_fraction(self,fraction,index):events.append(('CAP',fraction))
        def memory_allocated(self):return 0
        def memory_reserved(self):return 0
        def get_rng_state(self,index):return RNG()
        def synchronize(self):pass
        def empty_cache(self):events.append(('EMPTY_CACHE',))
    @contextmanager
    def no_grad():yield
    torch=NS(cuda=Cuda(),float32='FP32',get_default_dtype=lambda:'FP32',get_num_threads=lambda:2,
        get_num_interop_threads=lambda:2,get_rng_state=lambda:RNG(),equal=lambda a,b:True,no_grad=no_grad,
        is_autocast_enabled=lambda:False,_C=NS(_cuda_clearCublasWorkspaces=lambda:None),
        backends=NS(cudnn=NS(deterministic=True,benchmark=False,allow_tf32=False),cuda=NS(matmul=NS(allow_tf32=False))))
    c=NS(check=lambda:None,closed=lambda ref:ref['path'])
    source_refs={role:dict(path='/invented/'+role+'.py',bytes=1,sha256='a'*64) for role in subject.SOURCE_ROLES}
    modules={role:NS(__file__=source_refs[role]['path']) for role in subject.MODULE_ROLES}
    qids=list(reversed(range(100)))
    entries=[dict(dataset_index=i,image_id=1000+i,query_ids=qids[:],
        package_ref=dict(path='/invented/DEV_FULL100_%d.pt'%i,bytes=1,sha256='b'*64)) for i in range(187)]
    records=[dict(dataset_index=i,image_id=1000+i,qids=qids[:]) for i in range(187)]
    def build_pair(t,mods,refs):
        events.append(('BUILD_PAIR',));pair={a:Adapter(a,events) for a in ('PLAIN','FULL')}
        for value in pair.values():value.backbone.fail_remove=fail_hook_cleanup
        models.extend(weakref.ref(value) for value in pair.values());return pair,{'synthetic_receipt':True}
    modules['shared_initialization'].build_pair=build_pair
    modules['shared_initialization'].SOURCE_SHA256=dict.fromkeys(subject.CONSTRUCTOR_ROLES,'a'*64)
    @contextmanager
    def measure(t,control,record,cuda):
        try:yield
        finally:record.update(CUDA_peak_allocated_bytes=0,sampled_RSS_peak_bytes=1,RSS_sampling_errors=[])
    modules['u36_helpers']._measure=measure
    modules['u39_training_ops'].HEAD_NAMES=('OLD_HEAD',)
    def partition(adapter):
        # Match the original U39 training-only precondition: restored models
        # are frozen and must use U58's already-verified JSON partition proof.
        if not all(p.requires_grad for p in adapter.parameters()):
            raise ValueError('SYNTHETIC_ORIGINAL_PARTITION_REQUIRES_TRAINABLE')
        return {'trained-head'},set(),dict(allocated=1,effective=1,inactive=0,effective_names=['trained-head'],
            inactive_names=[],manifest=[dict(name='trained-head',shape=[1],numel=1,dtype='FP32',requires_grad=True,inactive=False)])
    modules['u39_training_ops']._partition=partition
    def restore(t,np,control,adapter,arm,metadata,fresh,*,modules,source_refs,report):
        events.append(('RESTORE',arm,t.cuda.is_initialized()))
        if arm==fail_restore:raise RuntimeError('SYNTHETIC_RESTORE_FAILURE')
        _,_,parameter_report=modules['u39_training_ops']._partition(adapter)
        adapter.eval().requires_grad_(False)
        report.update(status='RESTORED_CPU_ONLY',restored_model_state_exact=True,formal_endpoint=True,
            trained_head_reset=False,CUDA_initialized=False,CPU_checkpoint_payload_released=True,
            parameter_report=parameter_report,elapsed_seconds=0.)
        return report
    modules['restore_state'].restore_arm=restore
    modules['restore_state'].validate_final_endpoint=lambda *args:dict(cursor=dict(global_updates=37050,completed_epochs=50,within_epoch=0))
    def load(index):
        events.append(('LOAD',index))
        item={name:Token(name) for name in ('rgb400','e','M','L')}
        item.update(identity={'image_id':1000+index},query_ids=qids[:],geometry=Token('original-geometry'))
        originals=[dict(image_id=1000+index,query_id=qid,category_id=1,score=qid/100.,
            segmentation=dict(size=[1,1],counts='EMPTY'),bbox=[0,0,0,0],area=0) for qid in qids]
        return item,deepcopy(records[index]),originals
    def prepare(t,control,values,*,image_geometry,source_refs):
        events.append(('PREPARE',));return NS(records=deepcopy(records),load=load)
    modules['portable_reader'].prepare=prepare
    modules['portable_reader'].validate_entries=lambda values:deepcopy(values)
    modules['prediction_core'].LIMITS=dict(allocator_bytes=4*1024**3,maximum_RSS_bytes=16*1024**3,deadline_seconds=1800)
    modules['prediction_core']._state_digest=lambda helpers,state:state['trained-head'].label
    def prediction(t,np,control,adapter,item,row,originals,mods,cost,collect_support):
        events.append(('PREDICT',adapter.arm,row['dataset_index']))
        seen.append((item['e'].name,item['M'].name,item['L'].name,item['query_ids'][:],row['qids'][:]))
        if fail_predict:raise ArithmeticError('SYNTHETIC_PREDICTION_FAILURE')
        adapter.backbone.call();adapter.backbone.legacy.CONV0_4.call()
        cost['encode_calls']+=1;cost['readout_calls']+=25;cost['queries']+=100
        values=deepcopy(originals)
        for value in values:value['segmentation']['counts']='SYNTHETIC_TRAINED_OUTPUT'
        support=[dict(dataset_index=row['dataset_index'],image_id=row['image_id'],query_id=qid,Omega={'synthetic':True})
            for qid in item['query_ids']] if collect_support else []
        proof=dict(dataset_index=row['dataset_index'],image_id=row['image_id'],queries=100,
            zero_residual_native_L_exact=True,native_RLE_and_prediction_dictionaries_exact=True,
            original_full_S0_class_order_preserved=True,elapsed_seconds=0.)
        return values,deepcopy(originals),support,proof
    modules['prediction_core']._calibration_and_prediction=prediction
    return torch,NS(),c,dict(endpoint_metadata={'SYNTHETIC_ONLY':True},entries=entries,modules=modules,
        source_refs=source_refs,shard_index=0,dataset_range=[0,47],report={}),events,models,seen


class CoreTests(unittest.TestCase):
    def test_both_arms_restore_cold_before_first_gpu_and_original_query_identity_is_preserved(self):
        torch,np,c,args,events,models,seen=fixture()
        result=subject.predict_shard(torch,np,c,**args)
        self.assertEqual([e for e in events if e[0]=='RESTORE'],[('RESTORE','PLAIN',False),('RESTORE','FULL',False)])
        self.assertLess(events.index(('RESTORE','FULL',False)),events.index(('CUDA_INIT',)))
        plain_to=events.index(('MODEL_TO','PLAIN'));full_to=events.index(('MODEL_TO','FULL'))
        self.assertIn(('EMPTY_CACHE',),events[plain_to+1:full_to])
        self.assertEqual(result['arms']['PLAIN']['cleanup'],dict(allocated_after=0,reserved_after=0))
        self.assertEqual(result['arms']['FULL']['cleanup'],dict(allocated_after=0,reserved_after=0))
        self.assertEqual(result['arms']['PLAIN']['parameters'],result['arms']['PLAIN']['restore_report']['parameter_report'])
        self.assertEqual(result['status'],'PREDICTIONS_COMPLETE_PENDING_ROOT_ACCEPTANCE')
        self.assertFalse(result['qualified'])
        self.assertEqual(len(result['predictions']['PLAIN']),4700)
        self.assertEqual([v['query_id'] for v in result['identity_predictions'][:100]],list(reversed(range(100))))
        self.assertEqual([v['score'] for v in result['predictions']['FULL'][:100]],[q/100. for q in reversed(range(100))])
        self.assertTrue(all(row==('e','M','L',list(reversed(range(100))),list(reversed(range(100)))) for row in seen))
        self.assertTrue(all(reference() is None for reference in models))

    def test_original_prediction_error_survives_cleanup_error_and_reader_success_is_not_prediction(self):
        torch,np,c,args,events,models,seen=fixture(fail_predict=True,fail_hook_cleanup=True)
        with self.assertRaisesRegex(ArithmeticError,'SYNTHETIC_PREDICTION_FAILURE'):
            subject.predict_shard(torch,np,c,**args)
        self.assertEqual(args['report']['status'],'FAILED')
        self.assertTrue(any(event[0]=='LOAD' for event in events))
        self.assertEqual(args['report']['predictions'],{})
        self.assertTrue(all(reference() is None for reference in models))

    def test_second_cpu_restore_failure_never_initializes_gpu_or_prepares_reader(self):
        torch,np,c,args,events,models,seen=fixture(fail_restore='FULL')
        with self.assertRaisesRegex(RuntimeError,'SYNTHETIC_RESTORE_FAILURE'):
            subject.predict_shard(torch,np,c,**args)
        self.assertFalse(torch.cuda.is_initialized())
        self.assertFalse(any(e[0] in ('PREPARE','LOAD','PREDICT','MODEL_TO') for e in events))
        self.assertEqual(args['report']['status'],'FAILED')
        self.assertTrue(all(reference() is None for reference in models))

    def test_malformed_three_value_reader_result_propagates_without_prediction_fallback(self):
        torch,np,c,args,events,models,seen=fixture()
        original_prepare=args['modules']['portable_reader'].prepare
        def malformed(*pos,**kwargs):
            reader=original_prepare(*pos,**kwargs);reader.load=lambda index:{'SYNTHETIC_BAD_ABI':True};return reader
        args['modules']['portable_reader'].prepare=malformed
        with self.assertRaisesRegex(ValueError,'U59 returns item,row,originals'):
            subject.predict_shard(torch,np,c,**args)
        self.assertFalse(any(e[0]=='PREDICT' for e in events))
        self.assertEqual(args['report']['status'],'FAILED')
        self.assertTrue(all(reference() is None for reference in models))


if __name__=='__main__':unittest.main()
