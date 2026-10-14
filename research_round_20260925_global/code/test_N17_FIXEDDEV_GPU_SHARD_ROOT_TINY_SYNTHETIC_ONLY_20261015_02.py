"""One ordinary GPU-entry ABI check; no Torch/CUDA/SAR/actual refs."""
import importlib.util
import ast
import copy
import json
import tempfile
from pathlib import Path
import unittest
ENTRY=Path(__file__).with_name('N17_FIXEDDEV_LOCAL_CENTER_ERROR_GPU_SHARD_ROOT_UNBOUND_DRAFT_20261015_02.py')
class Contract(unittest.TestCase):
    def test_new_header_and_new_arm_member_identifiers(self):
        s=importlib.util.spec_from_file_location('_n17_gpu_tiny',ENTRY);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
        value=dict(schema=m.SCHEMA,shard_id=0,dataset_ordinal_range=[0,47],normal_RAW_count=122,expected_normal_RAW_count=122,
            total_failed_RAW_count=18,total_terminal_identity_pin_count=140,initialization_seed=202610150171,
            deadline_seconds=1800,automatic_retry=False,allocator_limit_MiB=4096,whole_GPU_limit_MiB=8192,
            entry_source=dict(path=str(ENTRY.resolve()),bytes=0,sha256='0'*64))
        self.assertTrue(m.validate_gpu_header(value))
        for key,v in (('schema',m.TRAIN_SCHEMA),('normal_RAW_count',121),('shard_id',True),('dataset_ordinal_range',[0,46]),('total_failed_RAW_count',17)):
            with self.subTest(key=key),self.assertRaises(RuntimeError):m.validate_gpu_header(dict(value,**{key:v}))
        event=dict(query_id=0,control='LOCAL_ERROR',scout_epoch=24,evalmode='shared-atmost-exact',stage='scout_C_logits',chunk_first=0)
        first=m.new_point_member_id(0,event)
        other=m.new_point_member_id(0,dict(event,control='CENTER_ERROR'))
        self.assertNotEqual(first,other)
        self.assertNotEqual(m.new_profile_member_id(0,0,24,'LOCAL_ERROR','atmost'),m.new_profile_member_id(0,0,24,'LOCAL_ERROR','exact'))
        with self.assertRaises(RuntimeError):m.new_point_member_id(0,dict(event,control='UTILITY'))
        tree=ast.parse(ENTRY.read_text(encoding='utf-8'))
        run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        self.assertFalse(any(isinstance(n,ast.Name) and n.id=='reference' and isinstance(n.ctx,ast.Store) for n in ast.walk(run)))
        self.assertTrue(any(isinstance(n,ast.Delete) and any(isinstance(t,ast.Name) and t.id=='profile' for t in n.targets) for n in ast.walk(run)))
        budget=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='capture_budget_gate')
        self.assertTrue(any(isinstance(n,ast.Constant) and n.value==800 for n in ast.walk(budget)))
        stats=[.68,.8,.7,.5,.6,.7,.1,.3,.8,.4,.5,.6]
        names=('AP','AP50','AP75','APS','APM','APL','AR1','AR10','AR100','ARS','ARM','ARL')
        coco=dict(stats=stats,metrics=dict(zip(names,stats)),iouType='segm',maxDets=[1,10,100],original_GT_preserved=True,
            score_filter=None,official_COCO_crowd_ignore_semantics=True)
        result=dict(stats=stats,COCO=coco,predictions=dict(path=str(ENTRY.resolve()),bytes=0,sha256='0'*64))
        self.assertTrue(m.validate_legacy_COCO(result))
        missing=copy.deepcopy(result);missing['stats'][3]=-1;missing['COCO']['metrics']['APS']=-1
        self.assertTrue(m.validate_legacy_COCO(missing))
        for key,wrong in (('iouType','bbox'),('maxDets',[1,10,300]),('original_GT_preserved',False),('score_filter',.1),('official_COCO_crowd_ignore_semantics',False)):
            with self.subTest(COCO=key),self.assertRaises(RuntimeError):m.validate_legacy_COCO(dict(result,COCO=dict(coco,**{key:wrong})))
        mismatch=copy.deepcopy(result);mismatch['COCO']['stats'][0]=.9
        with self.assertRaises(RuntimeError):m.validate_legacy_COCO(mismatch)
        wrong_scale=copy.deepcopy(result);wrong_scale['COCO']['metrics']={k:v*100 for k,v in wrong_scale['COCO']['metrics'].items()}
        with self.assertRaises(RuntimeError):m.validate_legacy_COCO(wrong_scale)
        with tempfile.TemporaryDirectory(prefix='n17_fake_failure18_') as temporary:
            phase=Path(temporary)/'FAKE18';phase.mkdir();rawpath=phase/'PARENT_RECEIPT.json'
            raw=dict(status='ACTUAL_OWNED_TERMINAL',returncode=1,original_identity={'pid':123,'birth':'FAKE'},root_wait_reaped=True,
                held_handle_exit=dict(wait_state='WAIT_OBJECT_0',actual_exit_code=1),numeric_birth_absence=dict(original_birth_absent=True),
                error=None,abort=None,wait_errors=False,GPU_observation_errors=[],admission_sha256='1'*64,
                whole_GPU_observed_peak_MiB=1,whole_GPU_used_limit_MiB=8192,whole_leaf_VRAM_certified=False,
                GPU_scope='whole GPU observed, not per-process WDDM allocation',phase='FAKE18')
            rawpath.write_text(json.dumps(raw),encoding='utf-8')
            fake=lambda name:dict(path=str(phase/name),bytes=0,sha256='0'*64)
            failure=dict(RAW_ref=fake('PARENT_RECEIPT.json'),phase='FAKE18',pin=dict(phase='FAKE18',returncode=1,
                original_identity=raw['original_identity'],admission_sha256='1'*64),returncode=1,admission=fake('AD.json'),
                phase_binding=fake('PHASE.json'),bootstrap=fake('BOOT.py'),entry_source=fake('ENTRY.py'),spec=fake('SPEC.json'),original_report_absent=True)
            fake_spec=dict(terminal_identity_pins={},RAW_inputs=[])
            for key in m.FAILURE18_PRIOR_KEYS:fake_spec[key]=dict(RAW_ref=fake(key+'.json'))
            fake_spec['additional_preserved_N17_firstDEV_reference_ABI_failure']=failure
            self.assertEqual(m.eighteenth_firstDEV_reference_ABI_failure_raw_only(fake_spec)['returncode'],1)
            for key,wrong in (('returncode',True),('original_report_absent',False)):
                with self.subTest(failure=key),self.assertRaises(RuntimeError):m.eighteenth_firstDEV_reference_ABI_failure_raw_only(dict(fake_spec,
                    additional_preserved_N17_firstDEV_reference_ABI_failure=dict(failure,**{key:wrong})))
            bad=copy.deepcopy(raw);bad['held_handle_exit']['actual_exit_code']=0;rawpath.write_text(json.dumps(bad),encoding='utf-8')
            with self.assertRaises(RuntimeError):m.eighteenth_firstDEV_reference_ABI_failure_raw_only(fake_spec)
if __name__=='__main__':unittest.main()
