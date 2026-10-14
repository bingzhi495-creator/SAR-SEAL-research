"""One ordinary GPU-entry ABI check; no Torch/CUDA/SAR/actual refs."""
import importlib.util
import ast
import copy
import json
import tempfile
from pathlib import Path
import unittest
ENTRY=Path(__file__).with_name('N17_FIXEDDEV_LOCAL_CENTER_ERROR_GPU_SHARD_ROOT_UNBOUND_DRAFT_20261015_04.py')
class Contract(unittest.TestCase):
    def test_new_header_and_new_arm_member_identifiers(self):
        s=importlib.util.spec_from_file_location('_n17_gpu_tiny',ENTRY);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
        value=dict(schema=m.SCHEMA,shard_id=1,dataset_ordinal_range=[47,70],normal_RAW_count=123,expected_normal_RAW_count=123,
            total_failed_RAW_count=20,total_terminal_identity_pin_count=143,initialization_seed=202610150171,
            deadline_seconds=1800,automatic_retry=False,allocator_limit_MiB=4096,whole_GPU_limit_MiB=8192,
            entry_source=dict(path=str(ENTRY.resolve()),bytes=0,sha256='0'*64))
        self.assertTrue(m.validate_gpu_header(value))
        for key,v in (('schema',m.TRAIN_SCHEMA),('normal_RAW_count',122),('shard_id',True),('dataset_ordinal_range',[0,46]),('total_failed_RAW_count',19)):
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

        with tempfile.TemporaryDirectory(prefix='n17_fake_failure19_') as temporary:
            phase=Path(temporary)/'FAKE19';phase.mkdir();rawpath=phase/'PARENT_RECEIPT.json'
            raw=dict(status='ACTUAL_OWNED_TERMINAL',returncode=1,original_identity={'pid':456,'birth':'FAKE'},root_wait_reaped=True,
                held_handle_exit=dict(wait_state='WAIT_OBJECT_0',actual_exit_code=1),numeric_birth_absence=dict(original_birth_absent=True),
                error=None,abort=None,wait_errors=False,GPU_observation_errors=[],admission_sha256='2'*64,
                whole_GPU_observed_peak_MiB=1,whole_GPU_used_limit_MiB=8192,whole_leaf_VRAM_certified=False,
                GPU_scope='whole GPU observed, not per-process WDDM allocation',phase='FAKE19',minimum_ram_bytes=16*1024**3,elapsed_seconds=1)
            rawpath.write_text(json.dumps(raw),encoding='utf-8')
            fake=lambda name:dict(path=str(phase/name),bytes=0,sha256='0'*64)
            failure=dict(RAW_ref=fake('PARENT_RECEIPT.json'),phase='FAKE19',pin=dict(phase='FAKE19',returncode=1,
                original_identity=raw['original_identity'],admission_sha256='2'*64),returncode=1,admission=fake('AD.json'),
                phase_binding=fake('PHASE.json'),bootstrap=fake('BOOT.py'),entry_source=fake('ENTRY.py'),spec=fake('SPEC.json'),original_report_absent=True)
            fake_spec=dict(terminal_identity_pins={},RAW_inputs=[])
            for key in m.FAILURE19_PRIOR_KEYS:fake_spec[key]=dict(RAW_ref=fake(key+'.json'))
            fake_spec['additional_preserved_N17_secondDEV_SOURCE_closure_failure']=failure
            # RAW-only stage may not compute a SOURCE hash or import scientific modules.
            original_reference=m.reference
            m.reference=lambda *a,**kw: (_ for _ in ()).throw(AssertionError('Forbidden SHA in RAW-only gate'))
            m.load=lambda *a,**kw: (_ for _ in ()).throw(AssertionError('Forbidden import in RAW-only gate'))
            self.assertEqual(m.nineteenth_secondDEV_SOURCE_closure_failure_raw_only(fake_spec)['returncode'],1)
            for key,wrong in (('returncode',True),('original_report_absent',False),('pin',dict(failure['pin'],admission_sha256='3'*64))):
                with self.subTest(failure19=key),self.assertRaises(RuntimeError):m.nineteenth_secondDEV_SOURCE_closure_failure_raw_only(dict(fake_spec,
                    additional_preserved_N17_secondDEV_SOURCE_closure_failure=dict(failure,**{key:wrong})))
            duplicate=copy.deepcopy(fake_spec);duplicate[m.FAILURE19_PRIOR_KEYS[-1]]['RAW_ref']=failure['RAW_ref']
            with self.assertRaises(RuntimeError):m.nineteenth_secondDEV_SOURCE_closure_failure_raw_only(duplicate)
            for key,wrong in (('minimum_ram_bytes',16*1024**3-1),('minimum_ram_bytes',True),('elapsed_seconds',1801),('elapsed_seconds',True),
                ('error','FAKEERROR'),('abort','FAKEABORT'),('returncode',0),('held_handle_exit',dict(wait_state='WAIT_OBJECT_0',actual_exit_code=0)),
                ('root_wait_reaped',False),('numeric_birth_absence',dict(original_birth_absent=False)),('whole_GPU_used_limit_MiB',8193)):
                bad=copy.deepcopy(raw);bad[key]=wrong;rawpath.write_text(json.dumps(bad),encoding='utf-8')
                with self.subTest(raw19=key),self.assertRaises(RuntimeError):m.nineteenth_secondDEV_SOURCE_closure_failure_raw_only(fake_spec)
            rawpath.write_text(json.dumps(raw),encoding='utf-8')
            gate=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='inherited_source_gate')
            calls=[n.func.id for n in ast.walk(gate) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
            self.assertLess(calls.index('nineteenth_secondDEV_SOURCE_closure_failure_raw_only'),calls.index('pre_gate'))
            oldtree=ast.parse(ENTRY.with_name('N17_FIXEDDEV_LOCAL_CENTER_ERROR_GPU_SHARD_ROOT_UNBOUND_DRAFT_20261015_03.py').read_text(encoding='utf-8'))
            oldfuncs={n.name:ast.dump(n,include_attributes=False) for n in oldtree.body if isinstance(n,ast.FunctionDef)}
            newfuncs={n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}
            changed={name for name in oldfuncs if oldfuncs[name]!=newfuncs[name]}
            self.assertEqual(changed,{'ninth_failed_raw_only','inherited_source_gate','validate_gpu_header','legacy_gate','source_gate','run'})
            self.assertEqual(ast.dump(next(n for n in oldtree.body if isinstance(n,ast.ClassDef) and n.name=='NewImageWriter'),include_attributes=False),
                ast.dump(next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NewImageWriter'),include_attributes=False))

        self.assertEqual(m.LEGACY_SHARD_RANGES,((0,47),(47,94),(94,141),(141,187)))
        self.assertEqual(m.SHARD_RANGES,((0,47),(47,70),(70,94),(94,141),(141,187)))
        for sid in range(1,5):self.assertTrue(m.validate_gpu_header(dict(value,shard_id=sid,dataset_ordinal_range=list(m.SHARD_RANGES[sid]),
            normal_RAW_count=122+sid,expected_normal_RAW_count=122+sid,total_terminal_identity_pin_count=142+sid)))
        with self.assertRaises(RuntimeError):m.validate_gpu_header(dict(value,shard_id=0,dataset_ordinal_range=[0,47],normal_RAW_count=122,expected_normal_RAW_count=122,total_terminal_identity_pin_count=142))
        # Old four-partition gate is identical after renaming its dedicated range constant.
        oldlegacy=next(n for n in oldtree.body if isinstance(n,ast.FunctionDef) and n.name=='legacy_gate')
        newlegacy=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='legacy_gate')
        for node in ast.walk(newlegacy):
            if isinstance(node,ast.Name) and node.id=='LEGACY_SHARD_RANGES':node.id='SHARD_RANGES'
        self.assertEqual(ast.dump(oldlegacy,include_attributes=False),ast.dump(newlegacy,include_attributes=False))
        with tempfile.TemporaryDirectory(prefix='n17_fake_failure20_') as temporary:
            phase=Path(temporary)/'FAKE20';phase.mkdir();rawpath=phase/'PARENT_RECEIPT.json'
            raw.update(phase='FAKE20',abort=m.ABORT,elapsed_seconds=1800.329)
            rawpath.write_text(json.dumps(raw),encoding='utf-8')
            fake=lambda name:dict(path=str(phase/name),bytes=0,sha256='0'*64)
            failure=dict(RAW_ref=fake('PARENT_RECEIPT.json'),phase='FAKE20',pin=dict(phase='FAKE20',returncode=1,
                original_identity=raw['original_identity'],admission_sha256='2'*64),returncode=1,admission=fake('AD.json'),
                phase_binding=fake('PHASE.json'),bootstrap=fake('BOOT.py'),entry_source=fake('ENTRY.py'),spec=fake('SPEC.json'),original_report_absent=True)
            fake_spec=dict(terminal_identity_pins={},RAW_inputs=[])
            for key in m.FAILURE20_PRIOR_KEYS:fake_spec[key]=dict(RAW_ref=fake(key+'.json'))
            fake_spec['additional_preserved_N17_DEV_GPU_controlled_deadline_failure']=failure
            self.assertEqual(m.twentieth_DEV_GPU_controlled_deadline_failure_raw_only(fake_spec)['abort'],m.ABORT)
            for key,wrong in (('abort',None),('elapsed_seconds',1799),('elapsed_seconds',1806),('minimum_ram_bytes',True),('whole_GPU_used_limit_MiB',8193)):
                bad=copy.deepcopy(raw);bad[key]=wrong;rawpath.write_text(json.dumps(bad),encoding='utf-8')
                with self.subTest(raw20=key),self.assertRaises((RuntimeError,m.RawGateError)):m.twentieth_DEV_GPU_controlled_deadline_failure_raw_only(fake_spec)
            rawpath.write_text(json.dumps(raw),encoding='utf-8')
            bad=copy.deepcopy(fake_spec);bad[m.FAILURE20_PRIOR_KEYS[-1]]['RAW_ref']=failure['RAW_ref']
            with self.assertRaises(RuntimeError):m.twentieth_DEV_GPU_controlled_deadline_failure_raw_only(bad)
        purecallnames=[n.func.id for n in ast.walk(gate) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
        self.assertLess(purecallnames.index('twentieth_DEV_GPU_controlled_deadline_failure_raw_only'),purecallnames.index('pre_gate'))

        m.reference=original_reference
        supplement=dict(schema=m.PARTITION_SUPPLEMENT_SCHEMA,logical_partition_count=4,physical_partition_count=5,
            logical_partition_ranges=[list(v) for v in m.LEGACY_SHARD_RANGES],physical_partition_ranges=[list(v) for v in m.SHARD_RANGES],
            shard0_reused_from_qualified_Source03=True,failed_partials_reused=False,method_training_data_metric_stopping_unchanged=True,
            single_active_leaf=True,deadline_seconds=1800,automatic_retry=False,owner_indices=[122,123,124,125,126],total_evaluation_walltime_increased=True,
            extra_ROOT_detail='FAKE supplementary metadata; no actual execution')
        with tempfile.TemporaryDirectory(prefix='n17_fake_execution20_') as temporary:
            p=Path(temporary)/'SUPPLEMENT.json';p.write_text(json.dumps(supplement),encoding='utf-8');ref=m.reference(p)
            s=dict(execution_partition_supplement=ref,expected_execution_partition_supplement=supplement)
            self.assertEqual(m.validate_execution_partition_supplement(s,{ref['path']:ref}),supplement)
            bad=dict(s,expected_execution_partition_supplement=dict(supplement,physical_partition_count=4))
            with self.assertRaises(RuntimeError):m.validate_execution_partition_supplement(bad,{ref['path']:ref})
            bad=dict(supplement,failed_partials_reused=True);p.write_text(json.dumps(bad),encoding='utf-8');ref=m.reference(p)
            with self.assertRaises(RuntimeError):m.validate_execution_partition_supplement(dict(execution_partition_supplement=ref,
                expected_execution_partition_supplement=bad),{ref['path']:ref})
if __name__=='__main__':unittest.main()
