"""Minimal ordinary shard scoping/owner regression; no scientific I/O/GPU."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch

SOURCE=Path(__file__).with_name('N16_FIXED_E24_ALL100_SCOUT_SHARD_GPU_ROOT_UNBOUND_DRAFT_20261014_01.py')


def ordinary():
    d=importlib.util.spec_from_file_location('ordinary_shard',str(SOURCE));m=importlib.util.module_from_spec(d);d.loader.exec_module(m);return m


class FixedOrdinalShard(unittest.TestCase):
    def test_failed15_controlled_raw_and_abort_none_rejection(self):
        m=ordinary();phase='SYNTHETIC_ONLY_FAILURE15'
        ref=lambda s:dict(path='E:/SAR/seal/tmp/SYNTHETIC_ONLY/'+s,bytes=0,sha256='0'*64)
        raw=dict(status='ACTUAL_OWNED_TERMINAL',returncode=1,root_wait_reaped=True,original_identity={'SYNTHETIC_PID':15},
            held_handle_exit=dict(wait_state='WAIT_OBJECT_0',actual_exit_code=1),numeric_birth_absence=dict(original_birth_absent=True),
            error=None,abort=m.ABORT,wait_errors=False,GPU_observation_errors=[],minimum_ram_bytes=32*1024**3,elapsed_seconds=1800.3,
            admission_sha256='0'*64,whole_GPU_observed_peak_MiB=4000,whole_GPU_used_limit_MiB=8192,whole_leaf_VRAM_certified=False,GPU_scope=m.GPU_SCOPE)
        extra=dict(RAW_ref=ref(phase+'/PARENT_RECEIPT.json'),phase=phase,
            pin=dict(phase=phase,returncode=1,original_identity={'SYNTHETIC_PID':15},admission_sha256='0'*64),returncode=1,
            admission=ref('AD'),phase_binding=ref('PHASE'),bootstrap=ref('BOOT'),entry_source=ref('ENTRY'),spec=ref('SPEC'),original_report_absent=True)
        spec={'additional_preserved_N16_fullDEV_controlled_bundle_deadline_failure':extra,'terminal_identity_pins':{}}
        for i,k in enumerate(('additional_preserved_independent_CPU_failure','additional_preserved_N16_cache_failure',
            'additional_preserved_N16_cache_target_dtype_failure','additional_preserved_N16_controlled_resource_failure',
            'additional_preserved_N16_fullDEV_packet_ABI_failure','additional_preserved_N16_fullDEV_controlled_storage_failure')):
            spec[k]={'RAW_ref':ref('OTHER%d/PARENT_RECEIPT.json'%i)}
        m.read=lambda p:raw
        self.assertIs(m.fifteenth_controlled_bundle_deadline_failure_raw_only(spec),raw)
        raw['abort']=None
        with self.assertRaises(m.RawGateError):m.fifteenth_controlled_bundle_deadline_failure_raw_only(spec)

    def test_ranges_cover187_once_and_sink_retains_global47(self):
        m=ordinary();self.assertEqual([i for a,b in m.SHARD_RANGES for i in range(a,b)],list(range(187)))
        sink=m.ImageBundleCaptureSink(torch,SOURCE.parent,{'dataset_ordinal_range':[47,94]},lambda:None,
            SimpleNamespace(),{'SYNTHETIC_ONLY':True})
        sink.begin_image(47,1047);self.assertEqual(sink.index,47)
        bad=m.ImageBundleCaptureSink(torch,SOURCE.parent,{'dataset_ordinal_range':[47,94]},lambda:None,SimpleNamespace(),{})
        with self.assertRaises(RuntimeError):bad.begin_image(0,1047)

    def test_training_owners_stay107108_as_successful_shards_append(self):
        node=next(n for n in ast.parse(SOURCE.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='training_gate')
        stop=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
            and any(isinstance(a,ast.Constant) and a.value=='Actual original trainingGPU0/CPU0 latest chronological prefixes required' for a in n.value.args))
        node.body=node.body[:stop+1];tree=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
        def need(ok,message):
            if not ok:raise RuntimeError(message)
        for normal in (109,110,111,112,113,114):
            rows=list(range(normal));records={'GPU':{'normal_RAW_count':107,'RAW_inputs':rows[:107]},
                'CPU':{'normal_RAW_count':108,'RAW_inputs':rows[:108]},'REPORT':{},'Q':{}}
            namespace={'require':need,'closed':lambda c,r:r,'read':lambda r:records[r]};exec(compile(tree,str(SOURCE),'exec'),namespace)
            spec=dict(normal_RAW_count=normal,RAW_inputs=rows,training_GPU_spec='GPU',training_CPU_spec='CPU',training_GPU_report='REPORT',
                training_CPU_qualification='Q',training_GPU_RAW=107,training_CPU_RAW=108,training_GPU_admission_refs='A',training_CPU_admission_refs='B',
                training_GPU_source='GS',training_CPU_source='CS')
            gate=dict(closure={},context={},owners=[],lineage=SimpleNamespace(owner_ref_join=lambda c,o,raw,*args:raw))
            namespace['training_gate'](spec,gate,cpu=normal>=113)

    def test_firstpass_matches_global_selector_ordinals_for_second_shard(self):
        m=ordinary();torch.set_num_threads(2);m.closed=lambda c,r:r['path']
        selectors=[{'carrier':{'path':'SYNTHETIC_%d'%i},'L_digest':'SYNTHETIC_DIGEST_%d'%i} for i in range(187)]
        class FakeTorch:
            def __getattr__(self,name):return getattr(torch,name)
            def load(self,path,**kwargs):
                i=int(path.split('_')[-1]);return dict(native_L=torch.full((100,2,3),-3.),dataset_index=i,image_id=1000+i,
                    natural_query_ids=list(range(100)),L_digest='SYNTHETIC_DIGEST_%d'%i)
        fake_body=SimpleNamespace(domains_from_native=lambda np,engine,L:(L>0,np.abs(L)<=2,np.abs(L)<=2,
            np.flatnonzero(np.abs(L)<=2).astype(np.int64),int(np.count_nonzero(np.abs(L)<=2))))
        ids=list(range(1047,1094));rows=[{'band_counts_all100':[0]*100} for _ in ids]
        measured=m.firstpass_counts(FakeTorch(),np,None,fake_body,{'closure':{},'selector':{'manifest':{'rows':selectors}}},
            SimpleNamespace(imgs={i:{'height':2,'width':3} for i in ids}),ids,rows,
            SimpleNamespace(memory_info=lambda:SimpleNamespace(rss=64*1024**2)),lambda:None,16*1024**3,global_start=47)
        self.assertEqual([r['dataset_index'] for r in measured['rows']],list(range(47,94)))
        self.assertEqual([r['image_id'] for r in measured['rows']],ids)
        self.assertFalse(torch.cuda.is_initialized())

    def test_numerical_methods_remain_frozen_and_no_shard_COCO_call(self):
        new=ast.parse(SOURCE.read_text(encoding='utf-8'))
        old=ast.parse(SOURCE.with_name('N16_FIXED_E24_ALL100_SCOUT_FULLDEV_GPU_ROOT_UNBOUND_DRAFT_20261014_05.py').read_text(encoding='utf-8'))
        for name in ('model_load_cpu','candidate_engine','original_image','shared_actual_image','strip_event_to_shared','evaluate_image',
            'lookup_CPU_bundle_member','validate_CPU_image_bundle','decode_CPU_query_core_bundle'):
            self.assertEqual(ast.dump(next(n for n in old.body if getattr(n,'name',None)==name)),
                ast.dump(next(n for n in new.body if getattr(n,'name',None)==name)),name)
        run=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='coco_metrics' for n in ast.walk(run)))


if __name__=='__main__':unittest.main()
