"""Pure stdlib synthetic metadata regression. Never imports the scientific entry."""
import ast
import copy
import json
import math
from pathlib import Path
import unittest

TARGET = Path(__file__).with_name('N20_REPEAT202_FULL187_COCO_CPU_FRONT_UNBOUND_DRAFT_20261016_01.py')

def require(ok, message):
    if not ok:
        raise ValueError(message)

def api():
    tree = ast.parse(TARGET.read_text(encoding='utf-8'))
    names = {'n20_final_exact','n20_validate_final_header','n20_validate_final_partition','n20_validate_final_normal_provenance','n20_final_parent_metadata','n20_final_binding'}
    selected = [n for n in tree.body if (isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id.startswith('N20_FINAL_') for x in n.targets)) or (isinstance(n, ast.FunctionDef) and n.name in names)]
    module = ast.Module(body=selected, type_ignores=[])
    env = dict(json=json, math=math, Path=Path, require=require, __file__=str(TARGET),
               ref_shape=lambda ref: require(type(ref) is dict and set(ref)=={'path','bytes','sha256'} and type(ref['bytes']) is int and type(ref['path']) is str and type(ref['sha256']) is str, 'synthetic strictref'))
    exec(compile(ast.fix_missing_locations(module), str(TARGET), 'exec'), env)
    return env

def fixture(env):
    def ref(label):
        return dict(path='/SYNTHETIC_ONLY/'+label, bytes=1, sha256='a'*64)
    own=400;gowners=[183,185,188,192];cowners=list(range(210,274,2))
    gs=[dict(shard_id=i,dataset_ordinal_range=list(v),shard_owner_index=gowners[i],
             **{k:ref('g%d_%s'%(i,k)) for k in ('source','spec','report','RAW','admission','phase_binding','bootstrap')}) for i,v in enumerate(env['N20_FINAL_GPU_RANGES'])]
    cs=[dict(cpu_shard_id=i,cpu_dataset_ordinal_range=list(v),cpu_shard_owner_index=cowners[i],
             **{k:ref('c%d_%s'%(i,k)) for k in ('source','spec','report','RAW','admission','phase_binding','bootstrap')}) for i,v in enumerate(env['N20_FINAL_CPU_RANGES'])]
    plan=dict(schema='ROOT-N20-repeat202-fixedDEV-CPU32-execution-partition-lock-v1',status='PRELOCKED_BEFORE_REPEAT202_CPU_OR_AP',initialization_seed=202610150202,
              full_dataset_range=[0,187],CPU_dataset_ranges=[list(v) for v in env['N20_FINAL_CPU_RANGES']],GPU_owner_indices=gowners,CPU_owner_indices=cowners,
              normal_RAW_count_before_first_CPU=cowners[0],total_failed_RAW_count=26,budget_or_fusion_allowed=False,AP_evaluated=False)
    return dict(schema='ROOT-N20-repeat202-fixedDEV-full187-CPU-COCO-final-spec-v1',final_owner_index=own,normal_RAW_count=own,expected_normal_RAW_count=own,
                total_failed_RAW_count=26,total_terminal_identity_pin_count=own+26,initialization_seed=202610150202,deadline_seconds=1800,automatic_retry=False,
                maximum_process_RSS_bytes=16*1024**3,failed26_partials_reused=False,allocator_limit_MiB=4096,whole_GPU_limit_MiB=8192,
                entry_source=dict(path=str(TARGET),bytes=1,sha256='b'*64),numeric_CPU_shards=cs,completed_GPU_shards=gs,full_dataset_ordinal_range=[0,187],
                expected_N20_repeat202_CPU_execution_partition_lock=plan)


def preview_api():
    path=TARGET.with_name('N20_REPEAT202_FULL187_CPU_COCO_FINAL_RUN_TAIL_UNBOUND_DRAFT_20261016_01.py')
    tree=ast.parse(path.read_text(encoding='utf-8'))
    names={'n20_final_AP_percent','n20_final_main_preview'}
    env=dict(require=require,math=math,api=lambda module,name:getattr(module,name))
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])),str(path),'exec'),env)
    ipath=TARGET.with_name('N20_TWO_SEED_PRELOCKED_RESULT_INTERPRETER_UNBOUND_20261016_01.py')
    itree=ast.parse(ipath.read_text(encoding='utf-8'))
    ienv=dict(math=math)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[n for n in itree.body if isinstance(n,(ast.Assign,ast.FunctionDef))],type_ignores=[])),str(ipath),'exec'),ienv)
    from types import SimpleNamespace
    return env['n20_final_main_preview'],SimpleNamespace(**ienv)

class Full187Metadata(unittest.TestCase):
    def test_explicit_completed_cpu_owner_must_match_prelocked_slot(self):
        env=api();spec=fixture(env)
        self.assertEqual(env['n20_validate_final_header'](spec),spec['numeric_CPU_shards'])
        wrong=copy.deepcopy(spec);wrong['numeric_CPU_shards'][17]['cpu_shard_owner_index']+=1
        with self.assertRaises((ValueError,KeyError)):
            env['n20_validate_final_header'](wrong)

    def test_normal202_provenance_rejects_recovery_missing_or_false_flags(self):
        env=api();check=env['n20_validate_final_normal_provenance']
        good=dict(original_training_GPU_actual0_verified=True,training_saved_artifact_recovery=False,independent_endpoint_CPU_qualified=True)
        check(good)
        for key in good:
            wrong=dict(good);wrong[key]=not good[key]
            with self.subTest(key=key),self.assertRaises((ValueError,KeyError)):
                check(wrong)
            wrong=dict(good);del wrong[key]
            with self.subTest(missing=key),self.assertRaises((ValueError,KeyError)):
                check(wrong)
        wrong=dict(good,training_artifacts_recovery_qualified=True)
        with self.assertRaises(ValueError):
            check(wrong)

    def test_second_seed_preview_keeps_both_seed_identities(self):
        preview,interpreter=preview_api()
        def controls(local,center):
            return {'E24_LOCAL_ERROR_atmost':{'stats':[local]},'E24_CENTER_ERROR_atmost':{'stats':[center]},'strong_fixed_E24_Omega':{'stats':[68.09938494745545/100]}}
        first=dict(qualified=True,initialization_seed=202610150201,control_results=controls(.69,.68))
        gate=dict(first201_root_permission=dict(final_qualified_result=first))
        second=controls(.70,.69)
        result=preview(interpreter,second,gate)
        self.assertEqual(result['seeds'],[202610150201,202610150202])
        self.assertEqual(result['main_results'][1]['LOCAL_AP'],70.)
        self.assertFalse(result['qualification_performed'])
        self.assertEqual(result['main_results'][0]['seed'],202610150201)

    def test_parent_metadata_rejects_old13_or_wrong14_binding_before_consuming_cache(self):
        from types import SimpleNamespace
        class ReachedCache(Exception):
            pass
        for keys,accepted in ((None,True),('old13',False),('wrong14',False)):
            env=api();spec=fixture(env);numeric=SimpleNamespace()
            numeric.N20_DEV_CPU_SCHEMA=env['N20_FINAL_NUMERIC_SCHEMA']
            numeric.N20_CPU_RANGES=env['N20_FINAL_CPU_RANGES'];numeric.N20_CPU_PARENT_GPU_IDS=env['N20_FINAL_PARENT_IDS']
            binding=env['N20_FINAL_GPU_BINDING_KEYS']
            numeric.N20_GPU_BINDING_KEYS=binding if keys is None else (binding[:-1] if keys=='old13' else binding[:-1]+('old_recovery_root',))
            for name in ('n20_validate_cpu_header','n20_repeat202_CPU_partition','n20_completed_DEV_GPU_record','n20_scope_parent_GPU'):
                setattr(numeric,name,lambda *args:None)
            numeric.N20_CPU_SOURCE_KEYS=()
            for key in ('numeric_CPU_shard_gate_source','GPU_outer_source','interpreter_source','N20_repeat202_CPU_execution_partition_lock','N20_failed26_helper_source'):
                spec[key]=dict(path='/SYNTHETIC_ONLY/'+key,bytes=1,sha256='c'*64)
            env['closed']=lambda closure,ref:ref['path']
            env['read']=lambda path:copy.deepcopy(spec['expected_N20_repeat202_CPU_execution_partition_lock'])
            env['load']=lambda ref,name:numeric
            env['n20_repeat202_first_permission']=lambda *args:({}, {})
            env['n20_repeat202_history_view']=lambda *args:{}
            for number in range(22,27):
                env['n20_failed%d_raw_only'%number]=lambda *args:{}
                env['n20_failed%d_closed'%number]=lambda *args:None
            def cache(*args):
                raise ReachedCache('valid metadata proceeds past complete14 ABI')
            env['n20_repeat202_cache']=cache
            with self.subTest(keys=keys),self.assertRaises(ReachedCache if accepted else ValueError):
                env['n20_final_parent_metadata'](spec,dict(closure={}))

    def test_full187_header_rejects_numeric_child_alias_seed_or_owner_order(self):
        env=api();base=fixture(env)
        mutations=[lambda s:s.update(cpu_shard_id=0),lambda s:s.update(initialization_seed=202610150201),
                   lambda s:s.update(final_owner_index=True),lambda s:s['completed_GPU_shards'][1].update(shard_owner_index=183),
                   lambda s:s.update(full_dataset_ordinal_range=[0,6])]
        for mutate in mutations:
            spec=copy.deepcopy(base);mutate(spec)
            with self.assertRaises((ValueError,KeyError)):
                env['n20_validate_final_header'](spec)

if __name__=='__main__':
    unittest.main()
