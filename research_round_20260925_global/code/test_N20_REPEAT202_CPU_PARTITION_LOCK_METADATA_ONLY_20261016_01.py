"""Pure metadata regression; no tensors, actual records, datasets, or file payloads."""
import ast,copy,json,unittest
from pathlib import Path
BASE=Path(__file__).parent
FRONT=BASE/'N20_REPEAT202_FIXEDDEV_CPU_FRONT_UNBOUND_DRAFT_20261016_01.py'
HELPER=BASE/'N20_REPEAT202_TRAIN_GPU_FRONT_UNBOUND_DRAFT_20261016_01.py'
def extract(path,name):
    return next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name==name)
def require(ok,message):
    if not ok:raise AssertionError(message)
tree=ast.parse(FRONT.read_text(encoding='utf-8'))
RANGES=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='N20_CPU_RANGES' for t in n.targets)))
class PartitionLockMetadata(unittest.TestCase):
    def evaluate(self,plan,cid=0,owners=None,expected=None):
        owners=owners or list(range(187,219))
        strict={'path':'C:/SYNTHETIC_ONLY/plan.json','bytes':1,'sha256':'0'*64}
        spec={'N20_repeat202_CPU_execution_partition_lock':strict,
            'expected_N20_repeat202_CPU_execution_partition_lock':copy.deepcopy(plan if expected is None else expected),
            'completed_GPU_shards':[{'shard_owner_index':i} for i in (183,184,185,186)],
            'cpu_shard_id':cid,'qualification_owner_index':owners[cid]}
        gate={'closure':{strict['path']:strict}}
        def closed(closure,value):
            require(closure.get(value['path'])==value,'Strictref membership');return value['path']
        namespace={'json':json,'require':require,'N20_CPU_RANGES':RANGES,'closed':closed,'read':lambda path:copy.deepcopy(plan)}
        nodes=[extract(HELPER,'n20_repeat202_exact'),extract(FRONT,'n20_repeat202_CPU_partition')]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(FRONT),'exec'),namespace)
        result=namespace['n20_repeat202_CPU_partition'](spec,gate)
        self.assertEqual(result,plan);self.assertEqual(gate['N20_repeat202_CPU_execution_partition_lock'],plan)
    def plan(self,owners=None):
        owners=owners or list(range(187,219))
        return {'schema':'ROOT-N20-repeat202-fixedDEV-CPU32-execution-partition-lock-v1',
            'status':'PRELOCKED_BEFORE_REPEAT202_CPU_OR_AP','initialization_seed':202610150202,
            'full_dataset_range':[0,187],'CPU_dataset_ranges':[list(v) for v in RANGES],
            'GPU_owner_indices':[183,184,185,186],'CPU_owner_indices':owners,
            'normal_RAW_count_before_first_CPU':owners[0],'total_failed_RAW_count':26,
            'budget_or_fusion_allowed':False,'AP_evaluated':False}
    def test_every_original_scope_and_nonconsecutive_explicit_owners(self):
        for cid in range(32):
            with self.subTest(cid=cid):self.evaluate(self.plan(),cid)
        owners=list(range(197,261,2));self.evaluate(self.plan(owners),31,owners)
    def test_semantically_invalid_locks_fail_even_when_expected_JSON_matches(self):
        changes=[('schema','old'),('status','late'),('initialization_seed',202610150201),
            ('full_dataset_range',[0,186]),('CPU_dataset_ranges',[list(v) for v in RANGES[:-1]]),
            ('GPU_owner_indices',[183,184,185,187]),('CPU_owner_indices',[187]*32),
            ('CPU_owner_indices',list(reversed(range(187,219)))),('CPU_owner_indices',list(range(186,218))),
            ('CPU_owner_indices',[True]+list(range(188,219))),('normal_RAW_count_before_first_CPU',188),
            ('total_failed_RAW_count',25),('budget_or_fusion_allowed',True),('AP_evaluated',True)]
        for key,value in changes:
            with self.subTest(field=key,value=value):
                plan=self.plan();plan[key]=value
                with self.assertRaises(AssertionError):self.evaluate(plan)
    def test_current_owner_and_exact_expected_JSON_must_match(self):
        owners=list(range(188,220))
        with self.assertRaises(AssertionError):self.evaluate(self.plan(),owners=owners)
        expected=self.plan();expected['AP_evaluated']=True
        with self.assertRaises(AssertionError):self.evaluate(self.plan(),expected=expected)
if __name__=='__main__':unittest.main(verbosity=2)
