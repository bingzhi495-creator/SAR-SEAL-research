"""Stdlib-only regression over ordinary draft AST; no scientific I/O/import.

Execute only the training_gate owner-prefix statements, supplying synthetic
read/closure/lineage callbacks. No gate, model, Source hash or CLI is run.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

HERE=Path(__file__).parent
STEM='N16_FIXED_E24_ALL100_SCOUT_FULLDEV_GPU_ROOT_UNBOUND_DRAFT_20261014_'


def owner_prefix_fixture(version):
    path=HERE/(STEM+version+'.py')
    node=next(n for n in ast.parse(path.read_text(encoding='utf-8')).body
              if isinstance(n,ast.FunctionDef) and n.name=='training_gate')
    stop=next(i for i,n in enumerate(node.body) if isinstance(n,ast.Expr)
              and isinstance(n.value,ast.Call) and any(isinstance(a,ast.Constant)
              and a.value=='Actual original trainingGPU0/CPU0 latest chronological prefixes required'
              for a in n.value.args))
    node.body=node.body[:stop+1]
    node.body.append(ast.Return(value=ast.Name(id='cpu',ctx=ast.Load())))
    module=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
    records={}
    def require(condition,message):
        if not condition:raise RuntimeError(message)
    namespace=dict(require=require,closed=lambda closure,ref:ref,read=lambda ref:records[ref])
    exec(compile(module,str(path),'exec'),namespace)
    return namespace['training_gate'],records


def synthetic_call(version,cpu,slots=(107,108)):
    function,records=owner_prefix_fixture(version)
    raws=['SYNTHETIC_RAW_%d'%i for i in range(109+int(cpu))]
    spec=dict(normal_RAW_count=len(raws),RAW_inputs=raws,training_GPU_spec='gpu_spec',
        training_GPU_report='gpu_report',training_CPU_spec='cpu_spec',training_CPU_qualification='cpu_q',
        training_GPU_RAW='gpu_RAW',training_CPU_RAW='cpu_RAW',
        training_GPU_admission_refs='gpu_ad',training_CPU_admission_refs='cpu_ad',
        training_GPU_source='gpu_source',training_CPU_source='cpu_source')
    records.update(gpu_spec=dict(normal_RAW_count=107,RAW_inputs=raws[:107]),
        cpu_spec=dict(normal_RAW_count=108,RAW_inputs=raws[:108]),gpu_report={},cpu_q={})
    observed=[]
    def join(context,owners,raw,ad,source,parent_spec,cuda):
        observed.append((raw,cuda))
        return slots[0] if raw=='gpu_RAW' else slots[1]
    gate=dict(closure={},context={},owners=[],lineage=SimpleNamespace(owner_ref_join=join))
    result=function(spec,gate,cpu=cpu)
    return result,observed


class OwnerPrefixRegression(unittest.TestCase):
    def test_old_01_reproduces_shadowing_for_both_modes(self):
        for cpu in (False,True):
            with self.subTest(cpu=cpu),self.assertRaisesRegex(TypeError,r'int\(\) argument'):synthetic_call('01',cpu)

    def test_gpu_false109_joins107108_and_preserves_boolean(self):
        result,observed=synthetic_call('02',False)
        self.assertIs(result,False)
        self.assertEqual(observed,[('gpu_RAW','0'),('cpu_RAW','')])

    def test_cpu_true110_keeps107108_before_latest_dev_gpu(self):
        result,observed=synthetic_call('02',True)
        self.assertIs(result,True)
        self.assertEqual(observed,[('gpu_RAW','0'),('cpu_RAW','')])

    def test_cpu_rejects_shifted108109_training_owner_slots(self):
        with self.assertRaises(RuntimeError):synthetic_call('02',True,slots=(108,109))


if __name__=='__main__':unittest.main()
