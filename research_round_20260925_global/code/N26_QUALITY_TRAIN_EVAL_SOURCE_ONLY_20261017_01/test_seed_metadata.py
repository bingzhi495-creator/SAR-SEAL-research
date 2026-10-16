# encoding: utf-8
"""UNIT_FIXTURE: execute actual metadata AST only; no scientific imports/inputs."""
import ast
import copy
from pathlib import Path
import unittest

ROOT = Path(__file__).parent
SEEDS = [202610172601,202610172602,202610172603]


def source(name):
    return ast.parse((ROOT/name).read_text(encoding='utf-8'),filename=name)


def metadata(tree,names):
    nodes=[]
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('SEED','SEEDS') for t in node.targets):
            nodes.append(copy.deepcopy(node))
        elif isinstance(node,ast.FunctionDef) and node.name in names | {'need'}:
            nodes.append(copy.deepcopy(node))
    env={}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<UNIT_FIXTURE metadata AST>','exec'),env)
    return env


def evaluate(node,env):
    return eval(compile(ast.Expression(copy.deepcopy(node)),'<UNIT_FIXTURE expression>','eval'),env)


class SeedMetadataTests(unittest.TestCase):
    def setUp(self):
        self.pre=dict(schema='N26-fixed-query-mask-quality-train-prelock-v1',images=741,GT=1661,
            epochs=24,batch_size=64,steps_per_epoch=26,total_updates=624,seeds=SEEDS,
            AdamW_lr=.0003,AdamW_weight_decay=.0001,AdamW_betas=[.9,.999],AdamW_eps=1e-8)

    def test_train_exact_three_full_seeds_only(self):
        env=metadata(source('train_body.py'),{'validate_prelock','validate_seed'})
        for seed in SEEDS:
            env['validate_prelock'](self.pre,dict(operation='GPU_QUERY_MASK_QUALITY_TRAIN',seed=seed,allocator_limit_MiB=4096))
        for seed in (0,202610172604,True,float(SEEDS[0]),SEEDS[0] % 2**32):
            with self.subTest(seed=seed),self.assertRaises((RuntimeError,ValueError)):
                env['validate_prelock'](self.pre,dict(operation='GPU_QUERY_MASK_QUALITY_TRAIN',seed=seed,allocator_limit_MiB=4096))
        for locked in (SEEDS[:2],list(reversed(SEEDS)),SEEDS+[202610172604],[float(x) for x in SEEDS]):
            with self.subTest(locked=locked),self.assertRaises((RuntimeError,ValueError)):
                env['validate_prelock'](dict(self.pre,seeds=locked),dict(operation='GPU_QUERY_MASK_QUALITY_TRAIN',seed=SEEDS[0],allocator_limit_MiB=4096))

    def test_eval_seed_gate_is_called_by_real_run(self):
        tree=source('eval_quality_body.py');env=metadata(tree,{'validate_seed'})
        self.assertIn('validate_seed',list(env))
        run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        calls=[n for n in ast.walk(run) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='validate_seed']
        self.assertEqual(len(calls),1)
        self.assertEqual([ast.dump(a) for a in calls[0].args],[ast.dump(ast.Name(id='pre',ctx=ast.Load())),ast.dump(ast.Name(id='spec',ctx=ast.Load()))])
        for seed in SEEDS:
            self.assertEqual(env['validate_seed'](self.pre,dict(seed=seed)),seed)
        for seed in (202610172604,True,float(SEEDS[0]),SEEDS[0] % 2**32):
            with self.subTest(seed=seed),self.assertRaises((RuntimeError,ValueError)):
                env['validate_seed'](self.pre,dict(seed=seed))
        for locked in (SEEDS[:2],list(reversed(SEEDS)),[float(x) for x in SEEDS]):
            with self.subTest(locked=locked),self.assertRaises((RuntimeError,ValueError)):
                env['validate_seed'](dict(self.pre,seeds=locked),dict(seed=SEEDS[0]))

    def test_eval_actual_checkpoint_initialization_and_output_seed_are_dynamic(self):
        tree=source('eval_quality_body.py');env=metadata(tree,{'validate_seed'})
        run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        check=next(n for n in ast.walk(run) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
            and n.func.id=='need' and len(n.args)==2 and isinstance(n.args[1],ast.Constant)
            and str(n.args[1].value).startswith('Same factory/input bindings'))
        manual=next(n for n in ast.walk(run) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                    and n.func.attr=='manual_seed')
        binding=next(n.value for n in ast.walk(run) if isinstance(n,ast.Assign)
                     and any(isinstance(t,ast.Name) and t.id=='binding' for t in n.targets))
        binding_seed=next(k.value for k in binding.keywords if k.arg=='seed')
        for seed in SEEDS:
            spec=dict(seed=seed,merged_inputs='merged',inputs_qualified='accepted',quality_source='factory')
            checkpoint=dict(schema='N26-query-mask-quality-fixedE24-checkpoint-v1',seed=seed,
                            epoch=24,updates=624,parameters=299521,tensors=6,
                            merged_inputs='merged',inputs_qualified='accepted',quality_source='factory')
            scope=dict(env,spec=spec,checkpoint=checkpoint,seed=seed)
            self.assertTrue(evaluate(check.args[0],scope))
            self.assertEqual(evaluate(manual.args[0],scope),seed)
            self.assertEqual(evaluate(binding_seed,scope),seed)
            self.assertFalse(evaluate(check.args[0],dict(scope,checkpoint=dict(checkpoint,seed=SEEDS[(SEEDS.index(seed)+1)%3]))))


if __name__=='__main__':unittest.main()
