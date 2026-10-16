# encoding: utf-8
"""UNIT_FIXTURE: stdlib metadata only; no Torch/NumPy/project imports or data."""
import ast
import importlib.util
from pathlib import Path
import types
import unittest

SOURCE = Path(__file__).with_name('capture_body.py')
MODULE_SPEC = importlib.util.spec_from_file_location('n26_capture_unit', SOURCE)
body = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(body)


class CaptureMetadataTests(unittest.TestCase):
    def setUp(self):
        self.pre = dict(dataset_shards=[[0,371],[371,741]],images=741,GT=1661,
                        models='frozen_R_E24',no_new_training=True)
        self.ctx = dict(images=741,GT=1661,dataset_indices=list(range(741)))

    def test_only_prelocked_complete_shard_is_accepted(self):
        for first,last in self.pre['dataset_shards']:
            values=list(range(first,last))
            self.assertEqual(body.checked_scope(dict(dataset_indices=values),self.pre,self.ctx),values)
        for values in ([],[0,0],list(range(370)),list(range(372)),[740,741]):
            with self.subTest(values=values[:3]):
                with self.assertRaises(RuntimeError):
                    body.checked_scope(dict(dataset_indices=values),self.pre,self.ctx)
        bad=dict(self.pre,no_new_training=False)
        with self.assertRaises(RuntimeError):
            body.checked_scope(dict(dataset_indices=list(range(371))),bad,self.ctx)

    def test_matched_pairs_retain_raw_GT_order_and_all_quality_levels(self):
        indices=[-1]*100;indices[91]=0;indices[4]=1
        self.assertEqual(body.matched_query_pairs(indices,[81,92]),[(0,81,91),(1,92,4)])
        # This metadata function has no IoU input and cannot drop low-quality GT.
        for bad in (indices[:99],[-1]*100,[-1]*99+[2]):
            with self.assertRaises(RuntimeError):body.matched_query_pairs(bad,[81,92])
        duplicate=list(indices);duplicate[6]=0
        with self.assertRaises(RuntimeError):body.matched_query_pairs(duplicate,[81,92])

    def test_original_2048_chunk_and_true_tail(self):
        self.assertEqual(list(body.point_blocks(0)),[])
        self.assertEqual(list(body.point_blocks(2048)),[(0,2048)])
        self.assertEqual(list(body.point_blocks(4097)),[(0,2048),(2048,4096),(4096,4097)])
        with self.assertRaises(RuntimeError):list(body.point_blocks(-1))

    def test_real_run_binding_uses_execution_admission_ref(self):
        tree=ast.parse(SOURCE.read_text(encoding='utf-8'))
        run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        binding=next(n.value for n in ast.walk(run) if isinstance(n,ast.Assign)
                     and any(isinstance(t,ast.Name) and t.id=='binding' for t in n.targets))
        c=types.SimpleNamespace(execution_spec_ref={'spec':1},execution_admission_ref={'ad':1},
                                ref=lambda path:{'source':str(path)})
        spec={k:k for k in ('phase','context','prelock','setup_source','quality_source')}
        ctx={k:k for k in ('teacher_checkpoint','fixed_PR_checkpoint')}
        value=eval(compile(ast.Expression(binding),'<UNIT_FIXTURE run binding>','eval'),
                   dict(c=c,spec=spec,ctx=ctx,Path=Path,__file__=str(SOURCE)))
        self.assertEqual(value['GPU_admission'],c.execution_admission_ref)

    def test_original_N07_provenance_uses_only_actual_writer_fields(self):
        tree=ast.parse(SOURCE.read_text(encoding='utf-8'))
        capture=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_capture_image')
        call=next(n for n in ast.walk(capture) if isinstance(n,ast.Call)
                  and isinstance(n.func,ast.Name) and n.func.id=='require'
                  and len(n.args)==2 and isinstance(n.args[1],ast.Constant)
                  and n.args[1].value=='Real original packet schemas/provenance required')
        expr=compile(ast.Expression(call.args[0]),'<UNIT_FIXTURE original N07 provenance>','eval')
        report=dict(source={'source':1},spec={'spec':1},GPU_admission={'ad':1},phase='old')
        packet=dict(schema='N16-all100-FIT-action-sampled-cache-body-v1')
        carrier=dict(schema='N16-actual-GPU-natural100-L-carrier-v1')
        old_packet={k:report[k] for k in ('source','spec','phase')}
        old_packet['schema']='original-N07'
        env=dict(packet=packet,carrier=carrier,old_packet=old_packet,old_report=report,
                 cache7=types.SimpleNamespace(PACKET_SCHEMA='original-N07'))
        self.assertTrue(eval(expr,env))
        for key in ('source','spec','phase'):
            with self.subTest(key=key):
                changed=dict(old_packet);changed[key]='different'
                self.assertFalse(eval(expr,dict(env,old_packet=changed)))


if __name__=='__main__': unittest.main()
