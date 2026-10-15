"""One pure-stdlib invented-metadata scope test; no numerical or scientific execution."""
import ast
import copy
import importlib.util
from pathlib import Path
import unittest
ENTRY=Path(__file__).with_name('N17_FIXEDDEV_LOCAL_CENTER_ERROR_CPU_NUMERIC_SHARD_ROOT_UNBOUND_DRAFT_20261015_04.py')
class Contract(unittest.TestCase):
    def test_remaining_scopes_parent_mapping_and_numeric_body_unchanged(self):
        s=importlib.util.spec_from_file_location('_n17_cpu04_scope',ENTRY);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
        fake=lambda p:dict(path=str(ENTRY.resolve().parent/p),bytes=0,sha256='0'*64)
        def spec(sid):
            return dict(schema='ROOT-N17-fixedDEV-local-center-error-CPU-numeric-shard-spec-v4',cpu_shard_id=sid,CPU_parent_GPU_id=(0,1,2,3,3,4,4)[sid],
                cpu_dataset_ordinal_range=list(((0,47),(47,70),(70,94),(94,117),(117,141),(141,164),(164,187))[sid]),normal_RAW_count=127+sid,
                expected_normal_RAW_count=127+sid,cpu_shard_owner_index=127+sid,total_failed_RAW_count=21,total_terminal_identity_pin_count=148+sid,
                deadline_seconds=1800,automatic_retry=False,allocator_limit_MiB=4096,whole_GPU_limit_MiB=8192,maximum_process_RSS_bytes=16*1024**3,
                initialization_seed=202610150171,entry_source=fake(ENTRY.name),execution_partition_supplement=fake('GPU_LOCK.json'),CPU_execution_partition_supplement=fake('CPU_LOCK.json'))
        first=spec(3);self.assertTrue(m.validate_cpu_header(first))
        seen=[]
        for sid in range(3,7):
            child=spec(sid);self.assertTrue(m.validate_cpu_header(child));pid=child['CPU_parent_GPU_id'];a,z=m.SHARD_RANGES[pid]
            source=fake('GPU04.py');child['GPU_outer_source']=source
            item=dict(shard_id=pid,dataset_ordinal_range=[a,z],source=source,spec=fake('GPU_SPEC.json'),report=fake('GPU_REPORT.json'),RAW=fake('RAW.json'),
                admission=fake('AD.json'),phase_binding=fake('PHASE.json'),bootstrap=fake('BOOT.py'),shard_owner_index=122+pid)
            child['GPU_shard']=item;self.assertEqual(m.validate_parent_record(child),item)
            rows=[dict(dataset_index=i,image_id=1000+i,qids=list(reversed(range(100))),bundle=fake('BUNDLE_%03d.pth'%i)) for i in range(a,z)]
            groups={'p':{r['image_id']:[dict(image_id=r['image_id'],query_id=q,score=.5) for q in r['qids']] for r in rows}}
            manifest=dict(rows=rows);report=dict(scope_dataset_indices=list(range(a,z)),scope_image_ids=[r['image_id'] for r in rows])
            gate=dict(parent_GPU_manifest=manifest,parent_GPU_report=report,expected_prediction_groups=groups,context={'unchanged':'ROOT'},
                expected_predictions={'p':[v for r in rows for v in groups['p'][r['image_id']]]})
            before=copy.deepcopy(gate);m.scope_parent_GPU(child,gate);fa,fz=child['cpu_dataset_ordinal_range'];seen.extend(r['dataset_index'] for r in gate['scope_rows'])
            self.assertEqual([r['dataset_index'] for r in gate['scope_rows']],list(range(fa,fz)));self.assertEqual(len(gate['scope_expected_predictions']['p']),(fz-fa)*100)
            self.assertEqual(gate['scope_capture_refs'],[r['bundle'] for r in rows[fa-a:fz-a]])
            for key in before:self.assertEqual(gate[key],before[key])
            bad=dict(child,CPU_parent_GPU_id=4 if pid==3 else 3)
            with self.assertRaises(RuntimeError):m.validate_parent_record(bad)
            badgate=copy.deepcopy(before);badgate['parent_GPU_manifest']['rows'].pop()
            with self.assertRaises(RuntimeError):m.scope_parent_GPU(child,badgate)
        self.assertEqual(seen,list(range(94,187)));self.assertEqual(len(seen),len(set(seen)))
        for sid in range(3):
            with self.assertRaises(RuntimeError):m.validate_cpu_header(spec(sid))
        oldtree=ast.parse(ENTRY.with_name('N17_FIXEDDEV_LOCAL_CENTER_ERROR_CPU_NUMERIC_SHARD_ROOT_UNBOUND_DRAFT_20261015_03.py').read_text(encoding='utf-8'))
        tree=ast.parse(ENTRY.read_text(encoding='utf-8'));oldfuncs={n.name:n for n in oldtree.body if isinstance(n,ast.FunctionDef)};newfuncs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
        changes={name for name in oldfuncs if ast.dump(oldfuncs[name],include_attributes=False)!=ast.dump(newfuncs[name],include_attributes=False)}
        self.assertEqual(changes,{'validate_cpu_header','validate_parent_record','source_gate','run'})
        oldloop=next(n for n in oldfuncs['run'].body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='row')
        newloop=next(n for n in newfuncs['run'].body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='row')
        self.assertEqual(ast.dump(ast.Module(body=oldloop.body,type_ignores=[]),include_attributes=False),ast.dump(ast.Module(body=newloop.body,type_ignores=[]),include_attributes=False))
        for name in ('original_head','independent_PR','encode','check'):
            old=next(n for n in oldfuncs['run'].body if isinstance(n,ast.FunctionDef) and n.name==name)
            new=next(n for n in newfuncs['run'].body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(new,include_attributes=False))
if __name__=='__main__':unittest.main(verbosity=2)
