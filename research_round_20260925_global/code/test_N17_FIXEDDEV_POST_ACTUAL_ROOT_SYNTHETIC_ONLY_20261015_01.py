"""One stdlib post-only API fixture; invented metadata, no science/RAW/SOURCE IO."""
import ast
import copy
import importlib.util
import os
from pathlib import Path
import types
import unittest

ENTRY=Path(os.environ.get('N17_POST_SYNTHETIC_ENTRY',str(Path(__file__).with_name('N17_FIXEDDEV_FULL187_POST_ACTUAL_ROOT_CPU_UNBOUND_DRAFT_20261015_01.py'))))

class PostOnlyMetadata(unittest.TestCase):
    def test_only_real_post_interfaces_and_explicit_completed_owner(self):
        definition=importlib.util.spec_from_file_location('_N17_post_synthetic',ENTRY)
        module=importlib.util.module_from_spec(definition);definition.loader.exec_module(module)
        calls=[];gate=object();record=object();path=Path('SYNTHETIC_POST_SPEC')
        spec={'final_record':record}
        def inherited(actual_path,actual_spec,*,cpu):
            self.assertIs(actual_path,path);self.assertIs(actual_spec,spec);self.assertIs(cpu,True);calls.append('own_gate');return gate
        def qualify(actual_record,actual_gate):
            self.assertIs(actual_record,record);self.assertIs(actual_gate,gate);calls.append('post_actual');return {'SYNTHETIC_ONLY':True}
        def forbidden(*a,**k):raise AssertionError('No run/COCO/model path is permitted')
        final=types.SimpleNamespace(inherited_source_gate=inherited,root_qualified_result=qualify,run=forbidden,COCO=forbidden)
        result,returned_gate=module.call_completed_final(final,path,spec)
        self.assertEqual(calls,['own_gate','post_actual']);self.assertIs(returned_gate,gate);self.assertEqual(result,{'SYNTHETIC_ONLY':True})
        def ref(name):return dict(path=str((ENTRY.parent/('SYNTHETIC_ONLY_'+name)).resolve()),bytes=0,sha256='0'*64)
        final_ref=ref('FINAL.py');final_ref.update(bytes=82341,sha256='41a1e8861402d1df1b39bde367b86e64ff7bea50ca7b2f4b2eb8620e350290bc')
        metadata=dict(schema=module.CPU_SCHEMA,normal_RAW_count=133,expected_normal_RAW_count=133,total_failed_RAW_count=20,
            total_terminal_identity_pin_count=153,deadline_seconds=1800,automatic_retry=False,allocator_limit_MiB=4096,whole_GPU_limit_MiB=8192,
            maximum_process_RSS_bytes=16*1024**3,entry_source=dict(ref('POST.py'),path=str(ENTRY.resolve())),
            CPU_admission_file=ref('POST_AD.json')['path'],phase='SYNTHETIC_POST',completed_final_CPU_source=final_ref,
            RAW_inputs=[{'phase':'SYNTHETIC_ANCESTOR'} for _ in range(133)])
        metadata['final_record']={k:ref('FINAL_'+k) for k in ('source','spec','Q','RAW','admission','phase_binding','bootstrap')}
        metadata['final_record'].update(source=final_ref,owner_index=132)
        self.assertTrue(module.validate_post_header(metadata))
        longer=copy.deepcopy(metadata);longer.update(normal_RAW_count=135,expected_normal_RAW_count=135,total_terminal_identity_pin_count=155)
        longer['RAW_inputs']+=[{'phase':'SYNTHETIC_ANCESTOR'} for _ in range(2)]
        self.assertTrue(module.validate_post_header(longer))
        for key,value in (('normal_RAW_count',132),('total_failed_RAW_count',19),('total_terminal_identity_pin_count',154),
                          ('automatic_retry',True),('phase','SYNTHETIC_ANCESTOR'),('CPU_admission_file',metadata['final_record']['admission']['path'])):
            with self.subTest(key=key),self.assertRaises(RuntimeError):module.validate_post_header(dict(metadata,**{key:value}))
        broken=copy.deepcopy(metadata);broken['final_record']['owner_index']=True
        with self.assertRaises(RuntimeError):module.validate_post_header(broken)
        broken=copy.deepcopy(metadata);broken['completed_final_CPU_source']['sha256']='1'*64
        with self.assertRaises(RuntimeError):module.validate_post_header(broken)
        tree=ast.parse(ENTRY.read_text(encoding='utf-8'))
        runtime=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('run','coco_metrics','proof_query','model_load_cpu') for n in ast.walk(runtime)))

if __name__=='__main__':unittest.main()
