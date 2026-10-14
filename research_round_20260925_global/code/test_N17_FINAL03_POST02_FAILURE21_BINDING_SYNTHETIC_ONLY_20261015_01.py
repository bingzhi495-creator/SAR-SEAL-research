"""One stdlib metadata-only final/post failure21 interface regression; no science IO."""
import ast
import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import types
import unittest

HERE=Path(__file__).parent
FINAL=Path(os.environ.get('N17_FAILURE21_FINAL_SYNTHETIC',str(HERE/'N17_FIXEDDEV_FULL187_CPU_COCO_FINAL_ROOT_UNBOUND_DRAFT_20261015_03.py')))
POST=Path(os.environ.get('N17_FAILURE21_POST_SYNTHETIC',str(HERE/'N17_FIXEDDEV_FULL187_POST_ACTUAL_ROOT_CPU_UNBOUND_DRAFT_20261015_02.py')))
SCHEMA='ROOT-N17-fixedDEV-full187-CPU-COCO-final-spec-v2'
def load(path,name):
    definition=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(definition);definition.loader.exec_module(value);return value
def ref(name):return dict(path=str((HERE/('SYNTHETIC_ONLY_'+name)).resolve()),bytes=0,sha256='0'*64)

class Binding21(unittest.TestCase):
    def test_final_v2_post_counts21_and_post_only_completed_api(self):
        final=load(FINAL,'_N17_failure21_final_synthetic')
        header=dict(schema=SCHEMA,normal_RAW_count=132,expected_normal_RAW_count=132,total_failed_RAW_count=21,
            total_terminal_identity_pin_count=153,deadline_seconds=1800,initialization_seed=202610150171,maximum_process_RSS_bytes=16*1024**3,
            automatic_retry=False,allocator_limit_MiB=4096,whole_GPU_limit_MiB=8192,entry_source=dict(ref('FINAL_SOURCE'),path=str(FINAL.resolve())))
        self.assertTrue(final.validate_final_header(header))
        for key,value in (('schema','ROOT-N17-fixedDEV-full187-CPU-COCO-final-spec-v1'),('total_failed_RAW_count',20),('total_terminal_identity_pin_count',152)):
            with self.subTest(final=key),self.assertRaises(RuntimeError):final.validate_final_header(dict(header,**{key:value}))
        post=load(POST,'_N17_failure21_post_synthetic');raw=FINAL.read_bytes()
        actual_source=dict(path=str(FINAL.resolve()),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        record={key:ref('FINAL_'+key) for key in ('source','spec','Q','RAW','admission','phase_binding','bootstrap')};record.update(source=actual_source,owner_index=132)
        metadata=dict(header,normal_RAW_count=133,expected_normal_RAW_count=133,total_terminal_identity_pin_count=154,
            entry_source=dict(ref('POST_SOURCE'),path=str(POST.resolve())),CPU_admission_file=ref('POST_AD')['path'],phase='SYNTHETIC_POST',
            RAW_inputs=[dict(phase='SYNTHETIC_COMPLETED') for _ in range(133)],completed_final_CPU_source=actual_source,final_record=record)
        self.assertEqual(post.CPU_SCHEMA,SCHEMA);self.assertEqual(post.COMPLETED_FINAL_SHA,actual_source['sha256']);self.assertEqual(post.COMPLETED_FINAL_BYTES,len(raw))
        self.assertTrue(post.validate_post_header(metadata))
        for key,value in (('total_failed_RAW_count',20),('total_terminal_identity_pin_count',153),('normal_RAW_count',132)):
            with self.subTest(post=key),self.assertRaises(RuntimeError):post.validate_post_header(dict(metadata,**{key:value}))
        calls=[];gate=object()
        def inherited(path,spec,*,cpu):self.assertIs(cpu,True);calls.append('inherited');return gate
        def qualify(actual_record,actual_gate):self.assertIs(actual_record,record);self.assertIs(actual_gate,gate);calls.append('post');return {'SYNTHETIC_ONLY':True}
        def forbidden(*a,**k):raise AssertionError('No COCO/run/forward in post')
        fake=types.SimpleNamespace(inherited_source_gate=inherited,root_qualified_result=qualify,run=forbidden,COCO=forbidden)
        result,_=post.call_completed_final(fake,Path('SYNTHETIC_POST_SPEC'),metadata)
        self.assertEqual(calls,['inherited','post']);self.assertEqual(result,{'SYNTHETIC_ONLY':True})
        tree=ast.parse(FINAL.read_text(encoding='utf-8'))
        source=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='source_gate')
        self.assertTrue(any(isinstance(n,ast.Constant) and n.value=='ROOT-N17-fixedDEV-local-center-error-CPU-numeric-shard-spec-v3' for n in ast.walk(source)))
        post_tree=ast.parse(POST.read_text(encoding='utf-8'))
        post_run=next(n for n in post_tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        written_counts=[n.value.value for n in ast.walk(post_run) if isinstance(n,ast.keyword) and n.arg=='total_failed_RAW_count' and isinstance(n.value,ast.Constant)]
        self.assertEqual(written_counts,[21])

if __name__=='__main__':unittest.main()
