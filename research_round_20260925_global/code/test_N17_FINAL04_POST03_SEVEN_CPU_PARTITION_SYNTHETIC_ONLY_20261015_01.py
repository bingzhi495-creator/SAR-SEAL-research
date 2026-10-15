"""One stdlib metadata-only mixed CPU-source/seven-scope merge regression; no science IO."""
import ast
import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import types
import unittest

HERE=Path(__file__).parent
FINAL=Path(os.environ.get('N17_SEVEN_CPU_FINAL_SYNTHETIC',str(HERE/'N17_FIXEDDEV_FULL187_CPU_COCO_FINAL_ROOT_UNBOUND_DRAFT_20261015_04.py')))
POST=Path(os.environ.get('N17_SEVEN_CPU_POST_SYNTHETIC',str(HERE/'N17_FIXEDDEV_FULL187_POST_ACTUAL_ROOT_CPU_UNBOUND_DRAFT_20261015_03.py')))
RANGES=((0,47),(47,70),(70,94),(94,117),(117,141),(141,164),(164,187))
SCHEMA='ROOT-N17-fixedDEV-full187-CPU-COCO-final-spec-v3'
def load(path,name):
    definition=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(definition);definition.loader.exec_module(value);return value
def ref(name,version=0):return dict(path=str((HERE/('SYNTHETIC_ONLY_'+name)).resolve()),bytes=version,sha256=str(version)*64)

class SevenCPUScopes(unittest.TestCase):
    def test_mixed_sources_18700_merge_and_post_only_api(self):
        final=load(FINAL,'_N17_seven_CPU_final_synthetic')
        self.assertEqual(final.RANGES,RANGES)
        legacy=ref('LEGACY_CPU03',3);new=ref('NEW_CPU04',4)
        spec=dict(legacy_numeric_CPU_shard_gate_source=legacy,numeric_CPU_shard_gate_source=new)
        descriptors=[]
        for sid,bounds in enumerate(RANGES):
            value=dict(cpu_shard_id=sid,cpu_dataset_ordinal_range=list(bounds),cpu_shard_owner_index=127+sid)
            value.update({key:ref('%d_%s'%(sid,key)) for key in ('source','spec','report','RAW','admission','phase_binding','bootstrap')})
            value['source']=dict(legacy if sid<3 else new,path=ref('%d_actual_source'%sid)['path'])
            descriptors.append(value)
        self.assertEqual(final.validated_CPU_descriptors(descriptors,134),descriptors)
        for sid,value in enumerate(descriptors):
            self.assertEqual(final.numeric_source_for_CPU_descriptor(value,spec),legacy if sid<3 else new)
            broken=copy.deepcopy(value);broken['source'].update(bytes=4 if sid<3 else 3,sha256=('4' if sid<3 else '3')*64)
            with self.assertRaises(RuntimeError):final.numeric_source_for_CPU_descriptor(broken,spec)
        for mutation in ('missing','range','old_owner','duplicate_RAW'):
            broken=copy.deepcopy(descriptors)
            if mutation=='missing':broken.pop()
            if mutation=='range':broken[3]['cpu_dataset_ordinal_range']=[94,141]
            if mutation=='old_owner':broken[1]['cpu_shard_owner_index']=129
            if mutation=='duplicate_RAW':broken[6]['RAW']=broken[5]['RAW']
            with self.subTest(mutation=mutation),self.assertRaises(RuntimeError):final.validated_CPU_descriptors(broken,134)
        ids=list(range(187));native=[dict(image_id=i,query_id=q,category_id=1,score=q/100.,segmentation=dict(size=[1,1],counts='SYNTHETIC')) for i in ids for q in reversed(range(100))]
        original=copy.deepcopy(native)
        for value in original:value['segmentation']['counts']='SYNTHETIC_GPU_%d_%d'%(value['image_id'],value['query_id'])
        original_GPU_pieces=[original[a*100:b*100] for a,b in ((0,47),(47,70),(70,94),(94,141),(141,187))]
        original=[value for piece in original_GPU_pieces for value in piece]
        pieces=[copy.deepcopy(original[a*100:b*100]) for a,b in RANGES]
        self.assertEqual(final.merge_policy_predictions(pieces,ids,native,original),original)
        broken=copy.deepcopy(pieces);broken[4][0]['score']+=.01
        with self.assertRaises(RuntimeError):final.merge_policy_predictions(broken,ids,native,original)
        broken=copy.deepcopy(pieces);broken[5][:100]=list(reversed(broken[5][:100]))
        with self.assertRaises(RuntimeError):final.merge_policy_predictions(broken,ids,native,original)
        header=dict(spec,schema=SCHEMA,normal_RAW_count=134,expected_normal_RAW_count=134,total_failed_RAW_count=21,total_terminal_identity_pin_count=155,
            initialization_seed=202610150171,deadline_seconds=1800,maximum_process_RSS_bytes=16*1024**3,automatic_retry=False,allocator_limit_MiB=4096,
            whole_GPU_limit_MiB=8192,entry_source=dict(ref('FINAL_SOURCE'),path=str(FINAL.resolve())))
        self.assertTrue(final.validate_final_header(header))
        post=load(POST,'_N17_seven_CPU_post_synthetic');data=FINAL.read_bytes()
        actual_source=dict(path=str(FINAL.resolve()),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        record={key:ref('FINAL_'+key) for key in ('source','spec','Q','RAW','admission','phase_binding','bootstrap')};record.update(source=actual_source,owner_index=134)
        metadata=dict(header,normal_RAW_count=135,expected_normal_RAW_count=135,total_terminal_identity_pin_count=156,entry_source=dict(ref('POST_SOURCE'),path=str(POST.resolve())),
            CPU_admission_file=ref('POST_AD')['path'],phase='SYNTHETIC_POST',RAW_inputs=[dict(phase='SYNTHETIC_COMPLETED') for _ in range(135)],completed_final_CPU_source=actual_source,final_record=record)
        self.assertTrue(post.validate_post_header(metadata));self.assertEqual(post.COMPLETED_FINAL_SHA,actual_source['sha256']);self.assertEqual(post.COMPLETED_FINAL_BYTES,len(data))
        calls=[];gate=object()
        def inherited(path,actual_spec,*,cpu):self.assertIs(cpu,True);calls.append('own');return gate
        def qualify(actual_record,actual_gate):self.assertIs(actual_record,record);self.assertIs(actual_gate,gate);calls.append('post');return {'SYNTHETIC_ONLY':True}
        def forbidden(*a,**k):raise AssertionError('No COCO/run/forward in post')
        result,_=post.call_completed_final(types.SimpleNamespace(inherited_source_gate=inherited,root_qualified_result=qualify,run=forbidden,COCO=forbidden),Path('SYNTHETIC_SPEC'),metadata)
        self.assertEqual(calls,['own','post']);self.assertEqual(result,{'SYNTHETIC_ONLY':True})
        tree=ast.parse(FINAL.read_text(encoding='utf-8'));source=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='source_gate')
        literals={n.value for n in ast.walk(source) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
        self.assertIn('ROOT-N17-fixedDEV-local-center-error-CPU-numeric-shard-spec-v3',literals)
        self.assertIn('ROOT-N17-fixedDEV-local-center-error-CPU-numeric-shard-spec-v4',literals)

if __name__=='__main__':unittest.main()
