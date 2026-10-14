"""One stdlib synthetic five-partition merge test; no science/GT/model/ref IO."""
import copy
import importlib.util
import os
from pathlib import Path
import unittest

ENTRY=Path(os.environ.get('N17_SYNTHETIC_ENTRY',str(Path(__file__).with_name('N17_FIXEDDEV_FULL187_CPU_COCO_FINAL_ROOT_UNBOUND_DRAFT_20261015_02.py'))))
RANGES=((0,47),(47,70),(70,94),(94,141),(141,187))

class FivePartitionMetadata(unittest.TestCase):
    def test_complete_18700_merge_and_explicit_five_owners(self):
        definition=importlib.util.spec_from_file_location('_N17_final_five_synthetic',ENTRY)
        module=importlib.util.module_from_spec(definition);definition.loader.exec_module(module)
        def fake_ref(name):return dict(path=str((Path(__file__).parent/('SYNTHETIC_ONLY_'+name)).resolve()),bytes=0,sha256='0'*64)
        descriptors=[]
        for sid,bounds in enumerate(RANGES):
            value=dict(cpu_shard_id=sid,cpu_dataset_ordinal_range=list(bounds),cpu_shard_owner_index=127+sid)
            value.update({key:fake_ref('%d_%s'%(sid,key)) for key in ('source','spec','report','RAW','admission','phase_binding','bootstrap')})
            descriptors.append(value)
        self.assertEqual(module.validated_CPU_descriptors(descriptors,132),descriptors)
        ids=list(range(187));native=[]
        for image in ids:
            for query in reversed(range(100)):
                native.append(dict(image_id=image,query_id=query,category_id=1,score=query/100.,segmentation=dict(size=[1,1],counts='SYNTHETIC')))
        pieces=[copy.deepcopy(native[first*100:last*100]) for first,last in RANGES]
        self.assertEqual(module.merge_policy_predictions(pieces,ids,native,native),native)
        for broken in (descriptors[:4],descriptors+[descriptors[-1]]):
            with self.assertRaises(RuntimeError):module.validated_CPU_descriptors(broken,132)
        broken=copy.deepcopy(descriptors);broken[2]['cpu_dataset_ordinal_range']=[70,95]
        with self.assertRaises(RuntimeError):module.validated_CPU_descriptors(broken,132)
        broken=copy.deepcopy(descriptors);broken[1]['cpu_shard_owner_index']=127
        with self.assertRaises(RuntimeError):module.validated_CPU_descriptors(broken,132)
        broken=copy.deepcopy(descriptors);broken[4]['cpu_shard_owner_index']=132
        with self.assertRaises(RuntimeError):module.validated_CPU_descriptors(broken,132)
        broken=copy.deepcopy(descriptors);broken[3]['RAW']=broken[2]['RAW']
        with self.assertRaises(RuntimeError):module.validated_CPU_descriptors(broken,132)
        broken=copy.deepcopy(pieces);broken[1][0]['score']+=.01
        with self.assertRaises(RuntimeError):module.merge_policy_predictions(broken,ids,native,native)
        broken=copy.deepcopy(pieces);broken[2][:100]=list(reversed(broken[2][:100]))
        with self.assertRaises(RuntimeError):module.merge_policy_predictions(broken,ids,native,native)
        with self.assertRaises(RuntimeError):module.merge_policy_predictions(pieces[:4],ids,native,native)

if __name__=='__main__':unittest.main()
