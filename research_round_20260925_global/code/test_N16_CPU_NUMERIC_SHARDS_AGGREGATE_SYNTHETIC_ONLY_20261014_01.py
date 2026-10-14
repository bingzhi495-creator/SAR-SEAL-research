"""SYNTHETIC_ONLY stdlib fake metadata; never COCO/model/data/GT execution."""
import copy
import importlib.util
from pathlib import Path
import unittest
path=Path(__file__).with_name('N16_FIXED_E24_ALL100_SCOUT_CPU_NUMERIC_SHARDS_AGGREGATE_ROOT_UNBOUND_DRAFT_20261014_01.py')
spec=importlib.util.spec_from_file_location('_n16_cpu_aggregate_synthetic',path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
def descriptors():
    values=[]
    for i,(first,last) in enumerate(module.RANGES):
        value=dict(cpu_shard_id=i,cpu_dataset_ordinal_range=[first,last],cpu_shard_owner_index=114+i)
        for key in ('source','spec','report','RAW','admission'):
            value[key]=dict(path='E:/SYNTHETIC_ONLY/cpu%d_%s.json'%(i,key),bytes=1,sha256='0'*64)
        values.append(value)
    return values
def fixture():
    ids=[1000+i for i in range(187)]
    native=[dict(image_id=image,query_id=q,category_id=1,score=q/100,segmentation={'synthetic_only':True})
            for image in ids for q in reversed(range(100))]
    actual=copy.deepcopy(native)
    actual[37]['segmentation']={'synthetic_changed_mask':True}
    pieces=[actual[first*100:last*100] for first,last in module.RANGES]
    return ids,native,actual,pieces
class CPUAggregateTests(unittest.TestCase):
    def test_complete_four_cpu_ranges_and_full_order(self):
        values=module.validated_CPU_descriptors(descriptors())
        self.assertEqual([v['cpu_shard_owner_index'] for v in values],[114,115,116,117])
        self.assertEqual([j for v in values for j in range(*v['cpu_dataset_ordinal_range'])],list(range(187)))
        ids,native,actual,pieces=fixture()
        self.assertEqual(module.merge_policy_predictions(pieces,ids,native,actual),actual)
    def test_wrong_owner_gap_overlap_swapped_and_bool_reject(self):
        changes=[lambda v:v[0].update(cpu_shard_owner_index=109),lambda v:v[1].update(cpu_dataset_ordinal_range=[48,94]),
                 lambda v:v[1].update(cpu_dataset_ordinal_range=[46,94]),lambda v:v.reverse(),
                 lambda v:v[0].update(cpu_dataset_ordinal_range=[False,47])]
        for change in changes:
            values=descriptors();change(values)
            with self.assertRaises(RuntimeError):module.validated_CPU_descriptors(values)
    def test_partial_order_score_or_original_AP_prediction_mismatch_reject(self):
        ids,native,actual,pieces=fixture()
        changes=[lambda v:v[0].pop(),lambda v:v[1].reverse(),lambda v:v[2][0].update(score=9),
                 lambda v:v[3][0].update(segmentation={'wrong_actual_mask':True})]
        for change in changes:
            values=copy.deepcopy(pieces);change(values)
            with self.assertRaises(RuntimeError):module.merge_policy_predictions(values,ids,native,actual)
if __name__=='__main__':unittest.main(verbosity=2)
