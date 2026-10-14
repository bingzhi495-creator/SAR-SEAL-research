"""SYNTHETIC_ONLY: fake refs and row ordering; no COCO/data/weights IO."""
import copy
import importlib.util
from pathlib import Path
import unittest

path=Path(__file__).with_name('N16_FIXED_E24_ALL100_SCOUT_SHARDS_AGGREGATE_ROOT_UNBOUND_DRAFT_20261014_02.py')
spec=importlib.util.spec_from_file_location('_n16_synthetic_aggregate',path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def records():
    rows=[]
    for i,(first,last) in enumerate(module.RANGES):
        row=dict(shard_id=i,dataset_ordinal_range=[first,last],shard_owner_index=109+i)
        for key in ('source','spec','report','RAW','admission'):
            row[key]=dict(path='E:/SYNTHETIC_ONLY/%d_%s.json'%(i,key),bytes=1,sha256='0'*64)
        rows.append(row)
    return rows

def native_rows(ids):
    return [dict(image_id=image,query_id=q,category_id=1,score=q/100,
                 segmentation={'synthetic_only':True}) for image in ids for q in reversed(range(100))]

class AggregateTests(unittest.TestCase):
    def test_four_ranges_cover_exact187_and_have_last46_images(self):
        values=module.validated_descriptors(records())
        indices=[j for value in values for j in range(*value['dataset_ordinal_range'])]
        self.assertEqual(indices,list(range(187)))
        self.assertEqual([v['dataset_ordinal_range'][1]-v['dataset_ordinal_range'][0] for v in values],[47,47,47,46])

    def test_gaps_overlap_swap_owner_and_boolean_range_reject(self):
        changes=[lambda r:r[1].update(dataset_ordinal_range=[48,94]),
                 lambda r:r[1].update(dataset_ordinal_range=[46,94]),
                 lambda r:r.reverse(),lambda r:r[2].update(shard_owner_index=110),
                 lambda r:r[0].update(dataset_ordinal_range=[False,47])]
        for change in changes:
            value=records()
            change(value)
            with self.assertRaises(RuntimeError):
                module.validated_descriptors(value)

    def test_actual_export_order_preserved_without_fake_COCO(self):
        ids=[113,205]
        native=native_rows(ids)
        values=copy.deepcopy(native)
        values[0]['segmentation']={'synthetic_changed_mask':True}
        self.assertIs(module.prediction_segment(values,ids,native),values)
        for bad in (values[::-1],values[:-1]):
            with self.assertRaises(RuntimeError):
                module.prediction_segment(bad,ids,native)
        bad=copy.deepcopy(values)
        bad[0]['score']+=.01
        with self.assertRaises(RuntimeError):
            module.prediction_segment(bad,ids,native)

if __name__=='__main__':
    unittest.main(verbosity=2)
