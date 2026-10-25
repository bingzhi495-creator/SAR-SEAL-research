"""Only synthetic COCO identity metadata, no scientific imports or pixels."""
import importlib.util
import hashlib
import os
from pathlib import Path
import unittest
import worker

SOURCE = Path(os.environ.get('SAM_GEOMETRY_METADATA_SOURCE',
    'E:/SAR/seal/github_release_20261011/research_round_20260925_global/code/N28_FIXED24_DEV_EVAL_DRAFT_20261017_01/geometry_diagnostic.py'))
raw = SOURCE.read_bytes()
assert len(raw) == 13722 and hashlib.sha256(raw).hexdigest() == worker.PINS['geometry_diagnostic']['sha256']
spec = importlib.util.spec_from_file_location('_closed_geometry_metadata',SOURCE)
geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geometry)


def metadata():
    return [dict(image_id=7, category_id=1, aRng=[0,1e10], maxDet=100,
        gtIds=[1001,1002,1003], gtIgnore=[0,0,1], gtMatches=[[5,0,0]], dtIds=[5,6])], {
        5:dict(id=5,image_id=7,query_id=41,category_id=1),
        6:dict(id=6,image_id=7,query_id=88,category_id=1)}


class Contract(unittest.TestCase):
    def test_missing_original_query_never_falls_back_to_an_alternative_query(self):
        pair = dict(image_id=7,query_id=41,GT_annotation_id=1001,dt_id=5)
        maps = {method:{(7,41):method+'-original',(7,99):method+'-alternative'}
                for method in worker.METHODS}
        self.assertEqual(worker.same_query_values(pair,maps)['SAMRefiner'],'SAMRefiner-original')
        del maps['SAMRefiner'][(7,41)]
        with self.assertRaises(ValueError):
            worker.same_query_values(pair,maps)

    def test_grid_changes_are_rejected_before_any_decode(self):
        worker.require_original_grid(dict(size=[2,3],counts='6'),[2,3])
        with self.assertRaises(ValueError):
            worker.require_original_grid(dict(size=[3,2],counts='6'),[2,3])

    def test_ignored_and_unmatched_GT_are_kept_in_the_scope_denominator(self):
        rows, detections = metadata()
        result = worker.fixed_native_pairs(geometry,rows,detections,[0,1e10],{1001,1002,1003})
        self.assertEqual(result['coverage'],dict(total_GT=3,selected_native_nonignored_GT=1,
            ignored_GT=1,unmatched_nonignored_GT=1,selected_fraction_of_all_GT=1/3))

    def test_COCO_detection_id_is_resolved_to_its_actual_query(self):
        rows, detections = metadata()
        result = worker.fixed_native_pairs(geometry,rows,detections,[0,1e10],{1001,1002,1003})
        self.assertEqual(result['pairs'],[dict(image_id=7,GT_annotation_id=1001,dt_id=5,query_id=41)])


if __name__ == '__main__':
    unittest.main()
