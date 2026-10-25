"""Synthetic metadata/counts only; no GT payload, scientific import or AP."""
from copy import deepcopy
import unittest
import worker


def datasets():
    integer = dict(images=[dict(id=7,width=5,height=4,file_name='x.jpg')],
        categories=[dict(id=1,name='ship')],annotations=[dict(id=9,image_id=7,category_id=1,
        iscrowd=0,segmentation=[[0,0,4,0,4,3]],bbox=[0,0,4,3],area=6.0)])
    floating = deepcopy(integer)
    floating['annotations'][0].update(segmentation=[[.2,.8,4.7,.1,4.9,3.9]],bbox=[.2,.1,4.7,3.8],area=8.62)
    return integer,floating


class Contract(unittest.TestCase):
    def test_mapping_report_preserves_original_paths_across_byte_identical_relocation(self):
        integer = dict(path='/actual/DEV.json',bytes=10,sha256='1'*64)
        floating = dict(path='/actual/FLOAT_DEV.json',bytes=20,sha256='2'*64)
        report = dict(schema='Root-SAMRefiner-DEV187-local-float-restoration-map-v1',
            integer_DEV_ref=dict(integer,path='D:/original/DEV.json'),
            float_DEV_ref=dict(floating,path='D:/derived/FLOAT_DEV.json'),
            source_training_member_ref=dict(worker.SOURCE_MEMBER,path='D:/archive/train.json'),
            declared_rule=worker.RULE,unique_mappings=380,ambiguous=0,missing=0,
            internal_IDs_population_preserved=True,official_archive_authenticated=False,
            posthoc_additional_diagnostic=True)
        worker.validate_mapping_report(report,integer,floating)
        report['float_DEV_ref']['sha256'] = '3'*64
        with self.assertRaises(ValueError):
            worker.validate_mapping_report(report,integer,floating)

    def test_empty_raster_cases_keep_explicit_undefined_IoU_and_denominator(self):
        empty = worker.raster_counts(0,0,0)
        self.assertIsNone(empty['IoU'])
        self.assertTrue(empty['both_empty'])
        self.assertEqual(empty['XOR_pixels'],0)
        added = worker.raster_counts(0,3,0)
        self.assertEqual(added['IoU'],0)
        self.assertEqual(added['XOR_pixels'],3)
        self.assertIsNone(added['float_over_integer_area_ratio'])

    def test_image_dimensions_and_non_geometry_annotation_fields_are_preserved(self):
        for change in ('image_shape','iscrowd'):
            integer,floating = datasets()
            if change == 'image_shape':
                floating['images'][0]['width'] = 6
            else:
                floating['annotations'][0]['iscrowd'] = 1
            with self.assertRaises(ValueError):
                worker.validate_GT_pair(integer,floating,expected_images=1,expected_annotations=1)

    def test_only_the_single_declared_truncation_rule_is_accepted(self):
        integer,floating = datasets()
        floating['annotations'][0]['segmentation'][0][2] = 5.7
        with self.assertRaises(ValueError):
            worker.validate_GT_pair(integer,floating,expected_images=1,expected_annotations=1)

    def test_internal_annotation_ids_and_population_cannot_change(self):
        integer,floating = datasets()
        worker.validate_GT_pair(integer,floating,expected_images=1,expected_annotations=1)
        floating['annotations'][0]['id'] = 99
        with self.assertRaises(ValueError):
            worker.validate_GT_pair(integer,floating,expected_images=1,expected_annotations=1)


if __name__ == '__main__':
    unittest.main()
