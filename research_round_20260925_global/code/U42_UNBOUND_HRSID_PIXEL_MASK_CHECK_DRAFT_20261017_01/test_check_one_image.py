"""Stdlib failure injection/metadata only; no actual pixel or mask success."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

path=Path(__file__).with_name('check_one_image.py')
spec=importlib.util.spec_from_file_location('U42_METADATA_RULE_TARGET',str(path))
target=importlib.util.module_from_spec(spec)
spec.loader.exec_module(target)


def image_ref():
    return dict(path='E:/SYNTHETIC_ONLY/not_an_actual_image.jpg',bytes=1,sha256='a'*64)


def image_record():
    return dict(id=7,file_name='not_an_actual_image.jpg',height=800,width=800)


class ForbiddenNumericalAPI:
    def __getattr__(self,name):
        raise AssertionError('No numerical/pixel/mask API allowed in this metadata test: '+name)


class LoadFailureImage:
    format='JPEG';mode='RGB';size=(800,800)

    def __init__(self):self.load_calls=0;self.closed=False
    def __enter__(self):return self
    def __exit__(self,*args):self.closed=True
    def close(self):self.closed=True
    def getbands(self):return ('R','G','B')
    def getexif(self):return {274:1}
    def load(self):
        self.load_calls+=1
        raise OSError('SYNTHETIC_LOAD_REJECTION_BEFORE_PIXELS')
    def verify(self):raise AssertionError('verify must not replace actual load')


class MetadataRules(unittest.TestCase):
    def test_load_error_is_a_retained_failure_and_never_verify_success(self):
        opened=LoadFailureImage();closed=[];checks=[]
        result=target.check_one_image(SimpleNamespace(open=lambda path:opened),
            SimpleNamespace(LOAD_TRUNCATED_IMAGES=False),ForbiddenNumericalAPI(),ForbiddenNumericalAPI(),
            lambda reference:closed.append(reference) or reference['path'],lambda:checks.append(True),
            image_ref(),image_record(),[])
        self.assertEqual(opened.load_calls,1)
        self.assertTrue(opened.closed)
        self.assertEqual(result['status'],'FAILED')
        self.assertFalse(result['pixel_decode_complete'])
        self.assertIn('SYNTHETIC_LOAD_REJECTION_BEFORE_PIXELS',result['image_error']['message'])
        self.assertEqual(closed,[image_ref()]);self.assertTrue(checks)
        # No success image is fabricated: every allowed-orientation branch
        # still deliberately fails at load, before any NumPy/mask operation.
        for exif,policy,loads in [({},'absent_no_transform_applied',1),
            ({274:1},'declared_1_no_transform_applied',1),({274:2},None,0),({274:'1'},None,0)]:
            opened=LoadFailureImage();opened.getexif=lambda value=exif:value
            flags=SimpleNamespace(LOAD_TRUNCATED_IMAGES=False)
            result=target.check_one_image(SimpleNamespace(open=lambda path:opened),flags,
                ForbiddenNumericalAPI(),ForbiddenNumericalAPI(),lambda value:value['path'],lambda:None,
                image_ref(),image_record(),[])
            self.assertEqual(opened.load_calls,loads)
            self.assertEqual(result['orientation_policy'],policy)
            self.assertFalse(flags.LOAD_TRUNCATED_IMAGES)
            self.assertEqual(result['status'],'FAILED')

    def test_multiple_components_remain_one_original_annotation_even_when_decode_fails(self):
        annotation=dict(id=9,image_id=7,category_id=1,iscrowd=1,area=3.5,bbox=[1.0,1.0,5.0,5.0],
            segmentation=[[-1,1,800,1,800,3,-1,3],[4,4,801,4,801,6,4,6]])
        opened=LoadFailureImage()
        result=target.check_one_image(SimpleNamespace(open=lambda path:opened),
            SimpleNamespace(LOAD_TRUNCATED_IMAGES=False),ForbiddenNumericalAPI(),ForbiddenNumericalAPI(),
            lambda value:value['path'],lambda:None,image_ref(),image_record(),[annotation])
        self.assertEqual(len(result['annotations']),1)
        row=result['annotations'][0]
        self.assertEqual((row['annotation_id'],row['component_count'],row['iscrowd']),(9,2,1))
        self.assertEqual(row['status'],'NOT_ATTEMPTED')
        self.assertEqual(row['out_of_canvas_coordinate_count'],4)
        self.assertEqual(row['out_of_canvas_component_count'],2)
        self.assertEqual(annotation['segmentation'],[[-1,1,800,1,800,3,-1,3],[4,4,801,4,801,6,4,6]])

    def test_continuous_metadata_and_raster_measurements_only_get_difference_fields(self):
        # Given scalar metadata only: no raster algorithm is executed or mocked.
        original=dict(area=3.5,bbox=[0.5,1.25,2.5,2.25])
        result=target.geometry_metadata_difference(original,4,[1.0,1.0,2.0,2.0])
        self.assertEqual(result['area_difference_raster_minus_declared'],0.5)
        self.assertEqual(result['bbox_difference_raster_minus_declared'],[0.5,-0.25,-0.5,-0.25])
        self.assertEqual(result['declared_area'],3.5)
        self.assertFalse(result['metadata_equality_required'])
        self.assertNotIn('PASS',result.values())
        self.assertEqual(original,dict(area=3.5,bbox=[0.5,1.25,2.5,2.25]))


if __name__=='__main__':unittest.main()
