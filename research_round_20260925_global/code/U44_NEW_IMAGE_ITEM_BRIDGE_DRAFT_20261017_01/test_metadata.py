"""Only synthetic JSON metadata; no tensor, image or scientific library import."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest


def bridge():
    path=Path(__file__).with_name('image_item.py')
    spec=importlib.util.spec_from_file_location('U44_IMAGE_ITEM_METADATA_ONLY',str(path))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def fixture():
    source=dict(path='E:/synthetic-only/example.jpg',bytes=11,sha256='a'*64)
    qids=[99]+list(range(99))
    identity=dict(image_id=73,image=source,original_shape=[800,800],actual_img_shape=[800,800],
        batch_input_shape=[800,800],pad_shape=[800,800],scale_factor=[1.,1.],native_input_gamma=1.)
    predictions=[dict(image_id=73,query_id=q,category_id=1,score=float(q)/100,
        segmentation=dict(size=[800,800],counts='opaque-synthetic-only')) for q in qids]
    return source,dict(identity=identity,export_query_ids=qids,native_predictions=predictions,natural_query_ids=list(range(100)))


class ImageMetadata(unittest.TestCase):
    def test_nonidentity_export_order_and_originals_are_preserved_without_population_lock(self):
        api=bridge();source,capture=fixture();before=deepcopy(capture)
        result=api.prepare_metadata(capture,1961,source)
        self.assertEqual(result['row'],dict(dataset_index=1961,image_id=73))
        self.assertIs(result['identity'],capture['identity'])
        self.assertIs(result['query_ids'],capture['export_query_ids'])
        self.assertIs(result['originals'],capture['native_predictions'])
        self.assertEqual(result['query_ids'][0],99)
        self.assertEqual(capture,before)

    def test_GT_identity_or_query_conflicts_fail_before_any_injected_operation(self):
        api=bridge();source,metadata=fixture()
        def forbidden(*args,**kwargs):raise AssertionError('No I/O or scientific operation permitted in this metadata test')
        capture=dict(metadata,F=None,e=None,M=None,C=None,L=None,S0_by_query=None,
            score_factors_by_query=None,class_packet=None,category_ids_by_query=None,receipt=None)
        capture['targets']=[]
        with self.assertRaises(ValueError):
            api.make_image_item(None,None,None,capture,0,source,image_geometry=None,
                source_refs=None,closed_ref=forbidden,check=forbidden)
        changed=deepcopy(metadata);changed['native_predictions'][0]['query_id']=0
        with self.assertRaises(ValueError):api.prepare_metadata(changed,0,source)
        changed=deepcopy(metadata);changed['identity']['image']['sha256']='b'*64
        with self.assertRaises(ValueError):api.prepare_metadata(changed,0,source)


if __name__=='__main__':unittest.main()
