"""Stdlib-only contract checks; no scientific imports or data reads."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest

spec=importlib.util.spec_from_file_location('full_fit_candidate',Path(__file__).with_name('worker.py'))
worker=importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)

def ref(path):
    return dict(path=path,bytes=1,sha256='a'*64)

class SampleContract(unittest.TestCase):
    def test_only_root_attested_real_fit_record_with_zero_class_is_admitted(self):
        sample=dict(schema=worker.SAMPLE_SCHEMA,dataset='pseg_ssdd',partition='FIT',fit_image_count=741,
            selection_rule='FIRST_NONEMPTY_IN_ROOT_CANONICAL_FIT_ORDER',root_verified_fit_membership=True,
            source_image_key='metadata-only-placeholder',fit_split_origin_ref=ref('/origin/split.json'),
            annotations_origin_ref=ref('/origin/annotations.json'),image_ref=ref('/probe/image.png'),
            record=dict(file_name='/probe/image.png',image_id=1,height=20,width=30,annotations=[
                dict(category_id=0,bbox_mode=1,iscrowd=0,bbox=[1,2,4,5],segmentation=[[1,2,5,2,5,7]])]))
        worker.validate_sample(sample)
        for field,value in [('partition','TEST'),('root_verified_fit_membership',False)]:
            invalid=copy.deepcopy(sample);invalid[field]=value
            with self.assertRaises(ValueError):worker.validate_sample(invalid)
        invalid=copy.deepcopy(sample);invalid['record']['annotations'][0]['category_id']=1
        with self.assertRaises(ValueError):worker.validate_sample(invalid)
        self.assertFalse(any(n.split('.')[0] in ('torch','numpy','detectron2','mmcv') for n in sys.modules))

if __name__=='__main__':unittest.main()
