"""Standard-library input/accounting checks; no scientific imports or execution."""
import copy
import importlib.util
from pathlib import Path
import unittest

def load():
    p=Path(__file__).with_name('worker.py')
    sp=importlib.util.spec_from_file_location('health_worker_contract',str(p))
    m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m

def manifest():
    rows=[];ann=1
    for i in range(741):
        n=3 if i<179 else 2
        ids=list(range(ann,ann+n));ann+=n
        h=('%064x'%(i%730+1))
        rows.append(dict(dataset_index=i,image_id=i+10,ann_ids=ids,
            archive_image_entry='images/'+h+'.jpg',original_basename='%06d.jpg'%(i+1),
            original_image_ref=dict(bytes=100,sha256=h),original_shape=[100,200]))
    return dict(schema='SSDD-FIT741-portable-original-byte-input-manifest-v1',
        population=dict(FIT_annotation_instances=1661,logical_training_rows=741,
            unique_JPEG_byte_objects=730,deduplicate_training_population=False,
            canonical_row_order='dataset_index 0..740'),rows=rows)

class Contract(unittest.TestCase):
    def test_duplicate_encoded_images_remain_distinct_logical_training_samples(self):
        m=load();plan=m.canonical_input_plan(manifest(),'/tmp/private_inputs')
        self.assertEqual(len(plan),741)
        self.assertEqual([r['dataset_index'] for r in plan[:32]],list(range(32)))
        self.assertNotEqual(plan[0]['image_id'],plan[730]['image_id'])
        self.assertEqual(plan[0]['image_ref'],plan[730]['image_ref'])
        self.assertEqual(sum(len(x['ann_ids']) for x in plan),1661)

    def test_loader_sorting_cannot_change_canonical_sample_or_annotation_identity(self):
        m=load();plan=m.canonical_input_plan(manifest(),'/tmp/private_inputs')
        records=[dict(image_id=r['image_id'],file_name=r['original_basename'],height=100,width=200,
            annotations=[dict(id=a) for a in r['ann_ids']]) for r in reversed(plan)]
        ordered=m.align_coco_records(records,plan)
        self.assertEqual([r['image_id'] for r in ordered],[r['image_id'] for r in plan])
        self.assertEqual(ordered[730]['file_name'],ordered[0]['file_name'])
        bad=copy.deepcopy(records);bad[0]['annotations'][0]['id']=-1
        with self.assertRaises(ValueError):m.align_coco_records(bad,plan)

    def test_zero_mask_gradient_is_observation_but_missing_updates_fail(self):
        m=load()
        steps=[dict(dataset_index=i,image_id=i+10,optimizer_steps_completed=i+1,
                    all_present_gradients_finite=True,total_loss=1.0,
                    IAMR_output_gradient={'classification':'FINITE_ALL_ZERO'}) for i in range(32)]
        m.validate_completed_steps(steps,32)
        with self.assertRaises(ValueError):m.validate_completed_steps(steps[:-1],31)
        bad=copy.deepcopy(steps);bad[20]['dataset_index']=19
        with self.assertRaises(ValueError):m.validate_completed_steps(bad,32)

if __name__=='__main__':unittest.main()
