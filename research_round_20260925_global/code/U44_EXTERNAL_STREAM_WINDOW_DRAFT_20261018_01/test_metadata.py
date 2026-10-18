"""Only fabricated JSON metadata; never import or imitate a tensor library."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest


def api():
    path=Path(__file__).with_name('stream_window.py')
    spec=importlib.util.spec_from_file_location('U44_WINDOW_METADATA',str(path))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def rows():
    return [dict(dataset_index=17+i,image_id=701+i,original_shape=[800,800],
        image_ref=dict(path='E:/synthetic-only/%d.jpg'%i,bytes=7,sha256=str(i+1)*64)) for i in range(2)]


class Metadata(unittest.TestCase):
    def test_small_contiguous_window_projects_only_original_image_fields(self):
        target=api();source=rows();before=deepcopy(source)
        projected=target.prepare_window_metadata(source)
        self.assertEqual(projected['dataset_range'],[17,19])
        self.assertEqual(set(projected['images'][0]['image_metadata']),{'image_id','image_ref','original_shape'})
        self.assertEqual(source,before)
        for bad in ([],source*3):
            with self.assertRaises(ValueError):target.prepare_window_metadata(bad)
        bad=deepcopy(source);bad[1]['dataset_index']=20
        with self.assertRaises(ValueError):target.prepare_window_metadata(bad)
        bad=deepcopy(source);bad[0]['annotations']=[]
        with self.assertRaises(ValueError):target.prepare_window_metadata(bad)

    def test_receipt_is_not_qualification_and_must_match_actual_current_models(self):
        target=api();reference=dict(path='E:/synthetic-only/state.pth',bytes=7,sha256='a'*64)
        shared=dict(schema='U44-external-fixed50-CPU-arm-restore-pending-v1',status='RESTORED_CPU_ONLY',
            qualified=False,seed=202610173101,fixed_epoch=50,updates_per_arm=37050,
            endpoint_refs={k:reference for k in ('training_qualified','pair_commit','preparation','order')},
            checkpoint_refs={k:reference for k in ('PLAIN','FULL')},training_qualified=reference,
            pair_commit_ref=reference,order_ref=reference,training_sources={'fixture_only':reference},
            initialization_receipt_sha256='c'*64,all_modules_eval=True,all_parameters_frozen=True,
            all_parameter_grads_none=True,CPU_checkpoint_payload_released=True,optimizer_created=False,
            optimizer_restored=False,CUDA_initialized=False)
        receipts={arm:dict(deepcopy(shared),arm=arm,checkpoint_ref=reference,model_state_sha256=digest)
                  for arm,digest in [('PLAIN','1'*64),('FULL','2'*64)]}
        current={arm:value['model_state_sha256'] for arm,value in receipts.items()}
        result=target.validate_restoration_metadata(receipts,current)
        self.assertEqual(result['fixed_epoch'],50)
        bad=deepcopy(current);bad['FULL']='3'*64
        with self.assertRaises(ValueError):target.validate_restoration_metadata(receipts,bad)
        bad=deepcopy(receipts);bad['FULL']['updates_per_arm']=256
        with self.assertRaises(ValueError):target.validate_restoration_metadata(bad,current)
        bad=deepcopy(receipts);bad['FULL']['qualified']=True
        with self.assertRaises(ValueError):target.validate_restoration_metadata(bad,current)


if __name__=='__main__':unittest.main()
