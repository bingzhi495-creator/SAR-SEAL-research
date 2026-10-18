"""Tiny synthetic metadata guard tests, never an input/tensor validation result."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
import validation_body as body


class WorkerContracts(unittest.TestCase):
    def test_failed_load_distinguishes_GT_read_attempt_from_success(self):
        # Standard-library failure boundaries only: no package I/O, tensor,
        # numerical library or successful scientific read is simulated.
        class Control:
            def check(self):pass
        class StoppedReader:
            records=[dict(dataset_index=i,G=3 if i<179 else 2) for i in range(741)]
            visits=[]
            failed_loads=0
            def load(self,index):
                self.failed_loads+=1
                raise ValueError('Synthetic stop before package or tensor I/O')
        def prepare_stop(*args,**kwargs):
            raise ValueError('Synthetic stop before reader construction')
        torch=SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda:False))
        for prepare,attempted,loads in ((prepare_stop,False,0),
            (lambda *args,**kwargs:StoppedReader(),True,1)):
            report={}
            with self.subTest(load_attempted=attempted):
                with self.assertRaisesRegex(ValueError,'Synthetic stop'):
                    body.run(torch,Control(),{},reader_module=SimpleNamespace(prepare=prepare),
                        image_geometry=None,source_refs={},prelock=body.prelock_contract(),
                        report=report,record_image=lambda value:self.fail('No image may be audited'))
                self.assertIs(report['FIT_GT_tensor_read_attempted'],attempted)
                self.assertIs(report['FIT_GT_tensors_read'],False)
                self.assertEqual(report['attempted_images'],loads)
                self.assertEqual(report['validated_images'],0)
                self.assertEqual(report['matched_instances'],0)

    def test_partial_or_failed_population_cannot_report_full_validation(self):
        value=dict(schema='U54-full-FIT-CPU-validation-body-report-v1',status='COMPLETE_PENDING_ROOT',
            attempted_images=741,validated_images=741,released_images=741,recorded_images=741,
            matched_instances=1661,reader_visits=list(range(741)),reader_failed_loads=0,
            flags=dict(CUDA_uninitialized_before=True,CUDA_uninitialized_after=True,
                metadata_population_verified=True,all_field_proofs_verified=True,
                all_returned_tensor_objects_released=True),
            qualified=False,new_training=False,AP=False,DEV=False,TEST=False,main_training=False,
            model_constructed=False,model_forward_executed=False,optimizer_created=False,
            GT_matching_recomputed=False,formal_training_budget_credit=0)
        body.completed_population(value)
        for key,replacement in (('validated_images',740),('matched_instances',1660),('status','FAILED')):
            changed=deepcopy(value);changed[key]=replacement
            with self.subTest(key=key):
                with self.assertRaises(ValueError):body.completed_population(changed)
        changed=deepcopy(value);changed['reader_visits'][-1]=739
        with self.assertRaises(ValueError):body.completed_population(changed)

    def test_source_scope_is_derived_and_conflicting_ref_is_rejected(self):
        import entry
        def ref(name,sha='a'*64):return dict(path='/synthetic/'+name,bytes=1,sha256=sha)
        a,b=ref('a.py'),ref('b.py')
        self.assertEqual(entry.canonical_sources([b,a,a]),[a,b])
        with self.assertRaises(ValueError):entry.canonical_sources([a,ref('a.py','b'*64)])


if __name__=='__main__':unittest.main()
