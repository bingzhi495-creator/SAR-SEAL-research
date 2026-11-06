"""Metadata refusals/outcome classification only, no scientific imports or files."""
from copy import deepcopy
import unittest
from worker import validate_repeat_metadata,classify_numeric_results

def fixture():
    refs={k:dict(path='/private/'+k+'.pt',bytes=100,sha256='a'*64) for k in ('C4','S8','R8')}
    digests={k:'b'*64 for k in ('model','gradients','optimizer','scheduler','rng','loader')}
    def report(k):
        return dict(worker_exit_code=0,all_requested_updates_completed=True,optimizer_steps_completed=8,new_optimizer_updates=4,
            samples_consumed_after_segment=8,resume_checkpoint_ref=deepcopy(refs['C4']),resume_start_exact=True,resume_start_component_digests=deepcopy(digests),
            steps=[dict(global_update=i,dataset_index=i,image_id=i+1,mapped_image_shape=[3,400,512],mapped_instances=1,
                augmentation_and_target_sha256=('%064x'%i),used_learning_rates=[3.5e-5],post_scheduler_learning_rates=[3.5e-5]) for i in range(5,9)],
            checkpoints=[dict(updates=8,ref=deepcopy(refs[k]),component_digests=deepcopy(digests))])
    return report('S8'),report('R8'),refs

class ContractTests(unittest.TestCase):
    def test_different_initial_checkpoint_is_rejected(self):
        s,r,refs=fixture();r['resume_checkpoint_ref']['sha256']='c'*64
        with self.assertRaises(ValueError):validate_repeat_metadata(s,r,refs)
    def test_changed_actual_augmented_input_is_rejected(self):
        s,r,refs=fixture();self.assertTrue(validate_repeat_metadata(s,r,refs))
        r['steps'][2]['augmentation_and_target_sha256']='f'*64
        with self.assertRaises(ValueError):validate_repeat_metadata(s,r,refs)
    def test_numeric_excess_is_a_legal_characterization(self):
        components={'model':dict(passed=False,tensors=[dict(exact_required=False,passed=False)])}
        self.assertFalse(classify_numeric_results(components,[dict(passed=False)]))
    def test_integer_or_semantic_counter_excess_is_not_legal_variance(self):
        components={'optimizer':dict(passed=False,tensors=[dict(exact_required=True,passed=False)])}
        with self.assertRaises(ValueError):classify_numeric_results(components,[])

if __name__=='__main__':unittest.main()
