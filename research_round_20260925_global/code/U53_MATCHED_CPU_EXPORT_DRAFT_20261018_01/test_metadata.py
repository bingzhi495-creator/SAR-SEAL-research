"""Tiny stdlib-only U53 limits; no tensors, data reads or numerical imports."""
import unittest
import matched_export


class RemainingBatch(unittest.TestCase):
    def test_prelocked_remaining_and_maximum_population_are_single_ordered_batches(self):
        remaining=list(range(64,741))
        self.assertEqual(matched_export.validate_batch(remaining),remaining)
        maximum=list(range(740,-1,-1))
        result=matched_export.validate_batch(maximum)
        self.assertEqual(result,maximum)
        self.assertIsNot(result,maximum)
        for bad in ([],list(range(742)),[0,0],[True],[741],[-1],[0.0],(0,)):
            with self.subTest(batch_length=len(bad)):
                with self.assertRaises(ValueError):matched_export.validate_batch(bad)

    def test_new_worker_accepts_single_remaining_batch_only_under_explicit_1800_second_protocol(self):
        import copy
        import entry
        def ref(name,sha='a'*64,size=1):
            return dict(path='E:/synthetic_metadata_only/'+name,bytes=size,sha256=sha)
        roles={role:ref(role,entry.SOURCE_PINS.get(role,'b'*64),
                       entry.EXPORT_SOURCE_BYTES if role=='export_source' else 1)
               for role in entry.ROLES}
        libraries={name:ref(name) for name in entry.LIBRARIES}
        context=ref('context');inputs=[ref('explicit_input')]
        spec=dict(schema='U53-matched-CPU-export-spec-v1',operation='CPU_U53_MATCHED_TRAINING_INPUT_EXPORT',
            expected_versions=dict(torch='2.0.1+cu118',numpy='1.23.5'),limits=dict(
                allowed_cpus=[0,1],threads=2,maximum_RSS_bytes=16*1024**3,
                minimum_available_RAM_bytes=16*1024**3,deadline_seconds=1800,CUDA_visible_devices=''),
            dataset_indices=list(range(64,741)),phase='synthetic_metadata_only',source_roles=roles,
            library_sources=libraries,context_ref=context,input_sources=inputs,
            sources=list(roles.values())+list(libraries.values())+[context]+inputs)
        entry.validate_spec(spec)
        self.assertEqual(spec['dataset_indices'],list(range(64,741)))
        maximum=copy.deepcopy(spec);maximum['dataset_indices']=list(range(741))
        entry.validate_spec(maximum)
        wrong=copy.deepcopy(spec);wrong['limits']['deadline_seconds']=180
        with self.assertRaises(ValueError):entry.validate_spec(wrong)
        wrong=copy.deepcopy(spec);wrong['schema']='U51-matched-CPU-export-spec-v1'
        with self.assertRaises(ValueError):entry.validate_spec(wrong)


if __name__=='__main__':unittest.main()
