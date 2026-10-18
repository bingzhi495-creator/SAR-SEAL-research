"""Two small stdlib entry checks; never main, actual references or ML imports."""
from copy import deepcopy
import unittest

import entry


def spec_fixture():
    def ref(name, sha='a'*64, size=1):
        return dict(path='E:/synthetic_only/'+name, bytes=size, sha256=sha)
    roles = {role: ref(role+'.py') for role in entry.ROLES}
    for role, sha in entry.SOURCE_PINS.items():
        roles[role]['sha256'] = sha
    roles['export_source']['bytes'] = entry.EXPORT_SOURCE_BYTES
    libraries = {key: ref(key+'.py') for key in entry.LIBRARIES}
    context, inputs = ref('context.json'), [ref('input_'+str(i)+'.json') for i in range(12)]
    return dict(schema='U51-matched-CPU-export-spec-v1', operation='CPU_U51_MATCHED_TRAINING_INPUT_EXPORT',
        phase='tiny_metadata_only', output_directory='D:/synthetic_only/fresh', dataset_indices=[710,54],
        expected_versions=dict(torch='2.0.1+cu118',numpy='1.23.5'), limits=deepcopy(entry.LIMITS),
        source_roles=roles, library_sources=libraries, context_ref=context, input_sources=inputs,
        sources=list(roles.values())+list(libraries.values())+[context]+inputs)


class EntryMetadata(unittest.TestCase):
    def test_prelocked_order_batch_and_source_admission(self):
        spec = spec_fixture()
        before = deepcopy(spec)
        entry.validate_spec(spec)
        self.assertEqual(spec, before)
        for invalid in ([],[54,54],[True],[741],[-1],list(range(33))):
            bad = deepcopy(spec); bad['dataset_indices'] = invalid
            with self.assertRaises(ValueError):
                entry.validate_spec(bad)
        bad = deepcopy(spec); bad['limits']['CUDA_visible_devices'] = '0'
        with self.assertRaises(ValueError):
            entry.validate_spec(bad)
        bad = deepcopy(spec); bad['sources'].pop()
        with self.assertRaises(ValueError):
            entry.validate_spec(bad)

    def test_summary_retains_proofs_without_private_identity(self):
        digest = 'c'*64
        proof = dict(shape=[2,3],dtype='torch.float32',logical_bytes=24,source_sha256=digest,
            sha256=digest,bitexact=True,independent_storage=True,compact_storage=True)
        package = dict(metadata=dict(G=2,query_ids=[17,8],ann_ids=[7,6],image_id=99,
            private_path='E:/private'),equivalence={key:deepcopy(proof) for key in ('rgb400','e','M','L','targets')})
        receipt = dict(package_ref=dict(path='D:/private/package.pt',bytes=123,sha256=digest),
            weights_only_CPU_roundtrip_verified=True,qualified=False)
        result = entry.summarize_export(package,receipt)
        self.assertEqual(result['matched_instances'],2)
        self.assertEqual(result['package_sha256'],digest)
        for forbidden in ('query_ids','ann_ids','image_id','private'):
            self.assertNotIn(forbidden,repr(result))


if __name__ == '__main__':
    unittest.main()
