"""Two tiny stdlib entry contracts; no main/ML/data/file-reference execution."""
from copy import deepcopy
import unittest

import entry


def spec_fixture():
    def ref(name, digest='a'*64, size=1):
        return dict(path='E:/synthetic_only/'+name, bytes=size, sha256=digest)
    roles = {key: ref(key+'.py') for key in
             ('entry_source', 'export_source', 'rgb_reader_source', 'matched_reader_source', 'geometry_source')}
    roles['export_source'] = ref('export.py', '30755d77c81632f36e0bc88d442b9fa4a9ee8a9afb121670c04b681b06b8e015', 12281)
    for key, sha in (
        ('rgb_reader_source', '678ce3781872e21e961b24a7b0a4d0ecc69f0f8a08b8feb7eb0b2e8624d57025'),
        ('matched_reader_source', '82565caa9c42b6c2ed501e13af9554c6ef47a99b6b525ccfce0bcef620534870'),
        ('geometry_source', '65db107de80c5f1d4dade7cb2ffd3f241fd6bf7f9b233cb1aa9c90cbfd00abf7')):
        roles[key]['sha256'] = sha
    libraries = {name: ref(name+'.py') for name in ('psutil', 'torch', 'numpy', 'PIL', 'PIL.Image',
        'PIL.ImageFile', 'pycocotools', 'pycocotools.coco', 'pycocotools.mask', 'pycocotools._mask')}
    context, inputs = ref('context.json'), [ref('input.json')]
    return dict(schema='ROOT-U49-matched-CPU-export-spec-v1', operation='CPU_U49_MATCHED_TRAINING_INPUT_EXPORT',
        phase='synthetic_metadata_only', output_directory='D:/synthetic_only/new_output', dataset_indices=[54, 710],
        expected_versions=dict(torch='2.0.1+cu118', numpy='1.23.5'),
        limits=dict(allowed_cpus=[0, 1], threads=2, maximum_RSS_bytes=16*1024**3,
                    minimum_available_RAM_bytes=16*1024**3, deadline_seconds=180, CUDA_visible_devices=''),
        source_roles=roles, library_sources=libraries, context_ref=context, input_sources=inputs,
        sources=list(roles.values())+list(libraries.values())+[context]+inputs)


class EntryMetadata(unittest.TestCase):
    def test_fixed_CPU_environment_indices_and_source_membership(self):
        spec = spec_fixture()
        before = deepcopy(spec)
        entry.validate_spec(spec)
        self.assertEqual(spec, before)
        for mutation in ('cuda', 'indices', 'source'):
            bad = deepcopy(spec)
            if mutation == 'cuda':
                bad['limits']['CUDA_visible_devices'] = '0'
            elif mutation == 'indices':
                bad['dataset_indices'] = [0, 1]
            else:
                bad['sources'] = bad['sources'][:-1]
            with self.assertRaises(ValueError):
                entry.validate_spec(bad)

    def test_sanitized_summary_omits_original_identities_paths_and_query_maps(self):
        digest = 'c'*64
        proof = dict(shape=[2, 3], dtype='torch.float32', logical_bytes=24,
                     source_sha256=digest, sha256=digest, bitexact=True,
                     independent_storage=True, compact_storage=True)
        package = dict(metadata=dict(G=2, query_ids=[19, 7], image_id=123,
                                      identity=dict(ann_ids=[88, 21]), provenance=dict(private_path='E:/private')),
                       equivalence={name: deepcopy(proof) for name in ('rgb400', 'e', 'M', 'L', 'targets')})
        receipt = dict(package_ref=dict(path='D:/private/package.pt', bytes=123, sha256=digest),
                       weights_only_CPU_roundtrip_verified=True, qualified=False)
        before = deepcopy((package, receipt))
        summary = entry.summarize_export(package, receipt)
        self.assertEqual(summary['matched_instances'], 2)
        self.assertEqual(summary['package_sha256'], digest)
        self.assertNotIn('query_ids', repr(summary))
        self.assertNotIn('ann_ids', repr(summary))
        self.assertNotIn('private', repr(summary))
        self.assertEqual((package, receipt), before)


if __name__ == '__main__':
    unittest.main()
