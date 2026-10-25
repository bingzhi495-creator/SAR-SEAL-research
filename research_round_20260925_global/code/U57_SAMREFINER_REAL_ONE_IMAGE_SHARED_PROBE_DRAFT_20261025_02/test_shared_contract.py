import copy
import sys
import unittest

import worker


def observation():
    return dict(schema='SAMRefiner-shared-GPU-background-observation-v02',
                boot_id='11111111-1111-4111-8111-111111111111',
                observed_UTC='2026-10-25T06:30:00Z',
                original_nvidia_smi_returncode=0, gpu_index=0,
                whole_device_exclusive=False,
                processes=[dict(pid=1234, start_ticks=5678, name='llama-server',
                                used_memory_MiB=37037)])


class SharedContractTest(unittest.TestCase):
    def test_background_observation_requires_explicit_nonexclusive_closed_metadata(self):
        value = observation(); before = copy.deepcopy(value)
        worker.validate_background_observation(value, value['boot_id'])
        self.assertEqual(value, before)
        for key, invalid in (('original_nvidia_smi_returncode', 1),
                             ('whole_device_exclusive', True), ('observed_UTC', 'not UTC')):
            bad = copy.deepcopy(value); bad[key] = invalid
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    worker.validate_background_observation(bad, value['boot_id'])
        with self.assertRaises(ValueError):
            worker.validate_background_observation(value, '22222222-2222-4222-8222-222222222222')
        bad = copy.deepcopy(value); bad['processes'][0]['cmdline'] = 'forbidden'
        with self.assertRaises(ValueError):
            worker.validate_background_observation(bad, value['boot_id'])
        self.assertFalse([name for name in sys.modules if name.split('.')[0] in ('torch', 'numpy')])

    def test_shared_lease_rejects_v01_or_falsely_exclusive_grants(self):
        bg = dict(path='/private/background.json', bytes=1, sha256='a' * 64)
        spec = dict(run_id='SAMRefiner-shared-v02-fixture',
                    exclusive_project_scientific_leaf=True, whole_device_exclusive=False,
                    gpu_background_observation_ref=bg,
                    source_refs=dict(worker=dict(path='/private/worker.py', bytes=1, sha256='b' * 64)),
                    prelock=dict(scope='metadata fixture only'))
        lease = dict(schema='Root-SAMRefiner-one-image-shared-lease-v02', status='ROOT_GRANTED',
                     run_id=spec['run_id'], operation=worker.OPERATION,
                     worker_ref=spec['source_refs']['worker'], prelock=spec['prelock'],
                     single_attempt=True, exclusive_project_scientific_leaf=True,
                     whole_device_exclusive=False, main_scientific_processes_reaped=True,
                     scope='one_image_shared_GPU_resource_and_function_probe_only',
                     gpu_background_observation_ref=bg, boot_id=observation()['boot_id'])
        worker.validate_shared_lease(spec, lease, observation()['boot_id'])
        for invalid in (dict(lease, schema='Root-SAMRefiner-one-image-lease-v1'),
                        dict(lease, exclusive_GPU=True), dict(lease, whole_device_exclusive=True),
                        dict(lease, exclusive_project_scientific_leaf=False)):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    worker.validate_shared_lease(spec, invalid, observation()['boot_id'])
        for bad_spec in (dict(spec, exclusive_GPU=True),
                         dict(spec, whole_device_exclusive=True)):
            with self.assertRaises(ValueError):
                worker.validate_shared_lease(bad_spec, lease, observation()['boot_id'])


if __name__ == '__main__':
    unittest.main()
