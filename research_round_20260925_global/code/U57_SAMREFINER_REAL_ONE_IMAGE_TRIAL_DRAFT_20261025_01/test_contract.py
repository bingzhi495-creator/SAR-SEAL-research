import copy
import math
import unittest

import one_image_trial as worker


def staged():
    checkpoint = dict(path='/private/sam_vit_h_4b8939.pth',
                      bytes=2564550879, sha256='a' * 64)
    spec = dict(checkpoint_ref=checkpoint, verified_checkpoint_sha256='a' * 64)
    stage = dict(schema='SAMRefiner-CHECKPOINT_STAGE-v1',
                 status='ROOT_VERIFIED_COMPLETE', checkpoint_ref=checkpoint,
                 verified_sha256='a' * 64, model_type='vit_h',
                 author_commit='4bb7f95738c7a2e805f189210fb09e02a2197557',
                 official_url='https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth')
    return spec, stage


def trial_spec():
    spec, _ = staged()
    pins = {
        'fixed_adapter': 'eb0609708bd189ebfda381c9ef5700c2bdba53fbeba0837febeff24fdbb9cac2',
        'bridge': '3da3bbd6d64ba5c4f9a28b243528f9bfe095c7b2348297885e3b6105ae22d855',
        'author_refiner': 'b467d8e3e3c2fc395466a293635774cb821c5ffebdb39c4e5d64ac93c49dd3ad',
        'author_utils': '0d0c8e4b257935968c39f4223993b5b7de1868a4a7d886092c6a0074131299ef',
        'sam_init': 'c8d8eb649610b2d51f138481a89617a3dcb61106465121dcb1c2f23affc84c30',
        'sam_registry': '77bc4faf728118625ddd184dc4e6375bd1b5c636fc66fb429f9fc5276f331cc5',
        'sam_model': '588d294a48bbd4ce5c5f4f560fa862aee0a8325a8dadda4e7d848a9f23d27946',
        'worker': 'b' * 64, 'resource_control': 'c' * 64}
    def ref(name, sha='d' * 64):
        return dict(path='/private/' + name, bytes=1, sha256=sha)
    spec.update(schema='ROOT-SAMRefiner-one-image-trial-spec-v1',
                operation='GPU_SAMREFINER_ORIGINAL_ORDINAL0_ONE_IMAGE_TRIAL', run_id='fixture',
                source_refs={name: ref(name + '.py', sha) for name, sha in pins.items()},
                library_sources={name: ref(name + '/__init__.py') for name in
                                 ('torch', 'numpy', 'psutil', 'cv2', 'PIL')},
                input_refs={name: ref(name) for name in ('png', 'masks', 'candidates')},
                input_report_ref=ref('input_report.json'), root_lease_ref=ref('root_lease.json'),
                checkpoint_stage_ref=ref('CHECKPOINT_STAGE.json'), output_directory='/private/output',
                prelock=dict(author_commit='4bb7f95738c7a2e805f189210fb09e02a2197557',
                    model_type='vit_h', ordinal=0, image_id=4, precision='float32', amp=False,
                    allow_cpu_fallback=False,
                    params=dict(iters=5, margin=0.0, gamma=4.0, strength=30, use_samhq=False),
                    startup_min_available_RAM_bytes=48*1024**3,
                    resource_plan='INDEPENDENT_EXTERNAL_BASELINE_PROSPECTIVE',
                    limits=dict(allowed_cpus=[0, 1], threads=2, allocator_bytes=16*1024**3,
                                maximum_RSS_bytes=16*1024**3,
                                minimum_available_RAM_bytes=16*1024**3, deadline_seconds=1800)))
    values = list(spec['source_refs'].values()) + list(spec['library_sources'].values())
    values += list(spec['input_refs'].values())
    values += [spec[k] for k in ('checkpoint_ref', 'input_report_ref', 'root_lease_ref', 'checkpoint_stage_ref')]
    spec['sources'] = sorted(values, key=lambda value: value['path'])
    return spec


class ContractTest(unittest.TestCase):
    def test_model_gate_requires_closed_root_verified_complete_checkpoint(self):
        spec, stage = staged()
        before = copy.deepcopy((spec, stage))
        self.assertIsNone(worker.validate_checkpoint_stage(spec, stage))
        self.assertEqual((spec, stage), before)
        for invalid in (None, dict(stage, status='DOWNLOADING'),
                        dict(stage, verified_sha256='a' * 12),
                        dict(stage, checkpoint_ref=dict(stage['checkpoint_ref'], sha256='b' * 64))):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    worker.validate_checkpoint_stage(spec, invalid)

    def test_trial_contract_refuses_implicit_precision_budget_or_source_changes(self):
        spec = trial_spec()
        worker.validate_spec(spec)
        changes = [('amp', True), ('allow_cpu_fallback', True), ('precision', 'float16'),
                   ('model_type', 'vit_b')]
        for key, value in changes:
            invalid = copy.deepcopy(spec); invalid['prelock'][key] = value
            with self.subTest(key=key):
                with self.assertRaises(ValueError): worker.validate_spec(invalid)
        for value in (None, 0):
            invalid = copy.deepcopy(spec)
            invalid['prelock']['limits']['allocator_bytes'] = value
            with self.assertRaises(ValueError): worker.validate_spec(invalid)
        invalid = copy.deepcopy(spec)
        invalid['prelock']['startup_min_available_RAM_bytes'] -= 1
        with self.assertRaises(ValueError): worker.validate_spec(invalid)
        invalid = copy.deepcopy(spec)
        invalid['source_refs']['bridge']['sha256'] = 'e' * 64
        with self.assertRaises(ValueError): worker.validate_spec(invalid)

    def test_original_candidate_order_class_and_raw_score_are_preserved(self):
        rows = [dict(image_id=4, query_id=q, category_id=1,
                     score=-0.0 if q == 99 else 1,
                     segmentation=dict(size=[317, 420], counts='opaque'))
                for q in reversed(range(100))]
        before = copy.deepcopy(rows)
        identities = worker.candidate_identities(rows, [317, 420])
        self.assertEqual([row['query_id'] for row in identities], list(reversed(range(100))))
        self.assertEqual(math.copysign(1.0, identities[0]['score']), -1.0)
        self.assertIs(type(identities[1]['score']), int)
        self.assertEqual(rows, before)
        invalid = copy.deepcopy(rows); invalid[1]['query_id'] = invalid[0]['query_id']
        with self.assertRaises(ValueError): worker.candidate_identities(invalid, [317, 420])


if __name__ == '__main__':
    unittest.main()

