"""Tiny standard-library metadata tests; no main, remote access, tensor or ML import."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


entry = load(Path(__file__).with_name('entry.py'), '_U50_ENTRY_META')
pilot = load(Path(__file__).parent.parent/'U50_DGX_PAIRED_PILOT_BODY_DRAFT_20261018_01/pilot_body.py', '_U50_BODY_META')


def spec_fixture():
    def ref(name, sha='a'*64, size=1):
        return dict(path='/synthetic/'+name, bytes=size, sha256=sha)
    modules = {role: ref(role+'.py', sha) for role, sha in entry.SOURCE_PINS.items()}
    spec = dict(schema=entry.SCHEMA, operation=entry.OPERATION, phase='tiny_metadata',
        entry_source=ref('entry.py'), owner_source=ref('linux_owner.py'),
        pilot_source=ref('pilot_body.py', entry.PILOT_SHA256, 20338), source_refs=modules,
        library_sources={name: ref(name+'.py') for name in ('psutil','torch','numpy')},
        portable_refs={str(index): ref(str(index)+'.pt') for index in (54,710)},
        prelock=pilot.pilot_contract(), output_directory='/synthetic_worker',
        owner_output_directory='/synthetic_owner')
    spec['sources'] = sorted([spec[k] for k in ('entry_source','pilot_source','owner_source')]
        +list(modules.values())+list(spec['library_sources'].values())+list(spec['portable_refs'].values()), key=lambda x:x['path'])
    return spec


class MetadataTests(unittest.TestCase):
    def test_source_and_loader_contract(self):
        spec = spec_fixture()
        self.assertEqual(len(entry.validate_spec(spec, pilot.validate_prelock)), 19)
        self.assertEqual(entry.LOAD_ORDER[:3], (('full_dsconv','dsconv'),('full_head','head'),('full_legacy','model_full')))
        cases = []
        bad = deepcopy(spec); bad['sources'].pop(); cases.append(bad)
        bad = deepcopy(spec); bad['source_refs']['full_dsconv']['sha256'] = 'b'*64; cases.append(bad)
        bad = deepcopy(spec); bad['prelock']['query_chunk'] = 4; cases.append(bad)
        bad = deepcopy(spec); bad['prelock']['environment']['torch'] = '2.0.1+cu118'; cases.append(bad)
        bad = deepcopy(spec); bad['output_directory'] = bad['owner_output_directory']; cases.append(bad)
        for bad in cases:
            with self.assertRaises(ValueError):
                entry.validate_spec(bad, pilot.validate_prelock)

    def test_failed_or_partial_body_cannot_complete(self):
        report = dict(status='COMPLETED_PENDING_ROOT_REVIEW', successful_optimizer_updates=6,
            optimizer_steps_entered=6, distinct_images=2, distinct_matched_GT=28,
            GT_visits_per_arm=29, total_GT_visits=58, image_order=[54,710,54],
            new_training=True,pilot_only=True,main_training=False,formal_training_budget_credit=0,
            U40_frontier_modified=False,weights_written=False,qualified=False,AP=False,DEV=False,TEST=False,
            caller_CPU_RNG_preserved=True, postinitialization_CUDA_RNG_preserved=True,
            final_CUDA_cleanup=dict(allocated_bytes=0,reserved_bytes=0), arms={})
        for arm in ('PLAIN','FULL'):
            report['arms'][arm] = dict(status='PASS', updates_completed=3,
                updates=[dict(dataset_index=i,G=g,all_matched_GT=g,status='PASS',optimizer_step_returned=True)
                         for i,g in zip((54,710,54),(1,27,1))],
                after_arm_release=dict(CUDA_allocated_bytes=0,CUDA_reserved_bytes=0))
        entry.completed_pilot(report)
        for key,value in (('status','FAILED'),('successful_optimizer_updates',5),('caller_CPU_RNG_preserved',False),('new_training',False)):
            bad = deepcopy(report); bad[key] = value
            with self.assertRaises(ValueError):
                entry.completed_pilot(bad)


if __name__ == '__main__':
    unittest.main()
