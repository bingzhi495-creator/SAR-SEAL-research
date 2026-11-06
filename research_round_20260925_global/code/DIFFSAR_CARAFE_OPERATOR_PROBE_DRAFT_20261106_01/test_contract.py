"""Two metadata-only admission tests; no scientific imports or CUDA execution."""
import importlib.util
from pathlib import Path
import sys
import unittest
from copy import deepcopy


class ProbeAdmission(unittest.TestCase):
    def load(self):
        path=Path(__file__).with_name('worker.py')
        spec=importlib.util.spec_from_file_location('_CARAFE_CANDIDATE_TEST',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    def fixture(self,m):
        ref=lambda name:dict(path='/private/'+name,bytes=1,sha256='0'*64)
        return dict(schema=m.SCHEMA,operation=m.OPERATION,run_id='CARAFE-fixed-FP32-v01-test',
            source_refs={'worker':ref('worker.py')},
            library_sources={k:ref(k) for k in m.LIBRARY_NAMES},root_lease_ref=ref('lease.json'),
            output_directory='/private/OUTPUT',prelock=deepcopy(m.PRELOCK))

    def test_import_is_inert_and_cap_or_case_changes_are_rejected(self):
        before=set(sys.modules);m=self.load();after=set(sys.modules)-before
        self.assertFalse(any(k.split('.')[0] in ('torch','numpy','mmcv') for k in after))
        s=self.fixture(m);m.validate_spec(s)
        changed=deepcopy(s);changed['prelock']['GPU_allocator_cap_bytes']*=2
        with self.assertRaises(ValueError):m.validate_spec(changed)

    def test_lease_rejects_run_or_worker_join_mismatch(self):
        m=self.load();s=self.fixture(m)
        lease=dict(schema=m.LEASE_SCHEMA,status='ROOT_GRANTED',operation=s['operation'],
            run_id=s['run_id'],worker_ref=deepcopy(s['source_refs']['worker']),
            prelock=deepcopy(s['prelock']),output_directory=s['output_directory'],
            single_attempt=True,main_scientific_processes_reaped=True,
            exclusive_project_scientific_leaf=True,whole_device_exclusive=False)
        m.validate_lease(s,lease)
        changed=deepcopy(lease);changed['run_id']+='-other'
        with self.assertRaises(ValueError):m.validate_lease(s,changed)
        changed=deepcopy(lease);changed['worker_ref']['sha256']='1'*64
        with self.assertRaises(ValueError):m.validate_lease(s,changed)
        changed=deepcopy(s);changed['prelock']['cases'][0]['shape'][-1]+=1
        with self.assertRaises(ValueError):m.validate_spec(changed)


if __name__=='__main__':unittest.main()
