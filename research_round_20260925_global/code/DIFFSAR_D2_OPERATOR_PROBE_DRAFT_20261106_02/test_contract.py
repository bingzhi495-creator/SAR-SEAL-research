"""Necessary source/lease admission check only; no scientific imports."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import unittest

class D2Admission(unittest.TestCase):
    def test_cublas_workspace_clear_is_required_and_recorded(self):
        from types import SimpleNamespace
        p=Path(__file__).with_name('worker.py');s=importlib.util.spec_from_file_location('_D2_CLEANUP_METADATA_TEST',p)
        m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
        calls=[];report={}
        standin=SimpleNamespace(_C=SimpleNamespace(_cuda_clearCublasWorkspaces=lambda:calls.append('clear')))
        m.clear_blas_workspaces(standin,report)
        self.assertEqual(calls,['clear'])
        self.assertTrue(report['cuBLAS_workspace_cleanup_entered'])
        self.assertTrue(report['cuBLAS_workspaces_cleared'])
        with self.assertRaises(ValueError):m.clear_blas_workspaces(SimpleNamespace(_C=SimpleNamespace()),{})

    def test_fixed_binary_and_run_lease_cannot_be_substituted(self):
        p=Path(__file__).with_name('worker.py');s=importlib.util.spec_from_file_location('_D2_PROBE_METADATA_TEST',p)
        m=importlib.util.module_from_spec(s);before=set(sys.modules);s.loader.exec_module(m)
        self.assertFalse(any(k.split('.')[0] in ('torch','torchvision','detectron2','numpy') for k in set(sys.modules)-before))
        ref=lambda name:dict(path='/private/'+name,bytes=1,sha256='0'*64)
        libs={k:ref(k) for k in m.LIBRARY_NAMES};libs['d2_extension'].update(m.D2_EXTENSION_PIN)
        spec=dict(schema=m.SCHEMA,operation=m.OPERATION,run_id='D2-fixed-FP32-v01-test',
            source_refs={'worker':ref('worker.py')},library_sources=libs,root_lease_ref=ref('lease.json'),
            output_directory='/private/OUTPUT',prelock=deepcopy(m.PRELOCK))
        m.validate_spec(spec)
        bad=deepcopy(spec);bad['library_sources']['d2_extension']['sha256']='1'*64
        with self.assertRaises(ValueError):m.validate_spec(bad)
        lease=dict(schema=m.LEASE_SCHEMA,status='ROOT_GRANTED',operation=spec['operation'],run_id=spec['run_id'],
            worker_ref=spec['source_refs']['worker'],prelock=spec['prelock'],output_directory=spec['output_directory'],
            single_attempt=True,main_scientific_processes_reaped=True,exclusive_project_scientific_leaf=True,
            whole_device_exclusive=False)
        m.validate_lease(spec,lease);lease['run_id']+='-other'
        with self.assertRaises(ValueError):m.validate_lease(spec,lease)

if __name__=='__main__':unittest.main()
