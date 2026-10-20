"""Small stdlib CPU8G delta tests; no main, ML import, real input or process."""
import ast
from pathlib import Path
import unittest

import entry


class ProfileTests(unittest.TestCase):
    def test_profile_identity_and_only_RSS_limit_change(self):
        self.assertEqual(entry.SCHEMA, 'U63-DEV-remaining186-full100-CPU8G-export-spec-v1')
        self.assertEqual(entry.OPERATION, 'CPU_U63_DEV_REMAINING186_FULL100_CPU8G_EXPORT')
        self.assertEqual(entry.LIMITS, dict(allowed_cpus=[0,1],threads=2,maximum_RSS_bytes=8*1024**3,
            minimum_available_RAM_bytes=16*1024**3,deadline_seconds=1800,CUDA_visible_devices=''))
        self.assertEqual(entry.MINIMUM_OUTPUT_FREE_BYTES,20*1024**3)
        self.assertEqual(entry.SELECTED_INDICES,tuple(range(1,187)))
        self.assertEqual(len(entry.SPEC_KEYS),14)

    def test_sampled_RSS_eight_GiB_boundary_refuses_more_without_relaxing_other_floors(self):
        def observation():
            return dict(peak_RSS_bytes=0,minimum_available_RAM_bytes=None,samples=0,violations=[],sampling_errors=[],
                minimum_output_disk_free_bytes=None,output_disk_samples=0,output_disk_sample_path='D:/never-open')
        observed=observation()
        entry.observe_resources(observed,rss_bytes=8*1024**3,available_ram_bytes=16*1024**3,output_free_bytes=20*1024**3)
        entry.validate_observations(observed)
        entry.observe_resources(observed,rss_bytes=8*1024**3+1,available_ram_bytes=16*1024**3,output_free_bytes=20*1024**3)
        with self.assertRaises(ValueError): entry.validate_observations(observed)
        self.assertIn('RSS/free-RAM boundary',observed['violations'])
        for available,disk in ((16*1024**3-1,20*1024**3),(16*1024**3,20*1024**3-1)):
            observed=observation()
            entry.observe_resources(observed,rss_bytes=1,available_ram_bytes=available,output_free_bytes=disk)
            with self.assertRaises(ValueError): entry.validate_observations(observed)

    def test_all_worker_functions_match_U62_AST_after_profile_label_mapping(self):
        old=Path(__file__).resolve().parent.parent/'U62_DEV_REMAINING186_EXPORT_DRAFT_20261021_01'/'entry.py'
        current=Path(entry.__file__)
        def functions(path):
            return {node.name:node for node in ast.parse(path.read_text(encoding='utf-8')).body
                    if isinstance(node,ast.FunctionDef)}
        previous,current_functions=functions(old),functions(current)
        self.assertEqual(set(previous),set(current_functions))
        class RestoreProfileLabels(ast.NodeTransformer):
            def visit_Constant(self,node):
                if type(node.value) is str:
                    node.value=node.value.replace('U63','U62').replace('CPU8G-','CPU-')
                return node
        mapping=RestoreProfileLabels()
        for name in previous:
            self.assertEqual(ast.dump(previous[name],include_attributes=False),
                ast.dump(mapping.visit(current_functions[name]),include_attributes=False),name)


if __name__=='__main__': unittest.main()
