# Pure-stdlib real-nesting/callsite regression; no Torch/project/model import.
import argparse
import ast
import copy
from pathlib import Path
import sys
import unittest

SOURCE = Path(__file__).with_name('N20_REPEAT202_PAIRED_TRAIN_CPU_RUN_TAIL_UNBOUND_DRAFT_20261016_03.py')
HELPER = 'n20_validate_repeat202_snapshot_root_refs'
EXTRA_KEYS = ('N20_first201_ROOT_qualified_result',
              'N20_repeat202_initialization_qualified_result',
              'N20_repeat202_cleanup_probe_qualified_result')
BASE_KEYS = ('source', 'spec', 'GPU_admission', 'phase', 'N20_locked_protocol', 'locked_protocol',
             'parent_cache_report', 'neighbor_report', 'scout_module_source', 'base_scout_source',
             'paired_training_body_source', 'original_paired_training_body_source', 'initialization_seed',
             'order_seed', 'expected_N20_initialization_digest', 'frozen_teacher_checkpoint', 'frozen_PR_checkpoint',
             'N20_root_cache_qualified_result', 'N20_initialization_lock', 'N20_cache_CPU_Q')


def fixture():
    report = {key: dict(path='C:/SYNTHETIC_ONLY/ref%d.json' % index,
                        bytes=index + 1, sha256=chr(97 + index % 6) * 64)
              for index, key in enumerate(BASE_KEYS + EXTRA_KEYS)}
    report.update(phase='SYNTHETIC_ONLY_REPEAT202', initialization_seed=202610150202,
                  order_seed=202610140160001, expected_N20_initialization_digest='d' * 64)
    snapshot = dict(metadata=dict(copy.deepcopy(report), endpoint_epoch=8), completed_epoch=8,
                    models={'LOCAL_ERROR': 'SYNTHETIC_ONLY', 'CENTER_ERROR': 'SYNTHETIC_ONLY'},
                    optimizers={'LOCAL_ERROR': 'SYNTHETIC_ONLY', 'CENTER_ERROR': 'SYNTHETIC_ONLY'})
    return snapshot, report


class Repeat202NestedRootReferenceMetadata(unittest.TestCase):
    def actual_call(self):
        tree = ast.parse(SOURCE.read_bytes().decode('utf-8'))
        helpers = [node for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name == HELPER]
        self.assertEqual(len(helpers), 1)
        runners = [node for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name == 'run']
        self.assertEqual(len(runners), 1)
        calls = [node for node in ast.walk(runners[0])
                 if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                 and isinstance(node.value.func, ast.Name) and node.value.func.id == HELPER]
        self.assertEqual(len(calls), 1)
        # Preserve the actual run() argument expression: flat-vs-nested wiring is under test.
        isolated = ast.Module(body=[helpers[0], calls[0]], type_ignores=[])
        code = compile(isolated, '<synthetic-helper-plus-actual-call-expression>', 'exec')

        def execute(snapshot, report):
            namespace = dict(__builtins__=__builtins__, snapshot=snapshot, report=report)
            exec(code, namespace)
        return execute

    def test_valid_nested_23_binding_is_accepted(self):
        snapshot, report = fixture()
        self.assertEqual(len(report), 23)
        self.assertTrue(all(key in snapshot['metadata'] and key not in snapshot for key in EXTRA_KEYS))
        self.actual_call()(snapshot, report)


    def test_each_missing_nested_or_report_ref_is_rejected(self):
        execute = self.actual_call()
        for side in ('metadata', 'report'):
            for key in EXTRA_KEYS:
                with self.subTest(side=side, key=key):
                    snapshot, report = fixture()
                    del (snapshot['metadata'] if side == 'metadata' else report)[key]
                    with self.assertRaisesRegex(AssertionError, key):
                        execute(snapshot, report)

    def test_each_tampered_nested_or_report_ref_is_rejected(self):
        execute = self.actual_call()
        for side in ('metadata', 'report'):
            for key in EXTRA_KEYS:
                for field in ('path', 'bytes', 'sha256'):
                    with self.subTest(side=side, key=key, field=field):
                        snapshot, report = fixture()
                        changed = (snapshot['metadata'] if side == 'metadata' else report)[key]
                        if field == 'bytes':
                            changed[field] += 1
                        elif field == 'sha256':
                            changed[field] = 'f' * 64
                        else:
                            changed[field] += '.changed'
                        with self.assertRaisesRegex(AssertionError, key):
                            execute(snapshot, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=SOURCE)
    arguments, remaining = parser.parse_known_args()
    SOURCE = arguments.source
    unittest.main(argv=[sys.argv[0]] + remaining, verbosity=2)
