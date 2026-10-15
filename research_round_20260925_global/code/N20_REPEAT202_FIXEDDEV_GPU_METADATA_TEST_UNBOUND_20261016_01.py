# Portable stdlib test of actual DEV-tail seed/nested-binding predicate nodes.
import argparse
import ast
import copy
from pathlib import Path
import sys
import unittest

SOURCE = Path(__file__).with_name('N20_REPEAT202_FIXEDDEV_GPU_RUN_TAIL_UNBOUND_DRAFT_20261016_01.py')
EXTRA_KEYS = ('N20_first201_ROOT_qualified_result',
              'N20_repeat202_initialization_qualified_result',
              'N20_repeat202_cleanup_probe_qualified_result')
BASE_KEYS = ('source', 'spec', 'GPU_admission', 'phase', 'N20_locked_protocol', 'locked_protocol',
             'parent_cache_report', 'neighbor_report', 'scout_module_source', 'base_scout_source',
             'paired_training_body_source', 'original_paired_training_body_source', 'initialization_seed',
             'order_seed', 'expected_N20_initialization_digest', 'frozen_teacher_checkpoint', 'frozen_PR_checkpoint',
             'N20_root_cache_qualified_result', 'N20_initialization_lock', 'N20_cache_CPU_Q')


def fixture(epoch=8):
    binding = {key: dict(path='C:/SYNTHETIC_ONLY/ref%d.json' % index,
                         bytes=index + 1, sha256=chr(97 + index % 6) * 64)
               for index, key in enumerate(BASE_KEYS + EXTRA_KEYS)}
    binding.update(phase='SYNTHETIC_ONLY_REPEAT202', initialization_seed=202610150202,
                   order_seed=202610140160001, expected_N20_initialization_digest='d' * 64)
    cp = dict(initialization_seed=202610150202, completed_epoch=epoch,
              metadata=dict(copy.deepcopy(binding), endpoint_epoch=epoch))
    return cp, binding, epoch


class Repeat202DEVCheckpointMetadata(unittest.TestCase):
    def actual_predicate(self):
        tree = ast.parse(SOURCE.read_bytes().decode('utf-8'))
        factories = [node for node in tree.body
                     if isinstance(node, ast.FunctionDef) and node.name == 'n20_model_load_cpu']
        self.assertEqual(len(factories), 1)
        selected = {}
        for node in ast.walk(factories[0]):
            if not (isinstance(node, ast.Compare) and isinstance(node.left, ast.Subscript)
                    and isinstance(node.left.value, ast.Name) and node.left.value.id == 'cp'):
                continue
            index = node.left.slice
            if isinstance(index, ast.Index):
                index = index.value
            key = ast.literal_eval(index)
            if key in ('initialization_seed', 'completed_epoch', 'metadata'):
                self.assertNotIn(key, selected)
                selected[key] = node
        self.assertEqual(set(selected), {'initialization_seed', 'completed_epoch', 'metadata'})
        # Execute the unchanged real predicate nodes, without importing/running its Torch function.
        expression = ast.Expression(body=ast.BoolOp(op=ast.And(), values=[
            selected[key] for key in ('initialization_seed', 'completed_epoch', 'metadata')]))
        code = compile(ast.fix_missing_locations(expression), '<actual-DEV-tail-metadata-predicates-only>', 'eval')

        def check(cp, original_binding, epoch):
            return eval(code, dict(__builtins__=__builtins__, cp=cp,
                                   original_binding=original_binding, epoch=epoch))
        return check

    def test_seed202_nested23_checkpoint_is_accepted(self):
        check = self.actual_predicate()
        for epoch in (8, 24):
            with self.subTest(epoch=epoch):
                cp, binding, epoch = fixture(epoch)
                self.assertEqual(len(binding), 23)
                self.assertTrue(all(key in cp['metadata'] and key not in cp for key in EXTRA_KEYS))
                self.assertTrue(check(cp, binding, epoch))


    def test_each_missing_root_ref_is_rejected(self):
        check = self.actual_predicate()
        for side in ('metadata', 'original_binding'):
            for key in EXTRA_KEYS:
                with self.subTest(side=side, key=key):
                    cp, binding, epoch = fixture()
                    del (cp['metadata'] if side == 'metadata' else binding)[key]
                    self.assertFalse(check(cp, binding, epoch))

    def test_each_tampered_root_ref_is_rejected(self):
        check = self.actual_predicate()
        for side in ('metadata', 'original_binding'):
            for key in EXTRA_KEYS:
                for field in ('path', 'bytes', 'sha256'):
                    with self.subTest(side=side, key=key, field=field):
                        cp, binding, epoch = fixture()
                        changed = (cp['metadata'] if side == 'metadata' else binding)[key]
                        if field == 'bytes':
                            changed[field] += 1
                        elif field == 'sha256':
                            changed[field] = 'f' * 64
                        else:
                            changed[field] += '.changed'
                        self.assertFalse(check(cp, binding, epoch))

    def test_historical_seed201_is_rejected(self):
        cp, binding, epoch = fixture()
        cp['initialization_seed'] = 202610150201
        self.assertFalse(self.actual_predicate()(cp, binding, epoch))

    def test_checkpoint_or_metadata_epoch_mismatch_is_rejected(self):
        check = self.actual_predicate()
        for side in ('checkpoint', 'metadata'):
            with self.subTest(side=side):
                cp, binding, epoch = fixture()
                if side == 'checkpoint':
                    cp['completed_epoch'] = 24
                else:
                    cp['metadata']['endpoint_epoch'] = 24
                self.assertFalse(check(cp, binding, epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=SOURCE)
    arguments, remaining = parser.parse_known_args()
    SOURCE = arguments.source
    unittest.main(argv=[sys.argv[0]] + remaining, verbosity=2)
