# Portable stdlib-only actual CPU-scope metadata-prefix regression.
import argparse
import ast
import copy
from pathlib import Path
import sys
import unittest

SOURCE = Path(__file__).with_name('N20_REPEAT202_FIXEDDEV_CPU_NUMERIC_RUN_TAIL_UNBOUND_DRAFT_20261016_01.py')


def fixture(sid=0, owner=197):
    ranges = ((0,6),(6,12),(12,18),(18,24),(24,30),(30,36),(36,42),(42,47),
        (47,53),(53,59),(59,65),(65,71),(71,77),(77,83),(83,89),(89,94),
        (94,100),(100,106),(106,112),(112,118),(118,124),(124,130),(130,136),(136,141),
        (141,147),(147,153),(153,159),(159,165),(165,171),(171,177),(177,183),(183,187))
    parent_ranges = ((0,47),(47,94),(94,141),(141,187))
    parents = [dict(shard_id=i, dataset_ordinal_range=list(parent_ranges[i]), shard_owner_index=index)
               for i, index in enumerate((183,185,189,192))]
    spec = dict(cpu_shard_id=sid, CPU_parent_GPU_id=sid//8, normal_RAW_count=owner,
                expected_normal_RAW_count=owner, qualification_owner_index=owner,
                cpu_shard_owner_index=owner, initialization_seed=202610150202,
                cpu_dataset_ordinal_range=list(ranges[sid]), GPU_shard=copy.deepcopy(parents[sid//8]))
    return spec, dict(completed_GPU_shards=parents)


class Repeat202CPUScopeMetadata(unittest.TestCase):
    def actual_prefix(self):
        tree = ast.parse(SOURCE.read_bytes().decode('utf-8'))
        values = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in (
                            'N20_CPU_RANGES', 'N20_CPU_PARENT_GPU_IDS', 'N20_CPU_GPU_RANGES'):
                        values[target.id] = ast.literal_eval(node.value)
        self.assertEqual(len(values), 3)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name=='n20_cpu_scope']
        self.assertEqual(len(functions), 1)
        prefix = []
        reached_report_load = False
        for statement in functions[0].body:
            if (isinstance(statement, ast.Assign) and isinstance(statement.targets[0], ast.Tuple)
                    and [target.id for target in statement.targets[0].elts
                         if isinstance(target, ast.Name)] == ['report','manifest']):
                reached_report_load = True
                break
            prefix.append(statement)
        self.assertTrue(reached_report_load)
        self.assertEqual(sum(isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                             and isinstance(node.value.func, ast.Name) and node.value.func.id=='require'
                             for node in prefix), 4)
        # Only the actual plain-metadata prefix before REPORT/manifest/payload access runs.
        code = compile(ast.Module(body=prefix, type_ignores=[]), '<actual-CPU-scope-metadata-prefix-only>', 'exec')

        def require(ok, message):
            if not ok:
                raise AssertionError(message)

        def execute(spec, gate):
            namespace = dict(values, __builtins__=__builtins__, spec=spec, gate=gate, require=require)
            exec(code, namespace)
            return namespace['first'], namespace['last']
        return execute

    def test_explicit_nonconsecutive_owners_are_accepted(self):
        execute = self.actual_prefix()
        for sid, owner, expected_range in ((0,197,(0,6)), (31,247,(183,187))):
            with self.subTest(sid=sid, owner=owner):
                spec, gate = fixture(sid, owner)
                self.assertEqual(execute(spec, gate), expected_range)


    def test_inconsistent_or_historical_current_owner_and_seed_are_rejected(self):
        execute = self.actual_prefix()
        for field in ('normal_RAW_count', 'expected_normal_RAW_count',
                      'qualification_owner_index', 'cpu_shard_owner_index'):
            with self.subTest(field=field):
                spec, gate = fixture()
                spec[field] += 1
                with self.assertRaises(AssertionError):
                    execute(spec, gate)
        spec, gate = fixture(owner=145)
        with self.assertRaises(AssertionError):
            execute(spec, gate)
        spec, gate = fixture()
        spec['initialization_seed'] = 202610150201
        with self.assertRaises(AssertionError):
            execute(spec, gate)

    def test_stale_unordered_duplicate_or_future_GPU_owners_are_rejected(self):
        execute = self.actual_prefix()
        for case in ('historical141', 'duplicate', 'unordered', 'not_before_CPU'):
            with self.subTest(case=case):
                spec, gate = fixture()
                owners = gate['completed_GPU_shards']
                if case == 'historical141':
                    owners[0]['shard_owner_index'] = 141
                elif case == 'duplicate':
                    owners[1]['shard_owner_index'] = owners[0]['shard_owner_index']
                elif case == 'unordered':
                    owners[1]['shard_owner_index'], owners[2]['shard_owner_index'] = (
                        owners[2]['shard_owner_index'], owners[1]['shard_owner_index'])
                else:
                    owners[3]['shard_owner_index'] = spec['qualification_owner_index']
                spec['GPU_shard'] = copy.deepcopy(owners[spec['CPU_parent_GPU_id']])
                with self.assertRaises(AssertionError):
                    execute(spec, gate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=SOURCE)
    arguments, remaining = parser.parse_known_args()
    SOURCE = arguments.source
    unittest.main(argv=[sys.argv[0]] + remaining, verbosity=2)
