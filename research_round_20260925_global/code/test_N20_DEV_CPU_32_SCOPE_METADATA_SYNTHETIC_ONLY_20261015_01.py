"""Stdlib-only synthetic execution-partition/header contract; no tensor/RAW IO."""
import argparse
import ast
from pathlib import Path
import sys
import types
import unittest

ROOT = Path('E:/SAR/seal/tmp').resolve(strict=True)
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--front',type=Path)
parser.add_argument('legacy_front',nargs='?',type=Path)
options,remaining = parser.parse_known_args()
sys.argv = [sys.argv[0]]+remaining
FRONT = (options.front or options.legacy_front or ROOT / 'N20_FIXEDDEV_FEATURE_ERROR_CPU_FRONT_UNBOUND_DRAFT_20261015_04.py').resolve(strict=True)
assert FRONT.parent == ROOT and 'UNBOUND' in FRONT.name and not FRONT.is_symlink()
EXPECTED = ((0,6),(6,12),(12,18),(18,24),(24,30),(30,36),(36,42),(42,47),
    (47,53),(53,59),(59,65),(65,71),(71,77),(77,83),(83,89),(89,94),
    (94,100),(100,106),(106,112),(112,118),(118,124),(124,130),(130,136),(136,141),
    (141,147),(147,153),(153,159),(159,165),(165,171),(171,177),(177,183),(183,187))
PARENTS = (0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3)


def load_front():
    # Only literal metadata constants and the single public header are loaded;
    # a complete entry is never imported and no scientific module can run.
    parsed = ast.parse(FRONT.read_text(encoding='utf-8-sig'))
    names = {'N20_DEV_CPU_SCHEMA','N20_CPU_RANGES','N20_CPU_PARENT_GPU_IDS','N20_COMPLETED_GPU_KEYS','N20_DEV_RANGES'}
    values = {'Path':Path,'__file__':str(FRONT),'CPU_SCHEMA':'ROOT-N19-fixedDEV-full187-fixed-half-readout-CPU-spec-v1'}
    for node in parsed.body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in names:
                    values[target.id] = ast.literal_eval(node.value)
    functions = [node for node in parsed.body if isinstance(node,ast.FunctionDef) and node.name=='n20_validate_cpu_header']
    assert len(functions)==1 and names <= set(values)
    def require(ok,message):
        if not ok:
            raise ValueError(message)
    def ref_shape(ref):
        require(type(ref) is dict and set(ref) == {'path','bytes','sha256'} and Path(ref['path']).is_absolute()
            and ref['bytes'] == 0 and ref['sha256'] == '0'*64,'Synthetic ref shape only')
    values.update(require=require,ref_shape=ref_shape)
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(FRONT),'exec'),values)
    return types.SimpleNamespace(**values)


def fixture(module,cid):
    def ref(role):
        return dict(path=str(ROOT/('SYNTHETIC_ONLY_'+role)),bytes=0,sha256='0'*64)
    GPU = []
    for pid,bounds in enumerate(((0,47),(47,94),(94,141),(141,187))):
        item = {key:ref('%d_%s'%(pid,key)) for key in ('source','spec','report','RAW','admission','phase_binding','bootstrap')}
        item.update(shard_id=pid,dataset_ordinal_range=list(bounds),shard_owner_index=141+pid);GPU.append(item)
    owner = 145+cid;pid = PARENTS[cid]
    return dict(schema=module.N20_DEV_CPU_SCHEMA,cpu_shard_id=cid,CPU_parent_GPU_id=pid,
        cpu_dataset_ordinal_range=list(EXPECTED[cid]),cpu_shard_owner_index=owner,
        normal_RAW_count=owner,expected_normal_RAW_count=owner,total_failed_RAW_count=26,
        total_terminal_identity_pin_count=owner+26,initialization_seed=202610150201,
        deadline_seconds=1800,automatic_retry=False,maximum_process_RSS_bytes=16*1024**3,
        allocator_limit_MiB=4096,whole_GPU_limit_MiB=8192,entry_source=dict(path=str(FRONT)),
        completed_GPU_shards=GPU,GPU_shard=GPU[pid],GPU_outer_source=GPU[pid]['source'],failed26_partials_reused=False)


class ThirtyTwoScopes(unittest.TestCase):
    def test_exact_coverage_headers_and_reject_old_current_count(self):
        module = load_front();historical = module.CPU_SCHEMA
        self.assertEqual(tuple(module.N20_CPU_RANGES),EXPECTED)
        self.assertEqual(tuple(module.N20_CPU_PARENT_GPU_IDS),PARENTS)
        self.assertEqual([index for a,z in EXPECTED for index in range(a,z)],list(range(187)))
        self.assertTrue(all(0 < z-a <= 6 for a,z in EXPECTED))
        for cid in range(32):
            with self.subTest(scope=cid):
                spec = fixture(module,cid)
                self.assertTrue(module.n20_validate_cpu_header(spec))
        old = fixture(module,0);old['total_failed_RAW_count']=25;old['total_terminal_identity_pin_count']=170
        with self.assertRaises(ValueError):module.n20_validate_cpu_header(old)
        partial = fixture(module,0);partial['failed26_partials_reused']=True
        with self.assertRaises(ValueError):module.n20_validate_cpu_header(partial)
        bad = fixture(module,8);bad['CPU_parent_GPU_id']=0
        with self.assertRaises(ValueError):module.n20_validate_cpu_header(bad)
        self.assertEqual(module.CPU_SCHEMA,historical)


if __name__ == '__main__':
    unittest.main()
