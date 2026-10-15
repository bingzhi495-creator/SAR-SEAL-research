"""Stdlib-only ordinary-header regression; ROOT runs RED, then GREEN.

Pass the ordinary front path explicitly. No parent spec/RAW/Source or tensor is
opened. Fake descriptors below test only the public metadata-header operation.
"""
import importlib.util
from pathlib import Path
import sys
import unittest


FRONT_PATH = Path(sys.argv.pop(1)).resolve(strict=True) if len(sys.argv) > 1 else (
    Path(__file__).with_name('N20_FIXEDDEV_FEATURE_ERROR_CPU_FRONT_UNBOUND_DRAFT_20261015_01.py'))


def load_front():
    spec = importlib.util.spec_from_file_location('_ordinary_N20_CPU_header_fixture', str(FRONT_PATH))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Keep the historical schema global. Current inheritance dispatch is a
    # separate assembly decision and must not mutate this ancestor constant.
    module.CPU_SCHEMA = 'ROOT-N19-fixedDEV-full187-fixed-half-readout-CPU-spec-v1'
    def require(ok, message):
        if not ok:
            raise ValueError(message)
    def ref_shape(ref):
        require(type(ref) is dict and set(ref) == {'path', 'bytes', 'sha256'}
            and Path(ref['path']).is_absolute() and type(ref['bytes']) is int
            and ref['bytes'] == 0 and ref['sha256'] == '0' * 64,
            'Synthetic strict-ref shape')
    module.require, module.ref_shape = require, ref_shape
    return module


def fixture(module):
    def ref(role):
        return {'path': str(FRONT_PATH.parent / ('SYNTHETIC_ONLY_' + role)),
                'bytes': 0, 'sha256': '0' * 64}
    records = []
    for sid, bounds in enumerate(((0,47),(47,94),(94,141),(141,187))):
        item = {key: ref('%d_%s' % (sid, key)) for key in
                ('source','spec','report','RAW','admission','phase_binding','bootstrap')}
        item.update(shard_id=sid, dataset_ordinal_range=list(bounds), shard_owner_index=141+sid)
        records.append(item)
    return dict(schema=module.N20_DEV_CPU_SCHEMA, cpu_shard_id=0, CPU_parent_GPU_id=0,
        cpu_dataset_ordinal_range=[0,24], cpu_shard_owner_index=145,
        normal_RAW_count=145, expected_normal_RAW_count=145, total_failed_RAW_count=24,
        total_terminal_identity_pin_count=169, initialization_seed=202610150201,
        deadline_seconds=1800, automatic_retry=False, maximum_process_RSS_bytes=16*1024**3,
        allocator_limit_MiB=4096, whole_GPU_limit_MiB=8192,
        entry_source={'path': str(FRONT_PATH)}, completed_GPU_shards=records,
        GPU_shard=records[0], GPU_outer_source=records[0]['source'])


class CPUHeaderLegacySchemaRegression(unittest.TestCase):
    def test_new_cpu_header_accepts_unchanged_historical_schema(self):
        module = load_front()
        historical = module.CPU_SCHEMA
        self.assertNotEqual(historical, module.N20_DEV_CPU_SCHEMA)
        self.assertTrue(module.n20_validate_cpu_header(fixture(module)))
        self.assertEqual(module.CPU_SCHEMA, historical)


if __name__ == '__main__':
    unittest.main()
