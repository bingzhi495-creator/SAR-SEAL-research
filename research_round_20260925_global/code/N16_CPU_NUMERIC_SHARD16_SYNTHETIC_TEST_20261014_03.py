"""SYNTHETIC_ONLY stdlib CPU numeric shard scope/failure16 test. No SCI I/O."""
import copy
import importlib.util
from pathlib import Path

p=Path(__file__).with_name('N16_FIXED_E24_ALL100_SCOUT_SHARDS_FULLDEV_CPU_PROOF_ROOT_UNBOUND_DRAFT_20261014_03.py')
s=importlib.util.spec_from_file_location('ordinary_cpu_numeric_shard03',p)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


def rejects(f):
    try:f()
    except (RuntimeError,ValueError,KeyError):return
    raise AssertionError('Invalid shard/failure accepted')


def main():
    for i,r in enumerate(m.RANGES):
        spec=dict(cpu_shard_id=i,cpu_dataset_ordinal_range=list(r),cpu_shard_owner_index=114+i,normal_RAW_count=114+i)
        assert m.cpu_shard_scope(spec)==r
        for key,value in [('cpu_shard_id',True),('cpu_dataset_ordinal_range',[False,r[1]]),('cpu_shard_owner_index',113),('normal_RAW_count',118)]:
            bad=dict(spec);bad[key]=value;rejects(lambda:m.cpu_shard_scope(bad))
    raw=dict(status='ACTUAL_OWNED_TERMINAL',returncode=1,root_wait_reaped=True,
        held_handle_exit=dict(wait_state='WAIT_OBJECT_0',actual_exit_code=1),numeric_birth_absence=dict(original_birth_absent=True),
        error=None,abort='QUALIFIED_ORIGINAL_LEAF_RESOURCE_OR_DEADLINE',wait_errors=False,GPU_observation_errors=[],
        original_identity={'SYNTHETIC_ONLY':True},admission_sha256='a'*64,minimum_ram_bytes=32*1024**3,
        whole_GPU_observed_peak_MiB=100.,whole_GPU_used_limit_MiB=8192,whole_leaf_VRAM_certified=False,
        GPU_scope='whole GPU observed, not per-process WDDM allocation',elapsed_seconds=1800.25)
    ref=lambda name:dict(path='E:/SYNTHETIC_ONLY/'+name,bytes=0,sha256='a'*64)
    phase='fakeCPU16';extra=dict(RAW_ref=ref(phase+'/PARENT_RECEIPT.json'),phase=phase,
        pin=dict(phase=phase,returncode=1,original_identity=raw['original_identity'],admission_sha256='a'*64),returncode=1,
        admission=ref('ad.json'),phase_binding=ref('phase.json'),bootstrap=ref('boot.py'),entry_source=ref('cpu.py'),spec=ref('spec.json'),original_Q_absent=True)
    spec=dict(additional_preserved_N16_fullDEV_controlled_CPU_proof_deadline_failure=extra,terminal_identity_pins={})
    for i,key in enumerate(m.FAILURE16_PRIOR_KEYS):spec[key]=dict(RAW_ref=ref('old%d/PARENT_RECEIPT.json'%i))
    original_read=m.read;m.read=lambda path:raw
    try:
        assert m.sixteenth_controlled_CPU_proof_deadline_failure_raw_only(spec)==raw
        for field,value in [('abort',None),('returncode',0),('elapsed_seconds',1799.)]:
            previous=raw[field];raw[field]=value;rejects(lambda:m.sixteenth_controlled_CPU_proof_deadline_failure_raw_only(spec));raw[field]=previous
        bad=copy.deepcopy(spec);bad['additional_preserved_N16_fullDEV_controlled_CPU_proof_deadline_failure']['original_Q_absent']=False
        rejects(lambda:m.sixteenth_controlled_CPU_proof_deadline_failure_raw_only(bad))
    finally:m.read=original_read
    print('SYNTHETIC_ONLY fixed4ranges/owners114117 + scope rejects + typed controlled CPU16/Qabsent PASS; no actual Source/RAW/science')


if __name__=='__main__':main()
