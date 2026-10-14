"""UNBOUND pure canonical controlled-failure RAW predicate; no I/O or hash.

ROOT must read and bind the actual record/pin, prove original ownership and
record locator phase, and invoke this BEFORE any Source read/hash. This helper
does not supply those proofs or qualify partial caches. Its embedded tests use
only invented dictionaries and do not establish a scientific execution result.
"""
import math

ABORT = 'QUALIFIED_ORIGINAL_LEAF_RESOURCE_OR_DEADLINE'
GPU_SCOPE = 'whole GPU observed, not per-process WDDM allocation'
MIN_RAM_BYTES = 16 * 1024**3
GPU_LIMIT_MIB = 8192


class RawGateError(ValueError):
    pass


def helpercheck(raw, pin):
    """Check canonical owned controlled-failure structure; allow extra fields.

    pin.phase is a record-locator obligation of the ROOT wrapper: the canonical
    actual RAW has no phase field. No synthetic field may be added to obtain a
    pass. This function never opens a record, computes SHA, or reads Source.
    """
    def need(condition, field):
        if not condition:
            raise RawGateError('Controlled resource RAW gate failed: '+field)

    def field(obj,key):
        need(type(obj) is dict and key in obj,key+' missing')
        return obj[key]

    def finite_number(value):
        return type(value) is int or (type(value) is float and math.isfinite(value))

    def same_json(left,right):
        if type(left) is not type(right):
            return False
        if type(left) is dict:
            return (all(type(k) is str for k in left) and set(left)==set(right)
                    and all(same_json(left[k],right[k]) for k in left))
        if type(left) is list:
            return len(left)==len(right) and all(same_json(a,b) for a,b in zip(left,right))
        if type(left) is float:
            return math.isfinite(left) and math.isfinite(right) and left==right
        return type(left) in (str,int,bool,type(None)) and left==right

    need(type(raw) is dict and type(pin) is dict,'raw/pin dictionaries')
    need(field(raw,'status')=='ACTUAL_OWNED_TERMINAL','status')
    rc=field(raw,'returncode')
    need(type(rc) is int and rc==1,'returncode must be exact failure1')
    need(field(raw,'root_wait_reaped') is True,'root_wait_reaped')
    held=field(raw,'held_handle_exit')
    need(field(held,'wait_state')=='WAIT_OBJECT_0','held wait_state')
    actual_exit=field(held,'actual_exit_code')
    need(type(actual_exit) is int and actual_exit==1,'held actual_exit_code must be failure1')
    birth=field(raw,'numeric_birth_absence')
    need(field(birth,'original_birth_absent') is True,'original birth absence')
    need(field(raw,'error') is None,'error must be None')
    need(field(raw,'abort')==ABORT,'exact controlled resource/deadline abort')
    need(field(raw,'wait_errors') is False,'wait_errors must be False')
    gpu_errors=field(raw,'GPU_observation_errors')
    need(type(gpu_errors) is list and gpu_errors==[],'GPU_observation_errors must be empty list')
    identity,pinned_identity=field(raw,'original_identity'),field(pin,'original_identity')
    need(type(identity) is dict and bool(identity) and type(pinned_identity) is dict
         and bool(pinned_identity) and same_json(identity,pinned_identity),'original_identity pin exact')
    sha,pinned_sha=field(raw,'admission_sha256'),field(pin,'admission_sha256')
    need(type(sha) is str and len(sha)==64 and all(c in '0123456789abcdef' for c in sha)
         and type(pinned_sha) is str and sha==pinned_sha,'admission SHA pin exact; no SHA computed')
    if 'returncode' in pin:
        need(type(pin['returncode']) is int and pin['returncode']==1,'optional pin returncode must be failure1')
    if 'phase' in raw and 'phase' in pin:
        need(same_json(raw['phase'],pin['phase']),'optional RAW phase matches pin')
    minimum_ram=field(raw,'minimum_ram_bytes')
    need(finite_number(minimum_ram) and minimum_ram>=MIN_RAM_BYTES,'minimum_ram_bytes >=16GiB')
    peak=field(raw,'whole_GPU_observed_peak_MiB')
    need(finite_number(peak) and 0<=peak<=GPU_LIMIT_MIB,'whole GPU observed peak <=8192MiB')
    limit=field(raw,'whole_GPU_used_limit_MiB')
    need(type(limit) is int and limit==GPU_LIMIT_MIB,'whole GPU limit must remain8192MiB')
    need(field(raw,'whole_leaf_VRAM_certified') is False,'per-leaf VRAM certification must remain False')
    need(field(raw,'GPU_scope')==GPU_SCOPE,'whole GPU observation scope')
    elapsed=field(raw,'elapsed_seconds')
    need(finite_number(elapsed) and 1800<=elapsed<=1805,'controlled resource/deadline elapsed1800..1805')
    return dict(passed=True,normal_success=False,resource_or_deadline_failure=True,
                numeric_or_method_failure_proven=False,partial_caches_qualified=False,
                returncode=1,held_actual_exit_code=1,abort=ABORT,
                minimum_ram_bytes=minimum_ram,whole_GPU_observed_peak_MiB=peak,
                elapsed_seconds=elapsed,phase_locator_proof_required_from_ROOT_wrapper=True,
                record_IO_performed=False,Source_read_or_hash_performed=False)


def _fake_tests():
    import copy
    raw = dict(status='ACTUAL_OWNED_TERMINAL',returncode=1,root_wait_reaped=True,
        original_identity={'pid':123,'birth':1000.25,'scope':'FAKE_SYNTHETIC_ONLY'},
        held_handle_exit={'wait_state':'WAIT_OBJECT_0','actual_exit_code':1},
        numeric_birth_absence={'original_birth_absent':True},error=None,abort=ABORT,
        wait_errors=False,GPU_observation_errors=[],minimum_ram_bytes=MIN_RAM_BYTES,
        elapsed_seconds=1800.,admission_sha256='1'*64,whole_GPU_observed_peak_MiB=8192,
        whole_GPU_used_limit_MiB=8192,whole_leaf_VRAM_certified=False,GPU_scope=GPU_SCOPE)
    pin = dict(original_identity=copy.deepcopy(raw['original_identity']),admission_sha256='1'*64,
               phase='FAKE_PHASE_LOCATOR_PROOF_IS_EXTERNAL',returncode=1,extra='allowed')
    baseline = helpercheck(raw,pin)
    assert baseline['passed'] and not baseline['normal_success'] and not baseline['partial_caches_qualified']
    assert baseline['phase_locator_proof_required_from_ROOT_wrapper']
    for elapsed in (1800,1805,1802.5):
        variant=copy.deepcopy(raw);variant['elapsed_seconds']=elapsed
        assert helpercheck(variant,pin)['passed']
    mutations = [
        ('status','COMPLETED'),('returncode',0),('returncode',True),('returncode',1.),
        ('root_wait_reaped',False),('root_wait_reaped',1),('error',False),('error',''),
        ('abort',None),('abort','ordinary-failure'),('wait_errors',True),('wait_errors',0),
        ('GPU_observation_errors',['failed']),('GPU_observation_errors',()),
        ('minimum_ram_bytes',MIN_RAM_BYTES-1),('minimum_ram_bytes',True),
        ('minimum_ram_bytes',float('nan')),('minimum_ram_bytes',float('inf')),
        ('elapsed_seconds',1799.999),('elapsed_seconds',1805.001),('elapsed_seconds',True),
        ('elapsed_seconds',float('nan')),('elapsed_seconds',float('inf')),('elapsed_seconds','1800'),
        ('whole_GPU_observed_peak_MiB',8192.01),('whole_GPU_observed_peak_MiB',-.01),
        ('whole_GPU_observed_peak_MiB',True),('whole_GPU_observed_peak_MiB',float('nan')),
        ('whole_GPU_observed_peak_MiB',float('inf')),('whole_GPU_used_limit_MiB',8191),
        ('whole_leaf_VRAM_certified',True),('whole_leaf_VRAM_certified',0),('GPU_scope','per-process'),
        ('admission_sha256','2'*64),('original_identity',{}),
        ('original_identity',dict(pid=124,birth=1000.25,scope='FAKE_SYNTHETIC_ONLY')),
        ('held_handle_exit',dict(wait_state='WAIT_TIMEOUT',actual_exit_code=1)),
        ('held_handle_exit',dict(wait_state='WAIT_OBJECT_0',actual_exit_code=0)),
        ('held_handle_exit',dict(wait_state='WAIT_OBJECT_0',actual_exit_code=True)),
        ('numeric_birth_absence',dict(original_birth_absent=False)),
        ('numeric_birth_absence',dict(original_birth_absent=1))]
    rejected=0
    for field,value in mutations:
        variant=copy.deepcopy(raw);variant[field]=value
        try: helpercheck(variant,pin)
        except RawGateError: rejected+=1
        else: raise AssertionError('Accepted fake violation '+field+'='+repr(value))
    for field in raw:
        variant=copy.deepcopy(raw);del variant[field]
        try: helpercheck(variant,pin)
        except RawGateError: rejected+=1
        else: raise AssertionError('Accepted missing required field '+field)
    for bad_pin in (dict(pin,returncode=0),dict(pin,admission_sha256='3'*64),
                    dict(pin,original_identity={'pid':True,'birth':1000.25,'scope':'FAKE_SYNTHETIC_ONLY'})):
        try: helpercheck(raw,bad_pin)
        except RawGateError: rejected+=1
        else: raise AssertionError('Accepted mismatched/normal-success pin')
    # Extras are permitted, without inventing a nonexistent RAW phase field.
    variant=copy.deepcopy(raw);variant['extra_field']='allowed'
    assert helpercheck(variant,pin)['passed'] and 'phase' not in variant
    return dict(status='SYNTHETIC_ONLY_FAKE_CHECKS_PASS',rejected_cases=rejected,
                accepted_cases=5,actual_RAW_read=False,Source_read_or_hash=False)


if __name__ == '__main__':
    import json
    print(json.dumps(_fake_tests(),sort_keys=True))
