"""UNBOUND_DRAFT: preserve original N20 CPU145 numeric deadline failure26.

Append after ordinary typed22..25 helpers. Call n20_failed26_raw_only(spec)
before inherited_source_gate/first SOURCE SHA; call n20_failed26_closed after.
ROOT supplies existing require/ref_shape/read/closed/check_pin/helpercheck and
FAILURE21_PRIOR_KEYS. No import-time IO, scientific tensor load or retry.
Current count is normal>=145 / failed26 / pins=normal+26. The old CPU spec stays
145/25/170. terminal_identity_pins contains normal+legacy8 only; record.pin is
independent, so its dictionary length is never compared with total pin count.
Thirteen failed partial proofs remain unqualified and are not read or reused.
"""
from pathlib import Path

N20_FAILURE26_IDENTITY = dict(pid=77760, birth_filetime_100ns='134365278288498169',
                            birth_unix=1792054228.8498173)
N20_FAILURE26_EMPTY_LOG_SHA = 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
N20_FAILURE26_REF_KEYS = ('RAW_ref','admission','phase_binding','bootstrap','entry_source','spec')
N20_FAILURE26_PRESERVED_KEYS = (
    'N20_failed_training_record','N20_failed_training_stderr',
    'N20_failed_DEV_capture_record','N20_failed_DEV_capture_stderr',
    'N20_root_training_qualified_result','expected_N20_root_training_qualified_result',
    'N20_root_cache_qualified_result','expected_N20_root_cache_qualified_result',
    'N20_initialization_lock','expected_N20_initialization_digest',
    'N20_locked_protocol','expected_N20_locked_protocol','locked_protocol',
    'scout_module_source','base_scout_source','paired_training_body_source',
    'original_paired_training_body_source','N20_training_tail_source','initialization_seed',
    'cache_GPU_report','region_common_source','legacy_qualified_result','legacy_shards',
    'n17_root_qualified_result_ref','expected_n17_root_qualified_result',
    'new_eval_body_source','legacy_eval_body_source','capture_body_source',
    'coordinates_source','labels_source')


def n20_failed26_raw_only(spec):
    extra = spec['N20_failed_CPU_numeric_record']
    require(type(extra) is dict and set(extra) == {
        'RAW_ref','phase','pin','returncode','admission','phase_binding',
        'bootstrap','entry_source','spec','original_report_absent'},
        'Exact10 original CPU numeric controlled-deadline failure26 record')
    require(type(extra['returncode']) is int and extra['returncode'] == 1
        and extra['original_report_absent'] is True and spec['failed26_partials_reused'] is False,
        'Failure26 stays exit1 with absent REPORT; failed partial numeric proofs never reused')
    for key in N20_FAILURE26_REF_KEYS:
        ref_shape(extra[key])
    ref_shape(spec['N20_failed26_helper_source'])
    normal = spec['normal_RAW_count']
    require(type(normal) is int and normal >= 145
        and type(spec['expected_normal_RAW_count']) is int and spec['expected_normal_RAW_count'] == normal
        and type(spec['total_failed_RAW_count']) is int and spec['total_failed_RAW_count'] == 26
        and type(spec['total_terminal_identity_pin_count']) is int
        and spec['total_terminal_identity_pin_count'] == normal + 26,
        'Current own count is normal>=145 plus preserved failed26; old parent counts are immutable')
    prior = [spec[key]['RAW_ref']['path'] for key in FAILURE21_PRIOR_KEYS]
    prior += [spec[key]['RAW_ref']['path'] for key in (
        'additional_preserved_N17_independent_CPU_SOURCE_closure_failure',
        'additional_preserved_N20_training_output_precondition_failure',
        'additional_preserved_N20_current_failure_counter_failure',
        'N20_failed_training_record','N20_failed_DEV_capture_record')]
    path = Path(extra['RAW_ref']['path'])
    require(path.name == 'PARENT_RECEIPT.json' and type(extra['phase']) is str and extra['phase']
        and path.parent.name == extra['phase']
        and extra['RAW_ref']['path'] not in spec['terminal_identity_pins']
        and extra['RAW_ref']['path'] not in prior
        and extra['RAW_ref']['path'] not in {row['receipt']['path'] for row in spec['RAW_inputs']},
        'Failure26 original RAW locator distinct from all normal owners and failed1..25')
    # terminal_raw_only deliberately is not used: that ordinary-exit helper rejects aborts.
    raw = read(path)
    decision = helpercheck(raw, extra['pin'])
    check_pin(extra['pin'], raw, extra['phase'], 1)
    require(decision['passed'] is True and decision['normal_success'] is False
        and decision['partial_caches_qualified'] is False
        and raw['original_identity'] == N20_FAILURE26_IDENTITY
        and raw['admission_sha256'] == extra['admission']['sha256']
        and raw['abort'] == 'QUALIFIED_ORIGINAL_LEAF_RESOURCE_OR_DEADLINE'
        and raw['elapsed_seconds'] == 1800.342999999877
        and raw['minimum_ram_bytes'] == 30790778880
        and raw['whole_GPU_observed_peak_MiB'] == 2146
        and raw['logs']['stdout.log'] == raw['logs']['stderr.log'] == N20_FAILURE26_EMPTY_LOG_SHA,
        'Exact original CPU pid/birth, controlled deadline, resources and empty logs; never normal0')
    return raw


def n20_failed26_closed(spec, gate, raw):
    extra, closure = spec['N20_failed_CPU_numeric_record'], gate['closure']
    for key in N20_FAILURE26_REF_KEYS:
        closed(closure, extra[key])
    closed(closure, spec['N20_failed26_helper_source'])
    row = dict(receipt=extra['RAW_ref'],phase=extra['phase'],admission=extra['admission'],
               phase_binding=extra['phase_binding'],bootstrap=extra['bootstrap'])
    owner = gate['lineage'].owner_join(row, raw, closure)
    require(owner['scientific_spec'] == extra['spec']
        and owner['entry'] == extra['entry_source']['path'] and owner['cuda_visible_devices'] == '',
        'Original failed26 CPU-empty owner/source/spec/admission/phase/bootstrap exact')
    original = read(closed(closure, extra['spec']))
    output = Path(original['output_directory'])
    require(original['schema'] == 'ROOT-N20-fixedDEV-local-center-feature-error-CPU-numeric-shard-spec-v1'
        and original['phase'] == extra['phase'] and original['entry_source'] == extra['entry_source']
        and extra['entry_source']['bytes'] == 159357
        and original['normal_RAW_count'] == original['expected_normal_RAW_count'] == 145
        and original['total_failed_RAW_count'] == 25 and original['total_terminal_identity_pin_count'] == 170
        and original['cpu_shard_id'] == 0 and original['cpu_shard_owner_index'] == 145
        and original['CPU_parent_GPU_id'] == 0 and original['cpu_dataset_ordinal_range'] == [0,24]
        and original['RAW_inputs'] == spec['RAW_inputs'][:145]
        and all(closure.get(ref['path']) == ref for ref in original['sources'])
        and all(spec['terminal_identity_pins'].get(key) == value
                for key,value in original['terminal_identity_pins'].items())
        and original['completed_GPU_shards'] == spec['completed_GPU_shards']
        and all(original[key] == spec[key] for key in N20_FAILURE26_PRESERVED_KEYS)
        and all(original[key] == spec[key] for key in original if key.startswith('additional_preserved_'))
        and original['deadline_seconds'] == 1800 and original['automatic_retry'] is False
        and original['maximum_process_RSS_bytes'] == 16*1024**3
        and original['allocator_limit_MiB'] == 4096 and original['whole_GPU_limit_MiB'] == 8192
        and output.is_dir() and not output.is_symlink()
        and not (output/'REPORT.json').exists() and not (output/'REPORT.json').is_symlink()
        and Path(original['CPU_admission_file']).resolve(strict=True) ==
            Path(extra['admission']['path']).resolve(strict=True)
        and spec['failed26_partials_reused'] is False,
        'Original CPU145/25/170 source/prefix/frozen science retained; absent REPORT and no partial reuse')
    return owner
