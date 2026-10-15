"""UNBOUND typed preservation of original N20 DEV GPU141 capture failure25.

Append after ordinary typed22/23/24 helpers. n20_failed25_raw_only(spec) must run
before inherited_source_gate/first SOURCE SHA; n20_failed25_closed afterwards.
ROOT updates only CURRENT entry/ninth counts to25 (first new attempt141/25/166).
Historical failed GPU141 spec remains141/24/165; prior failures stay unchanged.
No import-time IO, retry, tensor read or scientific eligibility is supplied.
The existing START/CAPTURES are preserved partials, never completed evidence.
"""


N20_FAILURE25_IDENTITY = dict(pid=73252, birth_filetime_100ns='134365182757714398',
                             birth_unix=1792044675.7714405)
N20_FAILURE25_ADMISSION_SHA = '3097ddd35fdb5a4c204f836e9e89c5ce3c9ff3f291048ec276c9acbe3b5ac26b'
N20_FAILURE25_STDERR_SHA = 'a51f6d29d9de97350d5b25ba71c8d4900b8c7d136fb683e072730ff2caac1ff4'
N20_FAILURE25_STDERR_LAST_LINE = (
    'RuntimeError: self.stride(-1) must be 1 to view Long as Byte (different element sizes), but got 0')


def n20_failed25_raw_only(spec):
    extra = spec['N20_failed_DEV_capture_record']
    require(type(extra) is dict and set(extra) == {
        'RAW_ref', 'phase', 'pin', 'returncode', 'admission', 'phase_binding',
        'bootstrap', 'entry_source', 'spec', 'original_report_absent'},
        'Exact10 preserved N20 DEV capture failure25 record')
    require(type(extra['returncode']) is int and extra['returncode'] == 1
        and extra['original_report_absent'] is True,
        'Original DEV GPU exit1/report absence retained; never completed normal owner')
    for key in ('RAW_ref', 'admission', 'phase_binding', 'bootstrap', 'entry_source', 'spec'):
        ref_shape(extra[key])
    prior = [spec[key]['RAW_ref']['path'] for key in FAILURE21_PRIOR_KEYS]
    prior += [spec[key]['RAW_ref']['path'] for key in (
        'additional_preserved_N17_independent_CPU_SOURCE_closure_failure',
        'additional_preserved_N20_training_output_precondition_failure',
        'additional_preserved_N20_current_failure_counter_failure',
        'N20_failed_training_record')]
    require(extra['RAW_ref']['path'] not in spec['terminal_identity_pins']
        and extra['RAW_ref']['path'] not in prior
        and extra['RAW_ref']['path'] not in {row['receipt']['path'] for row in spec['RAW_inputs']},
        'Failed25 stays separate from every normal and failed1..24 owner')
    raw = terminal_raw_only(extra['RAW_ref'], extra['phase'], 1)
    check_pin(extra['pin'], raw, extra['phase'], 1)
    require(raw['original_identity'] == N20_FAILURE25_IDENTITY
        and raw['admission_sha256'] == extra['admission']['sha256'] == N20_FAILURE25_ADMISSION_SHA
        and raw['logs']['stderr.log'] == N20_FAILURE25_STDERR_SHA
        and raw['error'] is None and raw['abort'] is None,
        'Exact original pid/birth/admission/stderr; ordinary exit1, held/reaped/birth absent, no abort')
    return raw


def n20_failed25_closed(spec, gate, raw):
    extra, closure = spec['N20_failed_DEV_capture_record'], gate['closure']
    for key in ('RAW_ref', 'admission', 'phase_binding', 'bootstrap', 'entry_source', 'spec'):
        closed(closure, extra[key])
    row = dict(receipt=extra['RAW_ref'], phase=extra['phase'], admission=extra['admission'],
        phase_binding=extra['phase_binding'], bootstrap=extra['bootstrap'])
    owner = gate['lineage'].owner_join(row, raw, closure)
    require(owner['scientific_spec'] == extra['spec']
        and owner['entry'] == extra['entry_source']['path'] and owner['cuda_visible_devices'] == '0',
        'Original failed25 GPU0/source/spec/phase/bootstrap exact original owner')
    original = read(closed(closure, extra['spec']))
    output = Path(original['output_directory'])
    preserved_keys = (
        'N20_failed_training_record', 'N20_failed_training_stderr',
        'N20_root_training_qualified_result', 'expected_N20_root_training_qualified_result',
        'N20_root_cache_qualified_result', 'expected_N20_root_cache_qualified_result',
        'N20_initialization_lock', 'expected_N20_initialization_digest',
        'N20_locked_protocol', 'expected_N20_locked_protocol', 'locked_protocol',
        'scout_module_source', 'base_scout_source', 'paired_training_body_source',
        'original_paired_training_body_source', 'N20_training_tail_source', 'initialization_seed',
        'cache_GPU_report', 'region_common_source', 'legacy_qualified_result', 'legacy_shards',
        'n17_root_qualified_result_ref', 'expected_n17_root_qualified_result',
        'new_eval_body_source', 'legacy_eval_body_source', 'capture_body_source',
        'coordinates_source', 'labels_source')
    require(original['schema'] == 'ROOT-N20-fixedDEV-local-center-feature-error-GPU-shard-spec-v1'
        and original['phase'] == extra['phase'] and original['entry_source'] == extra['entry_source']
        and original['normal_RAW_count'] == original['expected_normal_RAW_count'] == 141
        and original['total_failed_RAW_count'] == 24
        and original['total_terminal_identity_pin_count'] == 165
        and original['shard_id'] == 0 and original['dataset_ordinal_range'] == [0, 47]
        and original['RAW_inputs'] == spec['RAW_inputs'][:141]
        and all(closure.get(ref['path']) == ref for ref in original['sources'])
        and all(spec['terminal_identity_pins'].get(key) == value
                for key, value in original['terminal_identity_pins'].items())
        and all(original[key] == spec[key] for key in preserved_keys)
        and all(original[key] == spec[key] for key in original if key.startswith('additional_preserved_'))
        and original['deadline_seconds'] == 1800 and original['automatic_retry'] is False
        and original['maximum_process_RSS_bytes'] == 16 * 1024**3
        and original['allocator_limit_MiB'] == 4096 and original['whole_GPU_limit_MiB'] == 8192
        and output.is_dir() and not output.is_symlink()
        and (output / 'START.json').is_file() and not (output / 'START.json').is_symlink()
        and (output / 'CAPTURES').is_dir() and not (output / 'CAPTURES').is_symlink()
        and not (output / 'REPORT.json').exists() and not (output / 'REPORT.json').is_symlink()
        and Path(original['GPU_admission_file']).resolve(strict=True) ==
            Path(extra['admission']['path']).resolve(strict=True),
        'Original failed141/24/165 source/prefix/science and START/CAPTURES partials with absent REPORT')
    # CPU_adapter_source deliberately is not compared with the new own CPU's
    # F9 decoder: the original failed GPU used the old CEA2 common-PR decoder.
    # Entry/DEV-tail source may change for the isolated engineering fix; every
    # frozen model, scoring/capture body, protocol and prior carrier stays exact.
    stderr_ref = spec['N20_failed_DEV_capture_stderr']
    stderr_path = Path(closed(closure, stderr_ref))
    require(stderr_ref['bytes'] == 4679
        and stderr_ref['sha256'] == raw['logs']['stderr.log'] == N20_FAILURE25_STDERR_SHA
        and stderr_path.resolve(strict=True) ==
            (Path(extra['RAW_ref']['path']).parent / 'stderr.log').resolve(strict=True),
        'Original failed25 actual RAW-bound stderr locator/size/SHA exact')
    error_text = stderr_path.read_text('utf8')
    require(error_text.rstrip().endswith(N20_FAILURE25_STDERR_LAST_LINE)
        and extra['entry_source']['path'] in error_text and 'n20_same_bits' in error_text
        and '.view(torch.uint8)' in error_text,
        'Original exact Long-stride byte-view exception; no report reconstruction or partial qualification')
    return owner
