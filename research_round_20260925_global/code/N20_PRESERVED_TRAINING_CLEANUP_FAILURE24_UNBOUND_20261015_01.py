"""UNBOUND terminal/source closure for original trained GPU cleanup failure24.

This preserves original actual1 and missing REPORT. It does not certify saved
weights, regenerate an original report, run training or turn failed GPU into0.
CPU recovery must independently validate all saved tensors and endpoints.
"""


def n20_failed24_raw_only(spec):
    extra = spec['N20_failed_training_record']
    require(type(extra) is dict and set(extra) == {
        'RAW_ref', 'phase', 'pin', 'returncode', 'admission', 'phase_binding',
        'bootstrap', 'entry_source', 'spec', 'original_report_absent'},
        'Exact preserved failed training24 record')
    require(type(extra['returncode']) is int and extra['returncode'] == 1
        and extra['original_report_absent'] is True, 'Original GPU exit1/report absence retained')
    for key in ('RAW_ref', 'admission', 'phase_binding', 'bootstrap', 'entry_source', 'spec'):
        ref_shape(extra[key])
    prior = [spec[key]['RAW_ref']['path'] for key in FAILURE21_PRIOR_KEYS]
    prior += [spec[key]['RAW_ref']['path'] for key in (
        'additional_preserved_N17_independent_CPU_SOURCE_closure_failure',
        'additional_preserved_N20_training_output_precondition_failure',
        'additional_preserved_N20_current_failure_counter_failure')]
    require(extra['RAW_ref']['path'] not in spec['terminal_identity_pins']
        and extra['RAW_ref']['path'] not in prior
        and extra['RAW_ref']['path'] not in {row['receipt']['path'] for row in spec['RAW_inputs']},
        'Failed24 remains separate from every normal and failed1..23 owner')
    raw = terminal_raw_only(extra['RAW_ref'], extra['phase'], 1)
    check_pin(extra['pin'], raw, extra['phase'], 1)
    require(raw['error'] is None and raw['abort'] is None,
        'Original cleanup exception is ordinary exit1, never successful or resource-aborted training')
    return raw


def n20_failed24_closed(spec, gate, raw):
    extra, closure = spec['N20_failed_training_record'], gate['closure']
    for key in ('RAW_ref', 'admission', 'phase_binding', 'bootstrap', 'entry_source', 'spec'):
        closed(closure, extra[key])
    row = dict(receipt=extra['RAW_ref'], phase=extra['phase'], admission=extra['admission'],
        phase_binding=extra['phase_binding'], bootstrap=extra['bootstrap'])
    owner = gate['lineage'].owner_join(row, raw, closure)
    require(owner['scientific_spec'] == extra['spec']
        and owner['entry'] == extra['entry_source']['path'] and owner['cuda_visible_devices'] == '0',
        'Original failed24 GPU0 owner/source/spec/phase/bootstrap association')
    original = read(closed(closure, extra['spec']))
    output = Path(original['output_directory'])
    require(original['schema'] == 'ROOT-N20-paired-local-center-feature-error-training-GPU-spec-v1'
        and original['phase'] == extra['phase'] and original['entry_source'] == extra['entry_source']
        and original['normal_RAW_count'] == original['expected_normal_RAW_count'] == 140
        and original['total_failed_RAW_count'] == 23 and original['total_terminal_identity_pin_count'] == 163
        and original['RAW_inputs'] == spec['RAW_inputs'][:140]
        and all(closure.get(ref['path']) == ref for ref in original['sources'])
        and all(spec['terminal_identity_pins'].get(key) == value
                for key, value in original['terminal_identity_pins'].items())
        and all(original[key] == spec[key] for key in (
            'additional_preserved_N20_training_output_precondition_failure',
            'additional_preserved_N20_current_failure_counter_failure',
            'N20_root_cache_qualified_result', 'expected_N20_root_cache_qualified_result',
            'N20_initialization_lock', 'expected_N20_initialization_digest',
            'N20_locked_protocol', 'expected_N20_locked_protocol', 'locked_protocol',
            'scout_module_source', 'base_scout_source', 'paired_training_body_source',
            'original_paired_training_body_source', 'N20_training_tail_source', 'initialization_seed'))
        and original['deadline_seconds'] == 1800 and original['automatic_retry'] is False
        and original['allocator_limit_MiB'] == 4096 and original['whole_GPU_limit_MiB'] == 8192
        and output.is_dir() and not output.is_symlink()
        and not (output / 'REPORT.json').exists() and not (output / 'REPORT.json').is_symlink()
        and Path(original['GPU_admission_file']).resolve(strict=True) ==
            Path(extra['admission']['path']).resolve(strict=True),
        'Original failed24 source/history/science and saved directory with missing REPORT preserved')
    stderr_ref = spec['N20_failed_training_stderr']
    stderr_path = Path(closed(closure, stderr_ref))
    require(stderr_ref['sha256'] == raw['logs']['stderr.log']
        and stderr_path.resolve(strict=True) ==
            (Path(extra['RAW_ref']['path']).parent / 'stderr.log').resolve(strict=True),
        'Original failed24 actual terminal-bound stderr locator and bytes')
    error_text = stderr_path.read_text('utf8')
    require(error_text.rstrip().endswith(
        'RuntimeError: All model/optimizer CUDA tensors must be released after training')
        and extra['entry_source']['path'] in error_text,
        'Actual original source failed at the final CUDA cleanup check; no invented GPU success report')
    return owner
