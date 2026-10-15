"""UNBOUND metadata-only prior DEV hook, appended after N20 DEV front02.

Requires existing globals require/read/closed and inherited lineage/context/
owners. legacy_gate must already have populated legacy_rows/root/populations.
No source_gate/training_gate from N16/N17 is called; no Torch or tensor load.

Required existing spec fields (all refs must be in complete SOURCE):
  n17_root_qualified_result_ref, expected_n17_root_qualified_result (whole JSON),
  CPU_adapter_source (the same decoder ref as all seven original numeric CPUs).
Optional aliases prior_common_decoder_source/prior_shared_validator_source must
equal that decoder/original Source04 respectively. The latter is derived from
qualified parent1.source, never a fabricated locator. Neither old entry nor
training_gate may run. legacy_qualified_result/legacy_shards still mean N16;
N17 physical parents are obtained from qualified finalQ.numeric_CPU_shards.
Existing inherited context supplies N10_evaluation_source, reader_source,
renderer_source, DEV, COCO_source, export_source, native_reference_predictions.
Their values are not guessed or copied from an unrelated cache.

Outputs required by DEV tail: DEV_packet_module_refs exact{n10,reader,renderer},
DEV_packet_gate (separate gate copy whose report is the original prior report),
DEV_annotation_ref, DEV_packet_rows, DEV_packet_refs, prior_common_rows (187
items exact{row,binding}), prior_common_decoder_source. Actual shared/bundle
tensors, registry, common-C roles and same-input PR still require owned checks.
This hook preserves original gate parent/evaluation/common/context identities.
"""

import json
from pathlib import Path


N20_PRIOR_GPU_RANGES = ((0, 47), (47, 70), (70, 94), (94, 141), (141, 187))
N20_PRIOR_CPU_RANGES = ((0, 47), (47, 70), (70, 94), (94, 117), (117, 141), (141, 164), (164, 187))
N20_PRIOR_CPU_PARENT_IDS = (0, 1, 2, 3, 3, 4, 4)
N20_PRIOR_GPU_KEYS = {'shard_id', 'dataset_ordinal_range', 'source', 'spec', 'report',
    'RAW', 'admission', 'phase_binding', 'bootstrap', 'shard_owner_index'}
N20_PRIOR_CPU_KEYS = {'cpu_shard_id', 'cpu_dataset_ordinal_range', 'source', 'spec', 'report',
    'RAW', 'admission', 'phase_binding', 'bootstrap', 'cpu_shard_owner_index'}
N20_PRIOR_BINDING_KEYS = ('source', 'spec', 'GPU_admission', 'phase', 'ROOTtrain_result',
    'legacy_qualified_result', 'experiment_lock', 'new_eval_body_source', 'checkpoints',
    'shard_id', 'dataset_ordinal_range')


def _n20_prior_exact(left, right):
    return json.dumps(left, sort_keys=True, separators=(',', ':'), allow_nan=False) == json.dumps(
        right, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _n20_prior_completed_CPU(record, output_key, owner, gate, spec):
    """Only metadata joins of already-normal original N17 final134/post135."""
    require(type(record) is dict and set(record) == {
        'source', 'spec', output_key, 'RAW', 'admission', 'phase_binding', 'bootstrap', 'owner_index'}
        and type(record['owner_index']) is int and record['owner_index'] == owner
        and owner < spec['normal_RAW_count'], 'Explicit completed original N17 final/post record')
    closure = gate['closure']
    for key in set(record) - {'owner_index'}:
        closed(closure, record[key])
    row = spec['RAW_inputs'][owner]
    require(all(row[key] == record[other] for key, other in (
        ('receipt', 'RAW'), ('admission', 'admission'), ('phase_binding', 'phase_binding'),
        ('bootstrap', 'bootstrap'))), 'Original final/post exact normal RAW row')
    joined = gate['lineage'].owner_ref_join(gate['context'], gate['owners'], record['RAW'],
        record['admission'], record['source'], record['spec'], '')
    parent = read(closed(closure, record['spec']))
    output = read(closed(closure, record[output_key]))
    require(type(joined) is int and joined == owner == parent['normal_RAW_count']
        and parent['expected_normal_RAW_count'] == owner
        and parent['entry_source'] == record['source'] and parent['phase'] == row['phase']
        and parent['RAW_inputs'] == spec['RAW_inputs'][:owner]
        and parent['total_failed_RAW_count'] == 21 and parent['total_terminal_identity_pin_count'] == owner + 21
        and all(closure.get(ref['path']) == ref for ref in parent['sources'])
        and all(spec['terminal_identity_pins'].get(key) == value
                for key, value in parent['terminal_identity_pins'].items())
        and output['source'] == record['source'] and output['spec'] == record['spec']
        and output['CPU_admission'] == record['admission'] and output['phase'] == row['phase']
        and Path(parent['CPU_admission_file']).resolve(strict=True) ==
            Path(record['admission']['path']).resolve(strict=True)
        and Path(record[output_key]['path']).resolve(strict=True) ==
            (Path(parent['output_directory']) / ('Q.json' if output_key == 'Q' else 'ROOT_RESULT.json')).resolve(strict=True),
        'Original final/post owner/full-source/prefix/pins/admission/output metadata exact')
    return parent, output, read(closed(closure, record['RAW']))


def n20_prior_DEV_carrier_gate(spec, gate):
    closure = gate['closure']
    preserved = {key: gate[key] for key in ('parent', 'evaluation', 'common', 'context')}
    require(len(gate['legacy_rows']) == 187 and len(gate['legacy_reports']) == 4
        and len(gate['legacy_manifests']) == 4, 'Existing qualified N16 legacy metadata gate must run first')
    for key in ('n17_root_qualified_result_ref', 'CPU_adapter_source'):
        closed(closure, spec[key])
    decoder_ref = spec['CPU_adapter_source']
    if 'prior_common_decoder_source' in spec:
        closed(closure, spec['prior_common_decoder_source'])
        require(spec['prior_common_decoder_source'] == decoder_ref, 'Explicit decoder alias equals original CPU_adapter_source')

    # Reproduce only N16's prior/selector metadata setup, never its training_gate.
    context = dict(gate['context'])
    context['selector_mode'] = 'reuse'
    prior, prior_q = gate['lineage'].prior_gate(context, closure, gate['owners'])
    selector = gate['lineage'].selector_parent_gate(context, closure, gate['owners'])
    prior_rows = read(closed(closure, prior['image_rows']))
    packets = prior_q['packet_refs']
    cache = read(closed(closure, spec['cache_GPU_report']))
    require(prior['teacher_checkpoint'] == cache['teacher_checkpoint']
        and type(prior_rows) is list and type(packets) is list
        and len(prior_rows) == len(packets) == len(selector['manifest']['rows']) == 187
        and sum(row['raw_GT'] for row in prior_rows) == 380,
        'Original DEV prior teacher/187 packet rows/380 raw GT and selector population')
    for index, (row, packet, item) in enumerate(zip(prior_rows, packets, gate['legacy_rows'])):
        legacy = item['row']
        require(index == legacy['dataset_index'] and row['image_id'] == legacy['image_id']
            and packet == legacy['native_packet']
            and selector['manifest']['rows'][index] == legacy['selector'],
            'Original prior row/packet, frozen selector and qualified shared carrier stay associated')
        closed(closure, packet)
    module_refs = {name: context[key] for name, key in (
        ('n10', 'N10_evaluation_source'), ('reader', 'reader_source'), ('renderer', 'renderer_source'))}
    for ref in tuple(module_refs.values()) + tuple(context[key] for key in (
            'DEV', 'COCO_source', 'export_source', 'native_reference_predictions')):
        closed(closure, ref)
    packet_gate = dict(gate, context=context, report=prior, prior=prior, prior_q=prior_q,
                       packets=packets, selector=selector)

    # Existing published N18 metadata ABI for the fully qualified N17 carriers.
    root = read(closed(closure, spec['n17_root_qualified_result_ref']))
    require(_n20_prior_exact(root, spec['expected_n17_root_qualified_result'])
        and root['schema'] == 'Root-N17-full187-qualified-result-after-final-and-post-actual0-v1'
        and root['status'] == 'FULL187_FIXED24_MAIN_RESULT_QUALIFIED_CURRENT_RECIPE_STOP'
        and root['qualified'] is True and root['original_final_CPU_actual0_verified'] is True
        and root['original_post_CPU_actual0_verified'] is True and root['normal_RAW_count'] == 136
        and root['total_failed_RAW_count'] == 21 and root['total_terminal_identity_pin_count'] == 157
        and root['decision'] == 'STOP' and root['allowed_next_seeds'] == []
        and all(root[key] is False for key in ('additional_seed_or_fusion_executed', 'formal_TEST_evaluated',
            'cross_dataset_evaluated', 'innovation_established', 'TGRS_readiness_established')),
        'Exact completed original N17 Root scope/STOP; no new scientific qualification')
    final_spec, final_q, final_raw = _n20_prior_completed_CPU(root['final_record'], 'Q', 134, gate, spec)
    post_spec, post, post_raw = _n20_prior_completed_CPU(root['post_record'], 'ROOT_RESULT', 135, gate, spec)
    qualified = root['final_qualified_result']
    require(_n20_prior_exact(final_raw, root['original_final_actual'])
        and _n20_prior_exact(post_raw, root['original_post_actual'])
        and post['final_record'] == root['final_record'] and post['final_qualified_result'] == qualified
        and post['qualified'] is False and post['ROOT_post_actual_CPU0_verified'] is False
        and qualified['qualified'] is True and qualified['final_Q'] == root['final_record']['Q']
        and qualified['control_results'] == final_q['control_results']
        and qualified['numeric_CPU_shards'] == final_q['numeric_CPU_shards'] == final_spec['numeric_CPU_shards']
        and final_q['qualified'] is False and final_q['images'] == 187 and final_q['GT'] == 380
        and final_q['natural_queries'] == 18700 and root['primary_interpretation'] == qualified['primary_interpretation']
        and qualified['legacy_qualified_result'] == final_q['legacy_qualified_result'] == spec['legacy_qualified_result'],
        'Completed immutable final/post Qs and separately qualified Root carrier ancestry exact')

    descriptors = final_q['numeric_CPU_shards']
    require(type(descriptors) is list and len(descriptors) == 7, 'Existing seven qualified N17 numeric CPU scopes')
    parents, origins = {}, {}
    for cpu_id, descriptor in enumerate(descriptors):
        require(type(descriptor) is dict and set(descriptor) == N20_PRIOR_CPU_KEYS
            and descriptor['cpu_shard_id'] == cpu_id
            and descriptor['cpu_dataset_ordinal_range'] == list(N20_PRIOR_CPU_RANGES[cpu_id]),
            'Original canonical seven numeric CPU descriptors, never inferred owner arithmetic')
        for key in ('source', 'spec', 'report', 'RAW', 'admission', 'phase_binding', 'bootstrap'):
            closed(closure, descriptor[key])
        cpu_parent = read(closed(closure, descriptor['spec']))
        item = cpu_parent['GPU_shard']
        require(cpu_parent['entry_source'] == descriptor['source']
            and cpu_parent['CPU_adapter_source'] == decoder_ref
            and cpu_parent['cpu_shard_id'] == cpu_id
            and cpu_parent['cpu_dataset_ordinal_range'] == descriptor['cpu_dataset_ordinal_range']
            and type(item) is dict and set(item) == N20_PRIOR_GPU_KEYS
            and item['shard_id'] == N20_PRIOR_CPU_PARENT_IDS[cpu_id],
            'Qualified numeric CPU parent names the exact shared common-PR decoder and physical GPU')
        sid = item['shard_id']
        if sid in parents:
            require(parents[sid]['descriptor'] == item, 'Repeated split CPU scopes retain identical whole GPU parent')
            continue
        for key in ('source', 'spec', 'report', 'RAW', 'admission', 'phase_binding', 'bootstrap'):
            closed(closure, item[key])
        owner = gate['lineage'].owner_ref_join(gate['context'], gate['owners'], item['RAW'],
            item['admission'], item['source'], item['spec'], '0')
        parent = read(closed(closure, item['spec']))
        report = read(closed(closure, item['report']))
        manifest = read(closed(closure, report['capture_manifest']))
        first, last = N20_PRIOR_GPU_RANGES[sid]
        failed = 19 if sid == 0 else 20
        row = spec['RAW_inputs'][122 + sid]
        require(type(owner) is int and owner == item['shard_owner_index'] == 122 + sid
            and parent['normal_RAW_count'] == parent['expected_normal_RAW_count'] == owner
            and parent['RAW_inputs'] == spec['RAW_inputs'][:owner]
            and parent['entry_source'] == item['source'] and parent['total_failed_RAW_count'] == failed
            and parent['total_terminal_identity_pin_count'] == owner + failed
            and all(closure.get(ref['path']) == ref for ref in parent['sources'])
            and all(spec['terminal_identity_pins'].get(key) == value
                    for key, value in parent['terminal_identity_pins'].items())
            and row['receipt'] == item['RAW'] and row['admission'] == item['admission']
            and row['phase_binding'] == item['phase_binding'] and row['bootstrap'] == item['bootstrap']
            and item['dataset_ordinal_range'] == parent['dataset_ordinal_range'] == [first, last]
            and report['schema'] == 'N17-fixedDEV-local-center-error-GPU-shard-report-v1'
            and report['status'] == 'N17_FIXEDDEV_GPU_SHARD_CAPTURE_COMPLETE_PENDING_ROOT_ACTUAL0_AND_INDEPENDENT_CPU'
            and manifest['schema'] == 'N17-fixedDEV-GPU-shard-image-manifest-v1'
            and manifest['physical_capture_refs'] == report['capture_refs'] == [r['bundle'] for r in manifest['rows']]
            and manifest['images'] == report['images'] == last - first
            and manifest['queries'] == report['queries'] == (last - first) * 100
            and manifest['global_images'] == 187 and manifest['global_queries'] == 18700
            and report['scope_dataset_indices'] == list(range(first, last)),
            'Original five GPU actual0/source/prefix/19-or20/whole-manifest refs remain exact')
        binding = {key: manifest[key] for key in N20_PRIOR_BINDING_KEYS}
        if sid:
            binding['execution_partition_supplement'] = manifest['execution_partition_supplement']
            require(binding['execution_partition_supplement'] == qualified['execution_partition_supplement']
                and report['failed_partials_reused'] is False, 'Original failed partials never common-PR evidence')
        expected = dict(source=item['source'], spec=item['spec'], GPU_admission=item['admission'],
            phase=parent['phase'], ROOTtrain_result=qualified['ROOTtrain_result'],
            legacy_qualified_result=spec['legacy_qualified_result'], experiment_lock=qualified['experiment_lock'],
            new_eval_body_source=parent['new_eval_body_source'], checkpoints=qualified['checkpoints'],
            shard_id=sid, dataset_ordinal_range=[first, last])
        require(all(binding[key] == report[key] == value for key, value in expected.items())
            and all(report[key] == value for key, value in binding.items()), 'Exact original GPU/manifest binding, not current N20 training')
        for index, current in enumerate(manifest['rows'], start=first):
            old = gate['legacy_rows'][index]
            legacy = old['row']
            carrier = dict(bundle=legacy['bundle'], shared_member=legacy['shared_actual_image_inputs'],
                           original_GPU_binding=old['binding'])
            require(current['dataset_index'] == index and current['image_id'] == legacy['image_id']
                and current['identity'] == legacy['identity'] and current['qids'] == legacy['qids']
                and current['input_carrier'] == carrier and current['natural_queries'] == 100
                and len(current['query_profiles']) == 100
                and [entry['query_id'] for entry in current['query_profiles']] == list(range(100))
                and current['image_id'] not in origins, 'N17 common-C carrier has original identity/namespace/export order')
            closed(closure, current['bundle'])
            closed(closure, carrier['bundle'])
            origins[current['image_id']] = dict(row=current, binding=binding)
        parents[sid] = dict(descriptor=item, report=report, manifest=manifest)
    ordered = [origins[item['row']['image_id']] for item in gate['legacy_rows']]
    require(set(parents) == set(range(5)) and len(origins) == 187
        and [item['row']['dataset_index'] for item in ordered] == list(range(187)),
        'Five qualified physical parents cover exactly187 images and natural18700 queries')
    validator_ref = parents[1]['descriptor']['source']
    if 'prior_shared_validator_source' in spec:
        closed(closure, spec['prior_shared_validator_source'])
        require(spec['prior_shared_validator_source'] == validator_ref,
            'Explicit shared-validator alias equals original qualified Source04 parent source')
    gate.update(DEV_packet_module_refs=module_refs, DEV_packet_gate=packet_gate,
        DEV_annotation_ref=context['DEV'], DEV_packet_rows=prior_rows, DEV_packet_refs=packets,
        prior_common_rows=ordered, prior_common_decoder_source=decoder_ref,
        N17_common_PR_root=root, N17_common_PR_parents=parents,
        prior_shared_validator_source=validator_ref)
    require(all(gate[key] is value for key, value in preserved.items()),
        'Original N20 inherited parent/evaluation/common/context must not be mutated or replaced')
    return gate
