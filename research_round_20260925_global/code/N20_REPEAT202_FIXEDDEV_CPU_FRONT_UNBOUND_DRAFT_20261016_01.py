"""UNBOUND_NOT_EXECUTED: repeat202 fixed-DEV CPU32 metadata front, no run/CLI.

Assemble F typed prefix + its legacy metadata helpers + repeat202 GPUfront0886
pure metadata + endpointCPUfrontf2eb pure metadata + DEVfront9fa8 pure metadata
+ original prior-carrier hook/dependencies + THIS front + new CPU numeric tail.
Only the last source_gate owns the current CPU. No parent's source_gate/run is
called. Historical SCHEMA/CPU_SCHEMA/N20_TRAIN_SCHEMA and failures1..26 stay.

Current spec adds qualification_owner_index>=187 and the independent
N20_repeat202_CPU_execution_partition_lock/expected_N20_repeat202_CPU_execution_partition_lock.
cpu_shard_owner_index==qualification_owner_index==normal==expected; never145+cid.
The frozen current plan supplies32 explicit increasing CPU owners and four GPU
owners. Old CPU_execution_partition_supplement remains first201 HISTORY only.
Current exact10 completed_GPU_shards/GPU_shard retain whole original parents;
only scope_rows/predictions/captures are sliced. No input tensors read on import.
"""

from pathlib import Path

N20_DEV_CPU_SCHEMA = 'ROOT-N20-repeat202-fixedDEV-local-center-feature-error-CPU-numeric-shard-spec-v1'
N20_FINAL_CPU_SCHEMA = N20_DEV_CPU_SCHEMA
N20_CPU_RANGES = ((0,6),(6,12),(12,18),(18,24),(24,30),(30,36),(36,42),(42,47),
    (47,53),(53,59),(59,65),(65,71),(71,77),(77,83),(83,89),(89,94),
    (94,100),(100,106),(106,112),(112,118),(118,124),(124,130),(130,136),(136,141),
    (141,147),(147,153),(153,159),(159,165),(165,171),(171,177),(177,183),(183,187))
N20_CPU_PARENT_GPU_IDS = (0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3)
N20_DEV_RANGES = N20_REPEAT202_DEV_RANGES
N20_COMPLETED_GPU_KEYS = {'shard_id','dataset_ordinal_range','source','spec','report','RAW',
    'admission','phase_binding','bootstrap','shard_owner_index'}
N20_CPU_SOURCE_KEYS = ('GPU_outer_source','CPU_adapter_source','selector_cpu_proof_source',
    'legacy_selector_proof_source','endpoint_body_source','PR_formula_source','labels_source',
    'N20_failed_DEV_capture_stderr','N20_failed26_helper_source')
N20_GPU_BINDING_KEYS = ('source','spec','GPU_admission','phase','N20_repeat202_ROOT_training_qualification',
    'initialization_seed','N20_locked_protocol','legacy_qualified_result','new_eval_body_source',
    'capture_body_source','coordinates_source','checkpoints','shard_id','dataset_ordinal_range')
N20_REPEAT202_DEV_SOURCE_BYTES = 200187
N20_REPEAT202_DEV_SOURCE_SHA = 'a2aabdb78be8c91ffd93dcf902fc4f7941cfdb27bdc56a194ce448ef6156cf22'


def n20_cpu_policy_names():
    return ('native_official','strong_fixed_E24_Omega') + tuple(
        'E%d_%s_%s' % (epoch,arm,mode) for epoch in (24,8)
        for arm in ('LOCAL_ERROR','CENTER_ERROR') for mode in ('atmost','exact'))


def n20_validate_cpu_header(spec):
    owner = spec['qualification_owner_index']; cid = spec['cpu_shard_id']
    require(spec['schema'] == N20_DEV_CPU_SCHEMA and type(owner) is int and owner >= 187
        and type(cid) is int and 0 <= cid < 32
        and spec['normal_RAW_count'] == spec['expected_normal_RAW_count'] == spec['cpu_shard_owner_index'] == owner
        and type(spec['CPU_parent_GPU_id']) is int and spec['CPU_parent_GPU_id'] == N20_CPU_PARENT_GPU_IDS[cid]
        and spec['cpu_dataset_ordinal_range'] == list(N20_CPU_RANGES[cid])
        and spec['total_failed_RAW_count'] == 26 and spec['total_terminal_identity_pin_count'] == owner + 26
        and spec['initialization_seed'] == 202610150202 and spec['deadline_seconds'] == 1800
        and spec['automatic_retry'] is False and spec['failed26_partials_reused'] is False
        and spec['maximum_process_RSS_bytes'] == 16*1024**3
        and spec['allocator_limit_MiB'] == 4096 and spec['whole_GPU_limit_MiB'] == 8192
        and Path(__file__).resolve(strict=True) == Path(spec['entry_source']['path']).resolve(strict=True),
        'Independent repeat202 CPU own literal owner/CPU32 scope/source/resources; no inferred145+cid')
    records = spec['completed_GPU_shards']
    require(type(records) is list and len(records) == 4, 'All four new original completed DEV GPU parents precede CPU')
    owners = []
    for sid, item in enumerate(records):
        require(type(item) is dict and set(item) == N20_COMPLETED_GPU_KEYS
            and type(item['shard_id']) is int and item['shard_id'] == sid
            and item['dataset_ordinal_range'] == list(N20_DEV_RANGES[sid])
            and type(item['shard_owner_index']) is int and 183 <= item['shard_owner_index'] < owner
            and item['source']['bytes'] == N20_REPEAT202_DEV_SOURCE_BYTES
            and item['source']['sha256'] == N20_REPEAT202_DEV_SOURCE_SHA,
            'Exact original newDEV descriptor/scope/source and explicit earlier owner')
        owners.append(item['shard_owner_index'])
        for key in ('source','spec','report','RAW','admission','phase_binding','bootstrap'):
            ref_shape(item[key])
    require(owners == sorted(set(owners)) and spec['GPU_shard'] == records[spec['CPU_parent_GPU_id']]
        and spec['GPU_outer_source'] == spec['GPU_shard']['source'],
        'Four distinct ordered actual GPU owners, mapped complete parent source; no sid-based ownership guess')
    return True


def n20_repeat202_CPU_partition(spec, gate):
    plan = read(closed(gate['closure'], spec['N20_repeat202_CPU_execution_partition_lock']))
    require(type(plan) is dict and n20_repeat202_exact(plan, spec['expected_N20_repeat202_CPU_execution_partition_lock'])
        and plan['schema'] == 'ROOT-N20-repeat202-fixedDEV-CPU32-execution-partition-lock-v1'
        and plan['status'] == 'PRELOCKED_BEFORE_REPEAT202_CPU_OR_AP'
        and type(plan['initialization_seed']) is int and plan['initialization_seed'] == 202610150202
        and n20_repeat202_exact(plan['full_dataset_range'], [0,187])
        and n20_repeat202_exact(plan['CPU_dataset_ranges'], [list(r) for r in N20_CPU_RANGES])
        and type(plan['GPU_owner_indices']) is list and len(plan['GPU_owner_indices']) == 4
        and all(type(v) is int for v in plan['GPU_owner_indices'])
        and plan['GPU_owner_indices'] == [r['shard_owner_index'] for r in spec['completed_GPU_shards']]
        and type(plan['CPU_owner_indices']) is list and len(plan['CPU_owner_indices']) == 32
        and all(type(v) is int and v >= 187 for v in plan['CPU_owner_indices'])
        and plan['CPU_owner_indices'] == sorted(set(plan['CPU_owner_indices']))
        and max(plan['GPU_owner_indices']) < plan['CPU_owner_indices'][0]
        and type(plan['normal_RAW_count_before_first_CPU']) is int
        and plan['normal_RAW_count_before_first_CPU'] == plan['CPU_owner_indices'][0]
        and plan['CPU_owner_indices'][spec['cpu_shard_id']] == spec['qualification_owner_index']
        and type(plan['total_failed_RAW_count']) is int and plan['total_failed_RAW_count'] == 26
        and plan['budget_or_fusion_allowed'] is False
        and plan['AP_evaluated'] is False,
        'New prelocked repeat202 CPU32 ranges/explicit owners, not old first201 execution supplement')
    gate['N20_repeat202_CPU_execution_partition_lock'] = plan
    return plan


def n20_completed_DEV_GPU_record(spec, gate, item):
    c = gate['closure']
    for key in ('source','spec','report','RAW','admission','phase_binding','bootstrap'):
        closed(c, item[key])
    parent = read(closed(c, item['spec'])); report = read(closed(c, item['report']))
    owner = item['shard_owner_index']; sid = item['shard_id']; row = spec['RAW_inputs'][owner]
    joined = gate['lineage'].owner_ref_join(gate['context'], gate['owners'],
        item['RAW'], item['admission'], item['source'], item['spec'], '0')
    require(type(joined) is int and joined == owner == parent['DEV_owner_index']
        and parent['schema'] == N20_REPEAT202_DEV_SCHEMA and parent['entry_source'] == item['source']
        and parent['normal_RAW_count'] == parent['expected_normal_RAW_count'] == owner
        and parent['total_failed_RAW_count'] == 26 and parent['total_terminal_identity_pin_count'] == owner + 26
        and parent['RAW_inputs'] == spec['RAW_inputs'][:owner]
        and parent['shard_id'] == sid and parent['dataset_ordinal_range'] == list(N20_DEV_RANGES[sid])
        and all(row[k] == item[v] for k,v in (('receipt','RAW'),('admission','admission'),
            ('phase_binding','phase_binding'),('bootstrap','bootstrap'))) and row['phase'] == parent['phase']
        and all(c.get(ref['path']) == ref for ref in parent['sources'])
        and all(spec['terminal_identity_pins'].get(k) == v for k,v in parent['terminal_identity_pins'].items())
        and Path(parent['GPU_admission_file']).resolve(strict=True) == Path(item['admission']['path']).resolve(strict=True)
        and Path(item['report']['path']).resolve(strict=True) == (Path(parent['output_directory'])/'REPORT.json').resolve(strict=True),
        'Original completed actual0 GPU source/spec/ad/phase/bootstrap/prefix; not pending output self-certification')
    shared = N20_REPEAT202_CPU_SHARED_KEYS + ('N20_repeat202_ROOT_training_qualification',
        'expected_N20_repeat202_ROOT_training_qualification','legacy_qualified_result','legacy_shards',
        'n17_root_qualified_result_ref','expected_n17_root_qualified_result','new_eval_body_source',
        'legacy_eval_body_source','capture_body_source','coordinates_source','labels_source')
    require(all(n20_repeat202_exact(parent[k], spec[k]) for k in shared)
        and all(n20_repeat202_exact(v, spec[k]) for k,v in parent.items() if k.startswith('additional_preserved_')),
        'All original repeat202 models/qualified training/cache/protocol/old failed history exact')
    binding = dict(source=item['source'], spec=item['spec'], GPU_admission=item['admission'], phase=parent['phase'],
        N20_repeat202_ROOT_training_qualification=parent['N20_repeat202_ROOT_training_qualification'],
        initialization_seed=202610150202, N20_locked_protocol=parent['N20_locked_protocol'],
        legacy_qualified_result=parent['legacy_qualified_result'], new_eval_body_source=parent['new_eval_body_source'],
        capture_body_source=parent['capture_body_source'], coordinates_source=parent['coordinates_source'],
        checkpoints=gate['trained_checkpoints'], shard_id=sid, dataset_ordinal_range=list(N20_DEV_RANGES[sid]))
    names = n20_cpu_policy_names(); first, last = N20_DEV_RANGES[sid]
    require(set(binding) == set(N20_GPU_BINDING_KEYS)
        and report['schema'] == 'N20-fixedDEV-feature-error-GPU-shard-report-v1'
        and report['status'] == 'N20_FIXEDDEV_FEATURE_ERROR_SHARD_COMPLETE_PENDING_ROOT_ACTUAL0_AND_CPU'
        and all(report[k] == v for k,v in binding.items())
        and report['normal_RAW_count'] == owner and report['total_failed_RAW_count'] == 26
        and report['total_terminal_identity_pin_count'] == owner + 26
        and report['qualified'] is False and report['ROOT_actual_GPU0_verified'] is False
        and report['original_training_GPU_actual0_verified'] is True
        and report['training_saved_artifact_recovery'] is False and report['independent_endpoint_CPU_qualified'] is True
        and report['CPU_numerical_qualification_pending'] is True and report['models_frozen'] is True
        and report['teacher_forward_calls'] == report['new_fullC_PR_forward_calls'] == 0
        and report['fresh_selected_PR'] is True and report['reused_common_C_reference'] is True
        and report['AP_evaluated'] is False and report['local_AP_computed'] is False
        and report['full187_qualification_claimed'] is False and report['standalone_FPS_measured'] is False
        and report['cleanup'] == {'allocated_after':0,'reserved_after':0}
        and report['controls'] == list(names) and set(report['control_results']) == set(names),
        'Immutable newDEV pending report/normal-training provenance, no AP or current CPU qualification')
    manifest = read(closed(c, report['capture_manifest'])); start = read(closed(c, report['START']))
    output = Path(parent['output_directory'])
    require(Path(report['capture_manifest']['path']).resolve(strict=True) == (output/'CAPTURE_MANIFEST.json').resolve(strict=True)
        and Path(report['START']['path']).resolve(strict=True) == (output/'START.json').resolve(strict=True)
        and manifest['schema'] == 'N20-fixedDEV-feature-neighbors-GPU-shard-capture-manifest-v1'
        and start['schema'] == 'N20-fixedDEV-feature-GPU-shard-start-v1'
        and all(manifest[k] == start[k] == v for k,v in binding.items())
        and start['original_training_GPU_actual0_verified'] is True and start['training_saved_artifact_recovery'] is False
        and start['independent_endpoint_CPU_qualified'] is True and start['policies'] == list(names)
        and manifest['images'] == report['images'] == last-first
        and manifest['queries'] == report['queries'] == (last-first)*100
        and manifest['global_images'] == 187 and manifest['global_queries'] == 18700
        and manifest['internal_member_locators_are_strings'] is True
        and report['scope_dataset_indices'] == list(range(first,last))
        and len(manifest['rows']) == last-first
        and manifest['physical_capture_refs'] == report['capture_refs'] == [r['bundle'] for r in manifest['rows']],
        'Whole original GPU manifest/START/all100/capture refs unchanged before child slicing')
    for i,captured in enumerate(manifest['rows'], start=first):
        origin = gate['legacy_rows'][i]; old = origin['row']; common = gate['prior_common_rows'][i]
        carrier = dict(bundle=old['bundle'], shared_member=old['shared_actual_image_inputs'], original_GPU_binding=origin['binding'],
            common_PR_bundle=common['row']['bundle'], common_PR_original_GPU_binding=common['binding'])
        require(captured['dataset_index'] == i and captured['image_id'] == old['image_id']
            and captured['identity'] == old['identity'] and captured['qids'] == old['qids']
            and captured['input_carrier'] == carrier and captured['natural_queries'] == 100
            and captured['neighbor_input_member'] == 'i%03d/FEATURE_NEIGHBORS_INPUTS.pth' % i
            and type(captured['member_locators']) is list and all(type(v) is str for v in captured['member_locators'])
            and len(captured['member_locators']) == len(set(captured['member_locators']))
            and type(captured['query_profiles']) is list and len(captured['query_profiles']) == 100
            and [v['query_id'] for v in captured['query_profiles']] == list(range(100)),
            'Original F9 input/native/prior-common provenance/natural100; no conversion of exported qid order')
        closed(c, captured['bundle']); closed(c, carrier['bundle']); closed(c, carrier['common_PR_bundle'])
    image_ids = [r['image_id'] for r in manifest['rows']]
    require(report['scope_image_ids'] == image_ids and len(set(image_ids)) == last-first, 'Original image namespace/order')
    expected = {}; grouped = {}
    for name in names:
        value = report['control_results'][name]
        require(type(value) is dict and set(value) == {'predictions'}, 'Strictref predictions, never inline/local COCO')
        values = read(closed(c, value['predictions']))
        require(type(values) is list and len(values) == (last-first)*100, 'Full GPU scope prediction population')
        groups = {}
        for prediction in values:
            groups.setdefault(prediction['image_id'], []).append(prediction)
        require(list(groups) == image_ids and all([p['query_id'] for p in groups[r['image_id']]] == r['qids']
            for r in manifest['rows']), 'Original all100 exported query permutation, not forced natural sorting')
        for captured in manifest['rows']:
            baseline = gate['populations']['native_official'][captured['image_id']]; observed = groups[captured['image_id']]
            require(all({k:v for k,v in p.items() if k != 'segmentation'} ==
                {k:v for k,v in q.items() if k != 'segmentation'} for p,q in zip(observed,baseline)),
                'Original class/S0/query identity/all nonsegmentation fields preserved')
            if name in ('native_official','strong_fixed_E24_Omega'):
                require(observed == gate['populations'][name][captured['image_id']], 'Original native/R references exact')
        expected[name] = values; grouped[name] = groups
    return dict(descriptor=item, spec=parent, report=report, manifest=manifest, START=start,
        binding=binding, expected_predictions=expected, expected_prediction_groups=grouped)


def n20_scope_parent_GPU(spec, gate):
    cid = spec['cpu_shard_id']; pid = spec['CPU_parent_GPU_id']
    first, last = N20_CPU_RANGES[cid]; pa, pz = N20_DEV_RANGES[pid]
    rows = gate['parent_GPU_manifest']['rows']
    require(pa <= first < last <= pz and spec['cpu_dataset_ordinal_range'] == [first,last]
        and [r['dataset_index'] for r in rows] == list(range(pa,pz)), 'Immutable whole parent before child slicing')
    selected = rows[first-pa:last-pa]; ids = [r['image_id'] for r in selected]
    groups = {name:{image_id:parent[image_id] for image_id in ids}
        for name,parent in gate['expected_prediction_groups'].items()}
    expected = {name:[p for r in selected for p in parent[r['image_id']]] for name,parent in groups.items()}
    require(len(selected) == last-first and len(set(ids)) == last-first
        and all(len(v) == (last-first)*100 for v in expected.values()), 'Complete CPU child population; no failed partial')
    gate.update(scope_rows=selected, scope_capture_refs=[r['bundle'] for r in selected], scope_image_ids=ids,
        scope_expected_predictions=expected, scope_expected_prediction_groups=groups)
    return gate


def source_gate(path, spec):
    n20_validate_cpu_header(spec)
    raw22 = n20_failed22_raw_only(spec); raw23 = n20_failed23_raw_only(spec)
    raw24 = n20_failed24_raw_only(spec); raw25 = n20_failed25_raw_only(spec); raw26 = n20_failed26_raw_only(spec)
    gate = inherited_source_gate(path, spec, cpu=True)
    require(gate['source'] == spec['entry_source']
        and gate['context']['normal_RAW_count'] == spec['qualification_owner_index']
        and gate['context']['RAW_inputs'] == spec['RAW_inputs'], 'Current CPU own context must never become a parent GPU context')
    first201, history_spec = n20_repeat202_first_permission(spec, gate)
    history = n20_repeat202_history_view(spec, gate, history_spec)
    n20_failed22_closed(history, gate, raw22); n20_failed23_closed(history, gate, raw23)
    n20_failed24_closed(history, gate, raw24); n20_failed25_closed(history, gate, raw25); n20_failed26_closed(history, gate, raw26)
    c = gate['closure']
    for key in N20_CPU_SOURCE_KEYS:
        closed(c, spec[key])
    n20_repeat202_CPU_partition(spec, gate)
    protocol, cache_root, old_report, neighbor_report = n20_repeat202_cache(spec, gate)
    init_root, metadata = n20_repeat202_initialization(spec, gate, history_spec)
    probe_root = n20_repeat202_cleanup_probe(spec, gate)
    common = load(spec['region_common_source'], '_N20_repeat202_DEV_CPU_library_gate')
    require(callable(getattr(common,'imported_files',None)), 'Original independent CPU library ABI')
    gate.update(common=common, N20_protocol=protocol, initialization_metadata=metadata,
        N20_cache_root=cache_root, parent_cache_report=old_report, parent_cache_report_ref=spec['cache_GPU_report'],
        neighbor_report=neighbor_report, neighbor_report_ref=cache_root['GPU_report'], first201_root_permission=first201,
        repeat202_initialization_qualification=init_root, repeat202_cleanup_probe_qualification=probe_root,
        training_binding_refs=dict(N20_root_cache_qualified_result=spec['N20_root_cache_qualified_result'],
            N20_initialization_lock=spec['N20_initialization_lock'], N20_cache_CPU_Q=cache_root['CPU_Q'],
            N20_first201_ROOT_qualified_result=spec['N20_first201_ROOT_qualified_result'],
            N20_repeat202_initialization_qualified_result=spec['N20_repeat202_initialization_qualified_result'],
            N20_repeat202_cleanup_probe_qualified_result=spec['N20_repeat202_cleanup_probe_qualified_result']))
    for item in spec['completed_GPU_shards']:
        for key in ('source','spec','report','RAW','admission','phase_binding','bootstrap'):
            closed(c, item[key])
    original = read(closed(c, spec['GPU_shard']['spec']))
    n20_repeat202_DEV_training_metadata_gate(original, gate)
    require(legacy_gate(spec,gate) is gate, 'N16 four logical carriers unchanged')
    # Explicit original parent: its CPU_adapter_source is the old commonPR decoder;
    # own CPU_adapter_source is the new F9 decoder. Never swap aliases into parent.
    require(n20_prior_DEV_carrier_gate(original,gate) is gate, 'Original N17 five physical carrier roles, no parent own_gate')
    parents = [n20_completed_DEV_GPU_record(spec,gate,item) for item in spec['completed_GPU_shards']]
    require([r['dataset_index'] for p in parents for r in p['manifest']['rows']] == list(range(187))
        and len({r['image_id'] for p in parents for r in p['manifest']['rows']}) == 187,
        'All four actual completed GPU scopes cover full187 exactly once before any CPU child')
    parent = parents[spec['CPU_parent_GPU_id']]
    gpu = load(spec['GPU_outer_source'], '_N20_repeat202_completed_GPU_CPU_helpers')
    require(Path(gpu.__file__).resolve(strict=True) == Path(spec['GPU_outer_source']['path']).resolve(strict=True)
        and callable(getattr(gpu,'n20_model_load_cpu',None)) and callable(getattr(gpu,'n20_policy_names',None))
        and tuple(gpu.n20_policy_names()) == n20_cpu_policy_names(), 'Closed completed GPU pure CPU model/policy ABI')
    gate.update(gpu_helpers=gpu, completed_GPU_shards=spec['completed_GPU_shards'],
        parent_GPU=parent['spec'], parent_GPU_report=parent['report'], parent_GPU_manifest=parent['manifest'],
        parent_GPU_START=parent['START'], parent_GPU_binding=parent['binding'],
        expected_predictions=parent['expected_predictions'], expected_prediction_groups=parent['expected_prediction_groups'])
    n20_scope_parent_GPU(spec,gate)
    return gate
