"""N23 original full-FIT thin-cache intake. No model, sampling, optimizer or AP."""
import gc
import importlib.util
import json
from pathlib import Path


def need(ok, message):
    if not ok:
        raise AssertionError(message)


def check_population(report, accepted, pre):
    need(accepted['fully_qualified'] is True and accepted['scout_training_eligible'] is True,
         'Cache must have prior external Root acceptance')
    for key, expected in [('images',741),('raw_GT',1661),('natural_queries',74100),
                          ('sampled_points',3035136)]:
        need(report[key] == accepted[key] == expected, 'Original population total: ' + key)
    rows, refs = report['rows'], report['packet_refs']
    need(len(rows) == len(refs) == 741 and pre['dataset_indices'] == list(range(741)),
         'Complete original image and packet scope')
    for index, (row, ref) in enumerate(zip(rows, refs)):
        need(row['dataset_index'] == index and row['packet'] == ref
             and row['image_id'] == row['identity']['image_id'] == pre['fit_image_ids'][index]
             and row['identity']['ann_ids'] == pre['fit_ann_ids_by_image'][index]
             and row['raw_GT'] == len(row['identity']['ann_ids'])
             and row['points'] == 4096 and row['natural_queries'] == 100,
             'Original FIT row/identity/draw order drift at ' + str(index))
    need(sum(r['raw_GT'] for r in rows) == 1661, 'Real GT sum, not claimed metadata')


def load_module(control, ref, name):
    path = control.closed(ref)
    ms = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(ms)
    ms.loader.exec_module(result)
    return result


def run(torch, np, spec, control):
    read = lambda ref: json.loads(control.closed(ref).read_bytes())
    pre = read(spec['prelock'])
    parent = read(spec['cache_ROOT_qualified'])
    report = read(spec['cache_report'])
    need(parent['GPU_report'] == spec['cache_report'], 'Accepted report binding')
    check_population(report, parent, pre)
    binding = pre['expected_cache_binding']
    need(set(binding) == {'source','spec','GPU_admission','phase','teacher_checkpoint',
                         'fixed_PR_checkpoint','cache_body_source','C_engine_source'},
         'Exact original cache binding roles')
    for key, value in binding.items():
        if isinstance(value, dict):
            control.closed(value)
        if key not in ('cache_body_source','C_engine_source'):
            need(report[key] == value, 'Original cache parent metadata: ' + key)
    original_spec = read(binding['spec'])
    need(original_spec['cache_body_source'] == binding['cache_body_source']
         and original_spec['C_engine_source'] == binding['C_engine_source'],
         'Actual writer and C-domain source identity')
    adapter = load_module(control, spec['adapter_source'], 'N23_REAL_SHARED_POINT_ADAPTER')
    analysis = load_module(control, spec['coverage_source'], 'N23_REAL_COVERAGE_ANALYSIS')
    need(spec['captured_packet_refs'] == report['packet_refs'], 'Unchanged light packet references')
    output = Path(spec['output_directory'])
    output.mkdir()
    records = []
    for index, (row, ref) in enumerate(zip(report['rows'], report['packet_refs'])):
        control.check()
        packet = torch.load(str(control.closed(ref)), map_location='cpu')
        need(packet.get('natural_native_L') is None,
             'Only thin sampled-F packet; full native L is outside intake scope')
        shared = adapter.build_shared_batches(np, packet, row, binding)
        need(shared['synthetic_only'] is False and shared['training_executed'] is False,
             'Real reader route without training')
        record = analysis.analyze_image(index, row['image_id'], shared)
        for instance in record['GT_records']:
            K = int(row['K_counts_all100'][instance['query_id']])
            need(instance['K'] is None or instance['K'] == K, 'Original unsampled K metadata')
            instance['K'] = K
        records.append(record)
        del shared, packet
        gc.collect()
        control.check()
        if (index+1) % 64 == 0 or index == 740:
            print(json.dumps(dict(stage='N23_REAL_CACHE_COVERAGE_INTAKE', images_done=index+1,
                                  expected_images=741, training_executed=False)), flush=True)
    coverage = analysis.aggregate_images(records, expected_images=741, expected_real_GT=1661)
    coverage_ref = control.write_json(output / 'COVERAGE.json', coverage)
    value = dict(schema='N23-real-FIT-sampled-coverage-report-v1',
        status='COMPLETE_PENDING_ORIGINAL_ACTUAL0_AND_ROOT_ACCEPTANCE',
        images_completed=len(records), raw_GT=1661, original_draws=741*4096,
        adapter_source=spec['adapter_source'], coverage_source=spec['coverage_source'],
        cache_report=spec['cache_report'], cache_ROOT_qualified=spec['cache_ROOT_qualified'],
        original_cache_binding=binding, coverage_ref=coverage_ref, summary=coverage['summary'],
        full_native_L_loaded=False, original_images_loaded=False, model_constructed=False,
        weight_tensors_loaded=False, original_weight_file_identity_hashed=True,
        sampling_changed=False, new_training=False, DEV=False, TEST=False, AP=False,
        loss_normalization_selected=False, zero_effective_point_step_rule_selected=False,
        qualified=False, original_actual0_self_certified=False, innovation_established=False,
        scope='Accepted original cache samples only; coverage does not prove fair training or model efficacy')
    report_ref = control.write_json(output / 'CPU_REPORT.json', value)
    need(not torch.cuda.is_initialized(), 'CPU-only intake never initializes CUDA')
    return dict(report_ref=report_ref, coverage_ref=coverage_ref, summary=coverage['summary'],
                images_completed=len(records), training_executed=False, AP=False)
