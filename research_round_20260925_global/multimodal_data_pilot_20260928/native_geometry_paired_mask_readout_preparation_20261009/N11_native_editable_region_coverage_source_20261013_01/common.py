"""N11 post-RAW-gate provenance/identity helpers; no numerical imports.

ROOT must bind this independent SOURCE with a fresh SPEC. Entry scripts inspect
all original held/reaped RAW receipts before hashing or importing this module.
These helpers are shared provenance checks, not the independent pixel formulas.
"""
import ast
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import types

GPU_SCHEMA = 'ROOT-N11-native-editable-coverage-GPU-spec-v1'
CPU_SCHEMA = 'ROOT-N11-native-editable-coverage-CPU-spec-v1'
REPORT_SCHEMA = 'N11-native-editable-coverage-GPU-report-v1'
REPORT_STATUS = 'N11_NATIVE_EDITABLE_COVERAGE_GPU_COMPLETE_PENDING_CPU'
Q_SCHEMA = 'ROOT-N11-native-editable-coverage-CPU-qualification-v1'
Q_STATUS = 'ROOT_N11_NATIVE_EDITABLE_COVERAGE_CPU_QUALIFIED'
ANCHOR_SCHEMA = 'ROOT-N11-qualified-native-fixed-pair-anchors-v1'
DERIVATION_SCHEMA = 'ROOT-N11-qualified-native-fixed-pair-anchor-derivation-v1'
DERIVATION_STATUS = 'ROOT_N11_NATIVE_ANCHORS_EXACT_FROM_QUALIFIED_DETAILS'
N10_Q_SCHEMA = 'ROOT-N10-C-PR-FL91-fixed8-evaluation-CPU-qualification-v1'
N10_Q_STATUS = 'ROOT_N10_C_PR_FL91_FIXED8_EVALUATION_CPU_QUALIFIED'
N07_REPORT_SCHEMA = 'N07-original-detail-fixed-endpoint-evaluation-report-v1'
N07_Q_SCHEMA = 'ROOT-N07-original-detail-fixed-endpoint-evaluation-CPU-qualification-v1'
N07_Q_STATUS = 'ROOT_N07_ORIGINAL_DETAIL_FIXED_ENDPOINT_EVALUATION_CPU_QUALIFIED'
PACKET_SCHEMA = 'N07-original-detail-all100-native-fields-v1'
OWNER_SHA = 'ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
RAW_KEYS = ('receipt', 'phase', 'admission', 'phase_binding', 'bootstrap')
SLOTS = ((0, '0.50', 374, 6), (5, '0.75', 348, 32), (8, '0.90', 107, 273))
PAIR_KEYS = {'gt_id', 'dt_id', 'query_id', 'IoU', 'Dice', 'pixel_TP',
             'pixel_FP', 'pixel_FN', 'empty_union_IoU_convention'}
NUMERIC = dict(class_rtol=1e-6, class_atol=1e-6,
               quality_rtol=1e-4, quality_atol=1e-4,
               CPU_sign_and_band_rtol=1e-4, CPU_sign_and_band_atol=1e-4,
               native_CPU_render_batch=100, official_quality_eps=1e-6,
               COCO_stats_atol=1e-12, certified_GPU_error_bound=False)
RECIPE = dict(images=187, raw_GT=380, queries_per_image=100,
    predictions=18700, native_weights_epoch=24,
    native_band_closed_abs_logit=2.0, render_dtype='float32',
    native_baseline_render_queries=100, native_baseline_quality_queries=100,
    native_baseline_render_order='original natural all100, then original query reorder',
    fixed_native_assignment_slots=[0, 5, 8],
    pair_selection='only previously qualified native official area-all maxDet100 pairs',
    unmatched_or_ignored_query_selection=False,
    oracle_binary_inside_band='GT support; original native binary outside unchanged',
    oracle_is_conditional_fixed_query_diagnostic=True,
    oracle_AP_computed=False, GT_directed_inference=False,
    native_model_loaded=False, native_forward_calls=0,
    original_JPEG_decoded=False, F_e_projection_replayed=False,
    heads_loaded=False, optimizer_created=False, weights_updated=False,
    no_TEST=True, no_NMS=True, no_score_cutoff=True,
    all100_queries_retained=True, native_export_unchanged=True,
    allocator_limit_MiB=4096, whole_GPU_limit_MiB=8192,
    CPU_affinity=[0, 1], threads=2, deadline_seconds=1800)
REF_KEYS = ('entry_source', 'common_source', 'prior_N07_report',
    'prior_N07_CPU_qualification', 'qualified_native_Q',
    'qualified_native_Q_source', 'qualified_native_Q_spec',
    'qualified_native_Q_admission', 'qualified_native_Q_receipt',
    'pair_anchors', 'anchor_derivation_record', 'DEV', 'native_predictions',
    'reader_source', 'renderer_source', 'export_source')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    def reject(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=reject)


def reference(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def bound(ref):
    require(reference(ref['path']) == ref, 'Pinned bytes/SHA changed: ' + ref['path'])
    return Path(ref['path'])


def load(ref, name):
    path = bound(ref)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def deadline(started):
    require(time.monotonic() - started < 1800, 'Prelocked N11 1800s deadline exceeded')


def owner_join(row, raw, closure):
    owner = read(bound(row['admission']))
    pb = read(bound(row['phase_binding']))
    bootstrap = bound(row['bootstrap'])
    runtime = Path(row['receipt']['path']).parent
    require(Path(row['phase_binding']['path']) == runtime / 'PHASE_BINDING.json' and
        bootstrap == runtime / 'bootstrap.py' and
        owner['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and
        owner['phase'] == row['phase'] and owner['driver_sha256'] == OWNER_SHA and
        owner['automatic_retry'] is False and owner['allowed_cpus'] == [0, 1] and
        owner['cuda_visible_devices'] in ('', '0') and
        owner['whole_GPU_used_limit_MiB'] == 8192 and
        raw['admission_sha256'] == row['admission']['sha256'] == pb['admission_sha256'] and
        pb['bootstrap_sha256'] == row['bootstrap']['sha256'] and
        pb['argv'][-1] == str(bootstrap) and pb['CPU_affinity'] == [0, 1] and
        pb['CUDA_visible_devices'] == owner['cuda_visible_devices'] and
        pb['automatic_retry'] is False, 'Original RAW/owner/PB/bootstrap join differs')
    entry, args = owner['entry'], owner['entry_args']
    require(entry in closure and
        owner['bound_source_sha256'].get(entry) == closure[entry]['sha256'],
        'Ancestor entry must remain source-bound')
    tree = ast.parse(bootstrap.read_text(encoding='utf-8'))
    argv = [ast.literal_eval(n.value) for n in ast.walk(tree)
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Attribute) and
        isinstance(t.value, ast.Name) and t.value.id == 'sys' and t.attr == 'argv'
        for t in n.targets)]
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and
        isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and
        n.func.value.id == 'runpy' and n.func.attr == 'run_path']
    require(argv == [[entry] + args] and len(calls) == 1 and
        ast.literal_eval(calls[0].args[0]) == entry and
        {k.arg: ast.literal_eval(k.value) for k in calls[0].keywords} == {'run_name': '__main__'},
        'Original bootstrap SOURCE differs')
    if '--spec' in args:
        require(args.count('--spec') == 1, 'Ambiguous ancestor SPEC')
        locator = args[args.index('--spec') + 1]
        require(locator in closure and
            owner['bound_source_sha256'].get(locator) == closure[locator]['sha256'],
            'Actual ancestor SPEC must remain source-bound')
    if 'scientific_spec' in owner:
        ref = owner['scientific_spec']
        require(ref in closure.values() and
            owner['bound_source_sha256'].get(ref['path']) == ref['sha256'],
            'Actual ancestor scientific SPEC differs')
    return owner


def joined(source, spec_ref, admission, phase, cuda, rows, owners):
    matches = [i for i, owner in enumerate(owners)
        if owner['entry'] == source['path'] and
        owner['entry_args'] == ['--spec', spec_ref['path'], '--owner-execute']]
    require(len(matches) == 1, 'Unique actual successful parent execution required')
    i = matches[0]
    require(rows[i]['phase'] == phase and rows[i]['admission'] == admission and
        owners[i]['scientific_spec'] == spec_ref and
        owners[i]['cuda_visible_devices'] == cuda, 'Actual parent identity/admission differs')
    return i


def source_gate(path, spec, entry_file, raws, closure, cpu=False):
    count, schema, cuda = (34, CPU_SCHEMA, '') if cpu else (33, GPU_SCHEMA, '0')
    extra = ('GPU_entry_source', 'GPU_spec', 'GPU_report', 'GPU_RAW', 'GPU_admission', 'Q_path') if cpu else ('output_directory',)
    require(spec['schema'] == schema and all(k in spec for k in REF_KEYS + extra) and
        len(spec['RAW_inputs']) == count and spec['recipe'] == RECIPE and
        spec['numeric_protocol'] == NUMERIC and spec['deadline_seconds'] == 1800 and
        spec['allocator_limit_MiB'] == 4096 and spec['whole_GPU_limit_MiB'] == 8192,
        'Complete locked ROOT N11 SPEC required')
    refs = [spec[k] for k in REF_KEYS] + [row[k] for row in spec['RAW_inputs'] for k in RAW_KEYS if k != 'phase']
    if cpu:
        refs += [spec[k] for k in extra if k != 'Q_path']
    require(all(ref in closure.values() for ref in refs), 'Actual primary inputs absent from closure')
    source, spec_ref = reference(entry_file), reference(path)
    require(source == spec['entry_source'] and reference(__file__) == spec['common_source'],
        'Actual N11 entry/common SOURCE differs')
    owners = [owner_join(row, raw, closure) for row, raw in zip(spec['RAW_inputs'], raws)]
    admission_path = Path(spec['CPU_admission_file' if cpu else 'GPU_admission_file'])
    require(admission_path.is_absolute(), 'Absolute own admission required')
    admission_ref, admission = reference(admission_path), read(admission_path)
    require(admission['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and
        admission['driver_sha256'] == OWNER_SHA and admission['phase'] == spec['phase'] and
        admission['scientific_spec'] == spec_ref and admission['entry'] == source['path'] and
        admission['entry_args'] == ['--spec', str(path), '--owner-execute'] and
        admission['allowed_cpus'] == [0, 1] and admission['cuda_visible_devices'] == cuda and
        admission['automatic_retry'] is False and admission['whole_GPU_used_limit_MiB'] == 8192 and
        admission['qualified_leaf_deadline_seconds'] == 1800 and
        admission['bound_source_sha256'].get(str(path)) == spec_ref['sha256'] and
        all(admission['bound_source_sha256'].get(ref['path']) == ref['sha256']
            for ref in spec['sources']) and os.environ.get('CUDA_VISIBLE_DEVICES') == cuda and
        all(os.environ.get(k) == '2' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')), 'Actual owner/resource closure differs')
    prior = read(bound(spec['prior_N07_report']))
    prior_q = read(bound(spec['prior_N07_CPU_qualification']))
    q = read(bound(spec['qualified_native_Q']))
    require(prior['schema'] == N07_REPORT_SCHEMA and
        prior_q['schema'] == N07_Q_SCHEMA and prior_q['status'] == N07_Q_STATUS and
        prior_q['evaluation_report'] == spec['prior_N07_report'] and
        prior_q['source'] == prior['source'] and prior_q['evaluation_spec'] == prior['spec'] and
        prior_q['evaluation_GPU_admission'] == prior['GPU_admission'] and
        prior_q['reader_source'] == spec['reader_source'] and
        prior['DEV'] == prior_q['DEV'] == spec['DEV'] and
        prior['images'] == prior_q['images'] == 187 and
        prior['raw_GT'] == prior_q['raw_GT'] == 380 and
        prior['predictions_per_control'] == prior_q['predictions_per_control'] == 18700 and
        prior['original187_all100_low_fields_saved'] is True and
        prior['GPU_original_logits_saved'] is False and prior['cleanup']['allocated_after'] == 0 and
        prior['no_TEST'] is True and prior_q['native_M200_original_render_CPU_replayed'] is True,
        'Qualified original N07 packets/native baseline required')
    prior_i = joined(prior['source'], prior['spec'], prior['GPU_admission'], prior['phase'], '0', spec['RAW_inputs'], owners)
    prior_cpu_i = joined(prior_q['reader_source'], prior_q['spec'], prior_q['CPU_admission'], prior_q['phase'], '', spec['RAW_inputs'], owners)
    require(prior_q['evaluation_GPU_RAW'] == spec['RAW_inputs'][prior_i]['receipt'] and
        prior_i < prior_cpu_i < 32, 'Original N07 held-terminal chronology differs')
    require(q['schema'] == N10_Q_SCHEMA and q['status'] == N10_Q_STATUS and
        q['reader_source'] == spec['qualified_native_Q_source'] and
        q['spec'] == spec['qualified_native_Q_spec'] and q['CPU_admission'] == spec['qualified_native_Q_admission'] and
        q['RAW_inputs'] == spec['RAW_inputs'][:32] and
        q['prior_failed_CPU_receipt'] == spec['prior_failed_CPU_receipt'] and
        q['additional_failed_RAW_inputs'] == spec['additional_failed_RAW_inputs'] and
        q['prior_N07_report'] == spec['prior_N07_report'] and
        q['N07_cpu_qualification'] == spec['prior_N07_CPU_qualification'] and
        q['native_reference_predictions'] == spec['native_predictions'] and q['DEV'] == spec['DEV'] and
        q['reader_helpers_source'] == spec['reader_source'] and q['renderer_source'] == spec['renderer_source'] and
        q['native_full18700_exact_GPU_export_prerequisite'] is True and
        q['actual_official_one_to_one_matching_and_matched_pixels_CPU_recomputed'] is True and
        q['CPU_approximate_masks_substituted_for_exported_metrics'] is False and q['no_TEST'] is True,
        'Actually qualified N10 native matching anchor required')
    q_i = joined(q['reader_source'], q['spec'], q['CPU_admission'], q['phase'], '', spec['RAW_inputs'], owners)
    require(q_i == 32 and spec['RAW_inputs'][32]['receipt'] == spec['qualified_native_Q_receipt'],
        'N10 CPU must join actual held-terminal0 normal ancestor 33')
    packets = prior['final_field_packets']
    require(packets == prior_q['packet_refs'] == q['packet_refs'] and len(packets) == 187 and
        len({ref['path'] for ref in packets}) == 187 and
        all(ref in closure.values() for ref in packets) and
        all(prior[k] in closure.values() for k in ('image_rows', 'trace', 'native_query_score_factors')) and
        prior['predictions']['native_official'] == spec['native_predictions'] ==
            prior_q['control_results']['native_official']['predictions'], 'Original packet/export closure differs')
    reader_spec = read(bound(prior_q['spec']))
    producer_spec = read(bound(prior['spec']))
    require(reader_spec['renderer_source'] == spec['renderer_source'] and
        producer_spec['official_export_source'] == spec['export_source'], 'Original renderer/exporter binding differs')
    anchors = read(bound(spec['pair_anchors']))
    derivation = read(bound(spec['anchor_derivation_record']))
    require(anchors['schema'] == ANCHOR_SCHEMA and derivation['schema'] == DERIVATION_SCHEMA and
        derivation['status'] == DERIVATION_STATUS and
        anchors['qualification'] == derivation['qualification'] == spec['qualified_native_Q'] and
        anchors['matching_details'] == derivation['matching_details'] == q['matching_details'] and
        anchors['DEV'] == spec['DEV'] and anchors['native_predictions'] == spec['native_predictions'] and
        derivation['anchors'] == spec['pair_anchors'] and derivation['fixed_slot_indices'] == [0, 5, 8] and
        derivation['images'] == 187 and derivation['raw_GT_per_slot'] == 380 and
        derivation['every_original_GT_identity_ignore_FN_and_pair_exact'] is True and
        derivation['new_query_selection'] is False and derivation['no_GT_directed_inference'] is True,
        'ROOT exact compact anchor derivation record required')
    return dict(source=source, spec_ref=spec_ref, admission_ref=admission_ref,
        admission=admission, owners=owners, closure=closure, prior=prior,
        prior_q=prior_q, q=q, packets=packets, anchors=anchors)


def imported_files(psutil, torch, np, COCO, COCOeval, masks, gate):
    modules = dict(psutil=psutil, torch=torch, torch_binary=torch._C,
        numpy=np, numpy_binary=np.core._multiarray_umath,
        COCO=sys.modules[COCO.__module__], COCOeval=sys.modules[COCOeval.__module__],
        masks=masks, mask_binary=masks._mask)
    result = {name: reference(module.__file__) for name, module in modules.items()}
    for name, obj in (('interpolation', torch.nn.functional.interpolate),
                      ('serialization', torch.load)):
        result[name] = reference(inspect.getsourcefile(obj))
    require(all(ref in gate['closure'].values() and
        gate['admission']['bound_source_sha256'].get(ref['path']) == ref['sha256']
        for ref in result.values()), 'Actual numerical library/binary SOURCE is unbound')
    return result


def packet_check(torch, reader, coco, packet, row, index, ref, gate):
    prior = gate['prior']
    identity = row['identity']
    iid = row['image_id']
    require(row['dataset_index'] == index and row['packet'] == ref and
        packet['schema'] == PACKET_SCHEMA and packet['dataset_index'] == index and
        packet['identity'] == identity and packet['source'] == prior['source'] and
        packet['spec'] == prior['spec'] and packet['phase'] == prior['phase'] and
        packet['teacher_checkpoint'] == prior['teacher_checkpoint'] and
        packet['head_checkpoints'] == prior['head_checkpoints'] and packet['head_endpoint_epoch'] == 8 and
        packet['prediction_query_ids'] == row['prediction_query_ids'] and
        packet['field_digests'] == row['final_fields'] and packet['GPU_original_logits_saved'] is False,
        'Original packet/producer identity differs')
    image = coco.imgs[iid]
    h, w = identity['actual_img_shape']
    require(identity['image_id'] == iid and
        identity['original_shape'] == [image['height'], image['width']] and
        type(h) is int and type(w) is int and 0 < h <= 800 and 0 < w <= 800 and
        identity['batch_input_shape'] == [800, 800] and
        identity['ann_ids'] == row['ann_ids'] == [a['id'] for a in coco.imgToAnns[iid]] and
        row['raw_GT'] == len(identity['ann_ids']), 'Canonical image/crop/GT identity differs')
    qids = packet['prediction_query_ids']
    require(len(qids) == 100 and all(type(q) is int for q in qids) and
        sorted(qids) == list(range(100)), 'All100 original query identities required')
    fields = packet['native_fields']
    require(set(fields) == set(packet['field_digests']) == {'F256', 'e0', 'M200', 'final_class_logits'},
        'Original packet field ABI differs')
    # F/e are retained in the bound original packet but never projected or used.
    for name, shape in (('M200', (1, 100, 200, 200)), ('final_class_logits', (1, 100, 2))):
        value = fields[name]
        require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and
            value.dtype == torch.float32 and tuple(value.shape) == shape and
            not value.requires_grad and bool(torch.isfinite(value).all()) and
            reader.digest(value) == packet['field_digests'][name], 'Original M/class tensor changed')
    require(dict(reader.digest(fields['final_class_logits'][0]),
        values=fields['final_class_logits'][0].tolist()) == row['native_class_packet'],
        'Original class values/digest changed')
    require(len(row['band_counts_all100']) == 100 and
        all(type(v) is int and 0 <= v <= image['height'] * image['width']
            for v in row['band_counts_all100']) and row['matching_or_loss_executed'] is False and
        all(row[k] is True for k in ('GT_removed_from_forward', 'no_GT_query_selection',
            'all100_queries_retained', 'no_NMS', 'no_score_cutoff', 'fixed_native_score')),
        'Original field producer band/inference scope differs')
    return fields, identity, qids


def validate_anchors(coco, ids, anchors, q):
    images = anchors['images']
    require(len(images) == 187 and [im['image_id'] for im in images] == ids,
        'Canonical compact anchor image order differs')
    for index, label, expected_tp, expected_fn in SLOTS:
        total = dict(TP=0, FN=0, ignored_GT=0)
        for image in images:
            slots = image['threshold_slots']
            require([s['index'] for s in slots] == [0, 5, 8], 'Fixed threshold slots differ')
            cell = slots[[0, 5, 8].index(index)]
            expected = q['control_results']['native_official']['one_to_one'][index]
            require(cell['label'] == label and cell['threshold'] == expected['threshold'],
                'Actual floating COCO threshold must retain its fixed index')
            rows = cell['GT']
            require([r['gt_id'] for r in rows] == image['original_GT_ids'] and
                set(image['original_GT_ids']) == {a['id'] for a in coco.imgToAnns[image['image_id']]},
                'Every original GT/FN/ignore identity must remain')
            queries, gt_matches = set(), set()
            for row in rows:
                require(set(row) == {'gt_id', 'ignored', 'FN', 'matched_DT', 'TP_pair_key', 'pair'} and
                    type(row['ignored']) is bool and type(row['FN']) is bool and
                    type(row['matched_DT']) is int, 'Complete compact original GT row required')
                pair = row['pair']
                if row['ignored']:
                    require(row['FN'] is False and pair is None and row['TP_pair_key'] is None,
                        'Ignored GT must not acquire a diagnostic query')
                    total['ignored_GT'] += 1
                elif row['FN']:
                    require(row['matched_DT'] == 0 and pair is None and row['TP_pair_key'] is None,
                        'Native FN must not acquire a diagnostic query')
                    total['FN'] += 1
                else:
                    require(type(pair) is dict and set(pair) == PAIR_KEYS and
                        pair['gt_id'] == row['gt_id'] and pair['dt_id'] == row['matched_DT'] > 0 and
                        row['TP_pair_key'] == '%d:%d' % (pair['dt_id'], pair['gt_id']) and
                        type(pair['query_id']) is int and 0 <= pair['query_id'] < 100 and
                        pair['query_id'] not in queries and row['gt_id'] not in gt_matches and
                        pair['empty_union_IoU_convention'] == 0 and
                        all(type(pair[k]) is int and pair[k] >= 0 for k in ('pixel_TP', 'pixel_FP', 'pixel_FN')) and
                        all(type(pair[k]) in (int, float) and math.isfinite(pair[k]) and
                            0 <= pair[k] <= 1 for k in ('IoU', 'Dice')),
                        'Unique original ordinary matched pair required')
                    queries.add(pair['query_id'])
                    gt_matches.add(row['gt_id'])
                    total['TP'] += 1
        require(total == dict(TP=expected_tp, FN=expected_fn, ignored_GT=0) and
            total['TP'] + total['FN'] + total['ignored_GT'] == 380,
            'Locked native threshold-specific population differs')
    return images


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    require(not path.exists() and not temporary.exists(), 'Fresh output file required')
    with temporary.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    os.replace(temporary, path)


def final_closure(spec, path, gate, entry_file, libraries, imported):
    for ref in spec['sources']:
        bound(ref)
    require(reference(path) == gate['spec_ref'] and reference(entry_file) == gate['source'] and
        reference(spec['CPU_admission_file' if spec['schema'] == CPU_SCHEMA else 'GPU_admission_file']) ==
            gate['admission_ref'] and imported() == libraries,
        'Final SPEC/SOURCE/admission/library closure changed')
