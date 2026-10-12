"""N01 CPU qualification of archived gradients and five local M200 VJPs.

No native model, matching, optimizer, training, DEV or AP is executed. The
native30/selected-parameter gradients are observations; only their arithmetic
is independently checked. Native uncertainty coordinates were not archived.
"""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

SCHEMA = 'ROOT-N01-fixed-FIT-gradient-CPU-qualification-spec-v1'
QUAL_SCHEMA = 'ROOT-N01-fixed-FIT-gradient-CPU-qualification-v1'
STATUS = 'ROOT_N01_FIXED_FIT_GRADIENT_CPU_QUALIFIED'
GPU_SCHEMA = 'ROOT-N01-uniform-point-supervision-gradient-probe-spec-v1'
GPU_STATUS = 'N01_FIXED_FIT_LOCAL_GRADIENT_PROBE_GPU_COMPLETE_PENDING_CPU'
OWNER_SHA = 'ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
RAW_KEYS = ('receipt', 'phase', 'admission', 'phase_binding', 'bootstrap')
REF_KEYS = ('GPU_spec', 'GPU_admission', 'GPU_RAW', 'GPU_phase_binding',
            'GPU_bootstrap', 'GPU_report', 'GPU_start', 'FIT', 'renderer_source')
GROUPS = ('native', 'foreground', 'background', 'balanced')
SHAPES = {'panoptic_head.pixel_decoder.mask_feature.weight': (256, 256, 1, 1),
          'panoptic_head.pixel_decoder.mask_feature.bias': (256,)}
for _layer in range(3):
    SHAPES['panoptic_head.mask_embed.%d.weight' % (2 * _layer)] = (256, 256)
    SHAPES['panoptic_head.mask_embed.%d.bias' % (2 * _layer)] = (256,)
NUMERIC = dict(forward_rtol=1e-4, forward_atol=1e-4,
               gradient_rtol=1e-3, gradient_atol=1e-7,
               metric_rtol=1e-9, metric_atol=1e-12)


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


def ref_shape(ref):
    require(type(ref) is dict and set(ref) == {'path', 'bytes', 'sha256'} and
            type(ref['path']) is str and Path(ref['path']).is_absolute() and
            type(ref['bytes']) is int and ref['bytes'] >= 0 and
            type(ref['sha256']) is str and len(ref['sha256']) == 64 and
            set(ref['sha256']) <= set('0123456789abcdef'), 'Malformed absolute reference')


def bound(ref):
    ref_shape(ref)
    require(reference(ref['path']) == ref, 'Pinned input changed: ' + ref['path'])
    return Path(ref['path'])


def terminal(row):
    require(type(row) is dict and set(RAW_KEYS) <= set(row), 'Explicit RAW bindings required')
    path, phase = Path(row['receipt']['path']), row['phase']
    require(type(phase) is str and phase and path.is_absolute() and
            path.name == 'PARENT_RECEIPT.json' and path.parent.name == phase,
            'Absolute RAW path must identify its actual phase')
    raw = read(path)
    require(raw['status'] == 'ACTUAL_OWNED_TERMINAL' and type(raw['returncode']) is int and
            raw['returncode'] == 0 and type(raw['original_identity']) is dict and raw['original_identity'] and
            raw['held_handle_exit'] == dict(wait_state='WAIT_OBJECT_0', actual_exit_code=0) and
            type(raw['held_handle_exit']['actual_exit_code']) is int and
            raw['root_wait_reaped'] is True and raw['numeric_birth_absence']['original_birth_absent'] is True and
            raw.get('error') is None and raw.get('abort') is None and raw['wait_errors'] is False and
            raw['GPU_observation_errors'] == [], 'Actual held/reaped terminal0 required')
    if 'phase' in raw:
        require(type(raw['phase']) is str and raw['phase'] == phase, 'RAW phase differs')
    sha, peak = raw['admission_sha256'], raw['whole_GPU_observed_peak_MiB']
    require(type(sha) is str and len(sha) == 64 and set(sha) <= set('0123456789abcdef') and
            type(peak) in (int, float) and math.isfinite(peak) and 0 <= peak <= 8192 and
            type(raw['whole_GPU_used_limit_MiB']) is int and raw['whole_GPU_used_limit_MiB'] == 8192 and
            raw['whole_leaf_VRAM_certified'] is False and
            raw['GPU_scope'] == 'whole GPU observed, not per-process WDDM allocation', 'RAW owner/resource scope differs')
    return raw


def bootstrap_join(row, raw, closure):
    owner = read(bound(row['admission']))
    pb = read(bound(row['phase_binding']))
    boot = bound(row['bootstrap'])
    runtime = Path(row['receipt']['path']).parent
    require(Path(row['phase_binding']['path']) == runtime / 'PHASE_BINDING.json' and boot == runtime / 'bootstrap.py',
            'Original binding/bootstrap location differs')
    require(owner['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and
            owner['phase'] == row['phase'] and owner['driver_sha256'] == OWNER_SHA and
            owner['automatic_retry'] is False and owner['allowed_cpus'] == [0, 1] and
            owner['cuda_visible_devices'] in ('', '0') and owner['whole_GPU_used_limit_MiB'] == 8192 and
            raw['admission_sha256'] == row['admission']['sha256'] == pb['admission_sha256'] and
            pb['bootstrap_sha256'] == row['bootstrap']['sha256'] and pb['argv'][-1] == str(boot) and
            pb['CPU_affinity'] == owner['allowed_cpus'] and pb['CUDA_visible_devices'] == owner['cuda_visible_devices'] and
            pb['automatic_retry'] is False, 'Actual owner/RAW/phase binding differs')
    entry = owner['entry']
    require(entry in closure and owner['bound_source_sha256'].get(entry) == closure[entry]['sha256'],
            'Original admitted entry SOURCE absent from current closure')
    tree = ast.parse(boot.read_text(encoding='utf-8'))
    argv = [ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n, ast.Assign) and
            any(isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and
                t.value.id == 'sys' and t.attr == 'argv' for t in n.targets)]
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and
             isinstance(n.func.value, ast.Name) and n.func.value.id == 'runpy' and n.func.attr == 'run_path']
    require(argv == [[entry] + owner['entry_args']] and len(calls) == 1 and
            ast.literal_eval(calls[0].args[0]) == entry and
            {k.arg: ast.literal_eval(k.value) for k in calls[0].keywords} == {'run_name': '__main__'},
            'Actual bootstrap entry/arguments differ')
    if '--spec' in owner['entry_args']:
        args = owner['entry_args']; require(args.count('--spec') == 1, 'Ambiguous original SPEC argument')
        original = args[args.index('--spec') + 1]
        require(original in closure and owner['bound_source_sha256'].get(original) == closure[original]['sha256'],
                'Original owner SPEC absent/unbound')
    if 'scientific_spec' in owner:
        original = owner['scientific_spec']
        require(original in closure.values() and owner['bound_source_sha256'].get(original['path']) == original['sha256'],
                'Original scientific SPEC reference absent/unbound')
    return owner


def source_gate(path, spec):
    require(spec['schema'] == SCHEMA and type(spec['phase']) is str and spec['phase'] and
            len(spec['RAW_inputs']) == 15 and len({r['receipt']['path'] for r in spec['RAW_inputs']}) == 15,
            'Exactly fifteen distinct RAW bindings required')
    raws = [terminal(row) for row in spec['RAW_inputs']]  # All fifteen before first SHA or import.
    for ref in spec['sources']:
        ref_shape(ref)
    closure = {ref['path']: ref for ref in spec['sources']}
    require(len(closure) == len(spec['sources']) and len(spec['packet_refs']) == 32, 'Unique complete input closure required')
    refs = [spec[k] for k in REF_KEYS] + spec['packet_refs']
    refs += [r[k] for r in spec['RAW_inputs'] for k in RAW_KEYS if k != 'phase']
    for ref in refs:
        require(ref['path'] in closure and closure[ref['path']] == ref, 'Input absent from sources')
    for ref in spec['sources']:
        bound(ref)
    reader, spec_ref = reference(__file__), reference(path)
    require(reader['path'] in closure and closure[reader['path']] == reader, 'Actual CPU SOURCE must be bound')
    owners = [bootstrap_join(row, raw, closure) for row, raw in zip(spec['RAW_inputs'], raws)]
    gpu = read(bound(spec['GPU_spec']))
    own = dict(receipt=spec['GPU_RAW'], phase=gpu['phase'], admission=spec['GPU_admission'],
               phase_binding=spec['GPU_phase_binding'], bootstrap=spec['GPU_bootstrap'])
    require(spec['RAW_inputs'][:-1] == gpu['RAW_inputs'] and spec['RAW_inputs'][-1] == own and
            len(gpu['RAW_inputs']) == 14 and gpu['schema'] == GPU_SCHEMA and
            spec['FIT'] == gpu['FIT'] and spec['renderer_source'] == gpu['renderer_source'] and
            spec['GPU_admission']['path'] == gpu['GPU_admission_file'] and
            all(r['path'] in closure and closure[r['path']] == r for r in gpu['sources']), 'Exact GPU/ancestor closure differs')
    owner = owners[-1]
    require(owner['scientific_spec'] == spec['GPU_spec'] and owner['entry'] == gpu['entry_source']['path'] and
            owner['entry_args'] == ['--spec', spec['GPU_spec']['path'], '--owner-execute'] and
            owner['cuda_visible_devices'] == '0' and
            owner['qualified_leaf_deadline_seconds'] == gpu['deadline_seconds'] and
            owner['bound_source_sha256'].get(spec['GPU_spec']['path']) == spec['GPU_spec']['sha256'] and
            all(owner['bound_source_sha256'].get(r['path']) == r['sha256'] for r in gpu['sources']), 'Actual GPU owner closure differs')
    output = Path(gpu['output_directory'])
    require(spec['GPU_report']['path'] == str(output / 'REPORT.json') and
            spec['GPU_start']['path'] == str(output / 'START.json') and
            [r['path'] for r in spec['packet_refs']] == [str(output / ('PACKET_%02d.pth' % i)) for i in range(32)],
            'Canonical N01 output references required')
    trace = closure.get(str(output / 'TRACE.jsonl')); require(trace is not None, 'TRACE must be explicitly sources-bound')
    if 'GPU_trace' in spec:
        require(spec['GPU_trace'] == trace, 'Explicit GPU_trace differs from canonical TRACE')
    report, start = read(bound(spec['GPU_report'])), read(bound(spec['GPU_start']))
    rows = [json.loads(line) for line in bound(trace).read_text(encoding='utf-8').splitlines() if line.strip()]
    require(report['schema'] == 'N01-fixed-FIT-local-gradient-report-v1' and report['status'] == GPU_STATUS and
            report['source'] == gpu['entry_source'] and report['spec'] == spec['GPU_spec'] and
            report['GPU_admission'] == spec['GPU_admission'] and report['phase'] == gpu['phase'] and
            report['images'] == 32 and len(report['rows']) == 32 and rows == report['rows'] and
            report['packets'] == spec['packet_refs'] and report['actual_backward_calls'] == 129 and
            report['cleanup']['allocated_after'] == 0, 'Complete GPU report/TRACE differs')
    for key in ('native610_unchanged', 'one_forward_per_image', 'no_DEV', 'no_TEST', 'no_AP'):
        require(report[key] is True, 'GPU report scope differs: ' + key)
    for key in ('optimizer_created', 'weights_updated', 'core_innovation_established', 'TGRS_success',
                'native30_CPU_replayed', 'harmful_training_drift_proven'):
        require(report[key] is False, 'GPU nonclaim differs: ' + key)
    require(start['schema'] == 'N01-start-v1' and start['source'] == report['source'] and start['spec'] == report['spec'] and
            start['GPU_admission'] == report['GPU_admission'] and start['phase'] == report['phase'] and
            set(start['parameter_names']) == set(SHAPES) and len(start['parameter_names']) == 8 and
            start['parameter_count'] == 263168 and start['CPU_public_fixture'] == report['CPU_public_fixture'], 'GPU START differs')
    require(start['CPU_public_fixture'] == dict(status='PUBLIC_POINT_OBJECTIVE_VALUES_AND_ANALYTIC_GRADIENTS_PASS',
            dtype='float64', native_model_executed=False, CUDA_used=False), 'Recorded public CPU fixture scope differs')
    for key in ('native345_active', 'typed_BN106_and159_frozen', 'no_DEV', 'no_TEST', 'no_AP'):
        require(start[key] is True, 'GPU START policy differs')
    require(start['optimizer_created'] is start['weights_updated'] is False and
            gpu['diagnostic_batches'] == 32 and len(gpu['diagnostic_dataset_indices']) == 32 and
            len(set(gpu['diagnostic_dataset_indices'])) == 32 and gpu['audit_positions'] == [0, 1, 2, 3, 4] and
            len(gpu['diagnostic_image_rows']) == 32, 'Prelocked population/scope differs')
    return gpu, report, start, rows, trace, reader, spec_ref


def close(actual, expected, label, rtol=None, atol=None):
    rtol = NUMERIC['metric_rtol'] if rtol is None else rtol
    atol = NUMERIC['metric_atol'] if atol is None else atol
    require(type(actual) in (int, float) and type(expected) in (int, float) and
            math.isfinite(actual) and math.isfinite(expected) and abs(actual - expected) <= atol + rtol * abs(expected), label)


def tensor(torch, value, shape, dtype, label):
    require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and value.layout == torch.strided and
            tuple(value.shape) == tuple(shape) and value.dtype == dtype and bool(torch.isfinite(value).all()), label)
    return value


def digest_tensor(value):
    return dict(shape=list(value.shape), dtype=str(value.dtype).replace('torch.', ''),
                sha256=hashlib.sha256(value.detach().contiguous().numpy().tobytes()).hexdigest())


def relation(torch, a, b):
    na, nb = float(torch.linalg.vector_norm(a)), float(torch.linalg.vector_norm(b))
    dot = float(torch.dot(a, b))
    cosine = None if min(na, nb) <= 1e-12 else max(-1., min(1., dot / (na * nb)))
    return dict(norm=na, other_norm=nb, dot=dot, cosine=cosine)


def check_relation(actual, observed, label):
    require(set(actual) == set(observed), label + ' keys')
    for key in actual:
        if actual[key] is None:
            require(observed[key] is None, label + ' undefined cosine')
        else:
            close(actual[key], observed[key], label + ' ' + key)


def point_losses(torch, z, y):
    # Independent stable BCE operator and class weights, not producer softplus arithmetic.
    terms = torch.nn.functional.binary_cross_entropy_with_logits(z, y.to(z.dtype), reduction='none')
    fg = terms.masked_fill(~y, 0); bg = terms.masked_fill(y, 0)
    nfg = y.sum(1).clamp_min(1).to(z.dtype); nbg = (~y).sum(1).clamp_min(1).to(z.dtype)
    return dict(uniform=terms.mean(1).mean(), foreground=fg.mean(1).mean(), background=bg.mean(1).mean(),
                balanced=((fg.sum(1)/nfg + bg.sum(1)/nbg)/2).mean())


def soft_statistics(torch, labels, observed):
    tensor(torch, labels, observed['shape'], torch.float32, 'Archived native soft labels')
    require(bool(((labels >= 0) & (labels <= 1)).all()) and observed['total_points'] == labels.numel() and
            observed['interpolation_soft_count'] == int(((labels > 0) & (labels < 1)).sum()) and
            observed['hard_positive_count'] == int((labels == 1).sum()) and
            observed['hard_negative_count'] == int((labels == 0).sum()), 'Native soft-label counts differ')
    masses = labels.double().sum(1).tolist()
    require(len(masses) == len(observed['positive_mass_by_row']), 'Native mass row count differs')
    for actual, expected in zip(masses, observed['positive_mass_by_row']):
        close(actual, expected, 'Independent native positive mass', 1e-4, 1e-4)
    return dict(stage=observed['stage'], soft_points=observed['interpolation_soft_count'],
                positive_mass_double=masses, sampling_coordinates_replayed=False)


def local_vjp(torch, packet, raw, qs, indices):
    M = tensor(torch, packet['M200'], (1, 100, 200, 200), torch.float32, 'Archived M200').clone().requires_grad_(True)
    h, w = packet['identity']['actual_img_shape']; H, W = packet['identity']['original_shape']
    canvas = torch.nn.functional.interpolate(M, size=(800, 800), mode='bilinear', align_corners=False)
    raster = torch.nn.functional.interpolate(canvas[0, :, :h, :w][:, None], size=(H, W), mode='bilinear', align_corners=False)[:, 0]
    z = raster[qs].flatten(1).gather(1, indices)
    torch.testing.assert_close(z, packet['point_logits'], rtol=1e-4, atol=1e-4)
    y = raw.flatten(1).gather(1, indices); losses = point_losses(torch, z, y); details = {}
    require(set(packet['local_M_gradients']) == {'foreground', 'background', 'balanced'}, 'Local M gradient population differs')
    for name in ('foreground', 'background', 'balanced'):
        close(float(losses[name].detach()), packet['losses'][name], 'Rendered local loss ' + name, 1e-4, 1e-4)
        gradient = torch.autograd.grad(losses[name], M, retain_graph=name != 'balanced')[0][0, qs]
        actual = tensor(torch, packet['local_M_gradients'][name], (len(qs), 200, 200), torch.float32, 'Archived local gradient')
        torch.testing.assert_close(gradient, actual, rtol=1e-3, atol=1e-7)
        details[name] = dict(max_abs_error=float((gradient-actual).abs().max()),
                             CPU_norm_double=float(gradient.double().norm()), archived_norm_double=float(actual.double().norm()))
    return dict(point_logit_max_abs_error=float((z.detach()-packet['point_logits']).abs().max()), losses_and_selected_M_VJP=details)


def summary(torch, values):
    valid = [v for v in values if v['uniform']['cosine'] is not None and v['balanced']['cosine'] is not None]
    u = [v['uniform']['cosine'] for v in valid]; d = [v['balanced']['cosine']-v['uniform']['cosine'] for v in valid]
    ratios = [v['uniform']['norm']/v['balanced']['norm'] for v in valid if v['balanced']['norm'] > 1e-12]
    med_u = statistics.median(u) if u else None; med_d = statistics.median(d) if d else None
    neg, improved = sum(x < 0 for x in u), sum(x > 0 for x in d)
    candidate = bool(len(valid) >= 24 and neg >= 20 and improved >= 20 and med_u < 0 and med_d >= .10 and len(ratios) >= 24)
    return dict(valid_images=len(valid), uniform_negative_cosine_images=neg, balanced_improves_cosine_images=improved,
                median_uniform_native_cosine=med_u, median_balanced_minus_uniform_cosine=med_d,
                FIT_initial_selected_parameter_norm_calibration_c=statistics.median(ratios) if ratios else None,
                N02_worth_finite_local_followup_pending_CPU=candidate, mechanism_established=False,
                harmful_training_drift_proven=False,
                criterion='valid>=24/32; U negative cosine>=20; balance improves>=20; median U cosine<0; median balance-minus-U>=0.10',
                gradient_scope='eight selected parameter tensors only; not whole native345', not_statistical_significance_or_novelty=True)


def check_summary(actual, observed):
    require(set(actual) == set(observed), 'Summary keys differ')
    for key, value in actual.items():
        if type(value) is float:
            close(value, observed[key], 'Summary ' + key)
        else:
            require(type(value) is type(observed[key]) and value == observed[key], 'Summary ' + key)


def check_packet(torch, np, coco, gpu, row, packet, names, position):
    identity = packet['identity']; image_id = identity['image_id']; image = coco.imgs[image_id]
    ids = [a['id'] for a in coco.imgToAnns[image_id]]; G = len(ids); H, W = image['height'], image['width']
    require(1 <= G <= 100 and identity['ann_ids'] == ids and row['annotation_ids'] == ids and
            identity['original_shape'] == [H, W] and row['identity'] == identity and
            packet['schema'] == 'N01-point-gradient-packet-v1' and packet['position'] == row['position'] == position and
            packet['dataset_index'] == row['dataset_index'] == gpu['diagnostic_dataset_indices'][position] and
            identity['dataset_index'] == row['dataset_index'] and row['image_id'] == image_id and
            image_id == gpu['diagnostic_image_rows'][position]['image_id'] and
            identity['image'] == gpu['diagnostic_image_rows'][position]['image'], 'Packet/TRACE/prelocked identity differs')
    scale = 800./max(H, W); require(identity['actual_img_shape'] == [int(H*scale+.5), int(W*scale+.5)] and
                                    identity['batch_input_shape'] == [800, 800], 'Original raster/crop identity differs')
    require(type(H) is type(W) is int and H > 0 and W > 0 and G == row['raw_GT_count'] and
            packet['optimizer_created'] is packet['weights_updated'] is row['optimizer_created'] is row['weights_updated'] is False,
            'Finite positive original GT/scope required')
    masks = []
    for ann in coco.imgToAnns[image_id]:
        require(ann['category_id'] == 1, 'Original category differs')
        mask = np.zeros((H, W), np.uint8) if ann.get('segmentation') == [] else coco.annToMask(ann)
        require(mask.shape == (H, W) and bool(np.isin(mask, [0, 1]).all()), 'Independent raw GT decode differs')
        masks.append(mask.astype(bool))
    raw = torch.from_numpy(np.stack(masks)); indices = tensor(torch, packet['point_indices'], (G, 12544), torch.int64, 'Point indices')
    require(not bool(((indices < 0) | (indices >= H*W)).any()), 'Original point index bounds differ')
    private = torch.Generator(device='cpu').manual_seed(gpu['point_seed_base'] + position)
    require(torch.equal(indices, torch.randint(H*W, (G, 12544), generator=private, dtype=torch.int64)) and
            row['point_seed'] == gpu['point_seed_base'] + position and row['loss_seed'] == gpu['loss_seed_base'] + position and
            row['point_indices_digest'] == digest_tensor(indices), 'Private point stream/digest differs')
    y = raw.flatten(1).gather(1, indices); z = tensor(torch, packet['point_logits'], (G, 12544), torch.float32, 'Point logits')
    require(torch.equal(y, tensor(torch, packet['point_targets'], (G, 12544), torch.bool, 'Hard point targets')) and
            row['original_GT_area'] == raw.flatten(1).sum(1).tolist() and row['sampled_foreground_count'] == y.sum(1).tolist() and
            row['sampled_zero_foreground_instances'] == int((y.sum(1) == 0).sum()) and
            row['hard_additional_labels'] is True and row['point_count_per_GT'] == 12544, 'Independent hard target checks differ')
    qs = packet['final_query_by_raw_GT']
    require(len(qs) == G and all(type(q) is int and 0 <= q < 100 for q in qs) and len(set(qs)) == G, 'Matched query identity differs')
    metadata = packet['native_metadata']; require(metadata['arm'] == 'A' and metadata['native_atoms'] == 30 and
            metadata['raw_annotation_ids'] == ids and metadata['final_query_by_raw_GT'] == qs and
            sorted(metadata['processed_GT_to_raw']) == list(range(G)) and len(metadata['current_stage_matches']) == 10,
            'Actual native matcher metadata differs')
    require(metadata['auxiliary_weight'] == 1. and metadata['processed_GT_observation'] ==
            dict(calls=1, correspondence_validator_passed=True) and metadata['native_forward_counts'] ==
            dict(head_forward_started=1, head_forward_finished=1, pixel_decoder=1, mask_embed=10) and
            metadata['independent_raw_GT_qualification_claimed'] is False and metadata['fixed_teacher_matching_used'] is False,
            'Current one-forward/native GT observation scope differs')
    for match in metadata['current_stage_matches']:
        matched_q, matched_g = match['positive_query_indices'], match['assigned_GT_indices']
        require(match['positives'] == len(matched_q) == len(matched_g) == G and
                all(type(q) is int and 0 <= q < 100 for q in matched_q) and len(set(matched_q)) == G and
                all(type(g) is int for g in matched_g) and sorted(matched_g) == list(range(G)),
                'Ten complete current assignment populations required')
    expected_q = [None] * G
    for query, processed in zip(matched_q, matched_g):
        expected_q[metadata['processed_GT_to_raw'][processed]] = query
    require(qs == expected_q, 'Final current assignment/raw GT query mapping differs')
    losses = point_losses(torch, z, y)
    require(set(packet['losses']) == {'native', 'uniform', 'foreground', 'background', 'balanced'} and row['losses'] == packet['losses'],
            'Loss packet/TRACE differs')
    for name, value in losses.items(): close(float(value), packet['losses'][name], 'Independent point loss ' + name, 1e-4, 1e-4)
    atoms = metadata['native_atom_values']; require(type(atoms) is dict and len(atoms) == 30 and all('loss' in k for k in atoms), 'Thirty observed atoms required')
    numbers = [v for value in atoms.values() for v in (value if isinstance(value, list) else [value])]
    require(all(type(v) in (int, float) and math.isfinite(v) for v in numbers), 'Observed native loss atoms nonfinite')
    close(math.fsum(numbers), packet['losses']['native'], 'Observed native atom sum', 1e-4, 1e-4)
    p = z.sigmoid()
    require(len(row['output_gradient_foreground_mass']) == len(row['output_gradient_background_mass']) == G,
            'Output gradient mass population differs')
    for actual, expected in zip((y*(1-p)).mean(1).tolist(), row['output_gradient_foreground_mass']): close(actual, expected, 'Foreground mass', 1e-4, 1e-4)
    for actual, expected in zip(((~y)*p).mean(1).tolist(), row['output_gradient_background_mass']): close(actual, expected, 'Background mass', 1e-4, 1e-4)
    groups = packet['parameter_gradients']; require(set(groups) == set(GROUPS), 'Four selected gradient groups required')
    for group in GROUPS:
        require(set(groups[group]) == set(SHAPES), 'Eight parameter gradient names differ')
        for name in names: tensor(torch, groups[group][name], SHAPES[name], torch.float32, 'Selected parameter gradient')
    vectors = {k: torch.cat([groups[k][name].reshape(-1).double() for name in names]) for k in GROUPS}
    # The producer defines U by FP32 addition of its archived CPU FG/BG tensors.
    vectors['uniform'] = torch.cat([(groups['foreground'][n]+groups['background'][n]).reshape(-1).double() for n in names])
    U, B = relation(torch, vectors['uniform'], vectors['native']), relation(torch, vectors['balanced'], vectors['native'])
    check_relation(U, row['parameters_uniform_vs_native'], 'Uniform/native concatenated-double relation')
    check_relation(B, row['parameters_balanced_vs_native'], 'Balanced/native concatenated-double relation')
    close(float(vectors['foreground'].norm()), row['foreground_parameter_norm'], 'Foreground concatenated norm')
    close(float(vectors['background'].norm()), row['background_parameter_norm'], 'Background concatenated norm')
    close(B['norm'], row['balanced_parameter_norm'], 'Balanced concatenated norm')
    require(packet['backward_count'] == (5 if position == 0 else 4) and row['native_full_backward_CPU_replayed'] is False and
            set(row['field_relations']) == {'F256', 'e0', 'M200'}, 'Observed backward/field scope differs')
    counts = row['actual_native_gradient_counts']
    require(set(counts) == set(GROUPS) | ({'uniform_direct'} if position == 0 else set()) and
            all(set(v) == {'non_none', 'nonzero'} and type(v['non_none']) is type(v['nonzero']) is int and
                0 <= v['nonzero'] <= v['non_none'] <= 345 for v in counts.values()), 'Observed native gradient counts differ')
    linear = row['direct_uniform_gradient_linearity']
    if position == 0:
        require(linear['status'] == 'DIRECT_U_VS_FOREGROUND_PLUS_BACKGROUND_PASS' and linear['rtol'] == 1e-3 and
                linear['atol'] == 1e-7 and type(linear['max_abs_error']) in (int, float) and
                math.isfinite(linear['max_abs_error']) and linear['max_abs_error'] >= 0, 'Recorded direct-U linearity differs')
    else:
        require(linear is None, 'Direct-U backward exists only for first image')
    observed = packet['native_point_labels_observed']; require(observed == row['native_point_labels'] and len(observed) == 10, 'Native label/TRACE join differs')
    for stage, value in enumerate(observed):
        require(value['stage'] == stage and value['shape'] == [G, 12544] and value['total_points'] == G*12544 and
                all(type(value[k]) is int and value[k] >= 0 for k in ('interpolation_soft_count', 'hard_positive_count', 'hard_negative_count')) and
                sum(value[k] for k in ('interpolation_soft_count', 'hard_positive_count', 'hard_negative_count')) == G*12544 and
                len(value['positive_mass_by_row']) == G and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 12544 for v in value['positive_mass_by_row']),
                'Observed native soft-label statistic structure differs')
    local, soft = None, None
    if position < 5:
        require(torch.equal(tensor(torch, packet['raw_GT'], (G, H, W), torch.bool, 'Archived raw GT'), raw), 'Archived raw GT differs independently')
        local = local_vjp(torch, packet, raw, qs, indices)
        require(set(packet['native_point_targets']) == {'stage_%d' % i for i in range(10)}, 'Ten soft target packets required')
        soft = [soft_statistics(torch, packet['native_point_targets']['stage_%d' % i], observed[i]) for i in range(10)]
    return dict(position=position, image_id=image_id, annotation_ids=ids, raw_GT=G, hard_labels_independently_decoded=True,
                private_uniform_indices_exact=True, duplicate_indices_preserved=True, losses={k: float(v) for k, v in losses.items()},
                uniform=U, balanced=B, local_M200_replay=local, archived_native_soft_target_statistics=soft,
                selected_parameter_gradients_backpropagated_on_CPU=False)


def run(spec_file):
    path = Path(spec_file); require(path.is_absolute(), 'Absolute CPU SPEC required'); path = path.resolve(strict=True)
    spec = read(path); target = Path(spec['Q_path'])
    require(target.is_absolute() and not target.exists() and str(target) not in {r['path'] for r in spec['sources']} and target != path,
            'Fresh absolute Q output required')
    gpu, report, start, rows, trace, reader, spec_ref = source_gate(path, spec)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and all(os.environ.get(k) == '2' for k in
            ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')), 'CPU-only environment2 required')
    import psutil
    require(psutil.Process().cpu_affinity() == [0, 1], 'Actual CPU01 affinity required before scientific imports')
    import torch
    require(not torch.cuda.is_initialized() and torch.__version__ == '2.0.1+cu118', 'CPU Torch/cold CUDA differs')
    torch.set_num_threads(2); torch.set_num_interop_threads(2); rng = torch.random.get_rng_state().clone()
    import numpy as np
    from pycocotools.coco import COCO
    coco = COCO(str(bound(spec['FIT'])))
    require(len(coco.imgs) == 741 and len(coco.anns) == 1661 and set(coco.cats) == {1}, 'Original FIT population differs')
    details = []
    for position, ref in enumerate(spec['packet_refs']):
        require(rows[position]['packet'] == ref, 'Actual TRACE packet reference differs')
        packet = torch.load(str(bound(ref)), map_location='cpu')
        details.append(check_packet(torch, np, coco, gpu, rows[position], packet, start['parameter_names'], position))
        del packet
        print(json.dumps(dict(CPU_checked_images=position+1, local_M_VJP_checked=min(position+1, 5))), flush=True)
    actual_summary = summary(torch, details); check_summary(actual_summary, report['summary'])
    require(not torch.cuda.is_initialized() and torch.equal(rng, torch.random.get_rng_state()) and
            psutil.Process().cpu_affinity() == [0, 1] and torch.get_num_threads() == torch.get_num_interop_threads() == 2,
            'CPU resource/RNG scope changed')
    for ref in spec['sources']: bound(ref)
    require(reference(path) == spec_ref and reference(__file__) == reader, 'Final CPU provenance changed')
    q = dict(schema=QUAL_SCHEMA, status=STATUS, phase=spec['phase'], source=gpu['entry_source'], reader_source=reader,
             spec=spec_ref, GPU_spec=spec['GPU_spec'], GPU_admission=spec['GPU_admission'], parent_receipt=spec['GPU_RAW'],
             GPU_phase_binding=spec['GPU_phase_binding'], GPU_bootstrap=spec['GPU_bootstrap'], report=spec['GPU_report'],
             start=spec['GPU_start'], trace=trace, packet_refs=spec['packet_refs'], FIT=spec['FIT'], renderer_source=spec['renderer_source'],
             RAW_inputs=spec['RAW_inputs'], CPU_execution=dict(device='cpu', affinity=[0, 1], CUDA_VISIBLE_DEVICES='',
             torch_num_threads=2, torch_num_interop_threads=2, CUDA_initialized=False, external_CPU_RNG_unchanged=True),
             numeric_protocol=NUMERIC, images_checked=32, selected_parameter_tensors=8, selected_parameter_scalars=263168,
             local_M200_VJP_images_checked=5, archived_native_soft_label_statistics_images_checked=5,
             independent_checks=details, independent_summary=actual_summary, optimizer_created=False, weights_updated=False,
             full_native_backward_CPU_replayed=False, selected_parameter_backward_CPU_replayed=False,
             native_uncertainty_sampling_CPU_replayed=False, F_e_M_native_gradient_relations_CPU_replayed=False,
             GPU_native610_unchanged_SOURCE_observed=True, local_gradient_alignment_not_training_drift=True,
             harmful_training_drift_proven=False, mechanism_established=False, efficacy_claimed=False, no_DEV=True, no_TEST=True, no_AP=True,
             evidence_limits=['32 hard-target/point/loss checks and stored eight-parameter concatenated-double gradient arithmetic only.',
             'Five conditional local M200 VJPs use independent CPU bilinear projection; no native backbone or native30 backward is repeated.',
             'Native soft-label statistics are recomputed from archived labels for five images; uncertainty coordinates were not saved.',
             'F/e/native-M gradient relations, first-image native parameter linearity and unchanged610 are pinned GPU SOURCE observations.',
             'The fixed E24/FIT local probe does not establish harmful training drift, statistical significance, novelty, AP or efficacy.',
             'This file cannot prove its own owned CPU terminal; ROOT must separately close actual CPU admission/RAW provenance.'])
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(q, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(status=STATUS, qualification=reference(target))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True); parser.add_argument('--owner-execute', action='store_true')
    args = parser.parse_args(); require(args.owner_execute, 'Explicit ROOT-owned CPU execution required')
    run(args.spec)
