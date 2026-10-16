# encoding: utf-8
"""N22 UNBOUND capture only. No imports of scientific libraries at module load."""

import gc
import importlib.util
import json
from pathlib import Path
import sys
import time
import types

PACKET_SCHEMA = 'N22-FIT32-natural100-fullC-actual-GPU-action-packet-v1'
REPORT_SCHEMA = 'N22-FIT32-natural100-fullC-capture-report-v1'
SOURCE_ROLES = (
    'native_loader_source', 'micro_preflight_source', 'initializer_source',
    'training_source', 'native_helpers', 'renderer_source', 'joint_losses_source',
    'joint_preflight_entry_source', 'original_fit_cache_entry_source',
    'cache_body_source', 'region_common_source', 'legacy_eval_body_source',
    'N10_evaluation_source', 'export_source', 'point_heads_source')

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def collect_all100(collect_one):
    rows = []
    for q in range(100):
        value = collect_one(q)
        require(value["query_id"] == q, "Natural all100 query identity/order drift")
        require(type(value["C_count"]) is int and type(value["K"]) is int
                and 0 <= value["K"] <= value["C_count"], "Invalid C/K")
        rows.append(value)
    return rows


def validate_selected_rows(spec):
    rows = spec["selected_fit_rows"]
    require(isinstance(rows, list) and len(rows) == 32, "Exactly prelocked32 FIT rows required")
    indices, image_ids = [], []
    for item in rows:
        require(set(item) == {"dataset_index", "old_row", "packet_ref", "native_L_ref"},
                "Explicit N16 row/packet/native-L references required")
        index, row = item["dataset_index"], item["old_row"]
        require(type(index) is int and index >= 0 and row["dataset_index"] == index,
                "Original dataset ordinal drift")
        require(type(row["image_id"]) is int and row["identity"]["image_id"] == row["image_id"]
                and type(row["matching_loss_seed"]) is int, "Original identity/seed missing")
        for key in ("packet_ref", "native_L_ref"):
            ref = item[key]
            require(set(ref) == {"path", "bytes", "sha256"} and isinstance(ref["path"], str)
                    and type(ref["bytes"]) is int and ref["bytes"] > 0
                    and isinstance(ref["sha256"], str) and len(ref["sha256"]) == 64
                    and all(c in "0123456789abcdef" for c in ref["sha256"]), "Strict input reference required")
        indices.append(index); image_ids.append(row["image_id"])
    require(len(set(indices)) == len(set(image_ids)) == 32, "Duplicate FIT identity rejected")
    require(indices == spec["expected_selected_dataset_indices"]
            and image_ids == spec["expected_selected_image_ids"], "Prelocked N21 FIT population/order drift")
    return rows


def _call(control, name, *args):
    fn = control[name] if isinstance(control, dict) else getattr(control, name)
    return fn(*args)


def _json(control, ref):
    return json.loads(Path(_call(control, 'closed', ref)).read_text(encoding='utf-8-sig'))


def _load(control, ref, role):
    path = Path(_call(control, 'closed', ref))
    loader_spec = importlib.util.spec_from_file_location('_n22_' + role, path)
    require(loader_spec is not None and loader_spec.loader is not None, 'Source loader missing')
    module = importlib.util.module_from_spec(loader_spec)
    sys.modules[loader_spec.name] = module
    loader_spec.loader.exec_module(module)
    return module


def _tensor(torch, value, dtype, shape, label, device='cpu'):
    require(isinstance(value, torch.Tensor) and value.layout == torch.strided
            and value.dtype == dtype and tuple(value.shape) == tuple(shape)
            and value.device.type == device and not value.requires_grad,
            label + ': tensor ABI drift')
    if value.is_floating_point():
        require(bool(torch.isfinite(value).all()), label + ': nonfinite')


def _same_bits(torch, a, b, label):
    require(a.dtype == b.dtype and a.shape == b.shape, label + ': dtype/shape drift')
    # NumPy byte strings handle singleton/zero-stride views without uint8-view hazards.
    require(a.detach().cpu().contiguous().numpy().tobytes()
            == b.detach().cpu().contiguous().numpy().tobytes(), label + ': byte drift')


def _frozen(model, label):
    require(not model.training and not any(p.requires_grad or p.grad is not None
            for p in model.parameters()), label + ': not frozen/eval')


def _cpu_tree(torch, value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone().contiguous()
    if isinstance(value, dict):
        return {k: _cpu_tree(torch, v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(torch, v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(torch, v) for v in value)
    return value


def bootstrap(torch, np, spec, control):
    """Direct accepted-native0 construction; never calls historical own/qualified_training gates."""
    import cv2
    from scipy import ndimage
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.model import revert_sync_batchnorm
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    from mmdet.datasets import CocoDataset
    from mmdet.datasets.api_wrappers import COCO

    _call(control, 'check')
    require(not torch.cuda.is_initialized(), 'CPU model construction must precede CUDA initialization')
    refs, mr = spec['direct_sources'], spec['model_refs']
    require(all(k in refs for k in SOURCE_ROLES), 'Direct source role missing')
    context = _json(control, mr['native_context_spec'])
    train = _json(control, mr['native0_training_spec'])
    qualification = _json(control, mr['native0_qualification'])
    require(context['FIT'] == mr['FIT'] and context['checkpointmap']['native0'] == mr['native0_checkpoint']
            and qualification['checkpoint'] == mr['native0_checkpoint']
            and qualification['spec'] == mr['native0_training_spec']
            and train['initializer_qualification'] == mr['initializer_qualification']
            and qualification['model_proof']['proof_refs']['state_proof'] == mr['native0_state_proof'],
            'Accepted native0/configuration metadata reference drift')
    for key in ('native0_checkpoint', 'native0_state_proof', 'official_mask2former_config',
                'official_COCO_initializer_checkpoint', 'fixed_LOG_PR_checkpoint', 'FIT'):
        _call(control, 'closed', mr[key])
    require(train['initializers']['mask2former']['config'] == mr['official_mask2former_config']
            and train['initializers']['mask2former']['checkpoint'] == mr['official_COCO_initializer_checkpoint']
            and train['image_root'] == mr['image_root'], 'Native construction roles differ')
    accepted_init = _json(control, mr['initializer_qualification'])
    for ref in accepted_init['initializers']['mask2former']['official_config_source_closure']:
        _call(control, 'closed', ref)
    for role in ('native_loader_source', 'micro_preflight_source', 'initializer_source',
                 'training_source', 'native_helpers', 'renderer_source',
                 'joint_losses_source', 'joint_preflight_entry_source'):
        require(refs[role]['sha256'] == context[role]['sha256']
                and refs[role]['bytes'] == context[role]['bytes'], 'Math source copy differs: ' + role)
    names = dict(native='native_loader_source', pref='micro_preflight_source',
                 initializer='initializer_source', native_common='training_source',
                 old='native_helpers', diag='renderer_source', joint='joint_losses_source',
                 preflight='joint_preflight_entry_source', cache7='original_fit_cache_entry_source',
                 targets='cache_body_source', regions='region_common_source',
                 selector='legacy_eval_body_source', n10='N10_evaluation_source',
                 factory='point_heads_source')
    modules = {name: _load(control, refs[role], role) for name, role in names.items()}
    native, pref = modules['native'], modules['pref']
    modules['old'].torch = torch
    register_all_modules(init_default_scope=False)
    init_default_scope('mmdet')
    construction = dict(train, training_source=context['training_source'],
                        checkpointmap=context['checkpointmap'], qualificationmap=context['qualificationmap'],
                        order_seed=native.ORDER_SEED, phase_loss_seed_base=native.LOSS_SEED_BASE)
    model, anchor, policy, loading = native.prepare_training_model(
        torch, pref, modules['initializer'], modules['native_common'], MODELS,
        revert_sync_batchnorm, Config, tuple(CocoDataset.METAINFO['classes']), construction,
        dict(training_spec=train, qualification=qualification))
    del anchor, policy
    cp = torch.load(str(_call(control, 'closed', mr['fixed_LOG_PR_checkpoint'])), map_location='cpu')
    require(isinstance(cp, dict) and {'state_dict', 'meta'}.issubset(cp),
            'Original PR state_dict/meta container required')
    state, meta = cp['state_dict'], cp['meta']
    require(meta['arm'] == 'LOG_PR' and meta['seed'] == 202610120701
            and meta['completed_head_epochs'] == 24 and meta['updates'] == 39864
            and meta['parameter_tensors'] == 8 and meta['parameter_scalars'] == 40497
            and meta['native_weights_updated'] is False and meta['no_pending_gradients'] is True
            and meta['head_factory_source']['sha256'] == refs['point_heads_source']['sha256'],
            'Fixed qualified LOG_PR endpoint identity drift')
    head = modules['factory'].make_point_head(torch, 202610120701)
    expected = head.state_dict()
    require(tuple(state) == tuple(expected) and len(state) == 8 and not list(head.buffers())
            and sum(v.numel() for v in state.values()) == 40497, 'Fixed PR architecture differs')
    for key, value in state.items():
        _tensor(torch, value, torch.float32, expected[key].shape, 'PR state ' + key)
    head.load_state_dict(state, strict=True)
    head.requires_grad_(False); head.eval()
    model.requires_grad_(False); model.eval(); model.backbone.with_cp = False
    require(len(model.state_dict()) == 610 and len(list(model.parameters())) == 451,
            'Original native610/451 required')
    require(spec['allocator_limit_MiB'] == 4096, 'Locked allocator4096MiB required')
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(4096 * 1024**2 / total, device=0)
    model.to('cuda:0'); head.to('cuda:0')
    _frozen(model, 'Teacher'); _frozen(head, 'PR')
    dataset = CocoDataset(ann_file=str(_call(control, 'closed', mr['FIT'])),
        data_prefix=dict(img=mr['image_root']), metainfo=dict(classes=('ship',)),
        pipeline=[], filter_cfg=None, test_mode=True)
    coco = COCO(str(_call(control, 'closed', mr['FIT'])))
    require(len(dataset) == 741 and len(coco.anns) == 1661 and set(coco.cats) == {1},
            'Original FIT741/1661/ship1 required')
    for item in validate_selected_rows(spec):
        index, row = item['dataset_index'], item['old_row']
        require(dataset.get_data_info(index)['img_id'] == row['image_id'], 'Selected FIT order/id drift')
        _call(control, 'closed', row['identity']['image'])
    modules['kernels'] = modules['n10'].original_export_kernels(refs['export_source'])
    del cp, state, expected
    return dict(modules=modules, model=model, head=head, dataset=dataset, coco=coco,
                context=context, cv2=cv2, ndimage=ndimage, loading=loading,
                PR_meta=meta, teacher_state_before=pref.state_proof(model.state_dict()),
                PR_state_before=pref.state_proof(head.state_dict()))


def capture_image(torch, np, spec, control, boot, item):
    """Exactly one frozen loss forward, then all100 complete-C ungated PR."""
    modules, model, pr = boot['modules'], boot['model'], boot['head']
    pref, old, diag, joint, preflight, cache7, targets, selector = (
        modules[k] for k in ('pref', 'old', 'diag', 'joint', 'preflight', 'cache7', 'targets', 'selector'))
    index, row = item['dataset_index'], item['old_row']
    packet = torch.load(str(_call(control, 'closed', item['packet_ref'])), map_location='cpu')
    carrier = torch.load(str(_call(control, 'closed', item['native_L_ref'])), map_location='cpu')
    require(packet['schema'] == 'N16-all100-FIT-action-sampled-cache-body-v1'
            and carrier['schema'] == 'N16-actual-GPU-natural100-L-carrier-v1'
            and packet['dataset_index'] == carrier['dataset_index'] == index
            and packet['identity'] == carrier['identity'] == row['identity']
            and packet['natural_native_L_carrier'] == item['native_L_ref']
            and row['packet'] == item['packet_ref'] and row['natural_native_L_carrier'] == item['native_L_ref'],
            'Qualified N16 packet/native-L/row binding drift')
    for key in ('source', 'spec', 'GPU_admission', 'phase'):
        require(packet[key] == carrier[key], 'N16 carrier parent mismatch: ' + key)
    require(packet['teacher_checkpoint'] == spec['model_refs']['native0_checkpoint']
            and packet['fixed_PR_checkpoint'] == spec['model_refs']['fixed_LOG_PR_checkpoint']
            and packet['natural_query_ids'] == list(range(100)), 'Frozen teacher/PR/all100 drift')
    identity = packet['identity']; H, W = identity['original_shape']
    old_L = carrier['native_L']
    _tensor(torch, old_L, torch.float32, (100, H, W), 'Qualified native-L carrier')
    seed = packet['matching_loss_seed']
    require(type(seed) is int and seed == row['matching_loss_seed'], 'Original matcher seed drift')
    _frozen(model, 'Teacher'); _frozen(pr, 'PR')
    outside = pref.rng_snapshot(torch)
    with torch.no_grad(), torch.random.fork_rng(devices=[0]):
        _call(control, 'check')
        data, stripped, gray800, gray, observed = old.load_image(
            model, boot['dataset'], boot['coco'], boot['dataset'].get_data_info(index))
        del stripped, gray800, gray
        observed['dataset_index'] = index
        require(observed == identity, 'Original image/parser/resize identity drift')
        processed = pref.check_targets(torch, data, observed)
        require(pref.tree_equal(torch, processed, packet['processed_target_proof']), 'Processed targets drift')
        raw_GT, mapping, validator = preflight.raw_GT_validator(
            torch, np, boot['cv2'], boot['coco'], data, observed)
        require(mapping == packet['raw_target_mapping'] and identity['ann_ids'] == packet['raw_GT_ids']
                and pref.digest_tensor(raw_GT) == packet['raw_GT_digest'], 'Original real-GT/mapping drift')
        torch.random.default_generator.manual_seed(seed); torch.cuda.manual_seed(seed)
        native_head = model.panoptic_head
        with joint._observe_processed_GT(torch, native_head, identity['ann_ids'], raw_GT,
                                         mapping, identity, validator) as gt_observation:
            with targets.observe_all100_targets(torch, native_head, old.digest_tensor) as (matches, observations):
                with cache7.observed_fields(torch, native_head) as fields:
                    losses = model.loss(data['inputs'], data['data_samples'])
        require(len(losses) == 30 and set(losses) == set(old.ATOMIC_KEYS) == set(packet['native_loss_atoms']),
                'Thirty original loss atoms required')
        for value in losses.values():
            values = value if isinstance(value, (tuple, list)) else [value]
            require(values and all(not v.requires_grad and bool(torch.isfinite(v).all()) for v in values),
                    'Frozen loss atoms must be finite')
        require(matches == packet['native_stage_matches'] and len(matches) == 10
                and pref.tree_equal(torch, observations, packet['all_ten_target_observations'])
                and pref.tree_equal(torch, gt_observation, packet['actual_raw_processed_GT_correspondence']),
                'Original all10 matcher/GT correspondence drift')
        F, e = fields['F'][0], fields['e'][-1]
        classes, mask_stages = fields['output'][0]
        native_class, M = classes[-1], mask_stages[-1]
        _same_bits(torch, e, packet['native_e0'], 'All100 e')
        _same_bits(torch, native_class, packet['native_class_logits'], 'All100 class')
        assignment = targets.final_assignment(torch, observations[-1], mapping,
                                              identity['ann_ids'], native_head.num_classes)
        require(pref.tree_equal(torch, assignment, packet['assignment']), 'Original full100 assignment drift')
        native_L = diag.render_original(torch, M, identity)
        _tensor(torch, native_L, torch.float32, (100, H, W), 'Actual original100 L', 'cuda')
        actual_L = native_L.detach().cpu().clone().contiguous()
        error = float((actual_L - old_L).abs().max())
        require(error <= 1.e-4, 'Native-L numeric drift exceeds absolute1e-4/rtol0')
        domains, drift = [], []
        for q in range(100):
            fresh = selector.domains_from_native(np, boot['ndimage'], actual_L[q].numpy())
            previous = selector.domains_from_native(np, boot['ndimage'], old_L[q].numpy())
            diffs = dict(query_id=q, P=int(np.count_nonzero(fresh[0] != previous[0])),
                         Omega=int(np.count_nonzero(fresh[1] != previous[1])),
                         C=int(np.count_nonzero(fresh[2] != previous[2])), K=int(fresh[4] - previous[4]))
            drift.append(diffs); domains.append(fresh)
        if any(any(d[k] != 0 for k in ('P', 'Omega', 'C', 'K')) for d in drift):
            _call(control, 'write_json', Path(spec['output_directory']) / ('DRIFT_%04d.json' % index),
                  dict(dataset_index=index, native_L_max_abs_error=error, differences=drift))
            raise RuntimeError('N22_NATIVE_SIGN_OMEGA_C_K_DRIFT: original input population changed')
        F_original = modules['n10'].original_feature(torch, diag, F, identity)
        _tensor(torch, F_original, torch.float32, (256, H, W), 'Actual rendered F256', 'cuda')
        e0 = e[0]
        cost = selector._cost()
        def collect(q):
            _call(control, 'check')
            P, Omega, C, indices, K = domains[q]
            _, regions, regionK, _ = modules['regions'].gpu_regions(np, boot['ndimage'], actual_L[q].numpy())
            S = np.flatnonzero(regions['LOG_PR']).astype(np.int64)
            require(regionK == K and np.array_equal(regions['LOG_PR'], Omega)
                    and np.all(C.reshape(-1)[S]), 'Original Omega router/S subset drift')
            checked_points = [0]
            def observe_chunk(event):
                _call(control, 'check')
                require(event['query_id'] == q and event['stage'] == 'full_C'
                        and event['synthetic_only'] is False and event['point_chunk'] == 2048
                        and event['chunk_first'] == checked_points[0], 'Actual PR callback scope/order drift')
                n = len(event['flat_indices'])
                require(np.array_equal(event['flat_indices'].numpy(), indices[checked_points[0]:checked_points[0]+n]),
                        'Actual PR chunk coordinate drift')
                checked_points[0] += n
            actual = selector._run_points(torch, np, F_original, e0, native_L[q], q, indices,
                pr, 'PR', cost, observe_chunk, 'N22_FIXED_PR', 'all_C_including_K0', 'full_C')
            require(checked_points[0] == len(indices), 'Full C capture incomplete')
            native = torch.from_numpy(actual['native_L'].copy())
            u = torch.from_numpy(actual['residual'].copy())
            combined = torch.from_numpy(actual['combined'].copy())
            _same_bits(torch, native, actual_L[q].flatten()[torch.from_numpy(indices)], 'Actual C input L')
            before, after = native > 0, combined > 0
            return dict(query_id=q, C_count=int(len(indices)), K=int(K),
                C_indices=torch.from_numpy(indices.copy()), native_L_on_C=native,
                PR_u=u, PR_combined=combined, native_positive=before,
                actual_after_positive=after, actual_flip=before != after,
                StrongR_S_indices=torch.from_numpy(S.copy()))
        queries = collect_all100(collect)
        top, order = modules['kernels'].order(torch, native_class, types.SimpleNamespace(num_classes=1))
        require(len(order) == 100 and sorted(order.cpu().tolist()) == list(range(100)), 'Original export100 order drift')
        binary, quality, scores = modules['kernels'].quality(torch, native_L[order], top)
        by_query = scores.new_empty(100); by_query[order] = scores
        payload = dict(schema=PACKET_SCHEMA, dataset_index=index, identity=identity,
            natural_query_ids=list(range(100)), native_L=actual_L,
            native_e0=e.detach().cpu().clone().contiguous(),
            native_class_logits=native_class.detach().cpu().clone().contiguous(),
            raw_GT=raw_GT.detach().cpu().clone().contiguous(), raw_GT_ids=list(identity['ann_ids']),
            raw_target_mapping=tuple(mapping), assignment=_cpu_tree(torch, assignment),
            native_stage_matches=matches, all_ten_target_observations=_cpu_tree(torch, observations),
            processed_target_proof=processed, actual_raw_processed_GT_correspondence=gt_observation,
            native_export_query_ids=order.detach().cpu().tolist(),
            native_scores_by_query=by_query.detach().cpu().clone().contiguous(),
            native_top_scores_by_export_order=top.detach().cpu().clone().contiguous(),
            native_quality_by_export_order=quality.detach().cpu().clone().contiguous(),
            category_id=1, queries=queries, matching_loss_seed=seed,
            native_L_comparison=dict(max_abs_error=error, atol=1.e-4, rtol=0., differences=drift),
            input_refs=dict(packet=item['packet_ref'], native_L=item['native_L_ref'],
                            teacher=spec['model_refs']['native0_checkpoint'],
                            fixed_PR=spec['model_refs']['fixed_LOG_PR_checkpoint']),
            Omega_router_action_source='same full-C actual GPU actions restricted to Omega; no separate S forward',
            old_wholeDEV_StrongR_bitexact_or_AP_reproduction_claimed=False,
            raw_GT_only_is_true_GT=True, matcher_negative_empty_proxy_is_true_GT=False,
            actual_source_terminal_qualified=False, CPU_numerical_qualification_pending=True)
        del binary, quality, scores, by_query, top, order, native_L, F_original, F, e, e0, M
        del native_class, fields, classes, mask_stages, raw_GT, data, losses, packet, carrier, old_L
    require(pref.tree_equal(torch, outside, pref.rng_snapshot(torch)), 'External CPU/CUDA RNG not restored')
    _frozen(model, 'Teacher'); _frozen(pr, 'PR')
    return payload


def run(torch, np, spec, control):
    """Own wrapper validates RAW/admission; this body writes only a fresh capture directory."""
    started = time.monotonic()
    selected = validate_selected_rows(spec)
    output = Path(spec['output_directory'])
    require(output.is_absolute() and not output.exists(), 'Absolute fresh output directory required')
    output.mkdir()
    captures = output / 'CAPTURES'; captures.mkdir()
    boot = bootstrap(torch, np, spec, control)
    rows, refs = [], []
    for ordinal, item in enumerate(selected):
        _call(control, 'check')
        payload = capture_image(torch, np, spec, control, boot, item)
        target = captures / ('PACKET_%04d.pth' % item['dataset_index'])
        with target.open('xb') as stream:
            torch.save(payload, stream)
        packet_ref = _call(control, 'ref', target)
        row = dict(dataset_index=item['dataset_index'], image_id=payload['identity']['image_id'],
            identity=payload['identity'], packet_ref=packet_ref, natural_queries=100,
            raw_GT=len(payload['raw_GT_ids']), C_points=sum(q['C_count'] for q in payload['queries']),
            Omega_points=sum(q['K'] for q in payload['queries']),
            K0_nonempty_C_queries=sum(q['K'] == 0 and q['C_count'] > 0 for q in payload['queries']),
            native_L_max_abs_error=payload['native_L_comparison']['max_abs_error'])
        rows.append(row); refs.append(packet_ref)
        del payload
        gc.collect(); _call(control, 'check')
        print(json.dumps(dict(N22_captured_images=ordinal+1, total_images=32)), flush=True)
    pref = boot['modules']['pref']
    require(pref.state_proof(boot['model'].state_dict()) == boot['teacher_state_before']
            and pref.state_proof(boot['head'].state_dict()) == boot['PR_state_before'],
            'Frozen teacher/PR state changed')
    metadata = dict(schema=REPORT_SCHEMA,
        status='N22_CAPTURE_COMPLETE_PENDING_ROOT_ACTUAL0_AND_CPU_QUALIFICATION',
        rows=rows, packet_refs=refs, images=32, queries=3200,
        dataset_indices=[r['dataset_index'] for r in rows], image_ids=[r['image_id'] for r in rows],
        raw_GT=sum(r['raw_GT'] for r in rows), C_points=sum(r['C_points'] for r in rows),
        Omega_points=sum(r['Omega_points'] for r in rows),
        one_teacher_loss_forward_per_image=True, all100_C_including_K0_complete=True,
        ungated_PR_residual_and_actual_GPU_combined_preserved=True,
        original10stage_GT_e_class_exact=True, native_L_max_abs_tolerance=1.e-4,
        native_sign_Omega_C_K_exact=True, same_fullC_actions_restricted_to_Omega=True,
        old_wholeDEV_StrongR_bitexact_or_AP_reproduction_claimed=False,
        counts_or_oracles_computed=False, no_training=True, no_AP=True, no_DEV=True,
        qualified=False, ROOT_actual_GPU0_verified=False, CPU_numerical_qualification_pending=True,
        elapsed_seconds=time.monotonic()-started)
    # Release every runtime model alias before the wrapper's CUDA cleanup/terminal gate.
    boot.clear(); gc.collect(); torch.cuda.empty_cache()
    metadata['manifest_ref'] = _call(control, 'write_json', output / 'CAPTURE_MANIFEST.json', metadata)
    return metadata
