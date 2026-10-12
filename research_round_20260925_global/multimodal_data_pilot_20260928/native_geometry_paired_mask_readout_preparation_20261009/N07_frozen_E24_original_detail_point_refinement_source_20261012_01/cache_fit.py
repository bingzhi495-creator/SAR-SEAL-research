"""ROOT-owned N07 FIT cache preparation; SOURCE-only until separately admitted.

No optimizer, backward, augmentation, DEV, TEST or AP. Original JPEG patches
are image intensities, not SAR phase/amplitude or scattering measurements.
F/e are coarse native fields projected through the original renderer. The
cache fixes all matching and sampled points before either C/R head is trained.
This entry does not implement training/evaluation or qualify its own numerics.
"""
import argparse
import ast
from contextlib import contextmanager
from functools import partial
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import time
import types

SCHEMA = 'ROOT-N07-frozen-E24-original-detail-point-cache-spec-v1'
STATUS = 'N07_FROZEN_E24_FIT_POINT_CACHE_GPU_COMPLETE_PENDING_CPU'
PACKET_SCHEMA = 'N07-frozen-E24-original-detail-point-cache-packet-v1'
OWNER_SHA = 'ef9a9f17d4fb1bc3464936095ef799021dd880f8b15488d558a871c08b6e1924'
N01_SCHEMA = 'ROOT-N01-uniform-point-supervision-gradient-probe-spec-v1'
N01_Q_SCHEMA = 'ROOT-N01-fixed-FIT-gradient-CPU-qualification-v1'
N01_Q_STATUS = 'ROOT_N01_FIXED_FIT_GRADIENT_CPU_QUALIFIED'
RAW_KEYS = ('receipt', 'phase', 'admission', 'phase_binding', 'bootstrap')
MODULES = ('native_loader_source', 'evaluation_source', 'micro_preflight_source',
           'initializer_source', 'training_source', 'native_helpers', 'renderer_source',
           'joint_losses_source', 'joint_preflight_entry_source')
RECIPE = dict(images=741, raw_GT=1661, native_queries=100, points_per_GT=1024,
              band_points=768, uniform_points=256, band_abs_logit=2., patch_size=7,
              patch_padding='replicate', feature_channel_chunk=16, no_augmentation=True,
              train_epochs=8, train_seeds=[202610120701, 202610120702], arms=['C', 'R'],
              head_parameters=40257, head_parameter_tensors=6, AdamW_lr=3e-4, AdamW_weight_decay=1e-4)


def require(ok, message):
    if not ok: raise RuntimeError(message)


def read(path):
    def reject(value): raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=reject)


def reference(path):
    path = Path(path).resolve(strict=True); digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def ref_shape(ref):
    require(type(ref) is dict and set(ref) == {'path', 'bytes', 'sha256'} and
            type(ref['path']) is str and Path(ref['path']).is_absolute() and
            type(ref['bytes']) is int and ref['bytes'] >= 0 and type(ref['sha256']) is str and
            len(ref['sha256']) == 64 and set(ref['sha256']) <= set('0123456789abcdef'), 'Malformed reference')


def bound(ref):
    ref_shape(ref); require(reference(ref['path']) == ref, 'Pinned input changed: ' + ref['path'])
    return Path(ref['path'])


def terminal(ref, phase, exit_code=0):
    ref_shape(ref); path = Path(ref['path'])
    require(type(phase) is str and phase and path.name == 'PARENT_RECEIPT.json' and
            path.parent.name == phase, 'Absolute original receipt/phase locator required')
    raw = read(path)
    require(raw['status'] == 'ACTUAL_OWNED_TERMINAL' and type(raw['returncode']) is int and raw['returncode'] == exit_code and
            type(raw['original_identity']) is dict and raw['original_identity'] and raw['root_wait_reaped'] is True and
            raw['held_handle_exit'] == dict(wait_state='WAIT_OBJECT_0', actual_exit_code=exit_code) and
            type(raw['held_handle_exit']['actual_exit_code']) is int and raw['numeric_birth_absence']['original_birth_absent'] is True and
            raw.get('error') is None and raw.get('abort') is None and raw['wait_errors'] is False and
            raw['GPU_observation_errors'] == [], 'Actual held/reaped terminal%d required' % exit_code)
    if 'phase' in raw: require(type(raw['phase']) is str and raw['phase'] == phase, 'Original RAW phase differs')
    sha, peak = raw['admission_sha256'], raw['whole_GPU_observed_peak_MiB']
    require(type(sha) is str and len(sha) == 64 and set(sha) <= set('0123456789abcdef') and
            type(peak) in (int, float) and math.isfinite(peak) and 0 <= peak <= 8192 and
            type(raw['whole_GPU_used_limit_MiB']) is int and raw['whole_GPU_used_limit_MiB'] == 8192 and
            raw['whole_leaf_VRAM_certified'] is False and raw['GPU_scope'] ==
            'whole GPU observed, not per-process WDDM allocation', 'Original RAW resource scope differs')
    return raw


def owner_join(row, raw, closure):
    owner, pb = read(bound(row['admission'])), read(bound(row['phase_binding']))
    bootstrap = bound(row['bootstrap']); runtime = Path(row['receipt']['path']).parent
    require(Path(row['phase_binding']['path']) == runtime/'PHASE_BINDING.json' and bootstrap == runtime/'bootstrap.py' and
            owner['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and owner['phase'] == row['phase'] and
            owner['driver_sha256'] == OWNER_SHA and owner['automatic_retry'] is False and owner['allowed_cpus'] == [0, 1] and
            owner['cuda_visible_devices'] in ('', '0') and owner['whole_GPU_used_limit_MiB'] == 8192 and
            raw['admission_sha256'] == row['admission']['sha256'] == pb['admission_sha256'] and
            pb['bootstrap_sha256'] == row['bootstrap']['sha256'] and pb['argv'][-1] == str(bootstrap) and
            pb['CPU_affinity'] == owner['allowed_cpus'] and pb['CUDA_visible_devices'] == owner['cuda_visible_devices'] and
            pb['automatic_retry'] is False, 'Original owner/admission/PB/bootstrap differs')
    entry, args = owner['entry'], owner['entry_args']
    require(entry in closure and owner['bound_source_sha256'].get(entry) == closure[entry]['sha256'], 'Original entry absent/unbound')
    tree = ast.parse(bootstrap.read_text(encoding='utf-8'))
    argv = [ast.literal_eval(node.value) for node in ast.walk(tree) if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'sys' and target.attr == 'argv'
        for target in node.targets)]
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and
             isinstance(node.func.value, ast.Name) and node.func.value.id == 'runpy' and node.func.attr == 'run_path']
    require(argv == [[entry]+args] and len(calls) == 1 and ast.literal_eval(calls[0].args[0]) == entry and
            {item.arg: ast.literal_eval(item.value) for item in calls[0].keywords} == {'run_name': '__main__'}, 'Original bootstrap differs')
    if '--spec' in args:
        require(args.count('--spec') == 1, 'Ambiguous original SPEC')
        path = args[args.index('--spec')+1]
        require(path in closure and owner['bound_source_sha256'].get(path) == closure[path]['sha256'], 'Original SPEC absent/unbound')
    if 'scientific_spec' in owner:
        ref = owner['scientific_spec']
        require(ref in closure.values() and owner['bound_source_sha256'].get(ref['path']) == ref['sha256'], 'Scientific SPEC absent/unbound')
    return owner


def load(ref, name):
    path = bound(ref); source = path.read_bytes()
    require(len(source) == ref['bytes'] and hashlib.sha256(source).hexdigest() == ref['sha256'], 'SOURCE bytes changed')
    module = types.ModuleType(name); module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def source_gate(path, spec):
    require(spec['schema'] == SCHEMA and type(spec['phase']) is str and spec['phase'] and
            type(spec['RAW_inputs']) is list and len(spec['RAW_inputs']) == 17 and
            len({row['receipt']['path'] for row in spec['RAW_inputs']}) == 17 and
            all(set(RAW_KEYS) <= set(row) for row in spec['RAW_inputs']), 'Exactly seventeen normal ancestor RAWs required')
    raws = [terminal(row['receipt'], row['phase']) for row in spec['RAW_inputs']]
    failed = terminal(spec['prior_failed_CPU_receipt'], Path(spec['prior_failed_CPU_receipt']['path']).parent.name, 1)
    require(spec['prior_failed_CPU_receipt']['path'] not in {row['receipt']['path'] for row in spec['RAW_inputs']}, 'Failed attempt cannot qualify an ancestor')
    # All eighteen terminal states above precede the first SHA, SOURCE import or scientific read.
    for ref in spec['sources']: ref_shape(ref)
    closure = {ref['path']: ref for ref in spec['sources']}
    require(len(closure) == len(spec['sources']), 'Unique complete input closure required')
    required = [spec[key] for key in ('entry_source', 'point_heads_source', 'context_GPU_spec', 'context_CPU_qualification',
                                     'FIT', 'prior_failed_CPU_receipt')]
    required += [row[key] for row in spec['RAW_inputs'] for key in RAW_KEYS if key != 'phase']
    require(all(ref in closure.values() for ref in required), 'All actual input refs must be sources-bound')
    for ref in spec['sources']: bound(ref)
    source, spec_ref = reference(__file__), reference(path)
    require(source == spec['entry_source'], 'Actual cache entry SOURCE differs')
    owners = [owner_join(row, raw, closure) for row, raw in zip(spec['RAW_inputs'], raws)]
    admission_path = Path(spec['GPU_admission_file']); require(admission_path.is_absolute(), 'Absolute current GPU admission required')
    admission_ref = reference(admission_path); admission = read(admission_path)
    require(admission['status'] == 'ROOT_OWNED_SINGLE_ATTEMPT_LOCAL_DIAGNOSTIC_PHASE' and admission['phase'] == spec['phase'] and
            admission['driver_sha256'] == OWNER_SHA and admission['scientific_spec'] == spec_ref and admission['entry'] == source['path'] and
            admission['entry_args'] == ['--spec', str(path), '--owner-execute'] and admission['allowed_cpus'] == [0, 1] and
            admission['cuda_visible_devices'] == '0' and admission['automatic_retry'] is False and admission['whole_GPU_used_limit_MiB'] == 8192 and
            type(spec['deadline_seconds']) is int and spec['deadline_seconds'] > 0 and
            admission['qualified_leaf_deadline_seconds'] == spec['deadline_seconds'] and
            admission['bound_source_sha256'].get(str(path)) == spec_ref['sha256'] and
            all(admission['bound_source_sha256'].get(ref['path']) == ref['sha256'] for ref in spec['sources']), 'Current owner/complete closure differs')
    context, q = read(bound(spec['context_GPU_spec'])), read(bound(spec['context_CPU_qualification']))
    require(context['schema'] == N01_SCHEMA and q['schema'] == N01_Q_SCHEMA and q['status'] == N01_Q_STATUS and
            q['GPU_spec'] == spec['context_GPU_spec'] and q['FIT'] == spec['FIT'] == context['FIT'] and
            q['images_checked'] == 32 and q['selected_parameter_tensors'] == 8 and q['selected_parameter_scalars'] == 263168 and
            q['local_M200_VJP_images_checked'] == 5 and q['optimizer_created'] is False and q['weights_updated'] is False,
            'Qualified original E24/FIT context differs; N01 does not qualify this new cache')
    for key in ('source', 'reader_source', 'spec', 'GPU_admission', 'parent_receipt', 'GPU_phase_binding', 'GPU_bootstrap', 'report', 'start', 'trace'):
        require(q[key] in closure.values(), 'N01 qualification primary lineage absent: '+key)
    require(all(ref in closure.values() for ref in context['sources']) and
            all(any(all(current[key] == row[key] for key in RAW_KEYS) for current in spec['RAW_inputs'])
                for row in context['RAW_inputs']), 'Original context/ancestor closure must be retained')
    require(q['source'] == context['entry_source'] and q['renderer_source'] == context['renderer_source'], 'Original N01 source/renderer differs')
    gpu_rows = [index for index, row in enumerate(spec['RAW_inputs']) if row['receipt'] == q['parent_receipt']]
    cpu_rows = [index for index, owner in enumerate(owners) if owner['entry'] == q['reader_source']['path'] and
                owner['entry_args'] == ['--spec', q['spec']['path'], '--owner-execute']]
    require(len(gpu_rows) == len(cpu_rows) == 1 and spec['RAW_inputs'][gpu_rows[0]]['admission'] == q['GPU_admission'] and
            spec['RAW_inputs'][gpu_rows[0]]['phase_binding'] == q['GPU_phase_binding'] and
            spec['RAW_inputs'][gpu_rows[0]]['bootstrap'] == q['GPU_bootstrap'], 'Actual N01 GPU/CPU owned ancestors required')
    require(spec['recipe'] == RECIPE and spec['allocator_limit_MiB'] == 4096 and spec['whole_GPU_limit_MiB'] == 8192 and
            spec['image_root'] == context['image_root'] and
            all(type(spec[key]) is int and 0 <= spec[key] < 2**63-741 for key in ('point_seed_base', 'matching_loss_seed_base')),
            'Locked paired cache population/point/resource recipe differs')
    for key in MODULES: require(context[key] in closure.values(), 'Inherited helper absent: '+key)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and all(os.environ.get(key) == '2' for key in
            ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')), 'Actual CUDA/thread environment differs')
    return context, closure, source, spec_ref, admission_ref, failed


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def imported_files(torch, np, cv2, CocoDataset, COCO, closure):
    """Bind actual additional decoder/array/import files, not version strings alone."""
    paths = dict(torch=Path(torch.__file__), numpy=Path(np.__file__),
                 numpy_core=Path(np.core._multiarray_umath.__file__), cv2=Path(cv2.__file__),
                 CocoDataset=Path(inspect.getsourcefile(CocoDataset)), COCO=Path(inspect.getsourcefile(COCO)))
    result = {}
    for role, path in paths.items():
        ref = reference(path); require(ref in closure.values(), 'Actual imported file absent from owner closure: '+role)
        result[role] = ref
    return result


@contextmanager
def observed_fields(torch, head):
    before = {name: (tuple(module._forward_pre_hooks.items()), tuple(module._forward_hooks.items())) for name, module in head.named_modules()}
    cache = dict(F=[], e=[], output=[]); handles = []
    def feature(module, args, result):
        require(isinstance(result, (tuple, list)) and len(result) == 2, 'Native pixel-decoder two outputs required')
        cache['F'].append(result[0]); return None
    def embedding(module, args, result): cache['e'].append(result); return None
    def output(module, args, result): cache['output'].append(result); return None
    try:
        handles = [head.pixel_decoder.register_forward_hook(feature), head.mask_embed.register_forward_hook(embedding),
                   head.register_forward_hook(output)]
        yield cache
        require(len(cache['F']) == len(cache['output']) == 1 and len(cache['e']) == 10, 'One original loss forward/ten embeddings required')
        classes, masks = cache['output'][0]
        require(len(classes) == len(masks) == 10, 'All ten native stages required')
        F, e, M, C = cache['F'][0], cache['e'][-1], masks[-1], classes[-1]
        for value, shape in ((F, (1,256,200,200)), (e, (1,100,256)), (M, (1,100,200,200)), (C, (1,100,2))):
            require(isinstance(value, torch.Tensor) and value.dtype == torch.float32 and tuple(value.shape) == shape and
                    not value.requires_grad and bool(torch.isfinite(value).all()), 'Finite frozen native field differs')
        require(torch.equal(torch.einsum('bqc,bchw->bqhw', e, F), M), 'Native final F/e projection differs')
    finally:
        for handle in handles: handle.remove()
        require(before == {name: (tuple(module._forward_pre_hooks.items()), tuple(module._forward_hooks.items()))
                           for name, module in head.named_modules()}, 'Passive hook inventory not restored')


def matched_queries(matches, mapping, G):
    require(len(matches) == 10 and 1 <= G <= 100 and sorted(mapping) == list(range(G)), 'Positive complete original FIT GT/mapping required')
    for row in matches:
        queries, targets = row['positive_query_indices'], row['assigned_GT_indices']
        require(row['positives'] == len(queries) == len(set(queries)) == G and all(type(q) is int and 0 <= q < 100 for q in queries) and
                all(type(i) is int for i in targets) and sorted(targets) == list(range(G)), 'Original one-to-one matching must retain every GT')
    output = [None]*G
    for query, target in zip(matches[-1]['positive_query_indices'], matches[-1]['assigned_GT_indices']): output[mapping[target]] = query
    require(None not in output and len(set(output)) == G, 'Final query-to-raw annotation assignment differs')
    return output


def sample_points(torch, L, seed):
    require(L.device.type == 'cpu' and L.dtype == torch.float32 and L.ndim == 3 and bool(torch.isfinite(L).all()), 'CPU original matched logits required')
    rng = torch.Generator(device='cpu'); rng.manual_seed(seed)
    area = L.shape[1]*L.shape[2]; rows = []; band_counts = []
    for row in L:
        band = torch.nonzero(row.flatten().abs() <= 2, as_tuple=False).flatten(); band_counts.append(len(band))
        if len(band):
            near = band[torch.randint(len(band), (768,), generator=rng)]
            uniform = torch.randint(area, (256,), generator=rng)
            indices = torch.cat((near, uniform))
        else: indices = torch.randint(area, (1024,), generator=rng)
        rows.append(indices)
    return torch.stack(rows), band_counts


def original_patches(torch, np, gray, indices):
    require(gray.dtype == np.uint8 and gray.ndim == 2 and indices.device.type == 'cpu' and indices.dtype == torch.int64,
            'Actual original uint8 JPEG grayscale and CPU indices required')
    H, W = gray.shape; index = indices.numpy(); padded = np.pad(gray, ((3,3),(3,3)), mode='edge')
    dy, dx = np.meshgrid(np.arange(-3,4), np.arange(-3,4), indexing='ij')
    y, x = index//W, index%W
    patches = padded[y[...,None]+3+dy.reshape(49), x[...,None]+3+dx.reshape(49)].astype(np.float32)/np.float32(255.)
    require(patches.shape == tuple(indices.shape)+(49,) and np.isfinite(patches).all() and ((patches >= 0)&(patches <= 1)).all(), 'Real JPEG patches differ')
    return torch.from_numpy(patches.copy())


def sampled_features(torch, renderer, F, identity, indices):
    G, P = indices.shape; on_device = indices.to(F.device).reshape(-1); parts = []
    for first in range(0, 256, 16):
        original = renderer(F[:,first:first+16], identity)
        require(tuple(original.shape) == (16,)+tuple(identity['original_shape']), 'Actual two-stage original F projection differs')
        values = original.flatten(1)[:,on_device].transpose(0,1).reshape(G,P,16)
        parts.append(values.cpu().clone()); del original, values
    result = torch.cat(parts, dim=2).contiguous()
    require(tuple(result.shape) == (G,1024,256) and bool(torch.isfinite(result).all()), 'Sampled original coarse features differ')
    return result


def cache_image(torch, np, cv2, modules, model, dataset, coco, spec, context, closure, index):
    native, legacy, pref, initializer, common, old, diag, joint, preflight = modules
    info = dataset.get_data_info(index)
    image = reference(info['img_path']); require(image in closure.values(), 'Every actual FIT image must be owner-bound before loading')
    data, stripped, gray800, gray, identity = old.load_image(model, dataset, coco, info)
    del stripped, gray800
    require(identity['image'] == image, 'Actual image loader identity differs'); identity['dataset_index'] = index
    target = pref.check_targets(torch, data, identity)
    raw_GT, mapping, validator = preflight.raw_GT_validator(torch, np, cv2, coco, data, identity)
    external = pref.rng_snapshot(torch); head = model.panoptic_head
    seed = spec['matching_loss_seed_base']+index; point_seed = spec['point_seed_base']+index
    with torch.no_grad(), torch.random.fork_rng(devices=[0]):
        torch.random.default_generator.manual_seed(seed); torch.cuda.manual_seed(seed)
        with joint._observe_processed_GT(torch, head, identity['ann_ids'], raw_GT, mapping, identity, validator) as gt_observation:
            with old.observe_targets(head) as matches:
                with observed_fields(torch, head) as fields:
                    losses = model.loss(data['inputs'], data['data_samples'])
        require(set(losses) == set(old.ATOMIC_KEYS) and len(losses) == 30, 'Exactly thirty original native loss atoms required')
        loss_values = {}
        for name, value in losses.items():
            items = list(value) if isinstance(value, (tuple,list)) else [value]
            require(items and all(isinstance(item,torch.Tensor) and not item.requires_grad and bool(torch.isfinite(item).all()) for item in items),
                    'Original frozen native loss nonfinite/live')
            loss_values[name] = float(sum(item.mean() for item in items).cpu())
        queries = matched_queries(matches, mapping, len(raw_GT))
        F, e = fields['F'][0], fields['e'][-1]
        classes, masks = fields['output'][0]; M, C = masks[-1], classes[-1]
        query_tensor = torch.tensor(queries, device=M.device, dtype=torch.int64)
        renderer = partial(diag.render_original, torch)
        L = renderer(M[:,query_tensor], identity).cpu().clone()
        require(tuple(L.shape) == (len(raw_GT),)+tuple(identity['original_shape']), 'Original matched native raster differs')
        indices, band_counts = sample_points(torch, L, point_seed)
        features = sampled_features(torch, renderer, F, identity, indices)
        logits = L.flatten(1).gather(1,indices).contiguous()
        targets = raw_GT.cpu().flatten(1).gather(1,indices).contiguous()
        patches = original_patches(torch,np,gray,indices)
        embeddings = e[0,query_tensor].cpu().clone().contiguous()
        packet = dict(schema=PACKET_SCHEMA, dataset_index=index, identity=identity, query_ids=torch.tensor(queries,dtype=torch.int64),
                      F_samples=features, query_embeddings=embeddings, native_logits=logits, patches=patches, targets=targets,
                      point_indices=indices, native_class_logits=C.cpu().clone(), raw_GT_digest=pref.digest_tensor(raw_GT),
                      raw_GT_area=raw_GT.flatten(1).sum(1).cpu().tolist(), band_counts=band_counts,
                      point_seed=point_seed, matching_loss_seed=seed, native_stage_matches=matches, native_loss_atoms=loss_values,
                      actual_raw_processed_GT_correspondence=gt_observation, processed_target_proof=target,
                      original_JPEG_gray_digest=dict(shape=list(gray.shape),dtype=str(gray.dtype),sha256=hashlib.sha256(gray.tobytes()).hexdigest()),
                      replacement=True, duplicates_preserved=True, GT_selected_points=False, features_are_coarse_native_bilinear=True,
                      patch_is_original_uint8_JPEG_intensity=True, patch_has_SAR_complex_measurements=False)
    require(pref.tree_equal(torch,external,pref.rng_snapshot(torch)), 'External RNG changed during frozen matching/cache')
    for name in ('F_samples','query_embeddings','native_logits','patches','native_class_logits'):
        require(packet[name].device.type == 'cpu' and packet[name].dtype == torch.float32 and
                not packet[name].requires_grad and bool(torch.isfinite(packet[name]).all()), 'Detached finite FP32 cache required')
    require(targets.dtype == torch.bool and indices.dtype == torch.int64 and tuple(targets.shape) == tuple(indices.shape) == (len(raw_GT),1024),
            'Hard original sampled labels/indices differ')
    require(not any(param.requires_grad or param.grad is not None for param in model.parameters()) and not model.training,
            'Frozen teacher must have no trainable parameters/gradients')
    row = dict(dataset_index=index, image_id=identity['image_id'], identity=identity, raw_GT=len(raw_GT), query_ids=queries,
               points=len(raw_GT)*1024, band_counts=band_counts, empty_band_instances=sum(n == 0 for n in band_counts),
               sampled_foreground_by_GT=targets.sum(1).tolist(), duplicate_points_by_GT=[1024-int(torch.unique(v).numel()) for v in indices],
               matching_loss_seed=seed, point_seed=point_seed, all_ten_matching_stages=len(matches),
               cache_digests={name: pref.digest_tensor(packet[name]) for name in
               ('query_ids','F_samples','query_embeddings','native_logits','patches','targets','point_indices','native_class_logits')},
               raw_GT_ids=identity['ann_ids'], native_loss_atoms=30, one_teacher_forward=True, backward_calls=0,
               optimizer_created=False, weights_updated=False, external_RNG_restored=True)
    return row, packet


def run(spec_path):
    started = time.monotonic(); path = Path(spec_path)
    require(path.is_absolute(), 'Absolute ROOT-authored SPEC required'); path = path.resolve(strict=True)
    spec = read(path); context, closure, source, spec_ref, admission_ref, failed = source_gate(path,spec)
    output = Path(spec['output_directory']); require(output.is_absolute() and not output.exists(), 'Fresh absolute cache output required')
    import psutil
    require(psutil.Process().cpu_affinity() == [0,1], 'Actual CPU01 required before scientific imports')
    import torch
    require(torch.__version__ == '2.0.1+cu118' and not torch.cuda.is_initialized(), 'Original Torch/cold CUDA required')
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    legacy = load(context['evaluation_source'],'_N07_original_E24_qualification')
    teachers = legacy.qualified_training(context)
    import numpy as np
    import cv2
    from mmengine import Config
    from mmengine.model import revert_sync_batchnorm
    from mmengine.registry import init_default_scope
    from mmdet.registry import MODELS
    from mmdet.utils import register_all_modules
    from mmdet.datasets import CocoDataset
    from mmdet.datasets.api_wrappers import COCO
    imported = imported_files(torch,np,cv2,CocoDataset,COCO,closure)
    modules = [load(context[key],'_N07_'+key) for key in MODULES]
    native, legacy, pref, initializer, common, old, diag, joint, preflight = modules; old.torch = torch
    register_all_modules(init_default_scope=False); init_default_scope('mmdet')
    construction = dict(teachers['native0']['training_spec'], training_source=context['training_source'],
                        checkpointmap=context['checkpointmap'], qualificationmap=context['qualificationmap'],
                        order_seed=native.ORDER_SEED, phase_loss_seed_base=native.LOSS_SEED_BASE)
    model, anchor, policy, loading = native.prepare_training_model(torch,pref,initializer,common,MODELS,revert_sync_batchnorm,Config,
                                                                  tuple(CocoDataset.METAINFO['classes']),construction,teachers['native0'])
    native.imported_library_gate(torch,model,context)
    require(Path(inspect.getsourcefile(torch.nn.functional.interpolate)).resolve() == bound(context['interpolation_SOURCE']).resolve(),
            'Original interpolation SOURCE differs')
    model.requires_grad_(False); model.eval(); model.backbone.with_cp=False
    require(len(anchor) == len(model.state_dict()) == 610 and len(tuple(model.parameters())) == 451 and
            not any(param.requires_grad or param.grad is not None for param in model.parameters()), 'Strict frozen native610/451 required')
    dataset = CocoDataset(ann_file=str(bound(spec['FIT'])),data_prefix=dict(img=spec['image_root']),metainfo=dict(classes=('ship',)),
                          pipeline=[],filter_cfg=None,test_mode=True)
    coco = COCO(str(bound(spec['FIT'])))
    require(len(dataset) == len(coco.imgs) == 741 and len(coco.anns) == 1661 and set(coco.cats) == {1}, 'Complete FIT741/1661 required')
    image_ids = [dataset.get_data_info(i)['img_id'] for i in range(741)]
    require(image_ids == list(coco.imgs) and len(set(image_ids)) == 741, 'Canonical complete FIT image order required')
    require(all(str(Path(dataset.get_data_info(i)['img_path']).resolve(strict=True)) in closure for i in range(741)), 'All741 actual images absent from closure')
    torch.cuda.init(); torch.cuda.set_per_process_memory_fraction(min(1.,4096*1024**2/torch.cuda.get_device_properties(0).total_memory),0)
    torch.cuda.reset_peak_memory_stats(0); model.to('cuda:0'); output.mkdir()
    write(output/'START.json',dict(schema='N07-point-cache-start-v1',source=source,spec=spec_ref,GPU_admission=admission_ref,phase=spec['phase'],
          context_GPU_spec=spec['context_GPU_spec'],context_CPU_qualification=spec['context_CPU_qualification'],teacher_loading=loading,
          frozen_state_entries=610,frozen_parameter_tensors=451,native_weights_epoch=24,recipe=RECIPE,
          point_heads_source=spec['point_heads_source'],imported_files=imported,optimizer_created=False,weights_updated=False,all_GT_training_images_only=True,
          zero_GT_carrier_qualification_claimed=False,no_DEV=True,no_TEST=True,no_AP=True,budget_is_unmeasured_prelock=True))
    packet_refs=[]; rows=[]; total_GT=0
    with (output/'TRACE.jsonl').open('x',encoding='utf-8',newline='\n') as trace:
        for index in range(741):
            require(time.monotonic()-started < spec['deadline_seconds'], 'N07 prelocked cache deadline exceeded; no fallback/retry')
            row, packet = cache_image(torch,np,cv2,modules,model,dataset,coco,spec,context,closure,index)
            require(row['image_id'] == image_ids[index], 'Canonical FIT row changed')
            packet.update(source=source,spec=spec_ref,phase=spec['phase'],teacher_checkpoint=context['checkpointmap']['native0'])
            target = output/('PACKET_%04d.pth'%index)
            with target.open('xb') as stream: torch.save(packet,stream)
            row['packet']=reference(target); packet_refs.append(row['packet']); rows.append(row); total_GT+=row['raw_GT']
            trace.write(json.dumps(row,allow_nan=False)+'\n'); trace.flush(); del packet
            print(json.dumps(dict(cached_images=index+1,raw_GT=total_GT)),flush=True)
    final=pref.cpu_tree(torch,model.state_dict())
    require(total_GT == 1661 and final.keys() == anchor.keys() and all(torch.equal(final[name],anchor[name]) for name in anchor),
            'Full FIT population/unchanged native610 state differs')
    require(not any(param.grad is not None for param in model.parameters()), 'No native gradient/update permitted')
    peak=torch.cuda.max_memory_allocated(0); model.cpu(); del model,anchor,final; gc.collect(); cleanup=pref.clear_gpu(torch)
    require(cleanup['allocated_after'] == 0 and psutil.Process().cpu_affinity() == [0,1] and
            torch.get_num_threads() == torch.get_num_interop_threads() == 2, 'Final cleanup/resources differ')
    for ref in spec['sources']: bound(ref)
    require(reference(path) == spec_ref and reference(__file__) == source and reference(spec['GPU_admission_file']) == admission_ref,
            'Final SPEC/SOURCE/admission changed')
    report=dict(schema='N07-frozen-E24-point-cache-report-v1',status=STATUS,source=source,spec=spec_ref,GPU_admission=admission_ref,phase=spec['phase'],
                start=reference(output/'START.json'),trace=reference(output/'TRACE.jsonl'),packet_refs=packet_refs,rows=rows,images=741,raw_GT=1661,
                point_count=1661*1024,recipe=RECIPE,imported_files=imported,all_raw_GT_retained=True,one_teacher_forward_per_image=True,teacher_forwards=741,
                native610_unchanged_SOURCE_observed=True,native_gradient_tensors=0,optimizer_created=False,weights_updated=False,backward_calls=0,
                independent_CPU_cache_qualification=False,heads_trained=False,head_epochs=0,AP_computed=False,no_DEV=True,no_TEST=True,
                budget_is_unmeasured_prelock=True,elapsed_seconds=time.monotonic()-started,torch_peak_allocated_bytes=peak,cleanup=cleanup,
                evidence_limits=['Original JPEG grayscale detail only, not complex SAR or physical scattering features.',
                'F/e are coarse native fields rendered through200->800->crop->original; interpolation does not create image detail.',
                'Cache stage freezes one E24 teacher and fixes original native matching/GT-free point sampling for every future arm and seed.',
                'The original native loader/GT validator requires positive GT per FIT image; empty-band sampling is supported, full zero-GT carrier is not qualified.',
                'CPU cache replay, head numerical qualification, C/R training and all100 fixed-score DEV evaluation remain separate ROOT prerequisites.',
                'Allocator4096/whole8192 and deadline are preset upper bounds, not measured suitability or a success guarantee.'])
    write(output/'REPORT.json',report)
    print(json.dumps(dict(status=STATUS,report=reference(output/'REPORT.json'))),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--spec',required=True)
    parser.add_argument('--owner-execute',action='store_true'); args=parser.parse_args()
    require(args.owner_execute,'Explicit single ROOT-owned execution required'); run(args.spec)
