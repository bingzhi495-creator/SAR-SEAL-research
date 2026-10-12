"""ROOT-owned CPU-only strict official/ship initializer inspection; no inference.

Only canonical state_dict and a uniform module. prefix are accepted. Class-row
selection is factual, not proof of the public pretraining recipe or superiority.
Scientific imports and checkpoint reads occur only after the terminal gate.
"""
import argparse
import ast
import copy
import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path

OWNER_SCHEMA = 'ROOT-native-coco-comparator-initializer-CPU-owner-spec-v1'
MODELS = ('mask2former', 'condinst', 'cascade_maskrcnn')
EXPECTED_CP = {
    'mask2former': 'E:/SAR/seal/pretrained/m2f_r50_coco.pth',
    'condinst': 'E:/SAR/seal/pretrained/condinst_r50_coco.pth',
    'cascade_maskrcnn': 'E:/SAR/seal/pretrained/cascade_maskrcnn_r50_3x_coco.pth'}
CONFIG_NAMES = {
    'mask2former': 'mask2former_r50_8xb2-lsj-50e_coco.py',
    'condinst': 'condinst_r50_fpn_ms-poly-90k_coco_instance.py',
    'cascade_maskrcnn': 'cascade-mask-rcnn_r50_fpn_ms-3x_coco.py'}
RECIPE_LABELS = {'mask2former': 'R50 50-epoch large-scale-jitter COCO instance config',
                 'condinst': 'R50 FPN multiscale/poly 90000-iteration COCO instance config',
                 'cascade_maskrcnn': 'R50 FPN multiscale 3x PyTorch-style COCO instance config'}


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def reference(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def bound(row):
    actual = reference(row['path'])
    need(actual == {key: row[key] for key in ('path', 'bytes', 'sha256')},
         'Pinned input changed: ' + str(row['path']))
    return Path(actual['path'])


def terminal_gate(owner):
    # Reading this small receipt is permitted before imports/CP hashing.
    receipt_path = Path(owner['owner_receipt']['path'])
    need(receipt_path.is_absolute(), 'Absolute owner receipt required')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8-sig'))
    need(receipt['status'] == 'ACTUAL_OWNED_TERMINAL' and
         type(receipt['returncode']) is int and receipt['returncode'] == 0 and
         receipt['root_wait_reaped'] is True and receipt['original_identity'] is not None and
         receipt['held_handle_exit']['wait_state'] == 'WAIT_OBJECT_0' and
         receipt['held_handle_exit']['actual_exit_code'] == 0 and
         receipt['numeric_birth_absence']['original_birth_absent'] is True and
         receipt['error'] is None and receipt['abort'] is None and
         receipt['wait_errors'] is False and receipt['GPU_observation_errors'] == [],
         'STOP: actual exit0/held reap/birth absence/error-free terminal required before scientific access')
    need(receipt['admission_sha256'] == owner['admission']['sha256'],
         'STOP: receipt does not bind the supplied admission')
    # Only after the actual terminal checks may pinned files be hashed.
    bound(owner['owner_receipt'])
    bound(owner['admission'])
    training_spec = json.loads(bound(owner['training_spec']).read_text(encoding='utf-8-sig'))
    if 'phase' in receipt:
        need(receipt['phase'] == training_spec['phase'], 'Training phase/receipt differs')
    if 'root_training_qualification' in owner:
        qualified = json.loads(bound(owner['root_training_qualification']).read_text(encoding='utf-8-sig'))
        need(qualified['schema'] == 'ROOT-native-same-budget-adapter-CPU-qualification-v1' and
             qualified['status'] == 'ROOT_SAME_BUDGET_ADAPTER_SEGMENT_CPU_QUALIFIED' and
             qualified['spec'] == owner['training_spec'], 'ROOT segment CPU qualification differs')
    for row in owner['sources']:
        bound(row)
    return receipt, training_spec


def config_closure(path, seen=None):
    seen = set() if seen is None else seen
    path = Path(path).resolve(strict=True)
    if path in seen:
        return []
    seen.add(path)
    tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    rows = [reference(path)]
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(isinstance(target, ast.Name) and
                                                     target.id == '_base_' for target in statement.targets):
            bases = ast.literal_eval(statement.value)
            bases = [bases] if isinstance(bases, str) else bases
            need(isinstance(bases, (tuple, list)) and all(isinstance(value, str) for value in bases),
                 'Explicit local config inheritance required')
            for value in bases:
                rows.extend(config_closure(path.parent / value, seen))
    return rows


def disable_initialization(value):
    if isinstance(value, dict):
        # Preserve ConfigDict and other dict-derived constructor interfaces.
        disabled = copy.deepcopy(value)
        for key, item in list(disabled.items()):
            disabled[key] = None if key in ('init_cfg', 'pretrained') else disable_initialization(item)
        return disabled
    if isinstance(value, list):
        return [disable_initialization(item) for item in value]
    if isinstance(value, tuple):
        return tuple(disable_initialization(item) for item in value)
    return value


def build_cpu(torch, registry, revert, model_config):
    before = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(2026100823)
        # No init_weights, runner, checkpoint loader, or pretrained download.
        model = revert(registry.build(disable_initialization(copy.deepcopy(model_config))))
        model.requires_grad_(False).eval()
    need(torch.equal(before, torch.get_rng_state()) and not torch.cuda.is_initialized(),
         'CPU-only constructor changed external RNG or initialized CUDA')
    need(all(value.device.type == 'cpu' for value in model.state_dict().values()), 'CPU model required')
    return model


def canonical_state(torch, checkpoint):
    need(isinstance(checkpoint, dict) and isinstance(checkpoint.get('state_dict'), dict),
         'Canonical checkpoint.state_dict required; no plain/EMA fallback')
    state = checkpoint['state_dict']
    need(bool(state) and all(isinstance(name, str) and isinstance(value, torch.Tensor)
                            for name, value in state.items()), 'Unknown/non-tensor state entry')
    prefixed = [name.startswith('module.') for name in state]
    need(all(prefixed) or not any(prefixed), 'Mixed module. prefixes are forbidden')
    normalized = {name[7:] if all(prefixed) else name: value for name, value in state.items()}
    need(len(normalized) == len(state) and all(value.device.type == 'cpu' for value in normalized.values()),
         'Canonical CPU key normalization failed')
    return normalized, bool(all(prefixed))


def minimal_meta(checkpoint, classes):
    meta = checkpoint.get('meta')
    meta = meta if isinstance(meta, dict) else {}
    labels = {}
    dataset_meta = meta.get('dataset_meta')
    if isinstance(dataset_meta, dict) and 'classes' in dataset_meta:
        labels['dataset_meta.classes'] = dataset_meta['classes']
    if 'CLASSES' in meta:
        labels['CLASSES'] = meta['CLASSES']
    summaries = {}
    for key, value in labels.items():
        valid = isinstance(value, (tuple, list)) and all(isinstance(item, str) for item in value)
        summaries[key] = dict(count=len(value) if valid else None,
                              exactly_official_COCO80=bool(valid and tuple(value) == classes))
    confirmed = bool(summaries) and all(row['exactly_official_COCO80'] for row in summaries.values())
    factual = {key: meta[key] for key in ('epoch', 'iter', 'mmdet_version', 'mmcv_version')
               if key in meta and isinstance(meta[key], (str, int, float, bool, type(None))) and
               (not isinstance(meta[key], float) or math.isfinite(meta[key]))}
    if isinstance(meta.get('config'), str):
        factual['config_text_characters'] = len(meta['config'])
        factual['config_text_sha256'] = hashlib.sha256(meta['config'].encode('utf-8')).hexdigest()
    return dict(top_level_keys=sorted(map(str, checkpoint)), meta_keys=sorted(map(str, meta)),
                class_metadata=summaries, factual_meta=factual, COCO80_class_order_metadata_matches=confirmed,
                boat_original_index=8, original_boat_label=classes[8],
                public_checkpoint_origin_and_training_recipe_verified=False)


def strict_load(torch, model, state):
    expected = model.state_dict()
    missing, extra = sorted(set(expected) - set(state)), sorted(set(state) - set(expected))
    need(not missing and not extra, 'State names incompatible: missing=' + repr(missing[:16]) +
         ' unexpected=' + repr(extra[:16]))
    for name, target in expected.items():
        saved = state[name]
        need(saved.shape == target.shape and saved.dtype == target.dtype,
             'State shape/dtype incompatible: ' + name + ' saved=' + str(tuple(saved.shape)) +
             '/' + str(saved.dtype) + ' expected=' + str(tuple(target.shape)) + '/' + str(target.dtype))
        need(bool(torch.isfinite(saved).all()), 'Nonfinite checkpoint tensor: ' + name)
    loaded = model.load_state_dict(state, strict=True)
    need(not loaded.missing_keys and not loaded.unexpected_keys and
         all(torch.equal(value, state[name]) for name, value in model.state_dict().items()),
         'Strict CPU load was not tensor-exact')
    return dict(state_entries=len(expected), parameter_tensors=len(tuple(model.parameters())),
                parameter_scalars=sum(value.numel() for value in model.parameters()),
                buffer_tensors=len(tuple(model.buffers())), all_state_finite=True,
                key_shape_dtype_equal=True, strict_load_all_tensors_TorchEqual=True)


def ship_state(torch, kind, config, model, state):
    ship_config, transformed, selections = copy.deepcopy(config), dict(state), {}
    def select(name, expected_rows, rows):
        need(name in state and state[name].ndim >= 1 and state[name].shape[0] == expected_rows,
             'Exact class-row shape required: ' + name)
        index = torch.tensor(rows, dtype=torch.long)
        transformed[name] = state[name].index_select(0, index).clone()
        selections[name] = dict(original_shape=list(state[name].shape), selected_rows=list(rows),
                                 ship_shape=list(transformed[name].shape))
    if kind == 'mask2former':
        need(model.panoptic_head.num_classes == 80 and model.panoptic_fusion_head.num_classes == 80,
             '80-instance-class M2F required; 134-row panoptic checkpoint is incompatible')
        ship_config['panoptic_head']['num_things_classes'] = 1
        ship_config['panoptic_head']['num_stuff_classes'] = 0
        ship_config['panoptic_head']['loss_cls']['class_weight'] = [1.0, 0.1]
        ship_config['panoptic_fusion_head']['num_things_classes'] = 1
        ship_config['panoptic_fusion_head']['num_stuff_classes'] = 0
        for suffix in ('weight', 'bias'):
            select('panoptic_head.cls_embed.' + suffix, 81, [8, 80])
    elif kind == 'condinst':
        need(model.bbox_head.num_classes == 80 and model.bbox_head.loss_cls.use_sigmoid,
             'Official80-class sigmoid CondInst head required')
        ship_config['bbox_head']['num_classes'] = 1
        for suffix in ('weight', 'bias'):
            select('bbox_head.conv_cls.' + suffix, 80, [8])
    else:
        boxes = ship_config['roi_head']['bbox_head']
        masks = ship_config['roi_head']['mask_head']
        need(len(boxes) == len(model.roi_head.bbox_head) == len(model.roi_head.mask_head) == 3,
             'Exact three-stage Cascade bbox/mask heads required')
        mask_configs = masks if isinstance(masks, list) else [masks]
        for cfg in mask_configs:
            need(cfg['num_classes'] == 80, 'Cascade mask head must originally have80 classes')
            cfg['num_classes'] = 1
        for stage, (cfg, bbox) in enumerate(zip(boxes, model.roi_head.bbox_head)):
            need(cfg['num_classes'] == bbox.num_classes == 80 and
                 cfg['reg_class_agnostic'] == bbox.reg_class_agnostic and
                 not model.roi_head.mask_head[stage].class_agnostic,
                 'Cascade declared regression/mask semantics differ')
            cfg['num_classes'] = 1
            for suffix in ('weight', 'bias'):
                select('roi_head.bbox_head.' + str(stage) + '.fc_cls.' + suffix, 81, [8, 80])
                select('roi_head.mask_head.' + str(stage) + '.conv_logits.' + suffix, 80, [8])
                reg_key = 'roi_head.bbox_head.' + str(stage) + '.fc_reg.' + suffix
                if bbox.reg_class_agnostic:
                    need(state[reg_key].shape[0] == 4, 'Class-agnostic4 regression must stay unchanged')
                else:
                    # Only a matching declared class-specific80 config can
                    # reach this branch after the preceding full strict load.
                    select(reg_key, 320, [32, 33, 34, 35])
    allowed = set(selections)
    actual = {name for name in state if state[name].shape != transformed[name].shape or
              not torch.equal(state[name], transformed[name])}
    need(actual == allowed and set(transformed) == set(state) and
         all(torch.equal(transformed[name], state[name]) for name in set(state) - allowed),
         'Changes outside the exact allowed classifier/boat regression rows')
    return ship_config, transformed, selections


def inspect_one(torch, registry, revert, Config, classes, kind, entry):
    row = dict(model=kind, checkpoint=entry['checkpoint'], official_config=entry['config'],
               declared_config_recipe=RECIPE_LABELS[kind], random_missing_key_fallback=False,
               inspection_stage='INPUT_BINDINGS', official80_strict_load_completed=False,
               ship1_strict_load_completed=False)
    try:
        cp_path = bound(entry['checkpoint'])
        cfg_path = bound(entry['config'])
        need(cp_path == Path(EXPECTED_CP[kind]).resolve() and cfg_path.name == CONFIG_NAMES[kind] and
             '.mim' in cfg_path.parts, 'Only the declared local checkpoint and installed official config are allowed')
        row['official_config_source_closure'] = config_closure(cfg_path)
        row['inspection_stage'] = 'OFFICIAL_CONFIG_LOAD'
        cfg = Config.fromfile(str(cfg_path))
        need(cfg.model.backbone.style == 'pytorch', 'PyTorch-style R50 config required; no Caffe conversion')
        row['inspection_stage'] = 'CHECKPOINT_CPU_LOAD'
        checkpoint = torch.load(str(cp_path), map_location='cpu')
        row['metadata'] = minimal_meta(checkpoint, classes) if isinstance(checkpoint, dict) else dict(container_type=type(checkpoint).__name__)
        row['inspection_stage'] = 'CANONICAL_STATE_DICT'
        state, prefix = canonical_state(torch, checkpoint)
        row['consistent_module_prefix_removed'] = prefix
        row['checkpoint_state_schema'] = {name: dict(shape=list(value.shape), dtype=str(value.dtype))
                                          for name, value in state.items()}
        row['inspection_stage'] = 'OFFICIAL80_CONSTRUCTION'
        official = build_cpu(torch, registry, revert, cfg.model)
        row['inspection_stage'] = 'OFFICIAL80_STRICT_LOAD'
        row['official80'] = strict_load(torch, official, state)
        row['official80_strict_load_completed'] = True
        row['inspection_stage'] = 'SHIP_ROW_MAPPING'
        ship_cfg, selected, row['allowed_changed_keys'] = ship_state(torch, kind, cfg.model, official, state)
        row['inspection_stage'] = 'SHIP1_CONSTRUCTION'
        ship = build_cpu(torch, registry, revert, ship_cfg)
        row['inspection_stage'] = 'SHIP1_STRICT_LOAD'
        row['ship1'] = strict_load(torch, ship, selected)
        row['ship1_strict_load_completed'] = True
        row['inspection_stage'] = 'FINAL_CPU_STATE_CHECKS'
        row['all_other_state_tensors_unchanged'] = True
        row['all_parameters_frozen_and_grad_None'] = all(not value.requires_grad and value.grad is None
                                                        for current in (official, ship) for value in current.parameters())
        need(row['all_parameters_frozen_and_grad_None'] and not torch.cuda.is_initialized(),
             'Unexpected gradients or CUDA initialization')
        confirmed = row['metadata']['COCO80_class_order_metadata_matches']
        row['status'] = ('TECHNICAL_SHAPES_AND_COCO_CLASS_METADATA_CONFIRMED' if confirmed else
                         'TECHNICAL_SHAPES_NOT_PROVENANCE_CONFIRMED')
        row['boat_class_semantics_confirmed_by_metadata_only'] = confirmed
        row['strict_official80_and_ship1_compatible'] = True
        row['initializer_saved'] = row['inference_executed'] = row['loss_executed'] = False
        row['checkpoint_compatibility_failure_proven'] = False
        row['inspection_stage'] = 'COMPLETE'
    except Exception as error:
        failed_stage = row['inspection_stage']
        construction_error = failed_stage in ('OFFICIAL80_CONSTRUCTION', 'SHIP1_CONSTRUCTION')
        row.update(status=('TECHNICAL_CONSTRUCTION_ERROR_NOT_CHECKPOINT_COMPATIBILITY_RESULT'
                           if construction_error else 'INCOMPATIBLE'),
                   failed_stage=failed_stage, strict_official80_and_ship1_compatible=False,
                   checkpoint_compatibility_failure_proven=(failed_stage in
                       ('OFFICIAL80_STRICT_LOAD', 'SHIP1_STRICT_LOAD')),
                   error_type=type(error).__name__, error=str(error)[:3000])
    return row


def run(owner_file, acceptance_file):
    started = time.monotonic()
    need(Path(owner_file).is_absolute() and Path(acceptance_file).is_absolute(), 'Absolute paths required')
    acceptance = Path(acceptance_file)
    need(not acceptance.exists(), 'Fresh acceptance path required')
    owner_path = Path(owner_file).resolve(strict=True)
    owner = json.loads(owner_path.read_text(encoding='utf-8-sig'))
    need(owner['schema'] == OWNER_SCHEMA and set(owner['initializers']) == set(MODELS),
         'Exact CPU owner manifest and three initializer refs required')
    receipt, training_spec = terminal_gate(owner)
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Empty CUDA_VISIBLE_DEVICES required')
    import psutil
    import torch
    from mmengine import Config
    from mmengine.model import revert_sync_batchnorm
    from mmengine.registry import init_default_scope
    from mmdet.registry import MODELS as registry
    from mmdet.utils import register_all_modules
    from mmdet.datasets import CocoDataset
    need(psutil.Process().cpu_affinity() == [0, 1] and not torch.cuda.is_initialized(),
         'CPU01 affinity and uninitialized CUDA required')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    register_all_modules(init_default_scope=False)
    init_default_scope('mmdet')
    classes = tuple(CocoDataset.METAINFO['classes'])
    need(len(classes) == 80 and classes[8] == 'boat' and not torch.cuda.is_initialized(),
         'Installed official COCO80 class order required')
    external_rng = torch.get_rng_state().clone()
    results = {}
    for kind in MODELS:
        need(time.monotonic() - started < owner.get('deadline_seconds', 600), 'CPU owner deadline exceeded')
        results[kind] = inspect_one(torch, registry, revert_sync_batchnorm, Config, classes,
                                    kind, owner['initializers'][kind])
        gc.collect()
        need(not torch.cuda.is_initialized(), 'CPU qualification initialized CUDA')
    need(torch.equal(external_rng, torch.get_rng_state()) and not torch.cuda.is_initialized(),
         'External CPU RNG changed or CUDA initialized')
    report = dict(schema='ROOT-native-coco-comparator-initializer-CPU-inspection-v1',
                  status='TECHNICAL_INITIALIZER_INSPECTION_COMPLETE_NO_TRAINING_OR_AP',
                  source=reference(__file__), owner_spec=reference(owner_path),
                  training_spec=owner['training_spec'], owner_receipt=owner['owner_receipt'],
                  admission=owner['admission'], root_training_qualification=owner.get('root_training_qualification'),
                  terminal_gate_passed_before_scientific_access=True, initializers=results,
                  all_initializers_strictly_compatible=all(row['strict_official80_and_ship1_compatible'] for row in results.values()),
                  CPU_affinity=[0, 1], torch_threads=torch.get_num_threads(), CUDA_initialized=False,
                  external_CPU_RNG_restored=True, torch_version=str(torch.__version__),
                  training_executed=False, inference_executed=False, loss_executed=False, AP_computed=False,
                  public_training_recipe_provenance_verified=False,
                  comparator_recipes_differ=True, method_only_superiority_comparison_supported=False,
                  no_checkpoint_saved=True, elapsed_seconds=time.monotonic() - started)
    acceptance.parent.mkdir(parents=True, exist_ok=True)
    with acceptance.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
    print(json.dumps(dict(status=report['status'], acceptance=str(acceptance),
                          all_initializers_strictly_compatible=report['all_initializers_strictly_compatible'])), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--owner-spec', required=True)
    parser.add_argument('--acceptance', required=True)
    args = parser.parse_args()
    run(args.owner_spec, args.acceptance)


if __name__ == '__main__':
    main()
