"""Pure metadata planning; no files, tensors, labels, predictions or results read.

Frozen manifests must precede result access. This module neither performs nor
qualifies evaluation. Dense descriptor/summary validation is not tensor replay.
"""
import hashlib
import json
import math
from pathlib import PurePosixPath, PureWindowsPath

SCHEMA = 'external-population-contract-v1'
CANDIDATES_PER_IMAGE = 100
MAX_DENSE_AUDIT_IMAGES = 16
BYTES_PER_DENSE_IMAGE = 73064000
CONTROL_DEFINITIONS = (
    ('A_native_official', 'A', 'native', 'native', 'official_foreground_sigmoid_mean'),
    ('A_native_zero_budget', 'A', 'native', 'native', 'posterior_intersection_union_ratio'),
    ('B_native_official', 'B', 'native', 'native', 'official_foreground_sigmoid_mean'),
    ('B_native_posterior_budget', 'B', 'native', 'L_plus_u', 'posterior_intersection_union_ratio'),
    ('B_posterior_official', 'B', 'L_plus_u', 'L_plus_u', 'official_foreground_sigmoid_mean'),
    ('B_posterior_budget', 'B', 'L_plus_u', 'L_plus_u', 'posterior_intersection_union_ratio'),
)
CONTROLS = tuple(row[0] for row in CONTROL_DEFINITIONS)


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _positive_int(value, name):
    _require(type(value) is int and value > 0, name + ' must be a positive integer, not bool')
    return value


def _text(value, name):
    _require(type(value) is str and value and value == value.strip() and
             not any(ord(c) < 32 or ord(c) == 127 for c in value), name + ' must be explicit clean text')
    return value


def _relative_path(value):
    value = _text(value, 'relative_path')
    parts = value.split('/')
    reserved = {'con', 'prn', 'aux', 'nul'} | {'com%d' % i for i in range(1, 10)} | {'lpt%d' % i for i in range(1, 10)}
    _require(not PurePosixPath(value).is_absolute() and not PureWindowsPath(value).drive and
             not any(c in value for c in '\\:*?"<>|') and
             all(p not in ('', '.', '..') and p == p.strip() and not p.endswith('.') and
                 p.split('.')[0].casefold() not in reserved for p in parts),
             'relative_path must be canonical POSIX relative text without escape or Windows aliases')
    return value


def _sha(value):
    _require(type(value) is str and len(value) == 64 and set(value) <= set('0123456789abcdef'),
             'sha256 must be 64 lowercase hexadecimal characters')
    return value


def deterministic_dense_audit(image_ids, domain, limit=MAX_DENSE_AUDIT_IMAGES):
    """SHA256([domain, integer ID]) lowest ranks; return selected IDs sorted.

    Selection uses neither labels nor predictions nor dimensions nor input order.
    A domain and cap must be frozen before result access; no post-hoc redraw.
    """
    domain = _text(domain, 'audit_domain')
    _positive_int(limit, 'audit limit')
    _require(limit <= MAX_DENSE_AUDIT_IMAGES, 'audit limit exceeds predeclared maximum')
    _require(type(image_ids) in (list, tuple) and image_ids and
             all(type(i) is int for i in image_ids) and len(set(image_ids)) == len(image_ids),
             'image_ids must be nonempty unique integer IDs, not bool')
    def rank(image_id):
        token = json.dumps([domain, image_id], ensure_ascii=True, separators=(',', ':')).encode('ascii')
        return hashlib.sha256(token).hexdigest(), image_id
    return sorted(sorted(image_ids, key=rank)[:limit])


def validate_dense_descriptors(fields, arm, numeric_summary=None):
    """Validate saved-field metadata and optional finite scalar summaries only.

    This does not read a packet or prove tensor finiteness, a SHA match or replay.
    Actual checks remain the responsibility of a future qualified CPU evaluator.
    """
    _require(type(arm) is str and arm in ('A', 'B'), 'arm must be A or B')
    shapes = dict(native_masks=(1, 100, 200, 200), classes=(1, 100, 2))
    if arm == 'B':
        shapes.update(feature=(1, 256, 200, 200), coefficient=(1, 100, 256))
    _require(type(fields) is dict and set(fields) == set(shapes), 'dense field population differs')
    result = {}
    for name, expected in shapes.items():
        item = fields[name]
        _require(type(item) is dict and set(item) == {'shape', 'dtype', 'sha256'}, 'descriptor keys differ')
        shape = item['shape']
        _require(type(shape) in (list, tuple) and all(type(n) is int and n > 0 for n in shape) and
                 tuple(shape) == expected and item['dtype'] == 'float32', 'descriptor shape/dtype differs')
        result[name] = dict(shape=list(shape), dtype='float32', sha256=_sha(item['sha256']))
    if numeric_summary is not None:
        _require(type(numeric_summary) is dict, 'numeric_summary must be a dict')
        for name, value in numeric_summary.items():
            _text(name, 'numeric_summary key')
            _require(type(value) in (int, float) and math.isfinite(value), 'numeric_summary must be finite and not bool')
    return result


def freeze_population(images, source, gt_pin=None, audit_domain='external-dense-audit-v1'):
    """Build a canonical, JSON-safe population plan without filesystem access.

    images: list/tuple of exact {image_id, relative_path, height, width} dicts.
    source: exact {dataset_id, version, split, source_id} explicit strings.
    gt_pin: None (pending) or exact {path, bytes, sha256} metadata; never opened.
    All supplied images are retained, including scenes without any instances.
    """
    _require(type(source) is dict and set(source) == {'dataset_id', 'version', 'split', 'source_id'},
             'fixed source identifiers required')
    origin = {k: _text(v, k) for k, v in source.items()}
    _require(type(images) in (list, tuple) and images, 'positive image population required')
    rows, ids, paths = [], set(), set()
    for image in images:
        _require(type(image) is dict and set(image) == {'image_id', 'relative_path', 'height', 'width'},
                 'image metadata keys differ; labels/results are not manifest inputs')
        image_id = image['image_id']
        _require(type(image_id) is int and image_id not in ids, 'unique integer image ID required, not bool')
        path = _relative_path(image['relative_path'])
        _require(path.casefold() not in paths, 'duplicate or Windows-case-aliased image path')
        rows.append(dict(image_id=image_id, relative_path=path,
                         height=_positive_int(image['height'], 'height'), width=_positive_int(image['width'], 'width')))
        ids.add(image_id); paths.add(path.casefold())
    rows.sort(key=lambda r: r['image_id'])
    pin = None
    if gt_pin is not None:
        _require(type(gt_pin) is dict and set(gt_pin) == {'path', 'bytes', 'sha256'}, 'GT metadata pin keys differ')
        locator = _text(gt_pin['path'], 'GT pin path')
        _require(PurePosixPath(locator).is_absolute() or PureWindowsPath(locator).is_absolute(), 'GT locator must be absolute')
        pin = dict(path=locator, bytes=_positive_int(gt_pin['bytes'], 'GT bytes'), sha256=_sha(gt_pin['sha256']))
    all_ids = [r['image_id'] for r in rows]
    audit_ids = deterministic_dense_audit(all_ids, audit_domain)
    count, retained = len(rows), len(audit_ids)
    controls = [dict(name=n, arm=a, geometry=g, probability_logits=p, quality_kind=q)
                for n, a, g, p, q in CONTROL_DEFINITIONS]
    contract = dict(
        schema=SCHEMA, status='SOURCE_METADATA_PLAN_ONLY_NOT_LAUNCH_READY', source=origin,
        images=rows, gt_metadata_pin=pin, controls=controls,
        score_contract=dict(support='geometry_logits > 0', probability='sigmoid(probability_logits)',
            official_quality='sum(support * probability) / (sum(support) + epsilon)',
            budget_quality='sum(support * probability) / (sum(support) + sum((1-support) * probability) + epsilon)',
            score='own_arm_ship_probability * quality', epsilon=1e-6, empty_quality_and_score=0,
            cross_arm_class_equality_assumed=False, no_NMS=True, no_score_cutoff=True),
        expected_population=dict(COCO_images=count, COCO_image_ids=all_ids, raw_GT_annotations=None,
            candidates_per_image=CANDIDATES_PER_IMAGE, predictions_per_control=count*CANDIDATES_PER_IMAGE,
            predictions_all_six_controls=count*CANDIDATES_PER_IMAGE*len(CONTROLS),
            candidate_export_checks_images=count, dense_replay_images=retained,
            dense_field_packets_per_arm=retained),
        dense_audit=dict(domain=audit_domain, limit=MAX_DENSE_AUDIT_IMAGES, image_ids=audit_ids,
            selection='lowest SHA256 of compact ASCII JSON [domain, image_id], then ID tie break',
            frozen_before_GT_predictions_results_required=True,
            planned_full_population_replay=retained == count, actual_dense_replay_performed=False),
        planned_storage=dict(bytes_per_dense_image=BYTES_PER_DENSE_IMAGE, retained_images=retained,
            dense_field_bytes=BYTES_PER_DENSE_IMAGE*retained,
            scope='estimate for current 800-input/native200 two-arm saved-field contract only',
            unknown_original_size_inference_or_training_budget=False),
        pending=['actual version/IDs/dimensions and image content', 'GT pin bytes and full annotation population',
                 'resolved image-root containment including symlinks/junctions', 'split/overlap/test exposure audit',
                 'qualified dynamic evaluator and original owned execution'],
        launch_ready=False, blind_test_claimed=False, efficacy_claimed=False,
    )
    encoded = json.dumps(contract, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')
    contract['metadata_contract_sha256'] = hashlib.sha256(encoded).hexdigest()
    return contract
