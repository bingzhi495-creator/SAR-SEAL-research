"""Thin external TEST merge/evaluation; scientific modules are injected."""
from copy import deepcopy
import math
from pathlib import Path
import sys


ORIGINAL_OFFICIAL_BYTES = 49923
ORIGINAL_OFFICIAL_SHA256 = '2357929fcd38ffc3020c5b8d204ee371da8da3972ff2e2e03398c8225c55e916'


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _manifest_ids(manifest):
    _need(type(manifest) is list and len(manifest) == 1962, 'Original TEST1962 manifest required')
    for ordinal, row in enumerate(manifest):
        _need(type(row) is dict and set(row) == {'dataset_index', 'image_id'}
              and type(row['dataset_index']) is int and row['dataset_index'] == ordinal
              and type(row['image_id']) is int, 'Original integer image ID and TEST ordinal required')
    ids = [row['image_id'] for row in manifest]
    _need(len(set(ids)) == 1962, 'Original TEST image IDs must be unique within this split')
    return ids


def _methods(method_names):
    _need(type(method_names) in (list, tuple) and len(method_names) >= 3
          and all(type(name) is str and name and name == name.strip() for name in method_names)
          and len(set(method_names)) == len(method_names)
          and {'native', 'PLAIN', 'FULL'} <= set(method_names), 'Explicit prelocked native/PLAIN/FULL methods required')
    return tuple(method_names)


def merge_population(manifest, method_names, ranges, shards, population_contract):
    """Merge only a complete TEST; outer actual owner supplies accepted shards.

    Shards contain dataset_range and predictions keyed by all prelocked methods.
    predictions['native'] is the single original query-order/metadata reference.
    No acceptance boolean or qualification schema is created or consumed here.
    """
    ids = _manifest_ids(manifest)
    methods = _methods(method_names)
    _need(type(ranges) is list and ranges and type(shards) is list and len(shards) == len(ranges),
          'Every fixed range must have exactly one ordered shard')
    cursor = 0
    for span in ranges:
        _need(type(span) is list and len(span) == 2 and all(type(v) is int for v in span)
              and span[0] == cursor and cursor < span[1] <= 1962,
              'Fixed shard ranges must cover TEST in order without gaps, overlaps or empty spans')
        cursor = span[1]
    _need(cursor == 1962, 'Only a complete TEST population can be merged')
    merged = {method: [] for method in methods}
    for span, shard in zip(ranges, shards):
        first, stop = span
        _need(type(shard) is dict and shard.get('dataset_range') == span
              and type(shard.get('dataset_range')) is list
              and all(type(v) is int for v in shard['dataset_range']), 'Original fixed shard order required')
        predictions = shard['predictions']
        _need(type(predictions) is dict and set(predictions) == set(methods),
              'Every prelocked method is required in every shard')
        _need(all(type(rows) is list and len(rows) == (stop-first)*100 for rows in predictions.values()),
              'Exactly100 candidates per original image and method required')
        for local, image_id in enumerate(ids[first:stop]):
            begin = local*100
            native = predictions['native'][begin:begin+100]
            qids = population_contract._check_image(native, image_id)
            signatures = [population_contract._non_geometry_signature(row) for row in native]
            for method in methods:
                block = predictions[method][begin:begin+100]
                population_contract._check_image(block, image_id, qids)
                _need(all(type(row['segmentation']) is dict for row in block), 'Opaque JSON RLE dictionaries required')
                if method != 'native':
                    _need([population_contract._non_geometry_signature(row) for row in block] == signatures,
                          'Only segmentation may differ at each original image/query ordinal')
                merged[method].extend(block)
    _need(all(len(rows) == 196200 for rows in merged.values()), 'Complete TEST196200 candidates required')
    return {method: deepcopy(rows) for method, rows in merged.items()}


def _closed_module(module, reference, closed_ref):
    # The actual owner/runtime checks the bound bytes. This only joins the
    # admitted reference to the module that is really about to be called.
    bound = closed_ref(reference)
    source = getattr(module, '__file__', None)
    _need(type(source) is str and Path(source).resolve() == Path(bound).resolve(),
          'Executing module must match its explicitly closed source')


def evaluate_one_method(method, method_names, manifest, predictions, *, COCO, COCOeval,
                        official, module_sources, test_ref, closed_ref, population_contract):
    """One complete original TEST COCO call; no runner or Root qualification.

    predictions must be the requested method's output of merge_population.
    The caller must evaluate every prelocked method and own actual source,
    endpoint and shard acceptance. No AP-based skipping is implemented here.
    """
    methods = _methods(method_names)
    _need(type(method) is str and method in methods, 'Requested method must be prelocked')
    ids = _manifest_ids(manifest)
    _need(type(module_sources) is dict and set(module_sources) == {'official', 'coco', 'cocoeval'},
          'Explicit official/COCO/COCOeval source references required')
    original = module_sources['official']
    _need(type(original) is dict and type(original.get('bytes')) is int
          and original['bytes'] == ORIGINAL_OFFICIAL_BYTES
          and original.get('sha256') == ORIGINAL_OFFICIAL_SHA256,
          'Only the original reviewed population-independent coco_metrics source is permitted')
    _closed_module(official, original, closed_ref)
    function = getattr(official, 'coco_metrics', None)
    _need(callable(function) and getattr(function, '__globals__', None) is vars(official)
          and getattr(function, '__module__', None) == getattr(official, '__name__', None),
          'coco_metrics must be defined in the explicitly closed original module')
    _need(isinstance(COCO, type) and isinstance(COCOeval, type), 'Explicit original COCO classes required')
    for role, cls in (('coco', COCO), ('cocoeval', COCOeval)):
        module = sys.modules.get(cls.__module__)
        _need(module is not None, 'Actual COCO class module must be loaded by the admitted owner')
        _closed_module(module, module_sources[role], closed_ref)
    _need(type(predictions) is list and len(predictions) == 196200, 'Complete TEST196200 predictions required')
    for index, image_id in enumerate(ids):
        block = predictions[index*100:(index+1)*100]
        population_contract._check_image(block, image_id)
        _need(all(type(row['segmentation']) is dict for row in block), 'All100 opaque RLE dictionaries required')
    coco = COCO(str(closed_ref(test_ref)))
    _need(len(coco.imgs) == 1962 and set(coco.imgs) == set(ids)
          and all(type(image_id) is int for image_id in coco.imgs)
          and len(coco.anns) == 5922 and set(coco.cats) == {1} and type(next(iter(coco.cats))) is int
          and len(coco.dataset['images']) == 1962 and len(coco.dataset['annotations']) == 5922,
          'Complete original TEST1962/5922/category1 required; no filtered GT view')
    _need(all(ann['image_id'] in coco.imgs and ann['category_id'] == 1 for ann in coco.anns.values()),
          'All original TEST annotation identities and categories must be retained')
    before = deepcopy(coco.dataset)
    # No bbox/area pop, mask-area rewrite, score cutoff or per-shard evaluation.
    # The original function itself deep-copies predictions before loadRes.
    result = function(coco, predictions, ids, COCOeval)
    _need(coco.dataset == before, 'Original TEST GT must remain unchanged by evaluation')
    _need(type(result) is dict and type(result.get('stats')) is list and len(result['stats']) == 12
          and all(type(value) in (int, float) and math.isfinite(value) for value in result['stats'])
          and result.get('iouType') == 'segm' and result.get('maxDets') == [1, 10, 100]
          and result.get('original_GT_preserved') is True and result.get('score_filter') is None
          and result.get('official_COCO_crowd_ignore_semantics') is True,
          'Original full-population COCO12 result required')
    return result
