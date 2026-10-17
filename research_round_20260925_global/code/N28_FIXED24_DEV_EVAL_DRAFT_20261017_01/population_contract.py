"""Synthetic-testable JSON population contract; no file or scientific I/O.

These functions preserve prediction identities and scores. They do not decode
RLEs, validate mask pixels, run a model or certify official COCO evaluation.
"""
from copy import deepcopy
import json
import math


def _need(ok,message):
    if not ok:
        raise ValueError(message)


def _check_image(values,image_id,query_ids=None):
    _need(type(values) is list and len(values)==100 and type(image_id) is int,
          'Exactly 100 predictions for one integer image id are required')
    observed=[]
    for row in values:
        _need(type(row) is dict and row.get('image_id')==image_id
              and type(row.get('image_id')) is int
              and type(row.get('query_id')) is int and 0<=row['query_id']<100
              and type(row.get('category_id')) is int and row['category_id']==1
              and type(row.get('score')) in (int,float) and math.isfinite(row['score'])
              and 'segmentation' in row,'Original image/query/category/finite score population')
        observed.append(row['query_id'])
    _need(set(observed)==set(range(100)),'Each natural query appears exactly once')
    if query_ids is not None:
        _need(type(query_ids) in (list,tuple) and len(query_ids)==100
              and all(type(q) is int for q in query_ids)
              and list(query_ids)==observed,'Provided query ids must equal original export order')
    return observed


def replace_segmentations(original,new_rles,image_id,query_ids):
    _need(type(query_ids) in (list,tuple),'Explicit original query order is required')
    _check_image(original,image_id,query_ids)
    _need(type(new_rles) is list and len(new_rles)==100 and all(type(rle) is dict for rle in new_rles),
          'One opaque JSON RLE dictionary per original query is required')
    changed=deepcopy(original)
    for row,rle in zip(changed,new_rles):
        row['segmentation']=deepcopy(rle)
    return changed


def merge_predictions(shards,original,ranges=[[0,47],[47,94],[94,141],[141,187]]):
    fixed=[[0,47],[47,94],[94,141],[141,187]]
    _need(type(ranges) is list and ranges==fixed
          and all(type(pair) is list and all(type(index) is int for index in pair) for pair in ranges),
          'Only the four fixed complete DEV ranges are permitted')
    _need(type(original) is list and len(original)==18700,'Canonical full18700 predictions required')
    image_ids=[]
    for index in range(187):
        values=original[index*100:(index+1)*100]
        _need(type(values[0]) is dict,'Canonical prediction dictionaries required')
        image=values[0].get('image_id')
        _check_image(values,image)
        image_ids.append(image)
    _need(len(set(image_ids))==187,'Canonical population must contain187 distinct images')
    _need(type(shards) is list and len(shards)==4,'Exactly four ordered shards are required')
    merged=[]
    for shard,span in zip(shards,ranges):
        _need(type(shard) is dict and shard.get('dataset_range')==span
              and type(shard.get('dataset_range')) is list
              and all(type(index) is int for index in shard['dataset_range']),
              'Shard ranges must retain the fixed canonical order')
        values=shard.get('predictions');first,stop=span
        _need(type(values) is list and len(values)==(stop-first)*100,'Complete shard prediction population')
        for offset,value in enumerate(values):
            _need(type(value) is dict and type(value.get('segmentation')) is dict,
                  'Every original ordinal requires its output geometry dictionary')
            expected=original[first*100+offset]
            _need(_non_geometry_signature(value)==_non_geometry_signature(expected),
                  'All non-segmentation fields must remain exact at every canonical ordinal')
        merged.extend(values)
    return deepcopy(merged)


def _non_geometry_signature(value):
    # JSON preserves finite float values (including signed zero) and numeric
    # types. Dictionary key order is immaterial; nested metadata is preserved.
    try:
        return json.dumps({key:item for key,item in value.items() if key!='segmentation'},
                          sort_keys=True,separators=(',',':'),allow_nan=False)
    except (TypeError,ValueError) as error:
        raise ValueError('Prediction metadata must be finite plain JSON') from error
