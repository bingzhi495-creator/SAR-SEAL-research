"""UNBOUND pure metadata conversion; never imports ML libraries or loads tensors."""
import re
import json
from copy import deepcopy
from math import prod

PREFIX = 'backbone.bottom_up.'
BLOCKS = (3, 4, 23, 3)
NORM_FIELDS = ('weight', 'bias', 'running_mean', 'running_var')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def parse_request(text):
    """Parse bounded JSON metadata, rejecting duplicates at every nesting level."""
    need(type(text) is str and len(text.encode('utf-8'))<=8*1024**2, 'Bounded UTF8 JSON metadata text')
    def unique(pairs):
        result = {}
        for key,value in pairs:
            need(key not in result, 'Duplicate JSON key: '+key)
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError('Non-finite JSON constant: '+value)
    value = json.loads(text,object_pairs_hook=unique,parse_constant=invalid_constant)
    need(type(value) is dict and set(value)=={'schema','source','target'} and
         value['schema']=='R101-explicit-metadata-request-v1', 'Exact explicit metadata request schema')
    need(type(value['source']) is dict and type(value['target']) is dict, 'Plain metadata dictionaries')
    return value


def map_key(key):
    """Return one explicit target key, or None for supported excluded state."""
    need(type(key) is str, 'Source key must be a plain string')
    if key in ('fc.weight', 'fc.bias', 'bn1.num_batches_tracked'):
        return None
    if key == 'conv1.weight':
        return PREFIX + 'stem.conv1.weight'
    stem = re.fullmatch(r'bn1\.(weight|bias|running_mean|running_var)', key)
    if stem:
        return PREFIX + 'stem.conv1.norm.' + stem[1]
    block = re.fullmatch(r'layer([1-4])\.(0|[1-9][0-9]*)\.(.+)', key)
    need(block is not None, 'Unknown source key: ' + key)
    layer, index, leaf = int(block[1]), int(block[2]), block[3]
    need(index < BLOCKS[layer-1], 'Block outside the explicit R101 layout: ' + key)
    out = PREFIX + 'res%d.%d.' % (layer+1, index)
    conv = re.fullmatch(r'conv([1-3])\.weight', leaf)
    if conv:
        return out + leaf
    norm = re.fullmatch(r'bn([1-3])\.(weight|bias|running_mean|running_var|num_batches_tracked)', leaf)
    if norm:
        return None if norm[2] == 'num_batches_tracked' else out + 'conv%s.norm.%s' % (norm[1], norm[2])
    if leaf.startswith('downsample.'):
        need(index == 0, 'Only each first block has a projection shortcut: ' + key)
        if leaf == 'downsample.0.weight':
            return out + 'shortcut.weight'
        projection = re.fullmatch(r'downsample\.1\.(weight|bias|running_mean|running_var|num_batches_tracked)', leaf)
        if projection:
            return None if projection[1] == 'num_batches_tracked' else out + 'shortcut.norm.' + projection[1]
    raise ValueError('Unknown source key: ' + key)


def _architecture_specs():
    """Fixed R101/FrozenBN/modulated-DCN groups1; never infer from input keys."""
    required_source, required_target = {}, {}
    def add(name, shape):
        metadata = dict(dtype='torch.float32', shape=list(shape), numel=prod(shape))
        required_source[name] = metadata
        required_target[map_key(name)] = deepcopy(metadata)
    def norm(name, channels):
        for field in NORM_FIELDS:
            add(name+'.'+field, [channels])
    add('conv1.weight', [64,3,7,7])
    norm('bn1', 64)
    incoming = 64
    for stage, count in enumerate(BLOCKS, 1):
        width = 64 * 2**(stage-1)
        for block in range(count):
            name = 'layer%d.%d.' % (stage,block)
            add(name+'conv1.weight', [width,incoming,1,1])
            norm(name+'bn1', width)
            add(name+'conv2.weight', [width,width,3,3])
            norm(name+'bn2', width)
            add(name+'conv3.weight', [4*width,width,1,1])
            norm(name+'bn3', 4*width)
            if block == 0:
                add(name+'downsample.0.weight', [4*width,incoming,1,1])
                norm(name+'downsample.1', 4*width)
            offset = PREFIX+'res%d.%d.conv2_offset.' % (stage+1,block)
            for field, shape in (('weight',[27,width,3,3]),('bias',[27])):
                required_target[offset+field] = dict(dtype='torch.float32',shape=shape,numel=prod(shape))
            incoming = 4*width
    return required_source, required_target


def _validate_metadata(value, expected, key):
    need(type(value) is dict and set(value) in (
         {'dtype','shape','numel'},{'dtype','shape','numel','sha256'}), 'Exact metadata fields: '+key)
    shape = value['shape']
    need(type(shape) is list and all(type(d) is int and d>0 for d in shape), 'Integer positive shape dimensions: '+key)
    need(type(value['numel']) is int and value['numel']>0 and value['numel']==prod(shape), 'Exact integer numel: '+key)
    need(type(value['dtype']) is str and value['dtype']==expected['dtype'], 'Architecture dtype mismatch: '+key)
    need(shape==expected['shape'], 'Architecture shape mismatch: '+key)
    need(value['numel']==expected['numel'], 'Architecture numel mismatch: '+key)
    if 'sha256' in value:
        need(type(value['sha256']) is str and re.fullmatch('[0-9a-f]{64}',value['sha256']) is not None,
             'Optional digest must be an explicit lower-case SHA256 string: '+key)


def check_coverage(source, target):
    """Build a metadata-only proposed transfer plan, never a loading receipt."""
    need(type(source) is dict and type(target) is dict, 'Explicit plain metadata dictionaries')
    need(all(type(k) is str for k in source) and all(type(k) is str for k in target), 'Plain string keys')
    required_source, required_target = _architecture_specs()
    missing_source = sorted(set(required_source)-set(source))
    need(not missing_source, 'Missing required source: '+repr(missing_source[:10]))
    missing_target = sorted(set(required_target)-set(target))
    need(not missing_target, 'Missing required target: '+repr(missing_target[:10]))
    unknown_target = sorted(set(target)-set(required_target))
    need(not unknown_target, 'Unknown target keys outside explicit R101/DCN subset: '+repr(unknown_target[:10]))
    for key, expected in required_target.items():
        _validate_metadata(target[key], expected, key)
    mappings, excluded, seen = [], [], set()
    stages = {stage:dict(mapped_keys=0,mapped_numel=0) for stage in ('stem','res2','res3','res4','res5')}
    for key in sorted(source):
        dest = map_key(key)
        if dest is None:
            if key == 'fc.weight':
                expected = dict(dtype='torch.float32',shape=[1000,2048],numel=2048000)
            elif key == 'fc.bias':
                expected = dict(dtype='torch.float32',shape=[1000],numel=1000)
            else:
                expected = dict(dtype='torch.int64',shape=[],numel=1)
            _validate_metadata(source[key], expected, key)
            excluded.append(dict(source=key, reason='classifier' if key.startswith('fc.') else 'BN_counter', metadata=deepcopy(source[key])))
            continue
        _validate_metadata(source[key], required_source[key], key)
        need(dest in target, 'Missing mapped target key: '+dest)
        need(dest not in seen, 'Duplicate target mapping: '+dest)
        seen.add(dest)
        mappings.append(dict(source=key,target=dest,source_metadata=deepcopy(source[key]),target_initial_metadata=deepcopy(target[dest])))
        stage = dest[len(PREFIX):].split('.')[0]
        stages[stage]['mapped_keys'] += 1
        stages[stage]['mapped_numel'] += source[key]['numel']
    for key, expected in required_source.items():
        stage = map_key(key)[len(PREFIX):].split('.')[0]
        stages[stage]['required_keys'] = stages[stage].get('required_keys',0)+1
        stages[stage]['required_numel'] = stages[stage].get('required_numel',0)+expected['numel']
    return dict(schema='R101-explicit-metadata-transfer-plan-v1',
                evidence_level='CALLER_SUPPLIED_METADATA_ONLY_NOT_A_LOADING_RECEIPT',
                source_key_count=len(source),target_key_count=len(target),
                mapped_key_count=len(mappings),mapped_numel=sum(x['mapped_numel'] for x in stages.values()),
                mapping=mappings,source_excluded=excluded,target_unloaded_plan=sorted(set(target)-seen),
                stages=stages,actual_tensor_load_verified=False,source_provenance_verified=False,
                target_model_identity_verified=False,DCN_function_equivalence_verified=False,
                actual_target_values_unchanged_verified=False,AP=False)
