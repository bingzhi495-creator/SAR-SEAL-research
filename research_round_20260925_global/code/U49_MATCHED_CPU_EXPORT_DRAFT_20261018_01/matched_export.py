"""U49 unbound matched-only CPU export; no ML imports or work on import."""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath


INDICES = (54, 710)
SCHEMA = 'U49-matched-CPU-portable-item-v1'
FIELDS = ('rgb400', 'e', 'M', 'L', 'targets')
SOURCE_SHA256 = {
    'rgb_matched_reader': '678ce3781872e21e961b24a7b0a4d0ecc69f0f8a08b8feb7eb0b2e8624d57025',
    'matched_reader': '82565caa9c42b6c2ed501e13af9554c6ef47a99b6b525ccfce0bcef620534870',
    'image_geometry': '65db107de80c5f1d4dade7cb2ffd3f241fd6bf7f9b233cb1aa9c90cbfd00abf7',
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def _primitive(value):
    if value is None or type(value) in (str, int, bool):
        return
    if type(value) is float:
        need(math.isfinite(value), 'Finite primitive metadata only')
    elif type(value) in (list, tuple):
        for child in value:
            _primitive(child)
    elif type(value) is dict:
        need(all(type(key) is str for key in value), 'String metadata keys only')
        for child in value.values():
            _primitive(child)
    else:
        raise ValueError('Only primitive metadata, not tensor/model/geometry/Python objects')


def _reference(value):
    need(type(value) is dict and set(value) == {'path', 'bytes', 'sha256'}
         and type(value['path']) is str
         and (PureWindowsPath(value['path']).is_absolute() or PurePosixPath(value['path']).is_absolute())
         and type(value['bytes']) is int and value['bytes'] >= 0
         and type(value['sha256']) is str and len(value['sha256']) == 64
         and set(value['sha256']) <= set('0123456789abcdef'), 'Strict absolute path/bytes/SHA256 reference')


def _shape(value):
    need(type(value) in (list, tuple) and len(value) == 2
         and all(type(x) is int and x > 0 for x in value), 'Positive integer H,W')
    return list(value)


def package_metadata(dataset_index, item, record, provenance):
    """Pure metadata contract; this does not authenticate actual reader I/O."""
    need(type(item) is dict and type(record) is dict, 'Original reader metadata dictionaries')
    need(type(dataset_index) is int and dataset_index in INDICES
         and record['dataset_index'] == dataset_index, 'Only original indices 54 and 710')
    need(type(item) is dict and type(record) is dict
         and type(item.get('identity')) is dict
         and item['identity'] == record['identity']
         and item['identity']['image_id'] == record['image_id']
         and item['G'] == record['G'] and item['query_ids'] == record['query_ids'],
         'Exact original reader record, identity and matched GT/query order')
    G, queries, annotations = item['G'], item['query_ids'], item['identity']['ann_ids']
    need(type(G) is int and 1 <= G <= 100
         and type(queries) is list and len(queries) == len(set(queries)) == G
         and all(type(q) is int and 0 <= q < 100 for q in queries)
         and type(annotations) is list and len(annotations) == len(set(annotations)) == G
         and all(type(a) is int for a in annotations),
         'Original one-to-one GT/query mapping; never sort, filter, duplicate or rematch')
    identity, rgb = item['identity'], item['rgb_receipt']
    original, content = _shape(identity['original_shape']), _shape(identity['actual_img_shape'])
    longest = max(original)
    need(identity['batch_input_shape'] == [800, 800]
         and content == [(2*x*800+longest)//(2*longest) for x in original],
         'Original integer keep-ratio geometry, no replacement resize')
    need(type(rgb) is dict and rgb['image'] == identity['image']
         and rgb['original_shape'] == original and rgb['actual_img_shape'] == content
         and type(rgb['canvas_size']) is int and rgb['canvas_size'] == 800
         and type(rgb['branch_size']) is int and rgb['branch_size'] == 400
         and rgb['decoded_mode'] == 'RGB' and rgb['auto_orientation'] is False
         and (rgb['exif_orientation'] is None or (type(rgb['exif_orientation']) is int and rgb['exif_orientation'] == 1))
         and rgb['teacher_photometric_reconstruction'] is False
         and rgb['input_convention'] == 'RGB uint8 -> FP32 /255; common reviewed geometry'
         and rgb['baseline'] == 'N16 native_L selected by unchanged original GT-matched query ids'
         and rgb['image_source_control'] in ('base', 'separate_explicit_admitted_control'),
         'Keep original RGB receipt and native-L baseline; no photometric or geometry substitution')
    need(type(provenance) is dict and set(provenance) == {'context', 'source_refs', 'input_refs'}
         and set(provenance['context']) == {'FIT_context', 'N13_qualified', 'N13_report'}
         and set(provenance['source_refs']) == set(SOURCE_SHA256), 'Explicit original context and three reviewed sources')
    _primitive(dict(identity=identity, rgb_receipt=rgb, provenance=provenance))
    _reference(identity['image'])
    for ref in provenance['context'].values():
        _reference(ref)
    for role, ref in provenance['source_refs'].items():
        _reference(ref)
        need(ref['sha256'] == SOURCE_SHA256[role], 'Original source bytes required: '+role)
    inputs = provenance['input_refs']
    need(type(inputs) is list and inputs, 'Caller supplies explicit actual input references, never discovery')
    seen = {}
    for ref in inputs:
        _reference(ref)
        key = ref['path'].replace('\\', '/').casefold()
        need(key not in seen or seen[key] == ref, 'Contradictory input references')
        seen[key] = ref
    need(identity['image'] in inputs and all(ref in inputs for ref in provenance['context'].values()),
         'Original image and context must be among caller-admitted inputs')
    return deepcopy(dict(dataset_index=dataset_index, image_id=item['identity']['image_id'],
        G=item['G'], query_ids=item['query_ids'], ann_ids=item['identity']['ann_ids'],
        identity=item['identity'], rgb_receipt=item['rgb_receipt'],
        geometry=dict(original_shape=original, actual_img_shape=content, canvas_size=800, branch_size=400),
        provenance=provenance))


def _digest(value):
    # Future owned tensor execution only. This imports no numerical library.
    return hashlib.sha256(value.detach().contiguous().numpy().tobytes(order='C')).hexdigest()


def export_item(torch, dataset_index, item, record, *, provenance, check):
    """Clone one genuine U37.load result; Root must authenticate that call/source.

    No file reads, model construction, F access, interpolation, rematching or
    forward occurs. Geometry objects are intentionally not serialized. This
    function proves equality to the supplied item, not independent teacher/GT
    reconstruction or authenticity of an arbitrary caller-created dictionary.
    """
    need(callable(check), 'Owned resource check callback required')
    check()
    need(type(item) is dict and set(item) == set(FIELDS) |
         {'F', 'G', 'query_ids', 'identity', 'geometry', 'rgb_receipt'}, 'Exact ordinary U37 returned-item ABI')
    metadata = package_metadata(dataset_index, item, record, provenance)
    need(callable(getattr(item['geometry'], 'prepare_rgb', None))
         and callable(getattr(item['geometry'], 'render_delta', None)), 'Keep original U37 geometry object at input')
    G, (H, W) = metadata['G'], metadata['identity']['original_shape']
    shapes = dict(rgb400=(1, 3, 400, 400), e=(G, 256), M=(G, 200, 200), L=(G, H, W), targets=(G, H, W))
    tensors, proofs = {}, {}
    for key in FIELDS:
        check()
        source = item[key]
        need(isinstance(source, torch.Tensor) and source.device.type == 'cpu'
             and source.dtype == torch.float32 and tuple(source.shape) == shapes[key]
             and source.layout == torch.strided and not source.requires_grad and source.grad_fn is None
             and bool(torch.isfinite(source).all()), 'Original finite detached CPU FP32 field: '+key)
        if key == 'rgb400':
            need(bool(((source >= 0) & (source <= 1)).all()), 'Original RGB400 in [0,1]')
        elif key == 'targets':
            need(bool(((source == 0) | (source == 1)).all()), 'Original full-grid binary FP32 instance targets')
        # Never index M by natural query_ids: all five fields already use GT order.
        copied = source.detach().clone().contiguous()
        logical_bytes = int(copied.numel()*copied.element_size())
        need(copied.storage_offset() == 0 and copied.is_contiguous()
             and copied.untyped_storage().nbytes() == logical_bytes
             and copied.untyped_storage().data_ptr() != source.untyped_storage().data_ptr(),
             'Independent compact storage, never unused F/all100 L storage: '+key)
        before, after = _digest(source), _digest(copied)
        need(torch.equal(source, copied) and before == after, 'Exact values and bytes preserved: '+key)
        tensors[key] = copied
        proofs[key] = dict(shape=list(copied.shape), dtype=str(copied.dtype), logical_bytes=logical_bytes,
                           source_sha256=before, sha256=after, bitexact=True,
                           independent_storage=True, compact_storage=True)
    # Recheck caller-owned metadata after all copies; never mutate the source item.
    need(package_metadata(dataset_index, item, record, provenance) == metadata, 'Source metadata unchanged during export')
    check()
    return dict(schema=SCHEMA, metadata=metadata, tensors=tensors, equivalence=proofs,
                selected_indices=list(INDICES), images=1, qualified=False, new_training=False,
                AP=False, DEV=False, TEST=False, whole_population_exported=False)


def save_export(torch, package, destination, *, check):
    """Future exclusive owned write + real weights_only CPU roundtrip, no retry.

    On failure keep the partial/complete file; the caller records the failure.
    Root owns output-directory guards and science admission. This grants no
    qualification and never opens original image/cache/qualification refs.
    """
    need(callable(check) and type(package) is dict and package.get('schema') == SCHEMA
         and set(package['tensors']) == set(package['equivalence']) == set(FIELDS), 'U49 item package required')
    scalar = {key: value for key, value in package.items() if key != 'tensors'}
    _primitive(scalar)
    need(package['qualified'] is False and package['whole_population_exported'] is False,
         'Pending two-image export only')
    path = Path(destination)
    need(path.is_absolute() and path.parent.is_dir() and not path.exists() and not path.is_symlink(),
         'Fresh absolute file in a caller-owned directory')
    check()
    with path.open('xb') as stream:
        torch.save(package, stream)
        stream.flush()
        os.fsync(stream.fileno())
    loaded = None
    try:
        check()
        loaded = torch.load(str(path), map_location='cpu', weights_only=True)
        need(set(loaded) == set(package)
             and {key: value for key, value in loaded.items() if key != 'tensors'} == scalar,
             'Portable metadata roundtrip exact')
        need(set(loaded['tensors']) == set(FIELDS), 'Exactly five saved training fields')
        for key in FIELDS:
            actual, expected = loaded['tensors'][key], package['tensors'][key]
            need(isinstance(actual, torch.Tensor) and actual.device.type == 'cpu'
                 and actual.dtype == torch.float32 and tuple(actual.shape) == tuple(expected.shape)
                 and actual.storage_offset() == 0 and actual.is_contiguous()
                 and actual.untyped_storage().nbytes() == actual.numel()*actual.element_size()
                 and torch.equal(actual, expected)
                 and _digest(actual) == package['equivalence'][key]['sha256'], 'Real weights_only CPU roundtrip: '+key)
        actual = expected = None
    finally:
        loaded = None
    check()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return dict(package_ref=dict(path=str(path), bytes=path.stat().st_size, sha256=h.hexdigest()),
                weights_only_CPU_roundtrip_verified=True, qualified=False)
