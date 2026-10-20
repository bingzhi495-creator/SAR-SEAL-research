"""UNBOUND full-query DEV transport helpers; no scientific imports on import.

Metadata checks do not authenticate a cache or qualify numerical work. Genuine
Torch and an admitted closed(ref)/check() control are supplied by future Root.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath


FIELDS = ('rgb400', 'e', 'M', 'L')
GEOMETRY_SHA256 = '65db107de80c5f1d4dade7cb2ffd3f241fd6bf7f9b233cb1aa9c90cbfd00abf7'
SOURCE_PINS = {
    'dev_reader': 'daabec8388e23d9ff646ae78eba69ba2168099a72971578a8d34cef04f1268c0',
    'dev_rgb_reader': 'b91375e5366da3f74272ce811f453be1379b46a4bb96103f9d8039a4998a1204',
    'image_geometry': GEOMETRY_SHA256,
}
INPUT_ROLES = ('dev_context', 'native_predictions', 'original_capture_manifest', 'bundle', 'dense_packet')
RGB_CONVENTION = 'RGB uint8 -> FP32 /255; unchanged reviewed geometry'
RGB_BASELINE = 'original all100 N16 native_L; not GT-matched selection'
IDENTITY_KEYS = ('image_id', 'image', 'original_shape', 'actual_img_shape', 'batch_input_shape')
RGB_KEYS = ('decoded_mode', 'exif_orientation', 'auto_orientation',
            'teacher_photometric_reconstruction', 'input_convention', 'baseline')
METADATA_KEYS = {'dataset_index', 'image_id', 'identity', 'geometry', 'query_ids',
                 'natural_query_ids', 'native_predictions', 'rgb_metadata', 'provenance'}
PACKAGE_SCHEMA = 'U59-full100-DEV-portable-item-v1'
PROOF_SCHEMA = 'U59-full100-DEV-field-proof-v1'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def _primitive(value):
    if value is None or type(value) in (str, int, bool):
        return
    if type(value) is float:
        need(math.isfinite(value), 'Finite primitive metadata only')
    elif type(value) is list:
        for child in value:
            _primitive(child)
    elif type(value) is dict:
        need(all(type(key) is str for key in value), 'String metadata keys only')
        for child in value.values():
            _primitive(child)
    else:
        raise ValueError('Only plain JSON metadata; tensors/custom objects are not metadata')


def canonical_json(value):
    _primitive(value)
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


def metadata_sha256(value):
    return hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()


def _reference(value):
    need(type(value) is dict and set(value) == {'path', 'bytes', 'sha256'}
         and type(value['path']) is str
         and (PureWindowsPath(value['path']).is_absolute() or PurePosixPath(value['path']).is_absolute())
         and type(value['bytes']) is int and value['bytes'] >= 0
         and type(value['sha256']) is str and len(value['sha256']) == 64
         and set(value['sha256']) <= set('0123456789abcdef'), 'Strict absolute path/bytes/SHA256 reference')
    return value


def _pair(value):
    need(type(value) is list and len(value) == 2
         and all(type(v) is int and v > 0 for v in value), 'Positive integer H/W list required')
    return value


def _queries(value):
    need(type(value) is list and len(value) == 100 and all(type(q) is int for q in value)
         and set(value) == set(range(100)), 'Exact full100 original export permutation; no filtering or sorting')


def _geometry(identity):
    original, content = _pair(identity['original_shape']), _pair(identity['actual_img_shape'])
    longest = max(original)
    need(identity['batch_input_shape'] == [800, 800]
         and content == [(2*v*800+longest)//(2*longest) for v in original],
         'Original integer keep-ratio 800 geometry required')
    return dict(original_shape=list(original), actual_img_shape=list(content), canvas_size=800, branch_size=400)


def _native(values, image_id, qids, shape):
    need(type(values) is list and len(values) == 100, 'Exactly100 original native predictions')
    required = {'image_id', 'query_id', 'category_id', 'score', 'segmentation'}
    allowed = required | {'bbox', 'area'}
    for qid, row in zip(qids, values):
        need(type(row) is dict and required <= set(row) <= allowed,
             'Unreviewed native prediction field ABI; do not drop fields or copy target associations')
        need(type(row['image_id']) is int and row['image_id'] == image_id
             and type(row['query_id']) is int and row['query_id'] == qid
             and type(row['category_id']) is int and row['category_id'] == 1
             and type(row['score']) in (int, float) and math.isfinite(row['score'])
             and 0 <= row['score'] <= 1, 'Original image/query/class/finite S0 and export order')
        rle = row['segmentation']
        need(type(rle) is dict and set(rle) == {'size', 'counts'}
             and _pair(rle['size']) == shape and type(rle['counts']) is str,
             'Original opaque JSON RLE, including empty masks; no decode or regeneration')
        if 'bbox' in row:
            box = row['bbox']
            need(type(box) is list and len(box) == 4
                 and all(type(v) in (int, float) and math.isfinite(v) for v in box), 'Unchanged finite original bbox')
        if 'area' in row:
            need(type(row['area']) in (int, float) and math.isfinite(row['area']) and row['area'] >= 0,
                 'Unchanged finite original area')
        _primitive(row)


def _provenance(value):
    need(type(value) is dict and set(value) == {'source_refs', 'input_refs'}
         and type(value['source_refs']) is dict and set(value['source_refs']) == set(SOURCE_PINS)
         and type(value['input_refs']) is dict and set(value['input_refs']) == set(INPUT_ROLES),
         'Three original reader/geometry sources and five explicit input refs required')
    for role, reference in value['source_refs'].items():
        need(_reference(reference)['sha256'] == SOURCE_PINS[role], 'Original published reader/geometry source: '+role)
    for reference in value['input_refs'].values():
        _reference(reference)
    _primitive(value)


def validate_metadata(metadata):
    """Pure whitelist contract; no byte, tensor, RLE or provenance authentication."""
    need(type(metadata) is dict and set(metadata) == METADATA_KEYS, 'Exact U59 minimal metadata whitelist')
    need(type(metadata['dataset_index']) is int and 0 <= metadata['dataset_index'] < 187,
         'Original DEV ordinal0..186')
    identity = metadata['identity']
    need(type(identity) is dict and set(identity) == set(IDENTITY_KEYS)
         and type(identity['image_id']) is int and identity['image_id'] >= 0
         and type(metadata['image_id']) is int and metadata['image_id'] == identity['image_id'],
         'Minimal original image identity only; no GT-derived associations')
    _reference(identity['image'])
    need(metadata['geometry'] == _geometry(identity), 'Exact original geometry constructor metadata')
    _queries(metadata['query_ids'])
    need(type(metadata['natural_query_ids']) is list and metadata['natural_query_ids'] == list(range(100))
         and all(type(q) is int for q in metadata['natural_query_ids']), 'Natural axis0..99 distinct from export order')
    _native(metadata['native_predictions'], identity['image_id'], metadata['query_ids'], identity['original_shape'])
    rgb = metadata['rgb_metadata']
    need(type(rgb) is dict and set(rgb) == set(RGB_KEYS) and rgb['decoded_mode'] == 'RGB'
         and (rgb['exif_orientation'] is None or type(rgb['exif_orientation']) is int and rgb['exif_orientation'] == 1)
         and rgb['auto_orientation'] is False and rgb['teacher_photometric_reconstruction'] is False
         and rgb['input_convention'] == RGB_CONVENTION and rgb['baseline'] == RGB_BASELINE,
         'Original all100 RGB/255 path, no photometric reconstruction or auto orientation')
    _provenance(metadata['provenance'])
    _primitive(metadata)
    return metadata


def project_metadata(dataset_index, item, originals, *, provenance):
    """Project only reviewed fields; never deepcopy/serialize item, F or geometry.

    The upstream original reader may have opened historical target containers.
    This projection excludes their contents; it cannot certify upstream I/O.
    """
    need(type(item) is dict and set(item) == set(FIELDS) |
         {'F', 'identity', 'query_ids', 'geometry', 'rgb_receipt'}, 'Exact original all100 DEV RGB-reader item ABI')
    identity, rgb = item['identity'], item['rgb_receipt']
    need(type(identity) is dict and set(IDENTITY_KEYS) <= set(identity), 'Original image identity constructor fields required')
    # Validate the projected references and native dictionaries before copying
    # anything. An unexpected nested object must never escape via deepcopy.
    selected = {key: identity[key] for key in IDENTITY_KEYS}
    geometry = _geometry(selected)
    need(type(rgb) is dict and set(RGB_KEYS) <= set(rgb)
         and rgb.get('image') == selected['image'] and rgb.get('original_shape') == selected['original_shape']
         and rgb.get('actual_img_shape') == selected['actual_img_shape']
         and type(rgb.get('canvas_size')) is int and rgb['canvas_size'] == 800
         and type(rgb.get('branch_size')) is int and rgb['branch_size'] == 400
         and rgb.get('image_source_control') == 'same explicitly admitted control', 'Genuine original DEV RGB receipt joins')
    result = dict(dataset_index=dataset_index, image_id=selected['image_id'], identity=selected,
        geometry=geometry, query_ids=item['query_ids'], natural_query_ids=list(range(100)),
        native_predictions=originals, rgb_metadata={key: rgb[key] for key in RGB_KEYS},
        provenance=provenance)
    return deepcopy(validate_metadata(result))


def validate_population_indices(dataset_indices, image_ids):
    """Canonical population metadata only; no tensor or input authentication."""
    need(type(dataset_indices) is list and dataset_indices == list(range(187))
         and all(type(index) is int for index in dataset_indices)
         and type(image_ids) is list and len(image_ids) == 187
         and all(type(image_id) is int and image_id >= 0 for image_id in image_ids)
         and len(set(image_ids)) == 187, 'Exact ordered187 unique-image DEV population')
    return dict(images=187, queries=18700, tensor_population_revalidated=False)


def _dependencies(c, image_geometry, source_refs):
    """Caller supplies a genuine admitted control and source-loaded module.

    closed(ref) must authenticate bytes, not merely return a path. This helper
    does not create that control or admit any source/input itself.
    """
    need(callable(getattr(c, 'closed', None)) and callable(getattr(c, 'check', None)),
         'Root-admitted closed(ref)/check() control required')
    need(type(source_refs) is dict and set(source_refs) == set(SOURCE_PINS),
         'Exact three reviewed original source refs required')
    for role, reference in source_refs.items():
        need(_reference(reference)['sha256'] == SOURCE_PINS[role], 'Original source pin: '+role)
    c.check()
    paths = {role: Path(c.closed(reference)).resolve() for role, reference in source_refs.items()}
    module_file = getattr(image_geometry, '__file__', None)
    need(type(module_file) is str and Path(module_file).resolve() == paths['image_geometry']
         and callable(getattr(image_geometry, 'make_geometry', None)),
         'Original make_geometry loaded from the admitted source path')
    c.check()


def _tensor(torch, value, shape, label):
    need(isinstance(value, torch.Tensor) and value.dtype == torch.float32
         and value.device.type == 'cpu' and tuple(value.shape) == tuple(shape)
         and not value.requires_grad and bool(torch.isfinite(value).all()),
         label+': finite detached CPU FP32 tensor with original shape')
    if label == 'rgb400':
        need(bool(((value >= 0) & (value <= 1)).all()), 'Original RGB400 range [0,1]')
    return value


def _shapes(metadata):
    height, width = metadata['identity']['original_shape']
    return dict(rgb400=(1, 3, 400, 400), e=(100, 256),
                M=(100, 200, 200), L=(100, height, width))


def _field_proof(value):
    # Torch is injected; its installed NumPy bridge supplies these CPU bytes.
    # No conversion, quantization, tensor permutation or values are regenerated.
    raw = value.detach().contiguous().numpy().tobytes()
    return dict(shape=list(value.shape), dtype='float32', sha256=hashlib.sha256(raw).hexdigest())


def _validate_package(torch, value):
    need(type(value) is dict and set(value) == {'schema', 'metadata', 'tensors', 'proof'}
         and value['schema'] == PACKAGE_SCHEMA, 'Exact U59 package ABI')
    metadata = validate_metadata(value['metadata'])
    tensors = value['tensors']
    need(type(tensors) is dict and set(tensors) == set(FIELDS), 'Exactly RGB400/e/M/L; no F or target payload')
    shapes = _shapes(metadata)
    for name in FIELDS:
        _tensor(torch, tensors[name], shapes[name], name)
    proof = value['proof']
    need(type(proof) is dict and set(proof) == {'schema', 'metadata_sha256', 'fields'}
         and proof['schema'] == PROOF_SCHEMA
         and proof['metadata_sha256'] == metadata_sha256(metadata)
         and type(proof['fields']) is dict and set(proof['fields']) == set(FIELDS),
         'Closed metadata and all four explicit field proofs')
    _primitive(proof)
    need(proof['fields'] == {name: _field_proof(tensors[name]) for name in FIELDS},
         'Every original CPU FP32 field byte digest exact')
    return value


def export_item(torch, c, dataset_index, item, originals, *, image_geometry, source_refs, input_refs):
    """Future Root calls once per complete original DEV image, using real Torch.

    No reader, matching, teacher, or launcher is constructed here. Inputs must
    already come from the admitted original reader; upstream historical bundle
    loading may have touched target containers. Only the whitelist escapes.
    """
    _dependencies(c, image_geometry, source_refs)
    metadata = project_metadata(dataset_index, item, originals,
        provenance=dict(source_refs=source_refs, input_refs=input_refs))
    shapes = _shapes(metadata)
    tensors = {}
    for name in FIELDS:
        c.check()
        tensors[name] = _tensor(torch, item[name], shapes[name], name).detach().contiguous().clone()
    proof = dict(schema=PROOF_SCHEMA, metadata_sha256=metadata_sha256(metadata),
                 fields={name: _field_proof(tensors[name]) for name in FIELDS})
    result = _validate_package(torch, dict(schema=PACKAGE_SCHEMA, metadata=metadata, tensors=tensors, proof=proof))
    c.check()
    return result


def _weights_only(torch, path):
    try:
        return torch.load(str(path), map_location='cpu', weights_only=True)
    except TypeError as error:
        # An old/unknown Torch ABI must not silently fall back to pickle load.
        raise ValueError('Explicit CPU weights_only=True load required; unsupported ABI') from error


def _file_reference(c, path):
    digest, size = hashlib.sha256(), 0
    with path.open('rb') as handle:
        while True:
            c.check()
            block = handle.read(1024*1024)
            if not block:
                break
            size += len(block)
            digest.update(block)
    return dict(path=str(path.resolve()), bytes=size, sha256=digest.hexdigest())


def save_export(torch, c, value, output_path, *, image_geometry, source_refs):
    """Exclusive one-image write and real weights-only roundtrip when executed.

    No retry or file replacement. A failed save/roundtrip leaves its partial
    output for Root to preserve, and returns no accepted entry. This code is a
    helper, not Root qualification or a proof that a run has taken place.
    """
    _dependencies(c, image_geometry, source_refs)
    _validate_package(torch, value)
    path = Path(output_path)
    need(path.is_absolute() and path.parent.is_dir() and not path.exists() and not path.is_symlink(),
         'New absolute one-image output in an existing parent; no overwrite')
    c.check()
    with path.open('xb') as handle:
        torch.save(value, handle)
    c.check()
    loaded = _validate_package(torch, _weights_only(torch, path))
    need(loaded['schema'] == value['schema'] and loaded['metadata'] == value['metadata']
         and loaded['proof'] == value['proof']
         and all(torch.equal(loaded['tensors'][name], value['tensors'][name]) for name in FIELDS),
         'Saved/reloaded original metadata and all four CPU FP32 fields must be exact')
    reference = _file_reference(c, path)
    metadata = value['metadata']
    return dict(dataset_index=metadata['dataset_index'], image_id=metadata['image_id'],
                query_ids=deepcopy(metadata['query_ids']), package_ref=reference)


def validate_entries(entries):
    """Full187 manifest whitelist. This validates metadata, not package bytes."""
    need(type(entries) is list and len(entries) == 187, 'Complete ordered187 package entries required')
    paths, indices, image_ids = set(), [], []
    for entry in entries:
        need(type(entry) is dict and set(entry) == {'dataset_index', 'image_id', 'query_ids', 'package_ref'},
             'Exact per-image package entry; no original row/bundle or target metadata')
        _queries(entry['query_ids'])
        reference = _reference(entry['package_ref'])
        key = reference['path'].replace('\\', '/').casefold()
        need(key not in paths, 'Distinct package path for each canonical DEV image')
        paths.add(key)
        indices.append(entry['dataset_index'])
        image_ids.append(entry['image_id'])
    validate_population_indices(indices, image_ids)
    _primitive(entries)
    return deepcopy(entries)


class _Reader:
    def __init__(self, torch, c, entries, image_geometry, source_refs):
        self._torch, self._c, self._geometry = torch, c, image_geometry
        self._entries = deepcopy(entries)
        self.source_references = deepcopy(source_refs)
        self.records = [dict(dataset_index=entry['dataset_index'], image_id=entry['image_id'],
                             qids=deepcopy(entry['query_ids'])) for entry in entries]
        self._records = deepcopy(self.records)

    def load(self, index):
        """Return (item,row,originals) for the original per-image calibration.

        This is intentionally not a drop-in replacement for the old _arm reader
        ABI: Root unpacks these three values and passes them to the unchanged
        _calibration_and_prediction. No GT/RLE decode or residual inference here.
        """
        need(type(index) is int and 0 <= index < 187 and self.records == self._records,
             'Unchanged canonical187 ordinal/record mapping')
        torch, c = self._torch, self._c
        c.check()
        entry = self._entries[index]
        reference = entry['package_ref']
        path = Path(c.closed(reference))
        need(path.resolve() == Path(reference['path']).resolve(), 'Exact admitted portable package path')
        value = _validate_package(torch, _weights_only(torch, path))
        metadata = value['metadata']
        need(metadata['dataset_index'] == index and metadata['image_id'] == entry['image_id']
             and metadata['query_ids'] == entry['query_ids'],
             'Package image/ordinal/export permutation joins original full187 manifest')
        shape = metadata['geometry']
        geometry = self._geometry.make_geometry(torch, shape['original_shape'], shape['actual_img_shape'],
                                               shape['canvas_size'], shape['branch_size'])
        need(callable(getattr(geometry, 'render_delta', None)) and callable(getattr(geometry, 'prepare_rgb', None)),
             'Original image geometry object ABI; no fallback renderer or inferred geometry')
        item = dict(value['tensors'])
        item.update(identity=deepcopy(metadata['identity']), query_ids=deepcopy(metadata['query_ids']),
                    geometry=geometry)
        originals = deepcopy(metadata['native_predictions'])
        c.check()
        return item, deepcopy(self._records[index]), originals


def prepare(torch, c, entries, *, image_geometry, source_refs):
    """Future injected-Torch reader; close actual package bytes only on load().

    The caller admits the 187 package refs and the three actual local source
    refs. Original provenance paths inside a package remain historical metadata;
    they are not opened on the destination host. No model or worker is started.
    """
    _dependencies(c, image_geometry, source_refs)
    return _Reader(torch, c, validate_entries(entries), image_geometry, source_refs)
