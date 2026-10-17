"""UNBOUND all-100 DEV RGB bridge; numerical path has not been executed.

Only stdlib imports occur here. This module neither qualifies an evaluation nor
changes GT, native tensors, scores, categories, or prediction export order.
"""
from copy import deepcopy
import hashlib
from pathlib import Path

SOURCE_SHA256 = {
    'dev_reader': 'daabec8388e23d9ff646ae78eba69ba2168099a72971578a8d34cef04f1268c0',
    'image_geometry': '65db107de80c5f1d4dade7cb2ffd3f241fd6bf7f9b233cb1aa9c90cbfd00abf7',
}
# Provenance only, not an imported runtime dependency. JPEG/EXIF/RGB handling
# below follows reviewed U37 rgb_matched_reader.py lines 103-138 verbatim in
# behavior; its FIT-specific G/GT checks, class and receipt are not reused.
JPEG_LOGIC_PROVENANCE_SHA256 = '678ce3781872e21e961b24a7b0a4d0ecc69f0f8a08b8feb7eb0b2e8624d57025'
BASE_KEYS = frozenset(('F', 'e', 'M', 'L', 'identity', 'query_ids'))
ADDED_KEYS = ('rgb400', 'geometry', 'rgb_receipt')


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def _reference(value):
    need(type(value) is dict and set(value) == {'path', 'bytes', 'sha256'},
         'Exact path/bytes/sha256 reference required')
    need(type(value['path']) is str and Path(value['path']).is_absolute()
         and type(value['bytes']) is int and value['bytes'] >= 0
         and type(value['sha256']) is str and len(value['sha256']) == 64
         and set(value['sha256']) <= set('0123456789abcdef'),
         'Absolute strict reference with lowercase SHA256 required')
    return deepcopy(value)


def _path_key(reference):
    return reference['path'].replace('\\', '/').casefold()


def prepare_image_references(records, original_image_refs):
    """Validate synthetic-testable metadata only; never open or infer a JPEG.

    Each image ref belongs to its unchanged original dataset ordinal. Its
    actual packet identity.image must still be joined when that item is loaded.
    """
    need(type(records) is list and type(original_image_refs) is list
         and len(records) == len(original_image_refs) == 187,
         'Complete original 187-image DEV mapping required')
    images, image_ids, paths = [], set(), set()
    for index, (record, value) in enumerate(zip(records, original_image_refs)):
        need(type(record) is dict and type(record.get('dataset_index')) is int
             and record['dataset_index'] == index
             and type(record.get('image_id')) is int,
             'Original DEV dataset ordinal and image id required')
        queries = record.get('qids')
        need(type(queries) is list and len(queries) == 100
             and all(type(q) is int for q in queries)
             and set(queries) == set(range(100)),
             'Original all-100 query export permutation required')
        reference = _reference(value)
        key = _path_key(reference)
        need(record['image_id'] not in image_ids and key not in paths,
             'Distinct canonical DEV image ids and admitted JPEG paths required')
        image_ids.add(record['image_id'])
        paths.add(key)
        images.append(reference)
    return images


def _check_native_item(item, record, image_reference):
    need(type(item) is dict and not any(key in item for key in ADDED_KEYS),
         'RGB bridge must not overwrite any existing item field')
    need(set(item) == BASE_KEYS,
         'Unchanged DEV native item required; no GT fields or synthetic count shim')
    reference = _reference(image_reference)
    identity = item['identity']
    need(type(identity) is dict and identity.get('image_id') == record['image_id']
         and _reference(identity.get('image')) == reference,
         'Actual packet identity.image must exactly match this admitted ordinal')
    queries = record['qids']
    need(type(queries) is list and len(queries) == 100
         and all(type(q) is int for q in queries)
         and set(queries) == set(range(100))
         and type(item['query_ids']) is list and item['query_ids'] == queries,
         'Preserve original query export permutation; native tensor axes stay natural')
    return reference


def compose_rgb_item(item, record, image_reference, *, rgb400, geometry, rgb_receipt):
    """Shallow metadata composition; preserves all original objects unchanged.

    No tensor checks or numerical execution occur in this public helper. The
    reader load path must validate the actual new RGB tensor before calling it.
    """
    reference = _check_native_item(item, record, image_reference)
    need(type(rgb_receipt) is dict and rgb_receipt.get('image') == reference,
         'RGB receipt must identify the same admitted image')
    result = dict(item)
    result.update(rgb400=rgb400, geometry=geometry, rgb_receipt=rgb_receipt)
    return result


def _references(values):
    unique, result = {}, []
    for value in values:
        reference = _reference(value)
        key = _path_key(reference)
        need(key not in unique or unique[key] == reference,
             'Conflicting admitted source/image/input reference')
        if key not in unique:
            unique[key] = reference
            result.append(reference)
    return result


def _source_module(c, role, module, value):
    reference = _reference(value)
    need(reference['sha256'] == SOURCE_SHA256[role],
         'Exact reviewed '+role+' source required')
    path = Path(c.closed(reference)).resolve(strict=True)
    module_file = getattr(module, '__file__', None)
    need(type(module_file) is str and Path(module_file).resolve(strict=True) == path,
         role+' must be loaded from its admitted source path')
    raw = path.read_bytes()
    need(len(raw) == reference['bytes']
         and hashlib.sha256(raw).hexdigest() == SOURCE_SHA256[role],
         role+' source bytes changed')


class _RGBReader:
    """Pending numerical path: no real image/tensor execution has been tested."""

    def __init__(self, torch, np, c, base, geometry_module, images, source_refs):
        self._torch, self._np, self._c = torch, np, c
        self._base, self._geometry_module = base, geometry_module
        self.records = base.records
        self._records = deepcopy(base.records)
        self._images = deepcopy(images)
        self.image_references = deepcopy(images)  # original ordinal order
        self.source_references = deepcopy(source_refs)
        self.input_references = _references(
            list(base.input_references) + images + list(source_refs.values()))

    def load(self, index):
        """Preserve natural F/e/M/L axes and the separate export query order.

        The caller must index e/M/L by the actual exported query id, not rank.
        Neither GT matching nor prediction scores/categories enter this path.
        """
        need(type(index) is int and 0 <= index < 187, 'Original DEV ordinal required')
        torch, np, c = self._torch, self._np, self._c
        c.check()
        need(self.records is self._base.records and self.records == self._records,
             'Original DEV metadata/order changed after bridge preparation')
        item = self._base.load(index)
        record = self._records[index]
        reference = _check_native_item(item, record, self._images[index])
        identity = item['identity']
        original, content = identity['original_shape'], identity['actual_img_shape']
        need(identity['batch_input_shape'] == [800, 800], 'Original 800 padded canvas required')
        geometry = self._geometry_module.make_geometry(torch, original, content, 800, 400)

        # Source: U37 rgb_matched_reader.py 103-138, SHA pinned above. No FIT
        # reader, GT field, G shim, filename inference, or registration fallback.
        image_path = Path(c.closed(reference)).resolve(strict=True)
        need(image_path == Path(reference['path']).resolve(strict=True),
             'Read precisely the admitted actual identity.image path')
        from PIL import Image
        with Image.open(image_path) as image:
            need(image.format == 'JPEG', 'The admitted original image must be JPEG')
            need(image.size == (original[1], original[0]),
                 'JPEG dimensions must match original L orientation')
            orientation = image.getexif().get(274)
            need(orientation is None or (type(orientation) is int and orientation == 1),
                 'Only absent/identity EXIF orientation; no automatic rotation')
            original_mode = image.mode
            rgb = image.convert('RGB')
            try:
                pixels = np.array(rgb, dtype=np.uint8, copy=True)
            finally:
                rgb.close()
        need(pixels.dtype == np.uint8 and pixels.shape == (original[0], original[1], 3),
             'Decoded original RGB uint8 grid required')
        image_uint8 = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).contiguous()
        need(image_uint8.device.type == 'cpu' and image_uint8.dtype == torch.uint8,
             'Image preparation must remain CPU uint8 before common geometry')
        rgb400 = geometry.prepare_rgb(image_uint8)
        need(tuple(rgb400.shape) == (1, 3, 400, 400)
             and rgb400.dtype == torch.float32 and rgb400.device.type == 'cpu'
             and not rgb400.requires_grad and bool(torch.isfinite(rgb400).all())
             and bool(((rgb400 >= 0) & (rgb400 <= 1)).all()),
             'RGB400 must be finite detached CPU FP32 in [0,1]')
        del pixels, image_uint8
        c.check()
        receipt = dict(image=reference, original_shape=list(original),
            actual_img_shape=list(content), canvas_size=800, branch_size=400,
            original_mode=original_mode, decoded_mode='RGB',
            exif_orientation=orientation, auto_orientation=False,
            input_convention='RGB uint8 -> FP32 /255; unchanged reviewed geometry',
            teacher_photometric_reconstruction=False,
            native_tensor_query_axis='natural query ids 0..99; unchanged',
            prediction_query_order='original export permutation; unchanged',
            baseline='original all100 N16 native_L; not GT-matched selection',
            image_source_control='same explicitly admitted control')
        return compose_rgb_item(item, record, reference, rgb400=rgb400,
                                geometry=geometry, rgb_receipt=receipt)


def prepare(torch, np, c, context, original_image_refs, *, dev_reader,
            image_geometry, source_refs):
    """Compose source-closed original DEV reader and 187 pre-admitted JPEGs.

    This is not an evaluation runner or eligibility gate. Future Root-owned
    execution must validate actual inputs/decoding/numerics and fixed endpoint.
    Torch/NumPy are caller-injected; PIL is imported only inside load().
    """
    need(callable(getattr(c, 'closed', None)) and callable(getattr(c, 'check', None)),
         'Already admitted closed(ref)/check() control required')
    need(type(source_refs) is dict and set(source_refs) == set(SOURCE_SHA256),
         'Exact DEV-reader and geometry source references required')
    c.check()
    _source_module(c, 'dev_reader', dev_reader, source_refs['dev_reader'])
    _source_module(c, 'image_geometry', image_geometry, source_refs['image_geometry'])
    base = dev_reader.prepare(torch, c, context)
    images = prepare_image_references(base.records, original_image_refs)
    # Pre-registration belongs to the caller's Source map; actual image bytes
    # are closed and joined to packet identity only for the requested load.
    result = _RGBReader(torch, np, c, base, image_geometry, images, source_refs)
    c.check()
    return result
