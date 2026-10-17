"""UNBOUND RGB400 composition of the unchanged public N28 matched FIT reader.

No Torch/NumPy/PIL import, image/cache read or numerical work at module import.
Future callers must supply admitted controls and the two source-closed modules.
This module adds an image branch; it neither changes the native baseline L nor
grants a population, training, DEV, export-score or scientific qualification.
"""

from copy import deepcopy
import hashlib
from pathlib import Path


SOURCE_SHA256 = {
    'matched_reader': '82565caa9c42b6c2ed501e13af9554c6ef47a99b6b525ccfce0bcef620534870',
    'image_geometry': '65db107de80c5f1d4dade7cb2ffd3f241fd6bf7f9b233cb1aa9c90cbfd00abf7',
}
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
         'Absolute source reference with lowercase SHA256 required')
    return deepcopy(value)


def _control(value):
    need(callable(getattr(value, 'closed', None)) and callable(getattr(value, 'check', None)),
         'Caller must supply an already admitted closed(ref)/check() control')
    return value


def _source_module(c, role, module, reference):
    reference = _reference(reference)
    need(reference['sha256'] == SOURCE_SHA256[role], 'Exact reviewed '+role+' source required')
    path = Path(c.closed(reference)).resolve(strict=True)
    module_file = getattr(module, '__file__', None)
    need(type(module_file) is str and Path(module_file).resolve(strict=True) == path,
         role+' must be loaded from its admitted source path')
    raw = path.read_bytes()
    need(len(raw) == reference['bytes']
         and hashlib.sha256(raw).hexdigest() == SOURCE_SHA256[role],
         role+' source bytes differ')


def _references(values):
    result, seen = [], {}
    for value in values:
        reference = _reference(value)
        key = reference['path'].replace('\\', '/').casefold()
        need(key not in seen or seen[key] == reference, 'Conflicting image/input references')
        if key not in seen:
            seen[key] = reference
            result.append(reference)
    return result


class _RGBReader:
    def __init__(self, torch, np, c, image_control, base, geometry_module):
        self._torch, self._np, self._c = torch, np, c
        self._image_control, self._base = image_control, base
        self._geometry_module = geometry_module
        self._records = deepcopy(base.records)
        self.records = deepcopy(self._records)
        images = []
        for index, record in enumerate(self._records):
            need(record['dataset_index'] == index and type(record['identity']) is dict,
                 'Original reader record order/identity required')
            need('image' in record['identity'],
                 'Original identity.image is missing; no filename inference or fallback')
            images.append(_reference(record['identity']['image']))
        # Metadata enumeration only. Each actual JPEG must pass the selected
        # admitted control.closed() before any PIL open in load().
        self.image_references = _references(images)
        self.input_references = _references(list(base.input_references) + images)

    def load(self, index):
        need(type(index) is int and 0 <= index < len(self._records),
             'Original matched-reader image ordinal required')
        torch, np, c = self._torch, self._np, self._c
        image_control = self._image_control
        c.check()
        if image_control is not c:
            image_control.check()

        # Keep the entire existing tensor/GT/source/matcher verification intact.
        item = self._base.load(index)
        record = self._records[index]
        need(type(item) is dict and item['identity'] == record['identity']
             and item['G'] == record['G'] and item['query_ids'] == record['query_ids'],
             'Base reader identity and GT-ordered matched queries must remain unchanged')
        need(not any(key in item for key in ADDED_KEYS), 'RGB addition must not overwrite a base field')
        identity = item['identity']
        reference = _reference(identity['image'])
        original, content = identity['original_shape'], identity['actual_img_shape']
        need(identity['batch_input_shape'] == [800, 800], 'Original 800 padded canvas required')
        geometry = self._geometry_module.make_geometry(torch, original, content, 800, 400)

        # No registration fallback: a separate control is an explicit caller
        # choice, not a retry after base-control rejection of an unknown JPEG.
        image_path = Path(image_control.closed(reference)).resolve(strict=True)
        need(image_path == Path(reference['path']).resolve(strict=True),
             'Read precisely the original identity.image path')
        from PIL import Image
        with Image.open(image_path) as image:
            need(image.format == 'JPEG', 'The admitted original image must be JPEG')
            need(image.size == (original[1], original[0]), 'JPEG dimensions must match original GT/L orientation')
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
             'New common RGB400 branch must be finite CPU FP32 in [0,1]')
        del pixels, image_uint8
        c.check()
        if image_control is not c:
            image_control.check()
        # Shallow composition preserves every original value/object, including
        # F/e/M/L/targets, identity, query_ids and G. Nothing is re-matched.
        result = dict(item)
        result.update(rgb400=rgb400, geometry=geometry, rgb_receipt=dict(
            image=reference, original_shape=list(original), actual_img_shape=list(content),
            canvas_size=800, branch_size=400, original_mode=original_mode,
            decoded_mode='RGB', exif_orientation=orientation, auto_orientation=False,
            input_convention='RGB uint8 -> FP32 /255; common reviewed geometry',
            teacher_photometric_reconstruction=False,
            baseline='N16 native_L selected by unchanged original GT-matched query ids',
            image_source_control='base' if image_control is c else 'separate_explicit_admitted_control'))
        return result


def prepare(torch, np, c, context, *, matched_reader, image_geometry, source_refs,
            image_control=None):
    """Compose the old prepare/load ABI; execute only under future admission.

    source_refs has exactly matched_reader and image_geometry. The caller must
    Source-close this wrapper and its normal libraries too. If images use a new
    pre-collected Source, supply its already admitted image_control explicitly.
    No missing identity bridge, teacher normalization or population filter is
    synthesized here. The original reader retains its FIT741/1661 contract.
    """
    _control(c)
    image_control = c if image_control is None else _control(image_control)
    need(type(source_refs) is dict and set(source_refs) == set(SOURCE_SHA256),
         'Exactly two reviewed dependency source references required')
    c.check()
    _source_module(c, 'matched_reader', matched_reader, source_refs['matched_reader'])
    _source_module(c, 'image_geometry', image_geometry, source_refs['image_geometry'])
    base = matched_reader.prepare(torch, np, c, context)
    result = _RGBReader(torch, np, c, image_control, base, image_geometry)
    c.check()
    return result
