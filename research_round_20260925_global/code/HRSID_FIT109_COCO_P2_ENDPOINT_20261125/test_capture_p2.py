"""Synthetic native image contract checks independent from the expensive backbone."""

from hashlib import sha256
from pathlib import Path
from runpy import run_path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image


worker = run_path(str(Path(__file__).with_name("capture_p2.py")))
decode_native = worker["decode_native"]
data = np.empty((800, 800, 3), dtype=np.uint8)
data[:] = (19, 41, 63)
with TemporaryDirectory() as folder:
    path = Path(folder) / "P001_test.png"
    Image.fromarray(data).save(path)
    raw = path.read_bytes()
    row = dict(native_basename=path.name, encoded_bytes=len(raw), encoded_sha256=sha256(raw).hexdigest(),
               decoded_rgb_sha256=sha256(data.tobytes()).hexdigest())
    assert np.array_equal(np.asarray(decode_native(path.parent, row)), data)
    row["decoded_rgb_sha256"] = "0" * 64
    try:
        decode_native(path.parent, row)
    except ValueError as error:
        assert "RGB" in str(error)
    else:
        raise AssertionError("Changed source RGB digest should fail closed")
print("synthetic exact native input reader passed")
