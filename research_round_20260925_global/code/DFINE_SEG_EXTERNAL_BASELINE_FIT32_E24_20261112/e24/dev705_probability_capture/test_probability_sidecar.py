from io import BytesIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from probability_sidecar import Capture, verify_probability_file


with TemporaryDirectory() as tmp:
    directory = Path(tmp)
    audit = {"image_id": 42, "candidates": [
        {"native_rank": i + 1, "query_id": 23 - i, "probability_row": i}
        for i in range(2)
    ]}
    planes = np.zeros((2, 160, 160), dtype=np.float32)
    planes[1, 30, 40] = .75
    with Capture(directory) as capture:
        detail = capture.write(1, 42, audit, planes)
        receipt = capture.receipt()
    row = json.loads((directory / "PROBABILITIES_PRIVATE.jsonl").read_bytes())
    assert row["image_id"] == 42 and row["query_ids"] == [23, 22]
    assert all("query_id" not in r and "probability_row" not in r for r in detail["candidates"])
    assert receipt["probability_rows"] == 2 and receipt["probability_files"] == 1
    data = verify_probability_file(directory, 1, row["probability"], 2)
    assert np.array_equal(data, planes)
    path = directory / row["probability"]["file"]
    path.write_bytes(path.read_bytes()[:-1] + b"!")
    try:
        verify_probability_file(directory, 1, row["probability"], 2)
    except ValueError:
        pass
    else:
        raise AssertionError("Changed probability bytes accepted")
print("synthetic sidecar roundtrip and tamper rejection passed")

