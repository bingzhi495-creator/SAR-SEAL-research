"""Independently read closed window PNGs and check raw tile and affine claims."""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image


def require(criterion, reason):
    if not criterion:
        raise ValueError(reason)


def parse_pinned(path, expected):
    payload = path.read_bytes()
    require(sha256(payload).hexdigest() == expected, f"Digest mismatch for {path.name}")
    return json.loads(payload)


def axis_origin(size):
    # range order resolves ties towards the smaller coordinate.
    return min(range(0, size - 1024 + 1, 512), key=lambda v: abs(2*v + 1024 - size))


def main():
    parser = ArgumentParser(description=__doc__)
    for arg in ("terminal", "output", "cohort", "result"):
        parser.add_argument("--" + arg, required=True, type=Path)
    args = parser.parse_args()
    terminal = json.loads(args.terminal.read_bytes())
    require(terminal["phase"] == "native-window-intake" and terminal["actual_exit"] in (0, 1)
            and terminal["output_footer_verified"] is True, "Observed terminal admission required")
    summary = parse_pinned(args.output / "RESULT_SAFE.json", terminal["result_sha256"])
    cohort = parse_pinned(args.cohort, summary["manifest_sha256"])
    provenance = parse_pinned(args.output / "PROVENANCE_PRIVATE.json", summary["provenance_sha256"])
    require(provenance["manifest_sha256"] == summary["manifest_sha256"], "Frozen manifest binding")
    rows = provenance["results"]
    ids = [row["window_id"] for row in rows]
    require(ids == [row["window_id"] for row in cohort["items"]] and len(ids) == summary["requested_windows"],
            "Original complete ordered request roster")
    names = set()
    raw_intervals = 0
    source_bytes = 0
    all_steps = []
    native_hashes = set()
    for item, row in zip(cohort["items"], rows):
        if row["status"] != "complete":
            require(row["status"] == "failed", "Unknown terminal window status")
            continue
        require(row["author_url"] == item["author_url"] and row["etag"] == item["etag"]
                and row["object_bytes"] == item["object_bytes"], "Exact author-object pin replay")
        name = row["window_id"] + ".png"
        path = args.output / "images" / name
        require(sha256(path.read_bytes()).hexdigest() == row["png_sha256"], "Independent PNG bytes SHA")
        with Image.open(path, formats=["PNG"]) as image:
            require(image.mode == "L" and image.size == (1024, 1024) and not image.info,
                    "Anonymous native uint8 image required")
            decoded = np.array(image)
        require(decoded.shape == (1024, 1024) and decoded.dtype == np.uint8
                and row["native_dtype"] == "|u1" and [1024, 1024] == row["native_shape_yx"],
                "Native numeric raster shape")
        values = sha256(decoded.tobytes()).hexdigest()
        require(values == row["native_array_sha256"], "Independent decoded pixel SHA")
        native_hashes.add(values)
        y, x = (axis_origin(size) for size in row["full_raster_shape_yx"])
        require([y, x] == row["native_origin_yx"], "Nearest complete zero-offset tie window")
        source_affine = np.array(row["source_affine_ij_to_xy"]).reshape(2, 3)
        shifted = source_affine @ np.array(((1, 0, x), (0, 1, y), (0, 0, 1)))
        require(np.allclose(shifted.ravel(), row["window_affine_ij_to_xy"], rtol=0, atol=1e-8),
                "Independent affine composition")
        steps = np.linalg.norm(source_affine[:, :2], axis=0)
        require(np.allclose(steps, row["projected_pixel_step_m_xy"], rtol=1e-10, atol=0),
                "Author affine projected pixel steps")
        all_steps.extend(steps.tolist())
        ranges = row["source_ranges"]
        require(row["source_header_bytes"] == 65536 and len(ranges) == 4,
                "Only complete native tile intervals")
        source_bytes += row["source_header_bytes"]
        for (dy, dx), rr in zip(((0, 0), (0, 512), (512, 0), (512, 512)), ranges):
            require([y+dy, x+dx] == rr["row_col"]
                    and 0 <= rr["offset"] < rr["offset"] + rr["length"] <= row["object_bytes"],
                    "Exact replayed tile row, column and source interval")
            source_bytes += rr["length"]
            if row["compression"] == "NONE":
                raw = decoded[dy:dy+512, dx:dx+512].tobytes()
                require(len(raw) == rr["length"] and sha256(raw).hexdigest() == rr["sha256"],
                        "Raw origin tile SHA independently recovered from PNG")
                raw_intervals += 1
        names.add(name)
    require({entry.name for entry in (args.output/"images").iterdir()} == names
            and len(names) == summary["verified_native_windows"]
            and len(native_hashes) == summary["decoded_unique_windows"], "Exact admitted PNG roster")
    require(source_bytes <= summary["http_payload_bytes_reserved"] <= summary["http_payload_cap_bytes"],
            "Bounded network byte accounting")
    data = {
        "status": "INDEPENDENT_NATIVE_WINDOW_READBACK_COMPLETE",
        "extraction_actual_exit": terminal["actual_exit"],
        "extraction_result_sha256": terminal["result_sha256"],
        "manifest_sha256": summary["manifest_sha256"],
        "requested_windows": len(ids),
        "readback_windows": len(names),
        "unique_pixel_arrays": len(native_hashes),
        "matching_uncompressed_ranges": raw_intervals,
        "intake_payload_bytes_confirmed": source_bytes,
        "projected_spacing_min_max_m": [min(all_steps), max(all_steps)] if all_steps else [],
        "source_coords_exposed": False,
        "annotations_read": 0,
        "detector_forwards": 0,
    }
    payload = (json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with args.result.open("xb") as output:
        output.write(payload)
    print(sha256(payload).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
