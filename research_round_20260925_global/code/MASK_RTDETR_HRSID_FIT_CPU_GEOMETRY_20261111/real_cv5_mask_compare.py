"""Compare OpenCV 5 samplers on the exact original Mask RT-DETR FIT raster inputs."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from time import perf_counter


PINS = {
    "replay.py": "58ffa34c705e9fdb295ae25ef596ecaabebf495c6695697ad6346d30839cc834",
    "scope.py": "9ce6427b5123849a055874165af57fe913f48520f0e7d06e5c25c3a9ddd933c7",
    "footprint.py": "060a9101ca428ed48d74648bd04b049a969b028f71bbaa0b3db6e56fe78967ff",
    "paddle_geometry.py": "9508c2b1e852b0d0fb35c81e58399e162ae9d5fdda04b9c2f0d37b9cc4ab5240",
}
PLAN_SHA = "3d78297dd49a99be208a91f4bf45d2d56a5d0f1ae55b80328dbb4a42de6102ed"
OLD_RESULT_SHA = "da102a21ac7a81316061e2812c97db5761a8cad1b4af21558e7481141c7bfb88"
PREREG = "eb559b8a180f05ee56ff658a29fa0b3f4d0fef00"
VIEWS = 96
ROWS = 615
GROUPS = 24


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pack(rows):
    import numpy as np
    require(rows.dtype == np.uint8 and rows.ndim == 3 and not np.any(rows > 1),
            "Expected Boolean raster rows stored as uint8")
    return np.packbits(rows, axis=-1)


def unpack(packed, shape):
    import numpy as np
    n, h, w = shape
    require(packed.shape == (n, h, (w + 7) // 8), "Packed mask size mismatch")
    return np.unpackbits(packed, axis=-1, count=w)


def record_comparison(native4, center, native5, exact5, input_hw):
    import numpy as np
    require(native4.shape == center.shape == native5.shape == exact5.shape, "Mask sizes differ")
    h, w = center.shape
    ty = (2 * np.arange(h, dtype=np.int64) + 1) * input_hw[0] % (2 * h) == 0
    tx = (2 * np.arange(w, dtype=np.int64) + 1) * input_hw[1] % (2 * w) == 0
    diff = exact5 != center
    tie = int(np.count_nonzero(diff & (ty[:, None] | tx[None, :])))
    changed = int(np.count_nonzero(diff))
    return int(np.count_nonzero(native5 != native4)), changed, tie, changed - tie


def read_hash(path, pin):
    data = path.read_bytes()
    require(sha256(data).hexdigest() == pin, "Historical source or result hash differs")
    return data


def emit(path, report):
    payload = (json.dumps(report, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as file:
        file.write(payload)
    print(json.dumps(dict(event=report["status"], sha256=sha256(payload).hexdigest())), flush=True)


def capture(args):
    import numpy as np
    import replay

    old = json.loads(read_hash(args.old_result, OLD_RESULT_SHA))
    previous = replay.view_rows
    groups = {}
    captured = []

    def capture_view(tagged, stock, source_rows, group, census, records):
        previous(tagged, stock, source_rows, group, census, records)
        ordinal = len(captured)
        masks = tagged.before_resize["gt_segm"]
        a, b = tagged.output["gt_segm"], tagged.center_masks
        group_index = groups.setdefault(group, len(groups))
        if not len(masks):
            require(len(a) == len(b) == 0, "Empty view row count differs")
            captured.append(None)
            return
        path = args.output.parent / f"view_{ordinal:02d}.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, original=pack(masks), native=pack(a), center=pack(b),
                                input_shape=masks.shape, output_shape=a.shape, group=group_index)
        captured.append(dict(name=path.name, sha256=sha256(path.read_bytes()).hexdigest()))

    replay.view_rows = capture_view
    repeat = args.output.parent / "ORIGINAL_REPLAY_SAFE.json"
    replay.run(argparse.Namespace(bundle=args.bundle, plan=args.plan, plan_sha256=PLAN_SHA,
                                 upstream=args.upstream, protocol_commit=PREREG, output=repeat))
    repeated = json.loads(repeat.read_bytes())
    require({k:v for k,v in old.items() if k != "elapsed_seconds"}
            == {k:v for k,v in repeated.items() if k != "elapsed_seconds"},
            "OpenCV 4 original report was not reproduced")
    require(len(captured) == VIEWS and len(groups) == GROUPS
            and repeated["counts"]["rows_after_integer_ops"] == ROWS, "Incorrect original FIT population")
    emit(args.output, dict(status="MASK_RTDETR_CV4_FIT96_CAPTURE_CLOSED", views=captured,
                           rows=ROWS, groups=GROUPS, capture_opencv=version("opencv-python"),
                           original_report_reproduced=True, fit_only=True))


def compare(args):
    import cv2
    import numpy as np
    from footprint import exact_footprint_scores

    require((cv2.__version__, version("opencv-python"), np.__version__) == ("5.0.0", "5.0.0.93", "2.2.6"),
            "Exact CV5 runtime required")
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    captured = json.loads(read_hash(args.capture, args.capture_sha256))
    require(captured["status"] == "MASK_RTDETR_CV4_FIT96_CAPTURE_CLOSED"
            and captured["rows"] == ROWS and captured["groups"] == GROUPS
            and captured["fit_only"] and captured["original_report_reproduced"]
            and len(captured["views"]) == VIEWS, "Closed CV4 FIT capture required")

    counts = Counter()
    groups = defaultdict(list)
    changed_shapes = set()
    for ordinal, reference in enumerate(captured["views"]):
        counts["views"] += 1
        if reference is None:
            counts["zero_row_views"] += 1
            continue
        require(reference["name"] == f"view_{ordinal:02d}.npz", "Unexpected view file")
        from io import BytesIO
        data = read_hash(args.capture.parent / reference["name"], reference["sha256"])
        with np.load(BytesIO(data), allow_pickle=False) as f:
            pre = unpack(f["original"], tuple(f["input_shape"]))
            native = unpack(f["native"], tuple(f["output_shape"]))
            center = unpack(f["center"], tuple(f["output_shape"]))
            group = int(f["group"])
        require(0 <= group < GROUPS, "Unexpected group index")
        out_h, out_w = native.shape[-2:]
        view_changed = False
        for src, a, b in zip(pre, native, center, strict=True):
            input_hw = src.shape
            a5 = cv2.resize(src, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            e5 = cv2.resize(src, (out_w, out_h), interpolation=cv2.INTER_NEAREST_EXACT)
            n, e, tie, outside = record_comparison(a, b, a5, e5, input_hw)
            counts.update(rows=1, default_version_changed_masks=int(n > 0), default_version_xor=n,
                          exact_vs_center_changed_masks=int(e > 0), exact_vs_center_xor=e,
                          exact_vs_center_xor_at_axis_tie=tie, exact_vs_center_xor_outside_axis_tie=outside,
                          exact_vs_native4_changed_masks=int(np.any(e5 != a)),
                          exact_vs_center_lost_nonempty=int(b.any() and not e5.any()),
                          exact_vs_center_gained_nonempty=int(e5.any() and not b.any()))
            if e:
                view_changed = True
                changed_shapes.add((*input_hw, out_h, out_w))
            foreground = int(np.count_nonzero(src))
            if not 0 < foreground * out_h * out_w < 1024 * input_hw[0] * input_hw[1]:
                continue
            standard = exact_footprint_scores(src, (a, b))
            exact = exact_footprint_scores(src, (a, e5))["center"] if e else standard["center"]
            groups[group].append((standard["native"], exact, standard["center"]))
        counts["exact_vs_center_changed_views"] += view_changed
    require(counts["rows"] == ROWS and counts["views"] == VIEWS and len(groups) == GROUPS,
            "Incomplete FIT comparison")
    output = {}
    for arm, label in enumerate(("default4", "exact5", "integer_center")):
        iou, y = [], []
        for rows in groups.values():
            iou.append(np.mean([row[arm]["iou"] for row in rows]))
            common = [row[arm]["signed_y_px"] for row in rows
                      if all(m["signed_y_px"] is not None for m in row)]
            if common:
                y.append(np.mean(np.abs(common)))
        output[label] = dict(continuous_footprint_iou=float(np.mean(iou)),
                             absolute_y_px_common_three=float(np.mean(y)))
    emit(args.output, dict(status="MASK_RTDETR_CV5_REAL_FIT96_COMPARISON_COMPLETE",
                           cv4= captured["capture_opencv"], cv5=version("opencv-python"),
                           view_groups=len(groups), counts=dict(counts), differing_shape_pairs=len(changed_shapes),
                           small_rows=sum(map(len,groups.values())), small_group_equal_stats=output,
                           original_report_reproduced=True, fit_only=True, AP=False, DEV=False, TEST=False,
                           model_forward_calls=0, optimizer_updates=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("capture", "compare"))
    for name in ("runtime", "original", "output", "bundle", "plan", "upstream", "old-result", "capture"):
        parser.add_argument("--" + name, type=Path, required=name in ("runtime", "original", "output"))
    parser.add_argument("--capture-sha256")
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and os.environ.get("OMP_NUM_THREADS") == "1",
            "CPU-only one-thread experiment required")
    required = ("bundle","plan","upstream","old_result") if args.stage == "capture" else ("capture","capture_sha256")
    require(all(getattr(args, name) for name in required), "Required stage inputs missing")
    for name, pin in PINS.items():
        read_hash(args.original / name, pin)
    sys.path[:0] = [str(args.runtime), str(args.original)]

    def offline(event, payload):
        if event.startswith("socket."):
            raise RuntimeError("Offline FIT only")

    sys.addaudithook(offline)
    start = perf_counter()
    (capture if args.stage == "capture" else compare)(args)
    print(json.dumps(dict(stage=args.stage, seconds=perf_counter()-start)), flush=True)


if __name__ == "__main__":
    main()
