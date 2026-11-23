"""Build a label-free Phoenix request from the independently verified 74-image classical replay."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace


LOCK = "0160a2ee707bb95d2c8342cec229cc0c06e0edb4d9860e857701c23cf6b61052"
CFA_SHA = "b78131ada33688c7922fa62299000abc35712525940a1184f0ac780ca9738862"
CFA_RESULT = "af9ced9174b9363703bc54a1933cf033fd411fbc7035ac2f00acde9334d23d12"
CFA_TERM = "9af3d21e1c68baa61bed195a61657b3d09f777dc1ebee88fd0a914a2afe38222"
CFA_EDITS = "a94156db0cf7b55ea2a1d4b8ad9520a0a58fb0ff1fa59f850146e11a06d8a4ad"
REQUEST = "PHOENIX_RSDD74_EQUAL_COUNT_REQUEST"
STATUS = REQUEST + "_COMPLETE"
ARMS = ("dynamic", "static", "5217/learned", "5218/learned", "5219/learned")
TOTALS = dict(images=74, mapped_candidates=7346, eligible_planes=4137, positive_budget_candidates=4107, swaps_per_arm=29900)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def pin(path, expected):
    raw = path.read_bytes()
    require(digest(raw) == expected, "Input digest differs for " + path.name)
    return raw


def module(path, expected):
    obj = ModuleType(path.stem)
    obj.__file__ = str(path)
    exec(compile(pin(path, expected), str(path), "exec"), obj.__dict__)
    return obj


def write(path, raw):
    with path.open("xb") as stream:
        require(stream.write(raw) == len(raw), "Incomplete output file")
    return digest(raw)


def observed(directory, terminal, phase, status, result_sha=None, terminal_sha=None, worker=None):
    payload = (directory / "RESULT_SAFE.json").read_bytes()
    term = terminal.read_bytes()
    result, t = json.loads(payload), json.loads(term)
    rsha, tsha = digest(payload), digest(term)
    require((result_sha is None or rsha == result_sha) and (terminal_sha is None or tsha == terminal_sha)
            and (worker is None or result["worker_sha256"] == worker)
            and type(t["actual_exit"]) is int and t["actual_exit"] == 0 and t["output_footer_verified"] is True
            and t["result_sha256"] == rsha and t["worker_sha256"] == result["worker_sha256"]
            and t["phase"] == phase and result["status"] == status, "Observed successful " + phase + " required")
    return result, rsha, tsha


def native_inputs(a):
    cfa = module(a.code / "INSTANCE_RSDD_CLASSICAL_FAIR_MATCH_20261123/replay.py", CFA_SHA)
    verify, rsha, tsha = observed(a.cfa_verify, a.cfa_terminal, "verify-edits", cfa.VERIFIED,
                                   CFA_RESULT, CFA_TERM, CFA_SHA)
    require(verify["phase"] == "verify-edits" and verify["edits_sha256"] == CFA_EDITS
            and verify["protocol_sha256"] == cfa.LOCK and verify["target_labels_opened"] is verify["GT_opened"] is False
            and verify["exact_independent_edit_replay"] is verify["native_changed_counts_verified"] is True
            and all(verify["counts"][k] == v for k, v in TOTALS.items()), "Frozen 74-image reference required")
    inputs = cfa.module(a.code / cfa.INPUT_PATH, cfa.INPUT_SHA)
    rows, lineage = inputs.load_population(a)
    require(lineage == verify["input_lineage"], "Closed native capture or previous edit lineage differs")
    ledger = [json.loads(s) for s in pin(a.cfa_verify / cfa.LEDGER, CFA_EDITS).splitlines()]
    return rows, ledger, lineage, dict(edits_sha256=CFA_EDITS, result_sha256=rsha, terminal_sha256=tsha)


def initialize(a, items):
    global w
    import numpy as np
    import torch
    from pycocotools import mask as codec
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    require(os.environ["CUDA_VISIBLE_DEVICES"] == "-1" and not torch.cuda.is_initialized(), "CPU execution")
    cfa = module(a.code / "INSTANCE_RSDD_CLASSICAL_FAIR_MATCH_20261123/replay.py", CFA_SHA)
    common = cfa.module(a.code / cfa.COMMON_PATH, cfa.COMMON_SHA)
    portable = common.portable_module(a.code)
    native = portable.module(a.code / portable.NATIVE / "capture_instance_rsdd.py", portable.CAPTURE_SOURCE)
    native.install_guard(a.stock_root / "conversion", items)
    edit = portable.module(a.code / portable.UEIKAP / "edit_rf.py", common.EDIT)
    base = edit.load_module(a.code / edit.BASE / "replay.py", edit.REPLAY_SHA)
    methods = base.methods(a.code)
    pair, sampler = base.canvas(methods, (512, 512))
    attraction = cfa.module(a.code / cfa.ATTRACTION_PATH, cfa.ATTRACTION_SHA)
    w = SimpleNamespace(a=a, np=np, torch=torch, codec=codec, base=base, methods=methods, pair=pair,
                        sampler=sampler, attraction=attraction)


def validate_support(stock, coords, k, edits):
    np = w.np
    require(set(edits) == set(ARMS) and coords.ndim == 2 and coords.shape[1] == 2
            and np.issubdtype(coords.dtype, np.integer) and np.all((0 <= coords) & (coords < stock.size))
            and np.unique(coords).size == coords.size and not stock.flat[coords[:, 0]].any()
            and stock.flat[coords[:, 1]].all(), "Distinct native signed candidate endpoints required")
    for entry in edits.values():
        require(set(entry) == {"rle", "swaps"} and type(entry["swaps"]) is int and entry["swaps"] == k
                and entry["rle"]["size"] == [512, 512], "Five archived positive equal counts required")
        mask = w.codec.decode(entry["rle"]).astype(bool)
        adds, drops = np.flatnonzero(mask & ~stock), np.flatnonzero(stock & ~mask)
        require(len(adds) == len(drops) == k and np.isin(adds, coords[:, 0]).all()
                and np.isin(drops, coords[:, 1]).all(), "Archived mask violates the reconstructed endpoint support")


def build_image(work):
    from PIL import Image
    ordinal, rowset, archive = work
    item, audit, pred, _, _ = rowset
    image_id = item["image_id"]
    require(image_id == audit["image_id"] == pred["image_id"] == archive["image_id"]
            and audit["original_hw"] == [512, 512] and not audit["posterior_overrides"], "Native image identity differs")
    rgb = w.base.load_rgb(w.a.stock_root / "conversion/images", audit)
    require(digest(rgb.tobytes(order="C")) == item["rgb_sha256"] == audit["rgb_sha256"], "Original RGB required")
    features = w.methods.pair.image_features(rgb)
    require(w.torch.get_float32_matmul_precision() == "high", "Original RF-DETR numerical initialization")
    planes = w.base.load_array(w.a.stock_root / "capture", audit["logits_ref"])
    logits = dict(zip(audit["logit_ranks"], planes, strict=True))
    require(planes.shape == (len(logits), 144, 144), "Expected original native coarse planes")
    candidates = []
    for pos, (info, det) in enumerate(zip(audit["candidates"], pred["coco"], strict=True), 1):
        rank, area = info["native_rank"], int(w.codec.area(det["segmentation"]))
        require(info["mask_area"] == area and (rank in logits) == (32 <= area < 1024) and
                (info["position"], rank, info["query_id"], info["score"]) ==
                (pos, audit["native_ranks"][pos-1], audit["query_ids"][pos-1], det["score"]), "Candidate identity differs")
        edits = archive["changes"].get(str(pos))
        if rank not in logits:
            require(edits is None, "Unexpected edits on an ineligible native candidate")
            continue
        box = w.np.asarray(info["box_xyxy"], dtype=w.np.float32)
        if w.np.any(box[:2] == box[2:]):
            require(edits is None, "Unexpected edits on a degenerate box")
            continue
        stock = w.codec.decode(det["segmentation"]).astype(bool)
        prob = w.base.native_probability(logits[rank], stock, None)
        data = w.pair.paired_points(stock, box, area, det["score"], prob, features, w.sampler)
        coords = data[0] if data is not None else w.np.empty((0, 2), dtype=w.np.int32)
        w.base.check_support(w.methods, stock, box, prob, coords)
        cap = min(8, area // 10, len(coords))
        dynamic, static = w.attraction.select(stock, coords, cap) if cap else ([], [])
        k = len(dynamic)
        require(len(static) == k <= cap and (edits is not None) == bool(k), "Actual archived edit budget differs")
        if k:
            validate_support(stock, coords, k, edits)
            for arm, order in (("dynamic", dynamic), ("static", static)):
                require(w.base.mask_edit(stock, coords, order) == edits[arm], "Archived classical RLE differs")
            candidates.append(dict(position=pos, native_rank=rank, query_id=info["query_id"], score=det["score"],
                                   segmentation=det["segmentation"], k=k, coords=coords.tolist()))
    require(set(archive["changes"]) == {str(r["position"]) for r in candidates}, "Incomplete archived candidates")
    buf = BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    png = buf.getvalue()
    with Image.open(BytesIO(png)) as img:
        require(img.format == "PNG" and img.mode == "RGB" and img.size == (512, 512)
                and img.tobytes() == rgb.tobytes(order="C"), "Lossless original RGB PNG required")
    row = dict(image_id=image_id, image_file=f"image_{ordinal:03d}.png", encoded_sha256=digest(png),
               decoded_rgb_sha256=item["rgb_sha256"], candidates=candidates)
    counts = dict(images=1, mapped_candidates=len(pred["coco"]), eligible_planes=len(planes),
                  positive_budget_candidates=len(candidates), swaps_per_arm=sum(c["k"] for c in candidates))
    return row, png, counts


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ("code", "stock-root", "point-root", "cfa-verify", "cfa-terminal", "protocol", "output"):
        p.add_argument("--" + key, required=True, type=Path)
    p.add_argument("--protocol-sha256", required=True)
    p.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    a = p.parse_args()
    for k, v in vars(a).items():
        if isinstance(v, Path):
            setattr(a, k, v.resolve(strict=k != "output"))
    require(sys.flags.isolated and sys.dont_write_bytecode and sys.version_info[:3] == (3, 10, 13)
            and not a.output.exists() and digest(a.protocol.read_bytes()) == LOCK == a.protocol_sha256,
            "Qualified isolated runtime, published lock and fresh output required")
    runtime = {"torch":"2.10.0+cu130", "numpy":"1.26.4", "Pillow":"12.3.0", "pycocotools":"2.0.11", "rfdetr":"1.12.0"}
    require(all(version(k) == v for k, v in runtime.items()), "Original B5 CPU environment required")
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    a.cohort, a.support_root = 74, None
    start = perf_counter()
    rows, ledger, lineage, provenance = native_inputs(a)
    a.output.mkdir()
    images, counts = [], Counter()
    work = ((j, r, h) for j, (r, h) in enumerate(zip(rows, ledger, strict=True), 1))
    with ProcessPoolExecutor(max_workers=a.workers, initializer=initialize, initargs=(a, [row[0] for row in rows])) as pool:
        for row, png, n in pool.map(build_image, work, chunksize=1):
            require(write(a.output / row["image_file"], png) == row["encoded_sha256"], "Encoded PNG differs")
            images.append(row)
            counts.update(n)
            if len(images) in (1, 25, 50, 74):
                print(json.dumps(dict(images=len(images),positive_budget_candidates=counts["positive_budget_candidates"])), flush=True)
    require(dict(counts) == TOTALS, "Complete frozen candidate census required")
    payload = dict(status=REQUEST, protocol_sha256=LOCK, input_lineage=lineage, source_replay=provenance, images=images)
    request_sha = write(a.output / "REQUEST_PRIVATE.json", encode(payload))
    result = dict(status=STATUS, phase="request", protocol_sha256=LOCK, source_replay=provenance, input_lineage=lineage,
                  counts=dict(counts), request_sha256=request_sha, target_labels_opened=False, detector_forwards=0,
                  point_head_forwards=0, optimizer_updates=0, worker_sha256=digest(Path(__file__).read_bytes()),
                  seconds=perf_counter()-start)
    receipt = write(a.output / "RESULT_SAFE.json", encode(result))
    print(json.dumps(dict(status=STATUS, result_sha256=receipt)), flush=True)


if __name__ == "__main__":
    main()
