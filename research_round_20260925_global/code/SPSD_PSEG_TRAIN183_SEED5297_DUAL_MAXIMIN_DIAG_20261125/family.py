"""Label-free reconstruction and validation of finite dual-reference candidate families."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from types import ModuleType
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from pycocotools import mask as codec
import torch

LOCK = "d3e876ce1a253b7bd17826ecf78c252a0f2b31fd308fb03d3b310e7bfe155320"
GROUP_PROTOCOL = "4030a3cb5684a6dac0ea1e7765a48d05caec113c5c6e0e6de2c82b991e9d5f15"
OLD_SCORE = "e8268d0a4fdf3c36b92e8ac2ae604905b51a5337c9f6dc43463a0f42ccad53f9"
OLD_RENDER = "4681265afbbf7acae7389a937e988e85218f558ad7ff99e6d87fa5d0f69e51b3"
RENDER_WORKER = "17fb4db0858785a312d1c761d97a39f8de448e9b19230aeb727a8ff5d286f5a7"
SCORE_WORKER = "64b4df4f9b6af079eb56a5099d1552a05d2c701d020102f3007ad64e2d754a69"
MODES = ("static", "dual_robust", "dual_mean", "dual_cell_pessimistic")


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(blob):
    return sha256(blob).hexdigest()


def pin(path, expected):
    raw = path.read_bytes()
    need(digest(raw) == expected, "Exact bound input: " + path.name)
    return raw


def load(path, expected):
    code = ModuleType(path.stem)
    code.__file__ = str(path)
    exec(compile(pin(path, expected), str(path), "exec"), code.__dict__)
    return code


def closed(a):
    reader = load(a.gate / "score_gate.py", SCORE_WORKER)
    render = load(a.gate / "render_gate.py", RENDER_WORKER)
    sealed = json.loads(pin(a.gate/"score/RESULT_SAFE.json", OLD_SCORE))
    term = render.terminal(a.gate/"SCORE_TERMINAL.json", "score", OLD_SCORE, SCORE_WORKER)
    ready, end, _, previous, records, groups = reader.exact_readback(SimpleNamespace(render_root=a.gate))
    need(end["result_sha256"] == OLD_RENDER == sealed["render_result_sha256"]
         and sealed["fit_result_sha256"] == ready["fit_result_sha256"]
         and sealed["render_stream_sha256"] == ready["render_stream_sha256"]
         and sealed["historical_test_opened"] is sealed["fit_labels_opened"] is False,
         "Original 5297 closed fit-to-score chain")
    return reader, render, sealed, term, previous, records, groups


def top(values, count, add):
    idx = sorted(range(len(values)), key=lambda rank: (-values[rank] if add else values[rank], rank))
    return tuple(sorted(idx[:count]))


def family(values, incumbent, add):
    options = [incumbent]
    for i in range(11):
        q = i * values[:, 0] + (10-i) * values[:, 1]
        cell_set = top(q, len(incumbent), add)
        if cell_set not in options:
            options.append(cell_set)
    return options


def render_mask(native, loc, chosen):
    raw = codec.decode(native).astype(np.uint8, order="C")
    out = raw.copy(order="C")
    f = out.ravel()
    added, removed = (loc[np.asarray(indices, dtype=np.int64), s] for s, indices in enumerate(chosen))
    need(not f[added].any() and f[removed].all(), "Original signed endpoints")
    f[added], f[removed] = 1, 0
    encoded = codec.encode(np.asfortranarray(out))
    encoded["counts"] = encoded["counts"].decode("ascii")
    return encoded


def prepare(a):
    start = perf_counter()
    _, render, _, oldscoreterm, native, records, _ = closed(a)
    fit, _, points, head, selector, _, models, center, scale, terms = render.load_artifacts(a)
    root = a.points / "actual_retry/source_gate"
    index = [json.loads(line) for line in pin(
        root / "POINT_INDEX_PRIVATE.jsonl", points["partitions"]["source_gate"]["point_index_sha256"]).splitlines()]
    need(len(index) == len(native) == 183, "Complete point and candidate counterpart")
    stats = Counter()
    stream_sha = sha256()
    a.output.mkdir()
    with (a.output/"FAMILY_PRIVATE.jsonl").open("xb") as out:
        for item in index:
            iid = item["image_id"]
            stock = native[iid]
            old = records[iid]
            need(old["source_group"] == item["source_group"] and old["original_hw"] == item["original_hw"],
                 "Same source group and pointer")
            data = pin(root/item["filename"], item["sha256"])
            need(len(data) == item["bytes"], "Exact endpoint packet")
            with np.load(BytesIO(data), allow_pickle=False) as points:
                ranks, plan, loc, feats = (points[k] for k in ("native_ranks","point_offsets","signed_coordinates","features"))
                quotas, booked = (points[k] for k in ("static_offsets","static_side_indices"))
            result = {}
            estimates = head.predict(models["dual"], feats, center, scale).astype(np.float64)
            for j, rank_ in enumerate(ranks):
                rank = int(rank_)
                original = stock[rank-1]
                target = booked[quotas[j]:quotas[j+1]]
                k = len(target)
                need(k == original["quota"], "Earlier exact scored static budget replay")
                if k == 0:
                    continue
                endpoints = loc[plan[j]:plan[j+1]]
                p, d = estimates[plan[j]:plan[j+1]].transpose(1,0,2)
                inc_a, inc_d = (tuple(sorted(side.tolist())) for side in target.T)
                aas, dds = family(p, inc_a, True), family(d, inc_d, False)
                ap = p[np.asarray(aas)].sum(axis=1)
                dp = d[np.asarray(dds)].sum(axis=1)
                gain = ap[:,None,:] - dp[None,:,:] - (ap[0]-dp[0])
                ai, di = np.unravel_index(gain.min(axis=-1).argmax(), gain.shape[:2])
                choices = {
                    "static": (inc_a, inc_d), "robust": (aas[ai], dds[di]),
                    "mean": (top(p.mean(axis=1),k,True), top(d.mean(axis=1),k,False)),
                    "cell_pessimistic": (top(p.min(axis=1),k,True),top(d.max(axis=1),k,False)),
                }
                need(choices == selector.select(p,d,inc_a,inc_d,k=k),
                     "Independent finite, mean and pessimistic action parity")
                ordered = list(choices.values())
                for arm, value in zip(MODES, ordered, strict=True):
                    need(original["changes"][arm] == render_mask(original["native"], endpoints, value),
                         "Exact frozen original edited RLE reproduced")
                result[str(rank)] = dict(n=len(endpoints), k=k, coords=endpoints.tolist(),
                                         adds=aas, drops=dds, ap=ap.tolist(), dp=dp.tolist(),
                                         robust_pos=[int(ai),int(di)], choices=ordered)
                stats.update(active_candidates=1, positive_site_pairs=len(endpoints), exchanges=k,
                             add_sets=len(aas), drop_sets=len(dds), combinations=len(aas)*len(dds), mask_parity=4)
            raw = (json.dumps(dict(image_id=iid, group=item["source_group"], families=result),
                              sort_keys=True, separators=(",",":"), allow_nan=False)+"\n").encode()
            need(out.write(raw) == len(raw), "Complete private prelabel family stream")
            stream_sha.update(raw)
            stats["images"] += 1
    need(stats["active_candidates"] == 8785 and stats["exchanges"] == 57692 and stats["mask_parity"] == 35140
         and stats["images"] == 183, "Exact full-population action parity")
    doc = dict(status="SPSD_DUAL_5297_ORACLE_FAMILY_SEALED", prelock_sha256=LOCK,
               worker_sha256=digest(Path(__file__).read_bytes()), prior_score_sha256=OLD_SCORE,
               prior_score_terminal_sha256=oldscoreterm, prior_render_sha256=OLD_RENDER,
               upstream_terminal_hashes=terms, family_sha256=stream_sha.hexdigest(),
               fit_checkpoint_sha256=fit["output_checkpoint_sha256"], counts=stats,
               optimizer_updates=0, detector_forwards=0, gate_labels_opened=False, historical_test_opened=False,
               elapsed_seconds=perf_counter()-start)
    raw = (json.dumps(doc,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    with (a.output/"RESULT_SAFE.json").open("xb") as out:
        out.write(raw)
    print(json.dumps(dict(status=doc["status"],result_sha256=digest(raw))),flush=True)


def main():
    p = ArgumentParser(description=__doc__)
    for x in ("gate","fit","capture","points","core","selector","attraction","protocol","group-protocol","output"):
        p.add_argument("--"+x,type=Path,required=True)
    a = p.parse_args()
    pin(a.protocol,LOCK)
    pin(a.group_protocol,GROUP_PROTOCOL)
    torch.set_num_threads(1)
    prepare(a)


if __name__ == "__main__":
    main()

