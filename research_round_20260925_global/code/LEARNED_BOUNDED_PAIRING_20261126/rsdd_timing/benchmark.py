"""Time three frozen editing policies on the same closed, label-free candidate records."""
from argparse import ArgumentParser
from collections import Counter
import ctypes
from hashlib import sha256
import importlib.util
from itertools import permutations
import json
import os
from pathlib import Path
from statistics import median
import sys
from time import perf_counter, perf_counter_ns
from types import SimpleNamespace

import numpy as np
from pycocotools import mask as codec
from scipy.optimize import linear_sum_assignment


SOLVER_SHA = "99817f192798f52f03efe5e3ac422a453019bf458d33096a03844cb60360436c"
RENDER_SHA = "34d4e588dc757dd60c975ebe9498d479a80265bf568422d57b3fa0fa1f6975f5"
OLD_RESULT_SHA = "4a015eef90344a4d06bd77fedb5fd5a1ea9d5742811f7de01a96dcb6a24ac994"
OLD_TERMINAL_SHA = "92fea4ab053d319397b823bafe3cab7991c793c55a17802df0bd6bb64b11bcee"
PROTOCOL_SHA = "bda59439390f57f9e086cad2cd5dc96a9a84178ab3864f15f35f47021481ab43"
COUNT_EXPECTED = dict(images=74, candidates=7346, positive_candidates=4092, swaps=30841)
COLLECTED = "LEARNED_PAIRING_TIMING_INPUTS_COMPLETE"
TIMED = "LEARNED_PAIRING_TIMING_BENCHMARK_COMPLETE"
POLICIES = ("F", "L", "I")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path, expected):
    data = Path(path).read_bytes()
    need(digest(data) == expected, f"Unexpected SHA-256 for {Path(path).name}")
    return data


def module(path, expected):
    path = Path(path)
    key = "timing_" + path.stem + "_" + expected[:12]
    spec = importlib.util.spec_from_file_location(key, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[key] = loaded
    exec(compile(pin(path, expected), str(path), "exec"), loaded.__dict__)
    return loaded


def feed(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def closed(path, terminal, result_sha, terminal_sha, status, worker_sha):
    event = json.loads(pin(terminal, terminal_sha))
    report = json.loads(pin(path, result_sha))
    need(event["worker_sha256"] == report["worker_sha256"] == worker_sha
         and event["phase"] == "prepare-edits" and event["actual_exit"] == 0
         and event["output_footer_verified"] is True and event["result_sha256"] == result_sha
         and report["status"] == status, "Exact observed independent label-free result")
    return report


def choose_f(r):
    gains = r.q[:, 0] - r.q[:, 1]
    chosen = np.argsort(-gains, kind="stable")[:r.k]
    return chosen, chosen


def choose_i(r):
    return (np.argsort(-r.q[:, 0], kind="stable")[:r.k],
            np.argsort(r.q[:, 1], kind="stable")[:r.k])


def choose_l(r, solver):
    addition_priority = np.argsort(-r.q[:, 0], kind="stable")
    removal_priority = np.argsort(r.q[:, 1], kind="stable")
    free_add, free_remove = addition_priority[:r.k], removal_priority[:r.k]
    row, col = np.divmod(r.coords, 512)
    distance = (row[:, 0, None] - row[None, :, 1])**2 + (col[:, 0, None] - col[None, :, 1])**2
    free_dist = distance[np.ix_(free_add, free_remove)]
    try:
        a, d = linear_sum_assignment(np.where(free_dist <= r.cap, free_dist, np.inf))
    except ValueError:
        scale = solver.BINARY32_SCALE
        add = [int(float(p) * scale) for p in r.q[:, 0]]
        remove = [int(float(p) * scale) for p in r.q[:, 1]]
        return tuple(solver.best_matching(distance, r.cap, r.k, add, remove, np.argsort(addition_priority),
                                          np.argsort(removal_priority)))
    return free_add[a], free_remove[d]


def apply(r, selected):
    if selected is None:
        return r.native
    pixels = np.ascontiguousarray(codec.decode(r.native), dtype=np.uint8)
    add, drop = selected
    pixels.flat[r.coords[add,0]] = 1
    pixels.flat[r.coords[drop,1]] = 0
    encoded = codec.encode(np.asfortranarray(pixels))
    return dict(size=encoded["size"], counts=encoded["counts"].decode("ascii"))


def collect(a):
    renderer = module(a.render, RENDER_SHA)
    former_path = a.render.parent / "reserve74_5217_render/RESULT_SAFE.json"
    former_term = a.render.parent / "observed_reserve74_5217_render/TERMINAL.json"
    former = closed(former_path, former_term, OLD_RESULT_SHA, OLD_TERMINAL_SHA, renderer.STATUS, RENDER_SHA)
    need(former["view"] == "reserve74" and former["head_seed"] == 5217 and former["label_files_opened"] == 0,
         "Known archived label-free reserve")
    bounded_rows = tuple(json.loads(line) for line in pin(former_path.parent / "BOUNDED_PRIVATE.jsonl",
                                                          former["actual_bounded_stream_sha256"]).splitlines())
    archive_args = SimpleNamespace(code=a.code, archive=a.archive, point_head=a.head, workers=1)
    rows, prior, source, archive_lineage = renderer.reserve_archive(archive_args)
    need(archive_lineage == former["prior"] and len(rows)==len(prior)==len(bounded_rows)==74,
         "Exact native archive and original head lineage")
    frozen = module(a.code / renderer.RESERVE_REPLAY_PATH, renderer.RESERVE_REPLAY_SHA)
    frozen.initialize(source, True)
    edit, base, methods, predict, request, _ = frozen.worker
    packets = []

    def capture(m, stock, coords, q, posterior, *, independent):
        need(q.dtype == np.float32 and q.shape == coords.shape and np.isfinite(q).all(), "Frozen binary32 scores")
        cap = min(8, int(stock.sum())//10, len(coords))
        used = m.pair.choose_pairs(q[:,0]-q[:,1], cap)
        k = len(used)
        if k:
            picked = coords[used]
            row, col = np.divmod(picked,512)
            cap_squared = int(np.max((row[:,0]-row[:,1])**2+(col[:,0]-col[:,1])**2))
            packets.append(dict(coords=coords.tolist(), q=q.tolist(), k=k, cap=cap_squared))
        return dict(fixed=[(int(i),int(i)) for i in used],free_matched_count=edit.ordered_pairs(q,cap,k))

    edit.policies = capture
    sha = sha256()
    counts = Counter()
    with (a.output / "CANDIDATES_PRIVATE.jsonl").open("xb") as stream:
        for (row,pred), previous,bounded in zip(rows,prior,bounded_rows,strict=True):
            packets.clear()
            generated, _ = edit.image_edits(base,methods,predict,request,row,pred,independent=True)
            old = {k:{p:v for p,v in choices.items() if p in renderer.POLICIES}
                   for k,choices in previous["changes"].items() if "fixed" in choices}
            need(row["image_id"]==generated["image_id"]==previous["image_id"]==bounded["image_id"]
                 and generated["changes"] == old and set(old) == set(bounded["changes"])
                 and len(packets)==len(old), "Every historical fixed/independent mask and bounded slot must match")
            matched = dict(zip(sorted(old,key=int),packets,strict=True))
            records = []
            for ordinal, detection in enumerate(pred["coco"],1):
                key = str(ordinal)
                r = dict(native=detection["segmentation"])
                if key in matched:
                    r.update(matched[key])
                    r["expected"] = dict(F=old[key]["fixed"]["rle"],I=old[key]["free_matched_count"]["rle"],
                                         L=bounded["changes"][key]["rle"])
                    need(r["k"]==old[key]["fixed"]["swaps"]==old[key]["free_matched_count"]["swaps"]
                         ==bounded["changes"][key]["swaps"], "Exact same write count")
                    counts.update(positive_candidates=1,swaps=r["k"])
                records.append(r)
            raw = feed(records)
            sha.update(raw)
            stream.write(raw)
            counts.update(images=1,candidates=len(records))
        stream.flush()
        os.fsync(stream.fileno())
    need(dict(counts)==COUNT_EXPECTED,"Exact all-slot reserve roster")
    return dict(status=COLLECTED, counts=dict(counts), candidate_cache_sha256=sha.hexdigest(),
                prior_archive=archive_lineage, historical_bounded_result_sha256=OLD_RESULT_SHA,
                historical_fixed_and_independent_replayed_exactly=True)


def affinity_one():
    if os.name != "nt":
        allowed = os.sched_getaffinity(0)
        one = min(allowed)
        os.sched_setaffinity(0, {one})
        return one
    kernel = ctypes.WinDLL("kernel32",use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
    kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p,ctypes.c_size_t]
    current = kernel.GetCurrentProcess()
    available, system = ctypes.c_size_t(), ctypes.c_size_t()
    need(kernel.GetProcessAffinityMask(current,ctypes.byref(available),ctypes.byref(system)) and available.value,
         "Inspect current CPU affinity")
    one = available.value & -available.value
    need(kernel.SetProcessAffinityMask(current,one),"Bind timing process to one logical processor")
    return one.bit_length()-1


def parse_frames(payload):
    rows = []
    counts = Counter()
    for line in payload.splitlines():
        frame = json.loads(line)
        counts.update(images=1,candidates=len(frame))
        for m in frame:
            if "k" in m:
                m["coords"] = np.asarray(m["coords"],dtype=np.int32)
                m["q"] = np.asarray(m["q"],dtype=np.float32)
                counts.update(positive_candidates=1,swaps=m["k"])
            else:
                m["k"] = 0
            rows.append(SimpleNamespace(**m))
    need(dict(counts)==COUNT_EXPECTED, "Exact all-slot timing records")
    return rows, dict(counts)


def benchmark(a):
    cpu = affinity_one()
    worker = digest(Path(__file__).read_bytes())
    report = closed(a.collected, a.collect_terminal, a.collect_sha, a.collect_terminal_sha,COLLECTED,worker)
    need(report["protocol_sha256"]==PROTOCOL_SHA and report["counts"]==COUNT_EXPECTED
         and report["label_files_opened"]==report["detector_forwards"]==report["optimizer_updates"]==0,
         "Admitted no-label frozen timing inputs")
    rows, counts = parse_frames(pin(a.collected.parent/"CANDIDATES_PRIVATE.jsonl",report["candidate_cache_sha256"]))
    solver = module(a.code / "PHOENIX_BOUNDED_MATCH_20261123/matching.py",SOLVER_SHA)
    calls = dict(F=choose_f,I=choose_i,L=lambda r:choose_l(r,solver))

    def one(arm, verify):
        choose = calls[arm]
        start = perf_counter_ns()
        actions = [choose(r) if r.k else None for r in rows]
        middle = perf_counter_ns()
        masks = [apply(r,choice) for r,choice in zip(rows,actions,strict=True)]
        end = perf_counter_ns()
        if verify:
            for r, mask, action in zip(rows,masks,actions,strict=True):
                if not r.k:
                    need(mask == r.native,"Exact untouched native pass-through")
                    continue
                add, drop = action
                need(len(set(add))==len(set(drop))==r.k and mask==r.expected[arm],
                     "Exact archived signed mask for the timed implementation")
                if arm=="L":
                    x = r.coords[add,0]
                    y = r.coords[drop,1]
                    xa, xb = np.divmod(x,512)
                    ya, yb = np.divmod(y,512)
                    need(np.all((xa-ya)**2+(xb-yb)**2<=r.cap),"Exact archived individual distance cap")
        return [(middle-start)/1e9,(end-middle)/1e9,(end-start)/1e9]

    for arm in POLICIES:
        one(arm,True)
    print("All 7,346 exact archived F/L/I compact RLE masks and the signed quotas verified before timing.",flush=True)
    cases = []
    for order in permutations(sorted(POLICIES)):
        arms = {arm:one(arm,False) for arm in order}
        cases.append(dict(order=order,seconds=arms))
    names = ("selection","rle_application","combined_local")
    summary = {}
    for arm in POLICIES:
        summary[arm] = {}
        for i,key in enumerate(names):
            samples = [case["seconds"][arm][i] for case in cases]
            center = median(samples)
            summary[arm][key] = dict(median_seconds=center, range_seconds=[min(samples),max(samples)],
                                     ms_per_image=center*1000/counts["images"],
                                     us_per_positive_candidate=center*1e6/counts["positive_candidates"],
                                     us_per_all_candidates=center*1e6/counts["candidates"])
    return dict(status=TIMED, counts=counts, selected_logical_processor=cpu, numerical_threads=1,
                repeat_count=len(cases), timing_replays=cases, summary=summary, warmups_per_arm=1,
                exact_archived_F_L_I_rle_masks=True, whole_native_stream_including_noop=True,
                collected_result_sha256=a.collect_sha, solver_sha256=SOLVER_SHA)


def main():
    p = ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("phase", choices=("collect","benchmark"))
    for name in ("code","protocol","output"):
        p.add_argument("--"+name,type=Path,required=True)
    for name in ("render","archive","head","collected","collect-terminal"):
        p.add_argument("--"+name,type=Path)
    for name in ("collect-sha","collect-terminal-sha"):
        p.add_argument("--"+name)
    a=p.parse_args()
    need(sys.flags.isolated and sys.dont_write_bytecode and sys.version_info[:3]==(3,10,13)
         and np.__version__=="1.26.4" and all(os.environ.get(k)=="1"
         for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"))
         and os.environ.get("CUDA_VISIBLE_DEVICES")=="-1" and not a.output.exists(), "Qualified one-thread CPU timing runtime")
    pin(a.protocol,PROTOCOL_SHA)
    start=perf_counter()
    a.output.mkdir()
    data = collect(a) if a.phase=="collect" else benchmark(a)
    data.update(worker_sha256=digest(Path(__file__).read_bytes()), protocol_sha256=PROTOCOL_SHA,
                label_files_opened=0, detector_forwards=0, optimizer_updates=0,elapsed_seconds=perf_counter()-start)
    raw=feed(data)
    with (a.output/"RESULT_SAFE.json").open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=data["status"],result_sha256=digest(raw))),flush=True)


if __name__=="__main__":
    main()
