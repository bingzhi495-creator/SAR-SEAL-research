"""Paired single-thread CPU timing of three independently rendered reserve readouts."""

import argparse
from collections import Counter
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter, process_time
from types import ModuleType, SimpleNamespace


RESERVE = "INSTANCE_RSDD_RESERVE_POINTREND_20261123"
RESERVE_OP = "f8a0ab392bba3134a9a100731d0cbfdcefae0c590c5d0aa64536eeea6efb55fe"
POINT_EDITS = "07b87accf544a0558a55d00205ab4d56a04e0585a8665e415b0743b286f537e5"
ARM_NAME = {"L": "learned_common", "B": "point_equal", "P": "point_paste"}
HW = (512, 512)
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
               "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def source(path, expected):
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == expected, f"Unexpected operator {path.name}")
    module = ModuleType(path.stem)
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def open_inputs(args):
    op = source(args.code / RESERVE / "replay_reserve_pointrend.py", RESERVE_OP)
    a = SimpleNamespace(code=args.code, intake=args.stock_root/"conversion", roster=args.stock_root/"reserve.csv",
        capture=args.stock_root/"capture", occupancy=args.stock_root/"replay", basis=args.point_root/"basis",
        basis_terminal=args.point_root/"observed_basis/TERMINAL.json", fit_weights=args.point_weights,
        protocol=args.code.parent/"INSTANCE_RSDD_RESERVE_POINTREND_PRELOCK_20261123.md")
    population, _, _, lineage = op.read_population(a)
    safe = (args.point_root/"replay/RESULT_SAFE.json").read_bytes()
    gold = json.loads(safe)
    terminal = json.loads((args.point_root/"observed_replay/TERMINAL.json").read_bytes())
    require(terminal["actual_exit"] == 0 and terminal["output_footer_verified"] is True
            and terminal["result_sha256"] == op.digest(safe) and terminal["worker_sha256"] == RESERVE_OP
            and gold["status"] == op.STATUS and gold["edits_sha256"] == POINT_EDITS
            and gold["exact_independent_render"] and gold["GT_opened"] is False
            and all(gold[k] == v for k, v in lineage.items()), "Closed reserve PointRend reference required")
    rows = [json.loads(line) for line in op.pin(args.point_root/"replay/EDITS_PRIVATE.jsonl", POINT_EDITS).splitlines()]
    require(len(population) == len(rows) == 74, "Entire reserve must be included")
    portable = op.source(args.code/"INSTANCE_RSDD_B5_REPRO_20261123/portable_eval.py", op.PORTABLE)
    native = portable.module(args.code/portable.NATIVE/"capture_instance_rsdd.py", portable.CAPTURE_SOURCE)
    items, _ = portable.load_intake(a, native)
    native.install_guard(a.intake, items)
    original = op.source(args.code/op.TRANSFER/"replay_pointrend.py", op.REPLAY_OP)
    capture = op.source(args.code/op.TRANSFER/"capture_basis.py", op.ORIGINAL_BASIS)
    _, _, edit = capture.input_modules(args.code)
    base = edit.load_module(args.code/edit.BASE/"replay.py", edit.REPLAY_SHA)
    methods = base.methods(args.code)
    step = base.read_source(args.code/base.SPOT/"fit_step.py", methods.readout.PINS["fit_step.py"])
    plan = base.read_source(args.code/base.RF/"fit_plan.py", methods.readout.PINS["fit_plan.py"])
    trainer = step.load_trainer(args.code/base.BASE/"fit109_paired_point/train_point.py")
    models, center, scale = plan.restore_bundle(args.occupancy_weights, portable.POINT_HEAD, trainer, step)
    infer, adapter, render, _ = original.policies_modules(a, capture)
    point_head = original.restore_head(adapter, op.pin(args.point_weights, op.HEAD))
    pair, sampler = base.canvas(methods, HW)
    return SimpleNamespace(**vars(a), population=population, gold=rows, base=base, methods=methods,
        pair=pair, sampler=sampler, predict=lambda features: step.predict(models["S"], features, center, scale),
        adapter=adapter, render=render, infer=infer, point_head=point_head, lineage=lineage,
        learned_head_sha256=portable.POINT_HEAD, point_head_sha256=op.HEAD, expected_counts=gold["counts"])


def prepare(world, ordinal, loading):
    (audit, native), idx, original = world.population[ordinal]
    before = perf_counter()
    planes = world.base.load_array(world.capture, audit["logits_ref"])
    require(not audit["posterior_overrides"], "Reserve requires no CUDA sign overrides")
    by_rank = dict(zip(audit["logit_ranks"], planes, strict=True))
    golden = world.gold[ordinal]
    require(golden["image_id"] == native["image_id"] == original["image_id"], "Same native reserve image")
    # Only common metadata, the native RLE and eligible query logits are prepared here.
    rows, budgets = [], {}
    for pos, (info, pred) in enumerate(zip(audit["candidates"], native["coco"], strict=True), 1):
        key = str(pos)
        require(info["position"] == pos and pred["score"] == info["score"], "Same native ranked candidate")
        truth = golden["changes"].get(key, {})
        occ = original["changes"].get(key, {}).get("free_natural")
        require(truth.get("learned_common") == occ and truth.get("point_equal", {}).get("swaps", 0)
                == (occ["swaps"] if occ else 0), "Same independently recorded per-candidate natural budget")
        rank, area = info["native_rank"], info["mask_area"]
        require((32 <= area < 1024) == (rank in by_rank), "Declared native area gate differs")
        if rank not in by_rank:
            require(not truth, "Outside-gate candidate must stay native")
            continue
        box = np.asarray(info["box_xyxy"], dtype=np.float32)
        if np.any(box[2:] == box[:2]):
            require(not truth, "Degenerate original RoI must stay native")
            continue
        rows.append((key, pred["segmentation"], by_rank[rank], box, area, pred["score"]))
        budgets[key] = occ["swaps"] if occ else 0
    loading["native_planes_and_candidate_preparation"] += perf_counter()-before
    before = perf_counter()
    rgb = world.base.load_rgb(world.intake/"images", audit)
    loading["occupancy_rgb_decode_and_identity"] += perf_counter()-before
    before = perf_counter()
    spatial = world.base.load_array(world.basis, idx["spatial"])
    loading["pointrend_basis_read_and_identity"] += perf_counter()-before
    return SimpleNamespace(rows=rows, budgets=budgets, native=native["coco"], golden=golden["changes"],
                           rgb=rgb, spatial=spatial)


def occupancy(world, rows, rgb):
    result = {}
    image = world.methods.pair.image_features(rgb)
    for key, mask, plane, box, area, score in rows:
        stock = codec.decode(mask).astype(bool)
        prob = world.base.native_probability(plane, stock, None)
        data = world.pair.paired_points(stock, box, area, score, prob, image, world.sampler)
        if data is None:
            continue
        coords, features = data[:2]
        q = world.predict(features)
        choices = world.methods.free.choose_free(q, min(8, area//10, len(coords)))
        if choices:
            result[key] = world.base.mask_edit(stock, coords, choices)
    return result


def point_chunks(world, rows, spatial):
    spatial = torch.as_tensor(spatial).unsqueeze(0)
    for start in range(0, len(rows), 12):
        chunk = rows[start:start+12]
        # The original 512-pixel box normalization and denormalization is exact in FP32.
        boxes = torch.stack([torch.from_numpy(row[3]) for row in chunk])
        raw = torch.stack([torch.from_numpy(row[2]) for row in chunk])
        coarse = world.adapter.coarse_roi_logits(raw[:, None], boxes, HW)

        def predict(xy, samples):
            fine = world.adapter.fine_features_at_roi_points(spatial, boxes, xy, HW)
            return world.point_head(fine, samples)

        rendered = world.render.subdivide_binary_mask(coarse, predict, initial_size=(7, 7),
                                                      native_size=(224, 224), points_per_step=784)
        require(rendered.point_evaluations == 3136*len(chunk), "Unexpected PointRend execution budget")
        for i, row in enumerate(chunk):
            yield row, boxes[i:i+1], rendered.logits[i:i+1]


def encode(bitmap):
    rle = codec.encode(np.asfortranarray(bitmap, dtype=np.uint8))
    return {"size": rle["size"], "counts": rle["counts"].decode("ascii")}


def point_balanced(world, rows, spatial, budgets):
    result = {}
    for (key, mask, plane, box, area, score), boxes, logits in point_chunks(world, rows, spatial):
        k = budgets[key]
        if not k:
            continue
        stock = codec.decode(mask).astype(bool)
        prob = world.base.native_probability(plane, stock, None)
        adds, drops = world.base.expected_support(world.methods.rank, stock, box, prob)
        # B ranks the two sets independently. Its published tie-breaker is the native flat index.
        cells = torch.as_tensor(np.column_stack((adds, drops)).reshape(-1))
        xy = torch.stack((cells % HW[1], torch.div(cells, HW[1], rounding_mode="floor")), dim=-1)[None]
        values = world.adapter.paste_probabilities_at_pixels(logits, boxes, xy, HW).reshape(-1, 2).numpy()
        gain_add = values[:, 0].astype(np.float64)-0.5
        gain_drop = 0.5-values[:, 1].astype(np.float64)
        stock.flat[adds[np.lexsort((adds, -gain_add))[:k]]] = True
        stock.flat[drops[np.lexsort((drops, -gain_drop))[:k]]] = False
        result[key] = {"swaps": k, "rle": encode(stock)}
    return result


def point_paste(world, rows, spatial):
    result = {}
    canvas = np.zeros(HW, dtype=np.uint8, order="F")
    for (key, mask, plane, box, area, score), boxes, logits in point_chunks(world, rows, spatial):
        crop, xy = world.infer._crop_grid(boxes[0], HW)
        sampled = world.adapter.paste_probabilities_at_pixels(logits, boxes, xy, HW)
        canvas[crop] = sampled.reshape(canvas[crop].shape).numpy() >= 0.5
        rle = encode(canvas)
        canvas[crop] = 0
        if rle != mask:
            result[key] = {"rle": rle}
    return result


def verify(result, arm, data):
    name = ARM_NAME[arm]
    totals = Counter()
    require(set(result) == {k for k, modes in data.golden.items() if name in modes}, "Different changed candidate set")
    for pos, candidate in enumerate(data.native, 1):
        key = str(pos)
        reference = data.golden.get(key, {}).get(name)
        if reference is None:
            continue  # Exact changed set above guarantees native fallback, including outside the gate.
        out = result[key]
        require(out["rle"] == reference["rle"], f"{arm}: encoded native RLE differs")
        stock = codec.decode(candidate["segmentation"]).astype(bool)
        edited = codec.decode(out["rle"]).astype(bool)
        adds, drops = int(np.count_nonzero(edited & ~stock)), int(np.count_nonzero(stock & ~edited))
        if arm == "P":
            require((adds, drops) == (reference["adds"], reference["drops"]), "Full-paste edit count differs")
        else:
            require(adds == drops == reference["swaps"] == out["swaps"], "Matched actual signed budget differs")
        totals.update(changed=1, adds=adds, drops=drops, empty=int(not edited.any()))
    totals["candidates_checked"] = len(data.native)
    return totals


def dispatch(world, data, arm):
    if arm == "L":
        return occupancy(world, data.rows, data.rgb)
    if arm == "B":
        return point_balanced(world, data.rows, data.spatial, data.budgets)
    return point_paste(world, data.rows, data.spatial)


def one_pass(world, run):
    loading, counts = Counter(), {arm: Counter() for arm in ARM_NAME}
    elapsed = {arm: [] for arm in ARM_NAME}
    cpu_time, audit_time, rendered = Counter(), 0.0, 0
    with torch.inference_mode():
        for i in range(1 if run == 0 else len(world.population)):
            data = prepare(world, i, loading)
            rendered += len(data.rows)
            order = "LBP"[i%3:]+"LBP"[:i%3]
            if run == 2:
                order = order[::-1]
            for arm in order:
                # Reinstall the original per-head CPU calculation policy outside the timer.
                torch.set_float32_matmul_precision("high" if arm == "L" else "highest")
                start_cpu, start = process_time(), perf_counter()
                result = dispatch(world, data, arm)
                elapsed[arm].append(perf_counter()-start)
                cpu_time[arm] += process_time()-start_cpu
                start = perf_counter()
                counts[arm].update(verify(result, arm, data))
                audit_time += perf_counter()-start
            if i == 0 or (i+1)%10 == 0 or i == 73:
                print(json.dumps(dict(phase="warmup" if run == 0 else "measure", repeat=run, images=i+1)), flush=True)
    if run:
        expected = world.expected_counts
        require(rendered == expected["rendered"] and all(c["candidates_checked"] == expected["mapped_candidates"]
                for c in counts.values()), "Full archived reserve coverage required")
        for arm, c in counts.items():
            prefix = ARM_NAME[arm]
            require(c["changed"] == expected[prefix+"_candidates"] and
                    c["adds"] == expected[prefix+("_adds" if arm == "P" else "_swaps")] and
                    c["drops"] == expected[prefix+("_drops" if arm == "P" else "_swaps")], "Frozen counters differ")
    summaries = {}
    for arm, series in elapsed.items():
        summaries[arm] = dict(seconds=sum(series), process_cpu_seconds=cpu_time[arm],
            ms_per_image=1000*np.mean(series), image_median_ms=1000*np.median(series),
            image_p90_ms=1000*np.quantile(series, .90), counts=dict(counts[arm]),
            point_evaluations=3136*rendered if arm != "L" else 0)
    return dict(arms=summaries, compared_images=len(elapsed["L"]), renderable_rois=rendered,
                input_preparation_seconds=dict(loading), verification_seconds=audit_time)


def cpu_description():
    if os.name == "nt":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    return platform.processor() or platform.machine()


def main():
    global np, torch, codec
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("code", "stock-root", "point-root", "occupancy-weights", "point-weights", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    for key, path in vars(args).items():
        setattr(args, key, path.resolve(strict=key != "output"))
    require(not args.output.exists(), "Fresh benchmark directory required")
    os.environ.update({key: "1" for key in THREAD_VARS}, CUDA_VISIBLE_DEVICES="-1", OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE")
    import numpy as np
    import torch
    from pycocotools import mask as codec

    runtime = {key: version(key) for key in ("torch", "numpy", "rfdetr", "pycocotools", "Pillow", "scipy")}
    require(sys.version_info[:3] == (3, 10, 13) and runtime["torch"] == "2.10.0+cu130"
            and runtime["numpy"] == "1.26.4" and runtime["pycocotools"] == "2.0.11"
            and runtime["rfdetr"] == "1.12.0" and runtime["Pillow"] == "12.3.0", "Use the frozen reserve runtime")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir()
    start = perf_counter()
    world = open_inputs(args)
    setup_seconds = perf_counter()-start
    warmup = one_pass(world, 0)
    passes = [one_pass(world, repeat) for repeat in (1, 2)]
    require(torch.get_num_threads() == torch.get_num_interop_threads() == 1 and not torch.cuda.is_initialized(),
            "Single-thread CPU timing required")
    result = dict(status="RESERVE74_CPU_COST_COMPLETE", warmup_images=warmup["compared_images"],
        measured_passes=passes, setup_seconds=setup_seconds, process_wall_seconds=perf_counter()-start, cpu=cpu_description(),
        system=platform.system(), system_release=platform.release(), logical_cpu_count=os.cpu_count(), runtime=runtime,
        torch_intraop_threads=1, torch_interop_threads=1, thread_environment={k: os.environ[k] for k in THREAD_VARS},
        CPU_matmul_precision={"L":"high","B":"highest","P":"highest"}, cpu_affinity_pinned=False,
        crossed_order="LBP, BPL, PLB by image index; reverse each image's order in pass two",
        images=74, same_process=True, independent_point_renders=True, single_unmeasured_first_image_each_arm=True,
        B_uses_supplied_natural_k=True, detector_forwards=0, optimizer_updates=0, label_files_opened=False,
        exact_candidate_rle_and_signed_edit_counts=True, reference_edits_sha256=POINT_EDITS, lineage=world.lineage,
        occupancy_weights_sha256=world.learned_head_sha256, point_weights_sha256=world.point_head_sha256,
        worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    raw = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()
    with (args.output/"RESULT_SAFE.json").open("xb") as out:
        out.write(raw)
    print(json.dumps(dict(status=result["status"], result_sha256=sha256(raw).hexdigest())), flush=True)


if __name__ == "__main__":
    main()
