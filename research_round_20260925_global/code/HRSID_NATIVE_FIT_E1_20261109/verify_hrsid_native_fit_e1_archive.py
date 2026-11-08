"""Independently audit a closed HRSID E1 native Lightning archive."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import math
import os
from pathlib import Path

import torch


BATCHES = 2940
UPDATES = 735
EMA_KEY = "RFDETREMACallback"
WITNESS = "model.class_embed.weight"
E24_SHA = "30d6b22104122e6a38f8ec7127e30ab1ae46117d898a1131c202bc5b673530b5"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def load_closed(path, expected_sha):
    need(path.is_absolute() and path.resolve(strict=True) == path, "Exact closed archive path")
    digest = sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        need(0 < before.st_size <= 1024**3, "Bounded full checkpoint")
        for data in iter(lambda: stream.read(1024**2), b""):
            digest.update(data)
        need(digest.hexdigest() == expected_sha, "Exact checkpoint SHA-256")
        stream.seek(0)
        result = torch.load(stream, map_location="cpu", weights_only=True)
        after = os.fstat(stream.fileno())
    final = path.stat()
    ids = [(v.st_size, v.st_mtime_ns, v.st_ino) for v in (before, after, final)]
    need(ids[0] == ids[1] == ids[2], "Closed checkpoint file remained identical")
    return result, before.st_size


def finite(tensor):
    return bool(torch.isfinite(tensor).all())


def archive_accounting(saved):
    need(saved["pytorch-lightning_version"] == "2.6.6" and saved["epoch"] == 0
         and saved["global_step"] == UPDATES, "Native E1 checkpoint metadata")
    loop = saved["loops"]["fit_loop"]
    batch = loop["epoch_loop.batch_progress"]
    step = loop["epoch_loop.automatic_optimization.optim_progress"]["optimizer"]["step"]
    need(batch["is_last_batch"] is True, "Native checkpoint written at last batch")
    for count in ("current", "total"):
        need(loop["epoch_progress"][count] == dict(ready=1, completed=0, started=1, processed=1)
             and all(batch[count][key] == BATCHES for key in ("ready", "started", "processed", "completed"))
             and all(step[count][key] == UPDATES for key in ("ready", "completed")),
             "Native loop counters at the E1 boundary")
    optimizers, schedules = saved["optimizer_states"], saved["lr_schedulers"]
    need(len(optimizers) == len(schedules) == 1, "One native optimizer and scheduler")
    optim, sched = optimizers[0], schedules[0]
    state, groups = optim["state"], optim["param_groups"]
    counts = Counter()
    for item in state.values():
        count = float(item["step"])
        need(count.is_integer() and 0 < count <= UPDATES
             and finite(item["exp_avg"]) and finite(item["exp_avg_sq"]), "Finite AdamW state and actual counters")
        counts[int(count)] += 1
    need(bool(counts) and max(counts) == UPDATES and sched["last_epoch"] == UPDATES,
         "Native AdamW and LambdaLR reached 735")
    need(len(sched["base_lrs"]) == len(sched["_last_lr"]) == len(groups) > 0
         and all(math.isclose(last, base / 3, rel_tol=1e-9, abs_tol=1e-12)
                 and math.isclose(group["lr"], last, rel_tol=1e-9, abs_tol=1e-12)
                 for base, last, group in zip(sched["base_lrs"], sched["_last_lr"], groups, strict=True)),
         "All saved LR groups at warmup factor one third")
    ema = saved["callbacks"][EMA_KEY]
    average = ema["average_model_state_dict"]
    need(ema["latest_update_step"] == int(average["n_averaged"]) == UPDATES, "Native EMA at 735")
    return counts, len(groups), average


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("checkpoint_sha256")
    p.add_argument("initializer", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and not torch.cuda.is_available(), "CPU audit only")
    torch.set_num_threads(2)
    start, _ = load_closed(args.initializer, E24_SHA)
    before = start["state_dict"][WITNESS].clone()
    del start
    saved, size = load_closed(args.checkpoint, args.checkpoint_sha256)
    counts, groups, average = archive_accounting(saved)
    online = saved["state_dict"]
    config = saved["model_config"]
    need(config["resolution"] == 576 and config["group_detr"] == 13
         and config["num_queries"] == 164 and config["num_select"] == 100,
         "Serialized model preserves exact shared HRSID width")
    for name in ("model.refpoint_embed.weight", "model.query_feat.weight"):
        shape = online[name].shape
        need(shape[0] == 2132 and average["module." + name].shape == shape,
             "Each raw and EMA query table contains all 13 groups")
    after, smoothed = online[WITNESS], average["module." + WITNESS]
    need(before.shape == after.shape == smoothed.shape and after.shape[0] == 2
         and finite(after) and finite(smoothed), "Exact finite classifier witnesses")
    moved = float((after - before).abs().max())
    ema_difference = float((after - smoothed).abs().max())
    need(moved > 0 and ema_difference > 0, "Classifier changed and EMA is separately accumulated")
    result = dict(status="INDEPENDENT_HRSID_NATIVE_E1_ARCHIVE_VERIFIED", checkpoint_sha256=args.checkpoint_sha256,
                  checkpoint_bytes=size, optimizer_updates=UPDATES, ema_updates=UPDATES, microbatches=BATCHES,
                  adam_step_histogram=dict(counts), lr_groups=groups, lr_warmup_factor=1/3, serialized_resolution=576,
                  query_width=164, query_groups=13, inference_num_select=100, source_initializer_sha256=E24_SHA,
                  classifier_max_abs_movement=moved, classifier_raw_ema_max_abs=ema_difference,
                  weights_only_cpu_readback=True, AP=False, DEV=False, TEST=False)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

