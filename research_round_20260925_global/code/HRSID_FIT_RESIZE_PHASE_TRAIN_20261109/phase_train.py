"""Run a prelocked HRSID mask-resize arm through the unchanged native trainer."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import pickle
import random
import re
import sys
from time import perf_counter
from types import ModuleType

import numpy as np
import torch

from phase_patch import mask_resize_phase

NATIVE_SHA = "11d17cab878d62576b8d19e6231caab90c9990c03909cb01e1d04468add452a6"
WINDOW_UPDATES = 8
PHASE_MARKER = "sar_seal_hrsid_mask_resize_phase_v1"


class QualificationComplete(Exception):
    """Signal intentional normal short-window completion to the entry point."""


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def tensor_digest(tensor):
    cpu = tensor.detach().contiguous().cpu()
    hasher = sha256(repr((str(cpu.dtype), tuple(cpu.shape))).encode())
    hasher.update(cpu.numpy().tobytes(order="C"))
    return hasher.hexdigest()


def batch_identity(image, target):
    cpu_rng = pickle.dumps((random.getstate(), np.random.get_state(), torch.get_rng_state().numpy().tobytes()), protocol=4)
    return dict(image_id=int(target["image_id"].item()), image_sha256=tensor_digest(image.tensors),
                cpu_rng_sha256=sha256(cpu_rng).hexdigest(), labels=int(target["labels"].numel()),
                labels_sha256=tensor_digest(target["labels"]), masks_sha256=tensor_digest(target["masks"]),
                boxes_sha256=tensor_digest(target["boxes"]))


def exclusive_json(path, obj):
    with path.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(obj, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def attach_short_window(observer, trainer, native, base, args, marker):
    records = []
    witnesses = {}
    old_start = observer.on_train_start
    old_batch_start = observer.on_train_batch_start
    old_batch_end = observer.on_train_batch_end
    old_fit = trainer.fit
    visits = WINDOW_UPDATES * native.ACCUM
    start = perf_counter()

    def train_start(active, model):
        old_start(active, model)
        witnesses["classifier_initial_sha256"] = tensor_digest(model.model.class_embed.weight)

    def batch_start(active, model, batch, index):
        image, targets = batch
        target, = targets
        require(index == len(records) < visits, "Fixed native short-window batch cursor")
        record = batch_identity(image, target)
        old_batch_start(active, model, batch, index)
        record.update(batch=index + 1, postfilter_labels=int(target["labels"].numel()),
                      postfilter_masks_sha256=tensor_digest(target["masks"]),
                      postfilter_boxes_sha256=tensor_digest(target["boxes"]))
        records.append(record)

    def batch_end(active, model, outputs, batch, index):
        old_batch_end(active, model, outputs, batch, index)
        if index + 1 == visits:
            require(active.global_step == observer.updates == WINDOW_UPDATES,
                    "Real AdamW completed the fixed short window")
            # Native on_train_end may swap the active module to EMA.
            witnesses["classifier_online_ending_sha256"] = tensor_digest(model.model.class_embed.weight)
            require(witnesses["classifier_online_ending_sha256"] != witnesses["classifier_initial_sha256"],
                    "Actual online classifier tensor must change before EMA swap")
            active.should_stop = True

    def partial_epoch_end(active, model):
        require(active.should_stop and active.current_epoch == 0 and active.global_step == WINDOW_UPDATES
                and len(observer.visits) == visits and observer.completed == 0 and observer.armed is None,
                "Stop at the short window without claiming a completed epoch")

    def short_fit(*positional, **named):
        old_fit(*positional, **named)
        ema, = (cb for cb in trainer.callbacks if type(cb).__name__ == "RFDETREMACallback")
        base.ema_clock(ema.state_dict(), WINDOW_UPDATES)
        schedule = native.optimizer_state(observer.optim.state_dict(), observer.scheduler.state_dict(), WINDOW_UPDATES)
        require(trainer.global_step == observer.updates == WINDOW_UPDATES and len(records) == visits
                and len(observer.losses) == visits and len(observer.gradients) == WINDOW_UPDATES
                and max(observer.gradients) > 0 and observer.completed == 0 and not observer.saved
                and not list(args.output.glob("*.ckpt")), "Exact closed training window without model save")
        private = args.output / "QUALIFICATION_VISITS_PRIVATE.json"
        exclusive_json(private, records)
        pair_image_digest = sha256(json.dumps([(r["image_id"], r["image_sha256"], r["cpu_rng_sha256"])
                                               for r in records], separators=(",", ":")).encode()).hexdigest()
        report = dict(status="NATIVE_HRSID_MASK_RESIZE_QUALIFICATION_COMPLETE", **marker, **witnesses, **schedule,
                      initial_online_sha256=args.expected_initial_sha256, input_pair_sha256=pair_image_digest,
                      visit_ledger_sha256=digest(private), visits=visits, optimizer_updates=WINDOW_UPDATES,
                      ema_updates=WINDOW_UPDATES, completed_epochs=0, actual_trainer_epoch_cursor=trainer.current_epoch,
                      planned_updates=native.TOTAL, warmup_updates=native.WARMUP,
                      loss_min=min(observer.losses), loss_max=max(observer.losses),
                      classifier_gradient_norm_max=max(observer.gradients), removed_empty_rows=observer.removed_empty_rows,
                      targets_after_filter=sum(r["postfilter_labels"] for r in records),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      seconds=perf_counter()-start, FIT=True, DEV=False, TEST=False, AP=False, checkpoint_saved=False)
        exclusive_json(args.output / "QUALIFICATION_SAFE.json", report)
        native.event("HRSID_MASK_PHASE_SHORT_WINDOW_CLOSED", **report)
        raise QualificationComplete

    observer.on_train_start = train_start
    observer.on_train_batch_start = batch_start
    observer.on_train_batch_end = batch_end
    observer.on_train_epoch_end = partial_epoch_end
    trainer.fit = short_fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--expected-initial-sha256", required=True)
    parser.add_argument("--mode", required=True, choices=("native", "center"))
    parser.add_argument("--stage", required=True, choices=("qualification", "e24"))
    args = parser.parse_args()
    require(sys.version_info[:3] == (3, 10, 13) and os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
            and os.environ.get("HF_HUB_OFFLINE") == os.environ.get("TRANSFORMERS_OFFLINE") == "1"
            and bool(re.fullmatch(r"[0-9a-f]{40}", args.protocol))
            and bool(re.fullmatch(r"[0-9a-f]{64}", args.expected_initial_sha256))
            and args.native_source.is_absolute() and args.native_source.resolve(strict=True) == args.native_source
            and args.output.is_absolute() and args.output.parent.resolve(strict=True) == args.output.parent
            and not args.output.exists(), "Pinned original isolated training request")
    raw = args.native_source.read_bytes()
    require(sha256(raw).hexdigest() == NATIVE_SHA, "Exact original closed filtered trainer bytes")
    native = ModuleType("_pinned_native_filtered_trainer")
    native.__file__ = str(args.native_source)
    sys.modules[native.__name__] = native
    exec(compile(raw, str(args.native_source), "exec"), native.__dict__)
    base = native.pinned_module(native.E1, native.E1_SHA, "_closed_hrsid_e1_environment")
    require(not args.output.is_relative_to(base.LAYOUT) and args.output not in base.LAYOUT.parents,
            "Output must be isolated from FIT layout")
    sys.addaudithook(native.offline)
    base.admit_dependency_overlay()
    from rfdetr.datasets import _torchvision
    from rfdetr.training import trainer as trainer_api
    import phase_patch

    marker = dict(protocol=args.protocol, mask_resize_mode=args.mode, stage=args.stage, seed=native.SEED,
                  phase_trainer_sha256=digest(Path(__file__)),
                  phase_patch_sha256=digest(Path(phase_patch.__file__)),
                  native_trainer_sha256=NATIVE_SHA, initial_expanded_sha256=args.expected_initial_sha256)
    build_original = trainer_api.build_trainer

    def build(train, model, *, extra_callbacks, **kwargs):
        observer, = extra_callbacks
        trainer = build_original(train, model, extra_callbacks=extra_callbacks, **kwargs)
        if args.stage == "qualification":
            attach_short_window(observer, trainer, native, base, args, marker)
        else:
            save_original = observer.on_save_checkpoint

            def save_checkpoint(active, module, checkpoint):
                save_original(active, module, checkpoint)
                require(PHASE_MARKER not in checkpoint, "Fresh exact mask phase marker")
                checkpoint[PHASE_MARKER] = marker.copy()

            observer.on_save_checkpoint = save_checkpoint
        return trainer

    with mask_resize_phase(_torchvision, args.mode):
        trainer_api.build_trainer = build
        try:
            native.run(base, args.output, args.protocol, args.expected_initial_sha256)
        except QualificationComplete:
            require(args.stage == "qualification", "Only the qualification route may stop short")
            return
        finally:
            trainer_api.build_trainer = build_original
    require(args.stage == "e24", "A qualification return requires the fixed window receipt")
    result = args.output / "NATIVE_HRSID_COCO_FIT_E24_SAFE.json"
    exclusive_json(args.output / "PHASE_PROVENANCE_SAFE.json",
                   dict(status="NATIVE_HRSID_MASK_RESIZE_E24_CLOSED_PENDING_INDEPENDENT_ACCEPTANCE",
                        native_result_sha256=digest(result), **marker))
    native.event("HRSID_MASK_PHASE_E24_OUTPUT_COMPLETE", mask_resize_mode=args.mode, protocol=args.protocol)


if __name__ == "__main__":
    main()
