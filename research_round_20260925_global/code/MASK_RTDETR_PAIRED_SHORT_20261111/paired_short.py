"""Qualify one stock or center Mask RT-DETR-S arm through the exact ARM trainer."""
import argparse
from collections import deque
from contextlib import ExitStack
from hashlib import sha256
import json
import os
from pathlib import Path
import pickle
import random
import re
import sys
from types import ModuleType
from unittest.mock import patch


SOURCE_PINS = {
    "qualify.py": "4d00b5bff4fef2bb9445fd405a5c0b7e1cae5a4d483cbd73c66eb3e699263113",
    "arm_runtime.py": "a06aefa66a1cbf116323b2524230e57e057664d95342d68ed3a7128f8464bf8f",
    "arm_two_steps.py": "09ed1b3c0736c600423fa23d36e1d29af8d2f905e42493314ad3f935b15ed749",
    "paddle_geometry.py": "9508c2b1e852b0d0fb35c81e58399e162ae9d5fdda04b9c2f0d37b9cc4ab5240",
}
AUTHOR_TRANSFORMS = {
    "ppdet/data/transform/operators.py": "b5a298403f1e683011c0953823176d2eccdb4e56708117e0439a0649f6ba450f",
    "ppdet/data/transform/batch_operators.py": "286015c9fca629ac9fdf3d6cba3e77680ad5fb75f2592d1e5dc670c5af97d3be",
    "configs/mask_rtdetr/_base_/mask_rtdetr_reader.yml": "aa4d8a56678441fe1bb4e31f0deb2985aba1c537fb1cdc88278a47ab8bb66dad",
}
ROW_KEY = "gt_read_order"
PRIOR_PROTOCOL = "799d164213d1a08c2975d2d2e1b710bcd107209b"
ARM_STATUS = "P308_MASK_RTDETR_PAIRED_SHORT_ARM_PENDING_INDEPENDENT_ACCEPTANCE_V1"
EMPTY_STATUS = "P308_MASK_RTDETR_PAIRED_SHORT_EMPTY_ROWS_REJECTED_V1"
NATIVE_STATUS = "MASK_RTDETR_S_NATIVE_ARM_CPU_FIT8_TWO_UPDATE_TRAIN_QUALIFICATION_V1"
SCOPE = dict(FIT=True, DEV=False, TEST=False, AP=False)


class EmptyRows(ValueError):
    pass


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def canonical_json(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def stable_source(path, expected):
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(), "Canonical pinned source required")
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        raw = stream.read(512 * 1024 + 1)
        after = os.fstat(stream.fileno())
    signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    require(0 < len(raw) == before.st_size <= 512 * 1024
            and signature(before) == signature(after) == signature(path.stat())
            and sha256(raw).hexdigest() == expected, "Exact source bytes required: " + path.name)
    return raw


def load_source(path, name):
    raw = stable_source(path, SOURCE_PINS[path.name])
    require(name not in sys.modules, "Fresh isolated source namespace required: " + name)
    module = ModuleType(name)
    module.__file__ = path.as_posix()
    sys.modules[name] = module
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def write_audit(path, report):
    raw = canonical_json(report)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(event=report["status"], sha256=sha256(raw).hexdigest()), sort_keys=True), flush=True)


class Witness:
    def __init__(self, arm, np, paddle, center):
        require(arm in ("native", "center"), "Exact paired arm required")
        self.arm = arm
        self.np = np
        self.paddle = paddle
        self.center = center
        self.pending = deque()
        self.batches = []
        self.initial_model_sha256 = None

    def array_sha(self, value):
        if hasattr(value, "numpy"):
            value = value.numpy()
        value = self.np.ascontiguousarray(value)
        h = sha256(canonical_json(dict(dtype=value.dtype.str, shape=list(value.shape))))
        h.update(memoryview(value).cast("B"))
        return h.hexdigest()

    def input_identity(self, value):
        if hasattr(value, "numpy") or isinstance(value, self.np.ndarray):
            return {"array_sha256": self.array_sha(value)}
        if isinstance(value, self.np.generic):
            return self.input_identity(value.item())
        if type(value) in (list, tuple):
            return [self.input_identity(item) for item in value]
        if type(value) is dict:
            return {key: self.input_identity(item) for key, item in value.items()}
        require(value is None or type(value) in (str, int, float, bool), "Unexpected actual nonmask input type")
        return value

    def aug_rng(self):
        return sha256(pickle.dumps((random.getstate(), self.np.random.get_state()), protocol=4)).hexdigest()

    def paddle_rng(self):
        state, = self.paddle.get_rng_state(device="cpu")
        raw = str(state).strip()
        require(re.fullmatch(r"-?[0-9]+(?: [0-9]+){100,}", raw) is not None,
                "Paddle 2.6.2 CPU generator must expose the full numeric engine state")
        return sha256(raw.encode("ascii")).hexdigest()

    def tag_after_poly(self, original):
        def apply(operator, sample, context=None):
            out = original(operator, sample, context)
            require(ROW_KEY not in out, "Author already has the private row witness")
            out[ROW_KEY] = self.np.arange(len(out["gt_segm"]), dtype=self.np.int64).reshape(-1, 1)
            return out
        return apply

    def resize_type(self, original):
        witness = self

        class SelectedResize(original):
            def apply(self, sample, context=None):
                masks = sample["gt_segm"]
                rows = [int(value) for value in sample.pop(ROW_KEY).reshape(-1)]
                image_id = int(sample["im_id"].reshape(-1)[0])
                rng = witness.aug_rng()
                pre_mask_sha = witness.array_sha(masks)
                out = super().apply(sample, context)
                native = out["gt_segm"]
                hw = tuple(int(x) for x in out["im_shape"])
                center = witness.center(masks, hw)
                require(native.shape == center.shape and native.dtype == center.dtype == witness.np.uint8
                        and len(rows) == len(set(rows)) == len(native) == len(out["gt_bbox"]) == len(out["gt_class"])
                        and len(rows) > 0 and min(rows) >= 0, "Exact paired stock row, mask, dtype and shape required")
                require(masks.shape[-2:] != hw or witness.np.array_equal(native, center),
                        "Identity resize must preserve exact stock and center mask equality")
                witness.pending.append(dict(
                    image_id=image_id, rows=rows, resize_in_hw=list(masks.shape[-2:]), resize_out_hw=list(hw),
                    image_interpolation=int(self.interp), augmentation_rng_sha256=rng, pre_mask_sha256=pre_mask_sha,
                    native_mask_sha256=witness.array_sha(native), center_mask_sha256=witness.array_sha(center),
                    nonempty_native=[bool(row.any()) for row in native], nonempty_center=[bool(row.any()) for row in center],
                    changed_masks=sum(not witness.np.array_equal(a, b) for a, b in zip(native, center, strict=True)),
                ))
                if witness.arm == "center":
                    out["gt_segm"] = center
                return out

        return SelectedResize

    def pre_forward(self, layer, inputs):
        data, = inputs
        ids = [int(value) for value in data["im_id"].numpy().reshape(-1)]
        require(len(ids) == len(data["gt_segm"]) == len(data["gt_bbox"]) == len(data["gt_class"]) == 4
                and len(self.pending) >= 4 and ROW_KEY not in data and len(self.batches) < 2,
                "Exactly four original ordered rows per actual model batch required")
        rows = [self.pending.popleft() for _ in ids]
        require(ids == [row["image_id"] for row in rows] and len(set(ids)) == 4, "Actual model image order differs")
        model_input = data["image"].numpy()
        require(model_input.ndim == 4 and model_input.shape[0] == 4 and model_input.shape[1] == 3,
                "Four actual CHW images required")
        for i, row in enumerate(rows):
            masks, boxes, labels = (data[field][i] for field in ("gt_segm", "gt_bbox", "gt_class"))
            selected_sha = self.array_sha(masks)
            require(selected_sha == row[self.arm + "_mask_sha256"] and len(row["rows"]) == len(boxes) == len(labels)
                    and bool((labels.numpy() == 0).all()), "Actual consumed selected masks, boxes and class 0 required")
            row.update(image_sha256=self.array_sha(model_input[i]), boxes_sha256=self.array_sha(boxes),
                       labels_sha256=self.array_sha(labels))
        if self.initial_model_sha256 is None:
            state = layer.state_dict()
            require(len(state) == 646, "Exact stock initialized online model state size required")
            typed = {name: self.array_sha(value) for name, value in state.items()}
            self.initial_model_sha256 = sha256(canonical_json(typed)).hexdigest()
        explicit = {"image", "gt_segm", "gt_bbox", "gt_class", "im_shape", "scale_factor"}
        other = self.input_identity({key: value for key, value in data.items() if key not in explicit})
        self.batches.append(dict(ordinal=len(self.batches) + 1, rows=rows, paddle_cpu_rng_sha256=self.paddle_rng(),
                                 other_input_sha256=sha256(canonical_json(other)).hexdigest(),
                                 epoch_id=int(data["epoch_id"]), input_keys=sorted(data),
                                 im_shape_sha256=self.array_sha(data["im_shape"]),
                                 scale_factor_sha256=self.array_sha(data["scale_factor"])))
        if not all(all(row["nonempty_native"]) and all(row["nonempty_center"]) for row in rows):
            raise EmptyRows("At least one true retained FIT row is empty under a paired mask sampler")

    def report(self, status, scope):
        rows = [row for batch in self.batches for row in batch["rows"]]
        selected = [row[self.arm + "_mask_sha256"] for row in rows]
        model_rng = [batch["paddle_cpu_rng_sha256"] for batch in self.batches]
        common = [{key: value for key, value in batch.items() if key != "paddle_cpu_rng_sha256"}
                  for batch in self.batches]
        return dict(status=status, arm=self.arm, initial_model_sha256=self.initial_model_sha256,
                    initial_paddle_rng_sha256=model_rng[0], paddle_rng_stream_sha256=sha256(canonical_json(model_rng)).hexdigest(),
                    ordered_image_ids=[row["image_id"] for row in rows], fit_visits=len(rows),
                    changed_masks=sum(row["changed_masks"] for row in rows),
                    empty_native=sum(sum(not bit for bit in row["nonempty_native"]) for row in rows),
                    empty_center=sum(sum(not bit for bit in row["nonempty_center"]) for row in rows),
                    differential_empty=sum(sum(a != b for a, b in zip(row["nonempty_native"], row["nonempty_center"], strict=True)) for row in rows),
                    paired_batches=self.batches, common_ledger_sha256=sha256(canonical_json(common)).hexdigest(),
                    selected_mask_stream_sha256=sha256(canonical_json(selected)).hexdigest(), **scope)


def run(args):
    outputs = (args.output, args.artifacts, args.pair_audit)
    require(re.fullmatch(r"[0-9a-f]{40}", args.protocol_commit) is not None and args.protocol_commit != PRIOR_PROTOCOL,
            "Distinct prospective protocol commit required")
    require(all(path.is_absolute() and path.resolve(strict=True) == path and path.is_dir()
                for path in (args.source, args.fit, args.qualification_code, args.geometry_code)),
            "Exact canonical author, FIT and published source directories required")
    require(all(path.is_absolute() and path.parent.resolve(strict=True) == path.parent and not path.exists()
                and not path.is_symlink() and not any(path.is_relative_to(p) for p in (args.fit, args.source))
                for path in outputs) and len(set(outputs)) == 3
            and not any(path.is_relative_to(args.artifacts) for path in (args.output, args.pair_audit)),
            "Fresh distinct outputs outside data and author source required")
    require(args.backbone.is_absolute() and args.backbone.resolve(strict=True) == args.backbone
            and args.backbone.is_file(), "Exact canonical official backbone path required")
    require(sys.platform == "linux" and sys.flags.isolated == sys.flags.utf8_mode == 1 and sys.dont_write_bytecode,
            "Exact isolated Python invocation required")
    require({path.name for path in args.fit.iterdir()} == {"CLOSED_INPUT_PRIVATE.json", "images", "annotations.json"},
            "Exact isolated original SSDD FIT8 layout required")
    for relative, expected in AUTHOR_TRANSFORMS.items():
        stable_source(args.source / relative, expected)
    qualify = load_source(args.qualification_code / "qualify.py", "qualify")
    load_source(args.qualification_code / "arm_runtime.py", "arm_runtime")
    original = load_source(args.qualification_code / "arm_two_steps.py", "_p308_original_arm")
    geometry = load_source(args.geometry_code / "paddle_geometry.py", "_p308_fixed_geometry")
    require(geometry.UPSTREAM_COMMIT == qualify.SOURCE_COMMIT and qualify.SEED == 20261111, "Exact paired upstream and seed")
    original_runtime = original.load_runtime
    sys.addaudithook(qualify.deny_network)
    witness = None
    started = False
    argv = [str(original.__file__), "--protocol-commit", args.protocol_commit]
    for name in ("source", "fit", "backbone", "output", "artifacts"):
        argv.extend(("--" + name, str(getattr(args, name))))
    base_scope = dict(protocol_commit=args.protocol_commit, upstream_commit=qualify.SOURCE_COMMIT,
                      fit_manifest_sha256=qualify.MANIFEST_SHA, backbone_sha256=original.BACKBONE_SHA,
                      seed=qualify.SEED, **SCOPE)

    with ExitStack() as stack:
        def paired_runtime(source):
            nonlocal witness
            require(witness is None, "Only one exact author ARM runtime")
            np, paddle = original_runtime(source)
            from ppdet import engine
            from ppdet.data.transform import batch_operators, operators
            require(batch_operators.Resize is operators.Resize, "Author Resize binding must be the untouched stock class")
            witness = Witness(args.arm, np, paddle, geometry.center_masks)
            trainer = engine.Trainer

            def paired_trainer(*args, **kwargs):
                nonlocal started
                require(not started and kwargs.get("mode") == "train", "One exact stock train-only Trainer required")
                started = True
                active = trainer(*args, **kwargs)
                active.model.register_forward_pre_hook(witness.pre_forward)
                return active

            stack.enter_context(patch.object(operators.Poly2Mask, "apply",
                                            witness.tag_after_poly(operators.Poly2Mask.apply)))
            stack.enter_context(patch.object(batch_operators, "Resize", witness.resize_type(operators.Resize)))
            stack.enter_context(patch.object(engine, "Trainer", paired_trainer))
            return np, paddle

        stack.enter_context(patch.object(original, "load_runtime", paired_runtime))
        stack.enter_context(patch.object(sys, "argv", argv))
        try:
            original.main()
        except EmptyRows:
            require(witness is not None and not args.output.exists(), "Rejected arm cannot have an accepted native result")
            write_audit(args.pair_audit, witness.report(EMPTY_STATUS, base_scope))
            return 4
    require(started and witness is not None and len(witness.batches) == 2 and not witness.pending
            and witness.initial_model_sha256 is not None, "Exact consumed fit batches with no leftover fetches required")
    report = witness.report(ARM_STATUS, base_scope)
    require(report["fit_visits"] == len(set(report["ordered_image_ids"])) == 8 and report["changed_masks"] > 0
            and report["empty_native"] == report["empty_center"] == report["differential_empty"] == 0,
            "Exact distinct FIT8 visits and changed common nonempty supervision required")
    native_bytes = args.output.read_bytes()
    native = json.loads(native_bytes)
    require(native["status"] == NATIVE_STATUS and native["fit_visits"] == 8
            and native["counts"] == dict(optimizer=2, lr=2, ema=2) and native["completed_epochs"] == 1
            and native["official_DEV_opened"] is native["official_TEST_opened"] is native["AP_evaluated"] is False,
            "Original ARM two-update training qualification must finish")
    report.update(native_report_sha256=sha256(native_bytes).hexdigest(), optimizer_updates=2, lr_updates=2, ema_updates=2)
    write_audit(args.pair_audit, report)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("source", "fit", "backbone", "output", "artifacts", "pair-audit", "qualification-code", "geometry-code"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--protocol-commit", required=True)
    parser.add_argument("--arm", required=True, choices=("native", "center"))
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
