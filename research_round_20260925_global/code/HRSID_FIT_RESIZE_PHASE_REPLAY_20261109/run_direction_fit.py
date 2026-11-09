"""Replay three mask sampling phases on the frozen HRSID FIT views."""

from argparse import ArgumentParser
from collections import defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter
from types import ModuleType

import numpy as np
import torch

from common_mask_supervision import ROW_ID
from direction_replay import replay_direction_view, summarize_direction
from fit_query_export import INDEX_SHA, need, pinned, write
from replay_fit import AUG_SHA, BASE_SEED, COHORT_SHA, REPEATS


def run(args):
    start = perf_counter()
    source = ModuleType("_qualified_native_fit_augmentation")
    source.__file__ = str(args.audit_source)
    payload = pinned(args.audit_source, AUG_SHA, 65536)
    exec(compile(payload, str(args.audit_source), "exec"), source.__dict__)
    sys.addaudithook(source.offline)
    need(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
        and sys.version_info[:3] == (3, 10, 13)
        and not torch.cuda.is_available()
        and all(source.version(name) == version for name, version in source.VERSIONS.items())
        and all(sha256(Path(module.__file__).read_bytes()).hexdigest() == digest
                for module, digest in source.SOURCE_SHA.items()),
        "Exact frozen CPU package and native augmentation sources",
    )
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cohort = json.loads(pinned(args.cohort, COHORT_SHA, 1024**2))
    data = json.loads(pinned(args.train / "_annotations.coco.json", INDEX_SHA, 32 * 1024**2))
    fit = {row["id"]: row for row in data["images"]}
    annotations = defaultdict(list)
    for ann in data["annotations"]:
        annotations[ann["image_id"]].append(ann)
    need(
        len(fit) == 2937 and len(data["annotations"]) == 9260
        and len(cohort["images"]) == 114 and cohort["fit_only"] is True,
        "Exact qualification and 114 locked FIT images",
    )
    audit = source.NativeInputAudit()
    layers = {"core": [], "stress": []}
    args.output.mkdir()
    with (args.output / "VIEWS_PRIVATE.jsonl").open("x", encoding="utf8", newline="\n") as stream:
        for ordinal, row in enumerate(cohort["images"]):
            iid, name = row["image_id"], row["native_basename"]
            need(fit[iid]["file_name"] == name and row["core"] != row["stress"], "Exact frozen FIT row")
            png = pinned(args.train / name, row["encoded_sha256"], 4 * 1024**2)
            image, scale = source.io_utils.decode_pil_image_bytes(png, draft_size=None)
            need(
                image.mode == "RGB" and image.size == (800, 800) and scale == (1., 1.)
                and sha256(np.asarray(image).tobytes(order="C")).hexdigest() == row["decoded_rgb_sha256"],
                "Exact qualified image and pixels",
            )
            image, raw = audit.convert(image, dict(image_id=iid, annotations=annotations[iid]))
            source.target_summary(raw, 800, iid, normalized=False)
            need(not bool(source.mask_empty(raw).any()), "Qualified original FIT masks are nonempty")
            raw[ROW_ID] = torch.arange(len(raw["labels"]), dtype=torch.int64)
            layer = "core" if row["core"] else "stress"

            def qualified_transform(image, target):
                image, target = audit.transform(image, target)
                source.target_summary(target, 576, iid, normalized=True)
                return image, target

            for repeat in range(REPEATS):
                seed = BASE_SEED + ordinal * REPEATS + repeat
                view = replay_direction_view(source._torchvision, qualified_transform, source.set_seed, image, raw, seed)
                need(view["steps"][-1]["after"] == [576, 576], "Exact final FIT canvas")
                view.update(group=row["group"], layer=layer, ordinal=ordinal, seed=seed)
                layers[layer].append(view)
                stream.write(json.dumps(view, sort_keys=True, allow_nan=False) + "\n")
            if (ordinal + 1) % 10 == 0:
                print(json.dumps(dict(event="HRSID_FIT_VERTICAL_DIRECTION_REPLAY_PROGRESS", images=ordinal+1)),
                      flush=True)
        stream.flush()
        os.fsync(stream.fileno())
    need(len(layers["core"]) == 436 and len(layers["stress"]) == 20
         and len({view["group"] for view in layers["core"]}) == 109, "Exact fixed native replay views")
    result = dict(
        status="ACTUAL_HRSID_FIT_VERTICAL_DIRECTION_REPLAY",
        protocol=args.protocol, cohort_sha256=COHORT_SHA, fit_index_sha256=INDEX_SHA,
        groups_main=109, images=114, views=456,
        core=summarize_direction(layers["core"], primary=True),
        stress=summarize_direction(layers["stress"], primary=False),
        exact_shared_geometry_rng_images=True, exact_common_nonmask_fields=True,
        elapsed_seconds=perf_counter() - start,
        FIT=True, DEV=False, TEST=False, AP=False, forward_calls=0, optimizer_updates=0,
    )
    write(args.output / "RESULT_SAFE.json", result)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


def main():
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ("cohort", "train", "audit-source", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()
    need(
        re.fullmatch(r"[0-9a-f]{40}", args.protocol) and args.output.is_absolute()
        and args.output.parent.resolve(strict=True) == args.output.parent and not args.output.exists()
        and not args.output.is_symlink() and not args.output.is_relative_to(args.train.parent),
        "Exact prelock and private fresh output outside FIT",
    )
    run(args)


if __name__ == "__main__":
    main()
