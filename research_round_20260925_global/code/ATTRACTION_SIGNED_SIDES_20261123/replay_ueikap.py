"""Cross archived static-attraction and learned edit locations at UEIKAP's original dynamic count."""
import argparse
from collections import Counter
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import sys
from types import ModuleType

from pycocotools import mask as codec


PROTOCOL = "c0053d7cdf70372b1a2b8e41093a2cac3800982e7f05b3c3772618fabe7fd15a"
OLD_PROTOCOL = "fbcc4d067351a7a7d2db226e6261a13c2f399eed59be414226e6ae29b6553bf2"
CROSS = "aa590649fb2fa0a70d7fd239b26e9f865efac2d8c0dfe86c4525933f6720f214"
STATUS = "ATTRACTION_SIGNED_SIDES_UEIKAP342_REPLAY_COMPLETE"
ARMS = {"FF": "SS", "IF": "LS", "FI": "SL", "II": "LL"}
STATIC = "attraction_static_at_dynamic_count"
LEARNED = "learned_at_attraction_count"
DYNAMIC = "attraction_dynamic"
IMAGES, GROUPS = 342, 13
PHASES = {
    "intake": ("UEIKAP_ADDITIONAL_AUTHOR_INTAKE_COMPLETE",
               "ba7ec4395f6b96f8b20e2e33f0a04c0978451556275663fe02b63486e2902744",
               "adc37131157ee24a65ac4b9e33ea1623011f7e07f733664f4ec347c00db09be3"),
    "capture": ("UEIKAP_ADDITIONAL_RFDETR_CAPTURE_COMPLETE",
                "d7f99ff75782172fcfd263c8af43293b375b58b4ee2a2129139f892fc38ab5b9",
                "ad5ba3d2320699df213a83aa1128f39d89a71bcf3cc13d56c538e1c945f51a9e"),
    "verify-capture": ("UEIKAP_ADDITIONAL_RFDETR_CAPTURE_VERIFIED",
                       "1f7790e9858a19fadafc495b38592e93858fc00626bd92bcca0e548ed3b49ff6",
                       "8462bfc621fea88412ccee213e9c2c80e049dca0ce8fc67717b61d56f21e45e9"),
    "prepare-edits": ("UEIKAP_ADDITIONAL_RFDETR_EDITS_PREPARED",
                      "4582b7911caf3e086c2974f08a03f7d4005025534e1b095b35411aaca2763eaf",
                      "dd45302c8ff07fb6da24d8246c9da2a1874433a2cf7a312d4a676420ec45cb78"),
    "verify-edits": ("UEIKAP_ADDITIONAL_RFDETR_EDITS_VERIFIED",
                     "c689f7a75f42d4ffc6650a50a11438208553f54c28fb078230b3b2dec9235b75",
                     "4dbb93fec516bc0becf76b313b7e03d9af4f245dcfcc44a7fccec4ca6360bfd0"),
}
COUNT_FIELDS = ("mapped_candidates", "eligible_candidates", "positive_budget_candidates", "swaps_per_arm",
                "adds_differ_candidates", "drops_differ_candidates", "both_sides_differ_candidates",
                "adds_displaced_pixels", "drops_displaced_pixels")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pinned(path, expected):
    data = path.read_bytes()
    require(sha256(data).hexdigest() == expected, "Digest mismatch for " + path.name)
    return data


def old_writer(code):
    path = code / "INDEPENDENT_SIDE_FACTORIAL_20261123/replay_sides.py"
    module = ModuleType(path.stem)
    module.__file__ = str(path)
    exec(compile(pinned(path, CROSS), str(path), "exec"), module.__dict__)
    return module


def old_inputs(root):
    reports = {}
    for phase, (status, result_sha, terminal_sha) in PHASES.items():
        receipt = json.loads(pinned(root / phase / "RESULT_SAFE.json", result_sha))
        observer = "observed_capture_02" if phase == "capture" else "observed_" + phase
        event = json.loads(pinned(root / observer / "TERMINAL.json", terminal_sha))
        require(receipt["status"] == status and receipt["protocol_sha256"] == OLD_PROTOCOL
                and receipt["images"] == IMAGES and event["phase"] == phase and type(event["actual_exit"]) is int
                and event["actual_exit"] == 0 and event["output_footer_verified"] is True
                and event["worker_sha256"] == receipt["worker_sha256"] and event["result_sha256"] == result_sha,
                "An original phase did not close against its observed footer")
        if phase != "intake":
            require(receipt["groups"] == GROUPS and receipt["target_labels_opened"] is False
                    and receipt["GT_opened"] is False and receipt["optimizer_updates"] == 0, "Expected label-free provenance")
        reports[phase] = receipt
    intake, cap, readback, prepared, replay = [reports[phase] for phase in PHASES]
    manifest = intake["images_roster_sha256"]
    require(intake["acquisition_groups"] == GROUPS and cap["intake_result_sha256"] == PHASES["intake"][1]
            and readback["capture_sha256"] == PHASES["capture"][1]
            and readback["capture_terminal_sha256"] == PHASES["capture"][2]
            and readback["exact_native_parity"] is True and readback["counts"] == cap["counts"],
            "Native readback or intake binding differs")
    for entry in (cap, readback, prepared, replay):
        require(entry["manifest_sha256"] == manifest and entry["intake_terminal_sha256"] == PHASES["intake"][2]
                and entry["predictions_sha256"] == cap["predictions_sha256"], "Native manifest or candidate stream differs")
    require(readback["audit_sha256"] == cap["audit_sha256"] and readback["checkpoint_sha256"] == cap["checkpoint_sha256"],
            "Native audit or checkpoint binding differs")
    for entry in (prepared, replay):
        require(entry["capture_receipt_sha256"] == PHASES["verify-capture"][1]
                and entry["capture_terminal_sha256"] == PHASES["verify-capture"][2]
                and entry["candidate_audit_sha256"] == cap["audit_sha256"] and entry["model_forwards"] == 0,
                "Old signed choices came from a different carrier")
    require(replay["prepared_receipt_sha256"] == PHASES["prepare-edits"][1] and replay["exact_support_xor_area"] is True
            and replay["exact_replayed_policy_choices"] is True and replay["edits_sha256"] == prepared["edits_sha256"]
            and replay["counts"] == prepared["counts"], "Original independent mask replay differs")
    lineage = dict(manifest_sha256=manifest, predictions_sha256=cap["predictions_sha256"], audit_sha256=cap["audit_sha256"],
                   checkpoint_sha256=cap["checkpoint_sha256"], old_edits_sha256=replay["edits_sha256"])
    for phase, label in zip(PHASES, ("intake", "capture", "capture_readback", "old_prepare", "old_replay"), strict=True):
        lineage[label + "_result_sha256"], lineage[label + "_terminal_sha256"] = PHASES[phase][1:]
    return lineage, cap["counts"], replay["counts"]


def records(path, expected):
    digest = sha256()
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            yield json.loads(line)
    require(digest.hexdigest() == expected, "Full stream digest differs for " + path.name)


def render_image(i, item, audit, prediction, old, writer, totals):
    require(i == item["image_id"] == audit["image_id"] == prediction["image_id"] == old["image_id"]
            and set(old) == {"image_id", "changes"} and audit["original_hw"] == item["original_hw"] == [512, 512]
            and audit["rgb_sha256"] == item["rgb_sha256"] and audit["image_sha256"] == item["image_ref"]["sha256"]
            and audit["source_image_name"] == f"{i:06d}.png", "Image source identity or dimensions differ")
    native = prediction["coco"]
    ranks = audit["native_ranks"]
    require(ranks == prediction["native_ranks"] == sorted(set(ranks))
            and all(1 <= rank <= 100 for rank in ranks), "Native candidate scores are not in original rank order")
    choices, eligible = {}, []
    for pos, (rank, q, cand, det) in enumerate(zip(ranks, audit["query_ids"], audit["candidates"], native, strict=True), 1):
        stock = det["segmentation"]
        area = int(codec.area(stock))
        require((pos, rank, q, det["score"], area) ==
                (cand["position"], cand["native_rank"], cand["query_id"], cand["score"], cand["mask_area"])
                and det["image_id"] == i and det["category_id"] == 1 and stock["size"] == [512, 512],
                "Original ranked mask, query or confidence differs")
        if 32 <= area < 1024:
            eligible.append((rank, q))
        key = str(pos)
        opts = old["changes"].get(key, {})
        require((STATIC in opts) == (LEARNED in opts) == (DYNAMIC in opts), "Positive dynamic-count support differs")
        if STATIC not in opts:
            continue
        static, learned, dynamic = opts[STATIC], opts[LEARNED], opts[DYNAMIC]
        k = static["swaps"]
        require(set(dynamic) == {"swaps", "rle"} and type(dynamic["swaps"]) is int and dynamic["swaps"] == k
                and dynamic["rle"]["size"] == stock["size"] and int(codec.area(dynamic["rle"])) == area
                and area - int(codec.area(writer.intersection(stock, dynamic["rle"]))) == k,
                "Actual dynamic donor signed count or mask area differs")
        arms, add, drop = writer.cross_masks(stock, static, learned, area)
        choices[key] = {new: arms[prior] for prior, new in ARMS.items()}
        totals.update(positive_budget_candidates=1, swaps_per_arm=k, adds_differ_candidates=int(add > 0),
                      drops_differ_candidates=int(drop > 0), both_sides_differ_candidates=int(add > 0 and drop > 0),
                      adds_displaced_pixels=add, drops_displaced_pixels=drop)
    require(eligible == list(zip(audit["logit_ranks"], audit["logit_query_ids"], strict=True))
            and all(k.isdecimal() and str(int(k)) == k and 1 <= int(k) <= len(native) for k in old["changes"]),
            "Eligible native positions or old edit keys differ")
    totals.update(mapped_candidates=len(native), eligible_candidates=len(eligible))
    return {"image_id": i, "changes": choices}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ("code", "root", "protocol", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    require(args.protocol_sha256 == PROTOCOL and not args.output.exists() and sys.flags.isolated and sys.dont_write_bytecode,
            "Use the prelocked protocol, isolated process and a fresh output directory")
    pinned(args.protocol, PROTOCOL)
    require(version("pycocotools") == "2.0.11" and version("numpy") == "1.26.4", "Use the qualified B5 codec")
    writer = old_writer(args.code)
    lineage, native_counts, old_counts = old_inputs(args.root)
    manifest = json.loads(pinned(args.root / "intake/images_roster.json", lineage["manifest_sha256"]))
    items = manifest["items"]
    ids = list(range(1, IMAGES + 1))
    require(manifest["expected_ids"] == ids == [item["image_id"] for item in items]
            and len({item["acquisition_group"] for item in items}) == GROUPS, "Locked whole-population manifest required")
    preds = records(args.root / "capture/PREDICTIONS_PRIVATE.jsonl", lineage["predictions_sha256"])
    audits = records(args.root / "capture/CANDIDATE_AUDIT_PRIVATE.jsonl", lineage["audit_sha256"])
    history = records(args.root / "prepare-edits/EDITS_PRIVATE.jsonl", lineage["old_edits_sha256"])
    args.output.mkdir()
    totals = Counter(dict.fromkeys(COUNT_FIELDS, 0))
    digest = sha256()
    with (args.output / "EDITS_PRIVATE.jsonl").open("xb") as stream:
        for i, (item, audit, pred, old) in enumerate(zip(items, audits, preds, history, strict=True), 1):
            line = writer.encoded(render_image(i, item, audit, pred, old, writer, totals))
            stream.write(line)
            digest.update(line)
    require(totals["mapped_candidates"] == native_counts["mapped"] == old_counts["mapped_candidates"]
            and totals["eligible_candidates"] == native_counts["eligible"] and old_counts["images"] == IMAGES
            and all(totals["positive_budget_candidates"] == old_counts[arm + "_candidates"]
                    and totals["swaps_per_arm"] == old_counts[arm + "_swaps"] for arm in (STATIC, LEARNED, DYNAMIC)),
            "A whole-population census differs from the old comparison")
    report = dict(status=STATUS, phase="replay", images=IMAGES, groups=GROUPS, arms=list(ARMS.values()),
                  input_lineage=lineage, counts=dict(totals), original_arms_exact=True, per_candidate_signed_counts_exact=True,
                  cross_context_signed_xor_exact=True, model_forwards=0, optimizer_updates=0, target_labels_opened=False,
                  GT_opened=False, protocol_sha256=PROTOCOL, worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  cross_operator_sha256=CROSS, edits_sha256=digest.hexdigest())
    data = writer.encoded(report)
    with (args.output / "RESULT_SAFE.json").open("xb") as stream:
        stream.write(data)
    print(json.dumps({"status": STATUS, "result_sha256": sha256(data).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
