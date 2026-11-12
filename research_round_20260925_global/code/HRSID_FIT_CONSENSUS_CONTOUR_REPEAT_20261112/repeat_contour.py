"""Apply the previously frozen contour rules to the independently closed older FIT36."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

PROTOCOL = "5ba52ca3ed9e0fa3a4a29d3dcdb3d060d7f2c694b4db0acf3e6c7d49c2558cbe"
CONTOUR = "65bf443df93eda98d6772358ecf88e2ab815c5a2ef4fb41718ed554c2328ba85"
AREA = "448f37510bc87af1ce0bf2c483745b5966e243618783ebd8adc3b2d4815b52d1"
CAPTURE = "582d11542c3ec777fb2309493459045d51b51edc500a888970ccb03d2b57b3ac"
CAPTURE_TERMINAL = "d3a799c86550ea0060cf178dd8863a68a2168e740fd64a1634697ab52a457076"
CAPTURE_INDEPENDENT = "426bb58dd1035757a4e65225803fa999277e18e5ffd6d9103a4ca7b315beadf2"
STRUCTURAL = "ade28b44d1ef5701e8986be93624663bfd054e60f75b6eef1cef8c1acd6199ad"
STRUCTURAL_INDEPENDENT = "b93dda9f9a38115b02bced19ecbe5916c05aac012a38b34349935d1948a8531f"
WITNESS = "795884c58ed6573fd0b33a50331c831d26d597f79c872bd1fc802e7a26bd4818"
OLD_MANIFEST = "0453980c7c4f88b08506080b05ac58ccb7a89b948a055b6b2be7b719394dafac"
INITIAL_MANIFEST = "0b87d626884550f5617f450ebc6ca217c4bbc331a1ac41e3d975289e403e3629"
SEEDS = (5202, 5201, 5203)
MASK_MODES = {"anchor_count", "vote", "soft"}
PRELABEL_STATUS = "HRSID_FIT_CONSENSUS_CONTOUR_REPEAT_PRELABEL"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_module(name, path, source):
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def load_helpers(code):
    area_path = code / "HRSID_FIT_SEED_AREA_CONFIRM_20261112/score/score_seed_area_confirm.py"
    source = area_path.read_bytes()
    require(sha256(source).hexdigest() == AREA, "Exact published area helper")
    area = load_module("_repeat_area", area_path, source)
    contour_path = code / "HRSID_FIT_CONSENSUS_CONTOUR_20261112/score_consensus.py"
    source = area.pinned(contour_path, CONTOUR).decode()
    substitutions = (
        ("all(len(items) == 2620 for items in detections.values())",
         "all(len(items) == 2873 for items in detections.values())"),
        ('    require(sum(len(rr) for rr in geometry.values()) == 69 and len(geometry) == 31,\n'
         '            "Original exact eligible fixed geometry denominator")',
         '    fixed_pairs = sum(len(rr) for rr in geometry.values())\n'
         '    fixed_groups = len(geometry)\n'
         '    require(fixed_pairs > 0 and {f for _, f in geometry} == {0,1,2},\n'
         '            "Complete frozen valid matched geometry and all folds")'),
        ("fixed_geometry_pairs=69,\n                fixed_geometry_groups=31",
         "fixed_geometry_pairs=fixed_pairs,\n                fixed_geometry_groups=fixed_groups"),
    )
    for original, repeat in substitutions:
        require(source.count(original) == 1, "Exact candidate/geometry denominator adaptation")
        source = source.replace(original, repeat)
    return area, load_module("_repeat_contour", contour_path, source)


def assert_disjoint(old, initial):
    for field in ("group", "basename", "encoded_sha256", "decoded_rgb_sha256"):
        a, b = {r[field] for r in old}, {r[field] for r in initial}
        require(len(a) == len(old) and len(b) == len(initial) and a.isdisjoint(b),
                "Unique and source-disjoint cohort field: " + field)


def read_roster(area, args):
    old = area.json_value(area.pinned(args.old_manifest, OLD_MANIFEST))["images"]
    initial = area.json_value(area.pinned(args.initial_manifest, INITIAL_MANIFEST))["images"]
    for roster in (old, initial):
        require([row["ordinal"] for row in roster] == list(range(1, 37)), "Exact cohort ordinals")
        require(all(row["basename"].split("_", 1)[0] == row["group"] for row in roster),
                "Cohort basename and source prefix join")
    assert_disjoint(old, initial)
    return old


def read_capture(area, args):
    read = lambda path, digest: area.json_value(area.pinned(path, digest))
    cap = read(args.capture / "RESULT_SAFE.json", CAPTURE)
    terminal = read(args.capture / "ORIGINAL_TOOL_TERMINAL_PRIVATE.json", CAPTURE_TERMINAL)
    cap_independent = read(args.capture_independent, CAPTURE_INDEPENDENT)
    require(area.fit_only(cap) and cap["status"] == "HRSID_FIT_SEED_CONSENSUS_COVERAGE_ACTUAL"
            and cap["AP"] is False and cap["manifest_sha256"] == OLD_MANIFEST
            and cap["coverage"]["gate"] == "COVERAGE_STOP"
            and cap["images"] == 36 and cap["model_forwards"] == 108
            and terminal["exit_code"] == 0 and bool(terminal["chunk_id"]),
            "Exact naturally closed original capture")
    require(cap_independent["status"] == "CLOSED_SEED_CONSENSUS_COVERAGE_INDEPENDENT_READBACK"
            and cap_independent["result_sha256"] == CAPTURE and cap_independent["images"] == 36
            and cap_independent["coverage_verified"] is True
            and cap_independent["logit_file_hashes_checked"] is True
            and cap_independent["labels_opened"] is False and cap_independent["models_executed"] is False,
            "Independent original capture and plane checks")
    return cap


def read_structure(area, args):
    read = lambda path, digest: area.json_value(area.pinned(path, digest))
    structural = read(args.structural / "RESULT_SAFE.json", STRUCTURAL)
    independent = read(args.structural_independent, STRUCTURAL_INDEPENDENT)
    require(area.fit_only(structural) and structural["AP"] is False
            and structural["status"] == "HRSID_FIT_ASSOC_STRUCTURE_ACTUAL"
            and structural["capture_sha256"] == CAPTURE and structural["witness_sha256"] == WITNESS,
            "Exact original GT-free structural receipt")
    keys = ("rank_greedy_and_reciprocal_pairs", "representative_identities", "all_method_counts_and_offsets_match")
    require(independent["status"] == "HRSID_FIT_ASSOC_STRUCTURE_INDEPENDENT_READBACK"
            and independent["result_sha256"] == STRUCTURAL and independent["capture_sha256"] == CAPTURE
            and independent["witness_sha256"] == WITNESS and area.fit_only(independent["scopes"])
            and independent["scopes"]["AP"] is False and independent["checks"]["images"] == 36
            and all(independent["checks"][key] is True for key in keys)
            and independent["coverage"]["nms90"]["first"] == 540,
            "Independent structural identity and pairing checks")
    return area.stream(args.structural / "WITNESS_PRIVATE.jsonl", WITNESS)


def read_arm(area, args, cap, roster, seed):
    row, = [arm for arm in cap["arms"] if arm["seed"] == seed]
    rows = area.stream(args.capture / str(seed) / "CANDIDATES_PRIVATE.jsonl", row["prediction_sha256"])
    require(len(rows) == row["images"] == len(roster) == 36, "Original seed roster count")
    for image, recorded in zip(roster, rows, strict=True):
        require(all(image[key] == recorded[key] for key in ("ordinal", "group", "basename"))
                and image["decoded_rgb_sha256"] == recorded["rgb_sha256"], "Exact closed per-image identity")
        pool = recorded["candidates"]
        ranks = [cand["rank"] for cand in pool]
        require(ranks == sorted(set(ranks)) and all(cand["area"] > 0 for cand in pool),
                "Exact ordered nonempty original candidates")
    require(sum(len(x["candidates"]) for x in rows) == row["nonempty"], "Exact nonempty count")
    return rows


def validate_pairs(pairs, candidates):
    used = {5201: set(), 5203: set()}
    for key, donors in pairs.items():
        rank = int(key)
        require(key == str(rank), "Canonical anchor rank")
        anchor = candidates[5202][rank]
        require(32 <= anchor["area"] < 1024 and type(anchor["logit_row"]) is int, "Anchor eligibility")
        for seed, donor in zip((5201, 5203), donors, strict=True):
            record = candidates[seed][donor]
            require(donor not in used[seed], "Donors are exclusive within a seed and image")
            require(16 <= record["area"] < 2048 and type(record["logit_row"]) is int,
                    "Exact donor captured area and row")
            used[seed].add(donor)
        yield dict(rank=rank, donor_ranks={"5201": donors[0], "5203": donors[1]},
                   cap=2*min(8, anchor["area"]//10))


def write_stream(path, rows):
    digest = sha256()
    with path.open("xb") as out:
        for block in rows:
            line = (json.dumps(block, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
            out.write(line)
            digest.update(line)
        out.flush()
        os.fsync(out.fileno())
    return digest.hexdigest()


def read_variants(cases, edits):
    output = []
    for case, block in zip(cases, edits, strict=True):
        require(case["basename"] == block["basename"], "Exact variants image join")
        candidate_ranks = {cand["rank"] for cand in case["candidates"]}
        variants = {}
        for key, modes in block["variants"].items():
            rank = int(key)
            require(str(rank) == key and rank in candidate_ranks and set(modes) == MASK_MODES,
                    "Exact prelabel variant rank and modes")
            variants[rank] = modes
        output.append(dict(basename=case["basename"], variants=variants))
    return output


def prepare(args, area, contour, cap, roster, codec):
    from scipy.special import expit
    structure = read_structure(area, args)
    arms = {s: read_arm(area, args, cap, roster, s) for s in SEEDS}
    verify = area.module("_repeat_signals", args.code /
                         "HRSID_FIT_SEED_AREA_CONFIRM_20261112/verify_area_signals.py", contour.VERIFY)
    require(len(structure) == 36, "Exact structural roster")
    rows = []
    visits = additions = drops = sampled = fallbacks = associated = groups = 0
    for i, (image, witness) in enumerate(zip(roster, structure, strict=True)):
        require(all(image[key] == witness[key] for key in ("ordinal", "basename")), "Exact witness image")
        blocks = {s: arms[s][i] for s in SEEDS}
        candidates = {s: {row["rank"]: row for row in blocks[s]["candidates"]} for s in SEEDS}
        planes = {s: verify.array_from(args.capture / str(s), blocks[s]) for s in SEEDS}
        method = witness["methods"]["nms90_first"]["pairs"]
        representatives = witness["reps"]["90"]
        variants = {}
        for signal in validate_pairs(method, candidates):
            rank = signal["rank"]
            chosen = {5202: rank, **{int(s): r for s, r in signal["donor_ranks"].items()}}
            require(all(representatives[str(s)][str(r)] == r for s, r in chosen.items()),
                    "Exact chosen .90 representative")
            require(all(0 <= candidates[s][r]["logit_row"] < len(planes[s]) for s, r in chosen.items()),
                    "Exact bounded query-plane row")
            edited, (up, down), band, fallback = contour.band_edits(candidates, planes, signal, verify, codec, expit)
            sampled += band
            fallbacks += fallback
            if edited:
                variants[rank] = edited
                visits += 1
                additions += up
                drops += down
        associated += len(method)
        groups += bool(method)
        rows.append(dict(basename=image["basename"], variants=variants))
    require((associated, groups) == (540, 35), "Exact original .90 rank-greedy coverage")
    require(sum(len(x["candidates"]) for x in arms[5202]) == 2873, "Exact unchanged anchor pool")
    args.output.mkdir()
    variants_sha = write_stream(args.output / "MASK_EDITS_PRIVATE.jsonl", rows)
    anchor_sha, = [arm["prediction_sha256"] for arm in cap["arms"] if arm["seed"] == 5202]
    receipt = dict(status=PRELABEL_STATUS, FIT=True, DEV=False, TEST=False, AP=False, GT_opened=False,
                   original_candidates=2873, associated_candidates=associated, associated_groups=groups,
                   edited_candidates=visits, add_per_arm=additions, drop_per_arm=drops, native_band_positions=sampled,
                   author_torch_fallback_planes=fallbacks, variants_sha256=variants_sha, anchor_sha256=anchor_sha,
                   protocol_sha256=PROTOCOL, contour_source_sha256=CONTOUR, capture_sha256=CAPTURE,
                   structure_sha256=STRUCTURAL, witness_sha256=WITNESS, original_model_forwards=108,
                   source_prefix_overlap=0, rgb_content_overlap=0, source_groups=36,
                   source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                   model_forwards=0, optimizer_updates=0, older_capture_reciprocal_gate="COVERAGE_STOP")
    digest = area.write_new(args.output / "PRELABEL_SAFE.json", receipt)
    print(json.dumps(dict(status=PRELABEL_STATUS, prelabel_sha256=digest, edited_candidates=visits),
                     sort_keys=True), flush=True)


def score(args, area, contour, cap, roster, codec):
    import shapely
    require(shapely.__version__ == "2.0.2" and shapely.geos_version_string == "3.11.2", "Exact geometry")
    started = perf_counter()
    prelabel = area.json_value(area.pinned(args.prelabel / "PRELABEL_SAFE.json", args.prelabel_sha))
    terminal = area.json_value(area.pinned(args.prelabel_terminal, args.prelabel_terminal_sha))
    require(terminal["exit_code"] == 0 and terminal["chunk_id"]
            and json.loads(terminal["output"].strip().splitlines()[-1]) ==
            dict(status=PRELABEL_STATUS, prelabel_sha256=args.prelabel_sha,
                 edited_candidates=prelabel["edited_candidates"]), "Original natural prelabel terminal")
    require(prelabel["status"] == PRELABEL_STATUS and area.fit_only(prelabel) and prelabel["AP"] is False
            and prelabel["protocol_sha256"] == PROTOCOL and prelabel["source_sha256"] ==
            sha256(Path(__file__).read_bytes()).hexdigest(), "Exact label-free edit closure")
    cases = read_arm(area, args, cap, roster, 5202)
    require(prelabel["anchor_sha256"] == next(x["prediction_sha256"] for x in cap["arms"] if x["seed"] == 5202),
            "Exact original score, rank and RLE carrier")
    edits = read_variants(cases, area.stream(args.prelabel / "MASK_EDITS_PRIVATE.jsonl", prelabel["variants_sha256"]))
    require(sum(len(x["variants"]) for x in edits) == prelabel["edited_candidates"], "Exact prelabel variant count")
    stock = area.module("_repeat_match", args.code / "HRSID_FIT_HULL_CDF_20261112/stock_coco_match.py", contour.MATCH)
    endpoint = area.module("_repeat_geom", args.code /
                           "MASK_RTDETR_DEV705_CONTINUOUS_AP_20261112/continuous_endpoint.py", area.GEOMETRY_SHA)
    geometry = endpoint.geometry_modules(args.code)
    args.output.mkdir()
    scored = contour.run_score(args, area, stock, geometry, codec, cases, edits)
    counts = scored["edited_candidate_scope"]
    require(sum(counts[k] for k in ("edited_matches", "edited_unmatched", "edited_neutral")) ==
            prelabel["edited_candidates"], "Exact edited scope closure")
    result = {**prelabel, **scored, "status": "HRSID_FIT_CONSENSUS_CONTOUR_REPEAT_ACTUAL", "GT_opened": True,
              "prelabel_sha256": args.prelabel_sha, "prelabel_terminal_sha256": args.prelabel_terminal_sha,
              "fit_index_sha256": area.LABEL_SHA, "worker_seconds": perf_counter()-started}
    digest = area.write_new(args.output / "RESULT_SAFE.json", result)
    print(json.dumps(dict(status=result["status"], result_sha256=digest, advance=result["advance"],
                          gates=result["gates"]), sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stage = parser.add_subparsers(dest="stage", required=True)
    common = ("protocol", "code", "capture", "capture-independent", "old-manifest", "initial-manifest", "output")
    for kind, additional in (("prepare", ("structural", "structural-independent")),
                             ("score", ("prelabel", "prelabel-terminal", "fit-index", "shapely-site"))):
        command = stage.add_parser(kind)
        for key in common + additional:
            command.add_argument("--" + key, required=True, type=Path)
        if kind == "score":
            command.add_argument("--prelabel-sha", required=True)
            command.add_argument("--prelabel-terminal-sha", required=True)
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "CPU-only environment")
    import torch
    require(torch.__version__ == "2.10.0+cu130", "Exact native interpolation runtime")
    torch.set_num_threads(1)

    def offline(event, _args):
        if event.startswith(("socket.", "subprocess.")):
            raise RuntimeError("Offline CPU diagnostic")

    sys.addaudithook(offline)
    if args.stage == "score":
        sys.path.append(str(args.shapely_site.resolve(strict=True)))
    from pycocotools import mask as codec
    area, contour = load_helpers(args.code)
    area.pinned(args.protocol, PROTOCOL)
    roster = read_roster(area, args)
    capture = read_capture(area, args)
    action = prepare if args.stage == "prepare" else score
    action(args, area, contour, capture, roster, codec)


if __name__ == "__main__":
    main()