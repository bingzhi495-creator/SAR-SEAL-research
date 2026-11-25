"""Read the original B5 endpoint cells for the already closed cross-carrier census."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType

import numpy as np
from pycocotools import mask


ASSOC_SHA = "a329af68d5e7088d933bba4c0754c98997aa219ecb2e83167ad3d7a4418543b4"
CENSUS_SHA = "796384965253513302202131343aaa5cbfbd9ea0b9caa2375f4fab4939bd20c3"
REPLAY_SHA = "f29a85ec36134380f4d2fcc4bab860c695f24044bdac8ae5e76541073ac8dade"
REPLAY_TERM_SHA = "f172ecbfc3c606b874b6c7839b35f0c0c3c5a8ae69ae20e69a2e4c575d60df9d"
EDIT_SHA = "f8c20ce94715e651a1e5cd90e2213a3a2793c2b9454de45de212d07c69335a6b"
STATUS = "HRSID_DEV705_CROSS_CARRIER_B5_SUPPORT_CENSUS_V1"


def load_source():
    path = Path(__file__).with_name("census_cross_carrier.py")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != ASSOC_SHA:
        raise ValueError("Original association worker binding")
    source = ModuleType("sealed_association")
    source.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), source.__dict__)
    return source


def verify_edit_metadata(a, args):
    root = args.rf
    replay = a.data(root / "replay/RESULT_SAFE.json", REPLAY_SHA)
    term = a.data(root / "observed_replay/TERMINAL.json", REPLAY_TERM_SHA)
    a.need(term["actual_exit"] == 0 and term["phase"] == "replay"
           and term["result_sha256"] == REPLAY_SHA and term["output_footer_verified"] is True
           and term["worker_sha256"] == replay["worker_sha256"]
           and replay["status"] == "B5_RFDETR_CENTER5202_DEV705_REPLAY_COMPLETE"
           and replay["exact_per_candidate_budgets"] is replay["exact_native_support"]
           is replay["actual_XOR_area_RLE_verified"] is replay["exact_historical_sparse_bytes"] is True
           and replay["edits_sha256"] == EDIT_SHA and replay["capture_result_sha256"] == a.RF_RESULT_SHA
           and replay["GT_opened"] is replay["AP"] is False
           and replay["model_forward_calls"] == replay["point_optimizer_updates"] == 0,
           "Closed exact original B5 endpoint replay")
    return replay


def examine(a, ordinal, fields, points, counts, coverage):
    item, rf_image, rf, query, df, audit, prob = fields
    rf_masks, rf_areas, df_masks, df_areas = a.admit_image(*fields)
    original_ranks = query["native_ranks"]
    expected = [(query["native_ranks"][p], query["native_ids"][p]) for p, area in enumerate(rf_areas)
                if 32 <= area < 1024]
    a.need(points["image_id"] == item["image_id"] and points["original_hw"] == a.HW
           and points["group"] == query["source_prefix"] and points["ordinal"] == ordinal
           and [(r["rank"], r["query_id"]) for r in points["points"]] == expected,
           "B5 support attached to exact area-eligible native rank and query")
    support = {r["rank"]:r["coordinates"] for r in points["points"]}
    seed = points["seeds"]["5217"]
    aidx = np.flatnonzero(rf_areas > 0)
    didx = np.flatnonzero(df_areas[:100] > 0)
    eligible = np.flatnonzero((rf_areas[aidx] >= 32) & (rf_areas[aidx] < 1024)).tolist()
    stock = [rf_masks[i] for i in aidx]
    donors = [df_masks[i] for i in didx]
    counts.update(images=1, eligible_anchors=len(eligible),
                  stock_point_pairs=sum(map(len, support.values())), stock_supported=sum(map(bool, support.values())))
    flags = set()
    for row in a.match_donors(stock, donors, eligible) if eligible else []:
        if row["decision"] != "accepted":
            continue
        counts["accepted"] += 1
        ai, di = row["anchor"], row["donor"]
        rank = original_ranks[aidx[ai]]
        candidates = support[rank]
        sf = seed.get(str(rank), {}).get("SF", {})
        k = sf.get("adds", 0)
        a.need(type(k) is int and k == sf.get("drops", 0) and 0 <= k <= min(8, len(candidates), rf_areas[aidx[ai]] // 10),
               "Original positive-margin seed 5217 budget")
        if not candidates:
            continue
        indices = np.asarray(candidates, dtype=np.int64)
        a.need(indices.ndim == 2 and indices.shape[1] == 2 and len(indices) <= 32
               and np.unique(indices).size == indices.size and np.all((indices >= 0) & (indices < 640000)),
               "Distinct original signed endpoint positions")
        own = mask.decode(stock[ai]).ravel()[indices]
        a.need(not own[:,0].any() and own[:,1].all(), "Verified original addition and removal polarity")
        votes = mask.decode(donors[di]).ravel()[indices].astype(bool)
        plus = votes[:,0]
        minus = votes[:,1]
        add_vary, delete_vary = (plus.any() and not plus.all(), minus.any() and not minus.all())
        nonconstant = add_vary or delete_vary
        add_sites, delete_sites = int(plus.sum()), int((~minus).sum())
        counts.update(accepted_supported=1, accepted_positive_seed5217_k=k>0,
                      accepted_endpoint_pairs=len(indices), donor_fill_proposed_endpoints=add_sites,
                      donor_clear_proposed_endpoints=delete_sites, accepted_vary_add=add_vary,
                      accepted_vary_remove=delete_vary, accepted_nonconstant_either=nonconstant,
                      accepted_nonconstant_both=add_vary and delete_vary,
                      accepted_nonconstant_and_positive_seed5217_k=nonconstant and k>0,
                      accepted_any_donor_fill=add_sites>0, accepted_any_donor_clear=delete_sites>0,
                      accepted_both_directional_witnesses=add_sites>0 and delete_sites>0)
        flags.add("accepted_supported")
        if nonconstant:
            flags.add("accepted_nonconstant_either")
        if nonconstant and k > 0:
            flags.add("accepted_nonconstant_and_positive_seed5217_k")
    for flag in flags:
        coverage[flag][0].add(item["image_id"])
        coverage[flag][1].add(points["group"])


def run(args):
    started = perf_counter()
    a = load_source()
    a.pinned(args.protocol, args.protocol_sha)
    sys.addaudithook(a.audit_hook)
    root = Path(__file__).parent
    previous = a.data(root / "capture/RESULT_SAFE.json", CENSUS_SHA)
    a.need(previous["status"] == a.STATUS and previous["worker_sha256"] == ASSOC_SHA, "Closed association census")
    images, _, _ = a.read_inputs(args)
    replay = verify_edit_metadata(a, args)
    counts = Counter()
    coverage = defaultdict(lambda: [set(), set()])
    with (args.rf / "replay/SPARSE_EDITS_PRIVATE.jsonl").open("rb") as ledger:
        digest = sha256()
        for block in iter(lambda: ledger.read(8 * 1024 * 1024), b""):
            digest.update(block)
        a.need(digest.hexdigest() == EDIT_SHA, "Exact frozen B5 sparse replay bytes")
        ledger.seek(0)
        for index, (fields, line) in enumerate(zip(images, ledger, strict=True), 1):
            examine(a, index, fields, json.loads(line), counts, coverage)
    a.need(counts["images"] == a.IMAGES and counts["eligible_anchors"] == previous["census"]["eligible_anchors"]
           and counts["accepted"] == previous["census"]["accepted"]
           and counts["stock_point_pairs"] == replay["counts"]["point_pairs"] == 1298958
           and counts["stock_supported"] == replay["counts"]["supported"] == 40897,
           "Exact original donor receipt and stock support census")
    return dict(status=STATUS, phase="verify-capture", census={name: int(value) for name, value in counts.items()},
                source_proxy_coverage={flag: dict(images=len(images), source_prefixes=len(groups))
                                       for flag, (images, groups) in sorted(coverage.items())},
                worker_sha256=a.digest(Path(__file__).read_bytes()), protocol_sha256=args.protocol_sha,
                original_census_sha256=CENSUS_SHA, replay_receipt_sha256=REPLAY_SHA, sparse_edits_sha256=EDIT_SHA,
                target_labels_opened=False, mask_scoring=False, detector_forwards=0, optimizer_updates=0,
                exact_b5_endpoint_vote_census=True, seconds=perf_counter()-started)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("rf", "df", "protocol", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--protocol-sha", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Fresh support census receipt destination")
    result = run(args)
    args.output.mkdir()
    raw = (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    with (args.output / "RESULT_SAFE.json").open("xb") as stream:
        stream.write(raw)
    print(json.dumps(dict(status=STATUS, result_sha256=sha256(raw).hexdigest())), flush=True)


if __name__ == "__main__":
    main()

