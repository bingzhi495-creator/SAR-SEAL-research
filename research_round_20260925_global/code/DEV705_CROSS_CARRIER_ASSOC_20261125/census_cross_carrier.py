"""Count cross-carrier donor feasibility using closed image-only DEV705 ledgers."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from pycocotools import mask


HW = [800, 800]
IMAGES = 705
IOU_FLOOR = .50
DUPLICATE_FLOOR = .90
MARGIN = .10
EPS = 1e-12
STATUS = "HRSID_DEV705_CROSS_CARRIER_DONOR_CENSUS_V1"
IMAGE_SHA = "39a40d089d4a54448fe6efca8ea5cc98ad1eb7c0587815f3dd9a3b097173e78f"
RF_TERM_SHA = "8351b20a3f838f5f7a31453f974a4fe2259f7b246e1a517d4cf0755566adbaf4"
RF_RESULT_SHA = "5feb6ee353443b97a4a360f146ee0a0093747eb92f348d1c3c49462526f3a3fc"
DF_READBACK_SHA = "277965bef4230ec7fadac30e103095af3ce68f99d1b69f74a8e348c391cef035"
DF_OWNER_SHA = "8f60048eae69a4323039a205c2132561a56787f3fd74c6200afcad34f3c09661"


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(data):
    return sha256(data).hexdigest()


def pinned(path, expected):
    raw = path.read_bytes()
    need(digest(raw) == expected, "Frozen SHA-256 differs: " + path.name)
    return raw


def data(path, expected):
    return json.loads(pinned(path, expected))


def stream(path, expected):
    return [json.loads(line) for line in pinned(path, expected).splitlines()]


def duplicate_margin(values, chosen, rles):
    close = np.flatnonzero((values[chosen] - values + EPS) < MARGIN)
    close = [int(x) for x in close if x != chosen]
    if not close:
        return True, False
    overlaps = mask.iou([rles[chosen]], [rles[i] for i in close], [0] * len(close))[0]
    return bool(np.all(overlaps + EPS >= DUPLICATE_FLOOR)), True


def match_donors(anchors, donors, eligible):
    if not donors:
        return [dict(anchor=i, donor=None, iou=0., decision="no_donor") for i in eligible]
    matrix = mask.iou(anchors, donors, [0] * len(donors))
    forward = np.argmax(matrix, axis=1)
    reverse = np.argmax(matrix, axis=0)
    output = []
    for i in eligible:
        j = int(forward[i])
        iou = float(matrix[i, j])
        row = dict(anchor=i, donor=j, iou=iou)
        if iou < IOU_FLOOR:
            row["decision"] = "below_iou"
        elif reverse[j] != i:
            row["decision"] = "not_mutual"
        else:
            donor_ok, donor_dups = duplicate_margin(matrix[i], j, donors)
            anchor_ok, anchor_dups = duplicate_margin(matrix[:, j], i, anchors)
            row.update(decision="accepted" if donor_ok and anchor_ok else "ambiguous",
                       ambiguity_duplicates=donor_dups or anchor_dups)
        output.append(row)
    return output


def read_inputs(args):
    rf = args.rf
    df = args.df
    rf_term = data(rf / "observed_capture/TERMINAL.json", RF_TERM_SHA)
    rf_acc = data(rf / "capture/RESULT_SAFE.json", RF_RESULT_SHA)
    df_acc = data(df / "INDEPENDENT_RESULT_SAFE.json", DF_READBACK_SHA)
    df_owner = data(df / "CONTROL/OWNER_CLOSURE_PRIVATE.json", DF_OWNER_SHA)
    df_term = data(df / "CONTROL/CHILD_TERMINAL_PRIVATE.json", df_owner["child_terminal_sha256"])
    df_out = df / "CONTROL/EXPORT"
    df_cap = data(df_out / "RESULT_SAFE.json", df_acc["export_result_sha256"])
    need(rf_term["actual_exit"] == 0 and rf_term["output_footer_verified"] is True
         and rf_term["result_sha256"] == RF_RESULT_SHA and rf_term["worker_sha256"] == rf_acc["worker_sha256"]
         and rf_acc["status"] == "B5_RFDETR_CENTER5202_DEV705_CAPTURE_COMPLETE"
         and rf_acc["historical_CUDA_stock_exact"] is True
         and rf_acc["model_forward_calls"] == IMAGES
         and df_owner["status"] == "HRSID_DFINE_E24_DEV705_EXPORT_OUTER_NATURAL_CLOSE_V1"
         and df_owner["resource_closed"] is True and df_term["original_os_exit"] == 0
         and df_term["original_os_wait"] is True and df_term["job_active_after_wait"] == 0
         and df_owner["result_sha256"] == df_acc["export_result_sha256"]
         and df_acc["status"] == "HRSID_DFINE_E24_DEV705_PROBABILITY_INDEPENDENTLY_ACCEPTED_V1"
         and df_acc["historical_stream_byte_parity"] is True and df_acc["exact_author_binary_replay"] == 70500
         and df_acc["original_os_exit"] == 0 and rf_acc["images"] == df_acc["images"] == df_cap["images"] == IMAGES
         and rf_acc["GT_opened"] is df_acc["GT_opened"] is df_cap["GT_opened"] is False
         and rf_acc["input_manifest_sha256"] == df_acc["manifest_sha256"] == df_cap["manifest_sha256"] == IMAGE_SHA,
         "Exact independently closed capture pair and shared image-only manifest")
    need(df_cap["confidence_threshold"] == 0 and df_cap["mask_threshold"] == .5 and df_cap["box_nms_iou"] == .70
         and df_acc["predictions_sha256"] == df_cap["predictions_sha256"]
         and df_acc["candidate_audit_sha256"] == df_cap["candidate_audit_sha256"]
         and df_acc["probability_index_sha256"] == df_cap["probability_index_sha256"],
         "Original D-FINE mask mode and stock stream closure")

    manifest = data(rf / "intake/DEV_IMAGE_ONLY_MANIFEST_PRIVATE.json", IMAGE_SHA)
    rf_images = data(rf / "capture/IMAGE_MANIFEST_PRIVATE.json", rf_acc["image_manifest_sha256"])
    stock = stream(rf / "capture/STOCK_NATIVE_PREDICTIONS_PRIVATE.jsonl", rf_acc["stock_predictions_sha256"])
    queries = stream(rf / "capture/QUERIES_PRIVATE.jsonl", rf_acc["queries_sha256"])
    masks = stream(df_out / "DEV_MASK_PREDICTIONS_PRIVATE.jsonl", df_acc["predictions_sha256"])
    audits = stream(df_out / "DEV_CANDIDATE_AUDIT_PRIVATE.jsonl", df_acc["candidate_audit_sha256"])
    index = stream(df_out / "PROBABILITIES_PRIVATE.jsonl", df_acc["probability_index_sha256"])
    need(len(manifest) == len({r["image_id"] for r in manifest}) == len({r["member"] for r in manifest}) == IMAGES,
         "Complete original unique-source DEV chip roster")
    return zip(manifest, rf_images, stock, queries, masks, audits, index, strict=True), rf_acc, df_cap


def native_rles(rows, image_id):
    for row in rows:
        rle = row["segmentation"]
        need(row["image_id"] == image_id and row["category_id"] == 1 and 0 <= row["score"] <= 1
             and rle["size"] == HW and isinstance(rle["counts"], str), "Native stock ship raster")
    rles = [row["segmentation"] for row in rows]
    return rles, np.asarray(mask.area(rles), dtype=np.int64) if rles else np.empty(0, dtype=np.int64)


def admit_image(item, rf_image, rf, queries, df, audit, df_prob):
    image_id = item["image_id"]
    need([item["height"], item["width"]] == HW
         and all(row["image_id"] == image_id for row in (rf_image, rf, queries, df, audit, df_prob))
         and item["encoded_sha256"] == rf_image["encoded_sha256"] == queries["encoded_sha256"]
         and item["decoded_rgb_sha256"] == rf_image["decoded_rgb_sha256"] == queries["image_content_sha256"]
         and Path(item["file_name"]).stem + ".png" == queries["native_basename"],
         "Per-image exact encoded/decoded identity and native dimensions")
    rf_rows, df_rows = rf["coco"], df["coco"]
    rf_masks, rf_areas = native_rles(rf_rows, image_id)
    df_masks, df_areas = native_rles(df_rows, image_id)
    ranks, ids = queries["native_ranks"], queries["native_ids"]
    kept = [(k, q) for k, (q, label) in enumerate(zip(queries["joint_query_ids"], queries["joint_labels"], strict=True), 1)
            if label == 0]
    need(len(queries["joint_query_ids"]) == len(queries["joint_scores"]) == rf["native_count"] == 100
         and rf["raw_query_count"] == len(queries["all_rles"]) == len(queries["ship_scores"]) == 164
         and len(ids) == len(set(ids)) == len(ranks) == len(rf_masks)
         and kept == list(zip(ranks, ids, strict=True)) and ranks == rf["native_ranks"]
         and rf["unmapped_slot1_count"] == 100 - len(ranks)
         and all(0 <= q < 164 and queries["all_rles"][q] == row["segmentation"]
                 and queries["ship_scores"][q] == row["score"] == queries["joint_scores"][rank - 1]
                 for q, rank, row in zip(ids, ranks, rf_rows, strict=True)),
         "Stock RF-DETR ranks/query/mask metadata closure")
    eligible_meta = [(ids[pos], ranks[pos], int(area), rf_rows[pos]["score"]) for pos, area in enumerate(rf_areas)
                     if 32 <= area < 1024]
    captured = queries["logit_capture"]
    need(captured["shape"] == [len(eligible_meta), 144, 144]
         and eligible_meta == [(v["query_id"], v["native_rank"], v["predicted_area"], v["score"])
                               for v in captured["candidates"]],
         "Anchor area eligibility matches closed posterior sidecar metadata")
    need(audit["raw_queries"] == audit["internal_top"] == 300
         and audit["post_nms"] == len(df_rows) == len(audit["candidates"]) == len(df_prob["query_ids"])
         and len(set(df_prob["query_ids"])) == len(df_rows)
         and all(0 <= q < 300 for q in df_prob["query_ids"])
         and df_prob["probability"]["rows"] == min(100, len(df_rows))
         and all(x["score"] >= y["score"] for x, y in zip(df_rows, df_rows[1:])),
         "D-FINE native post-NMS candidate order and probability-index closure")
    for rank, (row, detail, area) in enumerate(zip(df_rows, audit["candidates"], df_areas, strict=True), 1):
        need(rank == detail["native_rank"] and row["score"] == detail["score"] and int(area) == detail["mask_area"]
             and len(detail["box_xyxy"]) == 4 and all(isfinite(x) for x in detail["box_xyxy"]),
             "D-FINE native candidate mask/score/rank metadata")
    return rf_masks, rf_areas, df_masks, df_areas


def audit_hook(event, args):
    # Imaging members, annotation files and model/archive containers are not inputs to this census.
    if event == "open" and isinstance(args[0], (str, bytes)):
        name = str(args[0]).casefold()
        forbidden = ("annotation", "original_gt", "empty_rasters", ".jpg", ".jpeg", ".png", ".tif", ".zip", ".ckpt", ".pt")
        if any(token in name for token in forbidden):
            raise PermissionError("Scientific input scope is sealed image-only metadata and native RLE")
    if event in ("socket.connect", "socket.getaddrinfo"):
        raise PermissionError("Offline image-only replay")


def summarize(args):
    started = perf_counter()
    pinned(args.protocol, args.protocol_sha)
    sys.addaudithook(audit_hook)
    images, rf_acc, df_acc = read_inputs(args)
    counts = Counter()
    coverage = defaultdict(lambda: [set(), set()])
    accepted_iou, xor_sizes = [], []
    for item, rf_image, rf, query, df, audit, prob in images:
        rf_masks, rf_areas, df_masks, df_areas = admit_image(item, rf_image, rf, query, df, audit, prob)
        aidx = np.flatnonzero(rf_areas > 0)
        didx = np.flatnonzero(df_areas[:100] > 0)
        area_ok = np.flatnonzero((rf_areas[aidx] >= 32) & (rf_areas[aidx] < 1024)).tolist()
        a = [rf_masks[p] for p in aidx]
        b = [df_masks[p] for p in didx]
        prefix = Path(item["file_name"]).name.split("_", 1)[0]
        need(query["source_prefix"] == prefix and bool(prefix), "Original source prefix proxy")
        counts.update(images=1, anchor_mapped=len(rf_masks), anchor_empty=len(rf_masks)-len(a),
                      anchor_rank_gaps=rf["unmapped_slot1_count"], eligible_anchors=len(area_ok),
                      donor_post_nms=len(df_masks), donor_top100=min(100, len(df_masks)),
                      donor_top100_empty=min(100, len(df_masks))-len(b), donor_top100_nonempty=len(b))
        image_flags = {"all", "eligible"} if area_ok else {"all"}
        for row in match_donors(a, b, area_ok) if area_ok else []:
            verdict = row["decision"]
            counts[verdict] += 1
            if row["iou"] >= IOU_FLOOR:
                counts["best_ge_050"] += 1
                image_flags.add("best_ge_050")
            if verdict in ("accepted", "ambiguous"):
                counts["mutual_ge_050"] += 1
                image_flags.add("mutual_ge_050")
            if verdict != "accepted":
                continue
            image_flags.add("accepted")
            counts["accepted_without_duplicate_exception"] += not row["ambiguity_duplicates"]
            counts["accepted_using_duplicate_exception"] += row["ambiguity_duplicates"]
            ai, di = row["anchor"], row["donor"]
            intersection = int(mask.area(mask.merge([a[ai], b[di]], intersect=True)))
            addition = int(df_areas[didx[di]]) - intersection
            removal = int(rf_areas[aidx[ai]]) - intersection
            changed = addition + removal
            counts.update(anchor_only_pixels=removal, donor_only_pixels=addition, common_pixels=intersection,
                          xor_pixels=changed, accepted_both_signs=addition > 0 and removal > 0,
                          accepted_donor_add_only=addition > 0 and removal == 0,
                          accepted_donor_drop_only=removal > 0 and addition == 0,
                          accepted_equal=changed == 0, accepted_different=changed > 0)
            if changed:
                image_flags.add("accepted_different")
            if addition > 0 and removal > 0:
                image_flags.add("accepted_both_signs")
            accepted_iou.append(row["iou"])
            xor_sizes.append(changed)
        for category in image_flags:
            coverage[category][0].add(item["image_id"])
            coverage[category][1].add(prefix)
    need(counts["images"] == IMAGES and counts["eligible_anchors"] == rf_acc["raw_logit_rows"] == 42325
         and counts["anchor_mapped"] == rf_acc["native_mapped_rows"] == 67386
         and counts["anchor_empty"] == 15114 and counts["anchor_rank_gaps"] == 70500 - 67386
         and counts["donor_post_nms"] == df_acc["post_nms_rows"] == 184687
         and counts["donor_top100"] == df_acc["rows_at_100"] == 70500
         and counts["donor_top100_empty"] == df_acc["empty_at_100"] == 18953,
         "Reproduced closed RF-DETR and D-FINE anchor/donor censuses")
    phases = ("no_donor", "below_iou", "not_mutual", "ambiguous", "accepted")
    for category in phases:
        counts[category] += 0
    need(sum(counts[k] for k in phases) == counts["eligible_anchors"]
         and counts["accepted"] == len(accepted_iou) == counts["accepted_equal"] + counts["accepted_different"]
         and counts["xor_pixels"] == counts["anchor_only_pixels"] + counts["donor_only_pixels"],
         "Complete exclusive outcomes and native signed disagreement conservation")
    return dict(status=STATUS, phase="capture", census=dict(counts), source_proxy_coverage={
                    flag: dict(images=len(im), source_prefixes=len(src)) for flag, (im, src) in sorted(coverage.items())},
                q10_median_q90={"accepted_cross_model_iou": np.quantile(accepted_iou, [.1,.5,.9]).tolist(),
                                "accepted_binary_xor_native_pixels": np.quantile(xor_sizes, [.1,.5,.9]).tolist()}
                if accepted_iou else {}, worker_sha256=digest(Path(__file__).read_bytes()),
                protocol_sha256=args.protocol_sha, image_only_manifest_sha256=IMAGE_SHA,
                rf_capture_receipt_sha256=RF_RESULT_SHA, df_capture_readback_sha256=DF_READBACK_SHA,
                source_scope="historical HRSID DEV705", target_labels_opened=False, mask_scoring=False,
                exact_b5_endpoint_vote_census=False, native_binary_disagreement=True,
                detector_forwards=0, optimizer_updates=0, seconds=perf_counter()-started)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("rf", "df", "protocol", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--protocol-sha", required=True)
    args = parser.parse_args()
    need(not args.output.exists(), "Fresh aggregate receipt destination")
    result = summarize(args)
    args.output.mkdir()
    raw = (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    with (args.output / "RESULT_SAFE.json").open("xb") as stream:
        stream.write(raw)
    print(json.dumps(dict(status=STATUS, result_sha256=digest(raw))), flush=True)


if __name__ == "__main__":
    main()
