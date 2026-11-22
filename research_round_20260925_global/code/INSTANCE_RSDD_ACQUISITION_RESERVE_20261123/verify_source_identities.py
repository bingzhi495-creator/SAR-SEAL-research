"""Compare accepted original RGB identities with the exact source-fit and author-test populations."""
import argparse
import csv
from hashlib import sha256
import io
import json
from pathlib import Path
import re


ROSTER = "2614e567e9ef1b2c791c1b357b03adfe7302ece53e911c1963c40e623ad405fc"
CHECKPOINT = "011bd30ed056c871be1d61e14d71b3b2b258330a9d20cf2f164ca4a7c4f23c11"
FIT_MANIFEST = "a7e4d2a7ff9e8ff238adbf94a1d0397553d8746e2558c316db4e417f86f8fec9"
FIT_INDEX = "e95cf26cc60a2c6fb651e907541380a256f2df7b1e153244a6c0f803b0b19f8f"
COHORT = "70e9a0e8fb30f5ba70b48b9067c1b95d07bfec23d6b248d92c5983727acceb6e"
TEST_MANIFEST = "16a546e96d6e76d3afbfbe4f380b182505494818b35f1c8bcc960e6d4d1b2184"
HEAD_FIT = "114ac529e2196fa4f60d8fe56e07381fb6814e0fdb09e7d5615626fd74a85f64"
PAIR = "ec6c36973a763344d0f063727e7e07c697a132573e6235eaa8525482e8d8da9f"
DECODE_AUDIT = "6f073cbf6dcf3ad08d68c405589323e321a04f441665e479d2cdd6505f48e6ae"
CAPTURE_AUDIT = "543fd840d917b81d1c11b16b20ea8aeb38c0ae0b4ab0d1db2d62d8f06412447c"


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def pin(path, expected):
    raw = path.read_bytes()
    need(sha256(raw).hexdigest() == expected, "Metadata fingerprint mismatch for " + path.name)
    return raw


def read(path, expected):
    return json.loads(pin(path, expected))


def sources(root, public_round):
    restart = root / "HRSID_MASK_COMMON_FIT_E24_5202_CENTER_RESTART_PRIVATE_20261110_01"
    receipts = restart / "INDEPENDENT_ACCEPTANCE_PRIVATE"
    pair = read(receipts / "PAIR_READY_PRIVATE.json", PAIR)
    arm = read(receipts / "CENTER_ARM_ACCEPTED_PRIVATE.json", pair["center_arm_receipt_sha256"])
    need(pair["center_checkpoint_sha256"] == arm["checkpoint_sha256"] == CHECKPOINT and
         arm["original_os_exit"] == 0 and arm["original_os_wait"] and arm["phase"]["seed"] == 5202 and
         arm["phase"]["mask_resize_mode"] == "center" and arm["completed_epochs"] == 24 and
         pair["fit_index_sha256"] == arm["common_support"]["fit_index_sha256"] == FIT_INDEX,
         "Accepted source center5202 checkpoint, fit index and closed training")
    layout = root / "HRSID_FIT_NATIVE_COCO_LAYOUT_ACTUAL_PRIVATE_20261109_01"
    manifest = read(layout / "NATIVE_FIT_LAYOUT_MANIFEST_PRIVATE.json", FIT_MANIFEST)
    pool = {r["image_id"]:r for r in manifest["images"]}
    need(len(pool) == len(manifest["images"]) == 2937 and manifest["native_index_sha256"] == FIT_INDEX,
         "Complete detector fit image roster")
    visit_sha = arm["visit_ledger_sha256"]
    visits = pin(restart/"center/NATIVE_ACTUAL_PRIVATE/FIT_VISITS_PRIVATE.jsonl", visit_sha).splitlines()
    need(len(visits) == 24, "All 24 source training epochs")
    for epoch, line in enumerate(visits, 1):
        row = json.loads(line)
        need(row["epoch"] == epoch and len(row["visits"]) == 2940 and
             {v[0] for v in row["visits"]} == pool.keys(), "Exact source training population in each epoch")
    decode = read(public_round/"results/HRSID_FIT2937_DGX_TRANSPORT_AND_STOCK_DECODE_ACTUAL_SAFE_20261111.json",
                  DECODE_AUDIT)["independent_readback"]
    need(decode["manifest_sha256"] == FIT_MANIFEST and decode["native_index_sha256"] == FIT_INDEX and
         decode["decoded_rgb_sha256_match"] and decode["image_size"] == [800,800] and decode["images"] == len(pool),
         "Complete independently verified native detector-source RGB population")
    cohort = read(root/"HRSID_FIT_NATIVE_QUERY_RETENTION_ACTUAL_PRIVATE_20261109_01/COHORT_PRIVATE.json", COHORT)
    core = [r for r in cohort["images"] if r["core"]]
    need(len(core) == len({r["image_id"] for r in core}) == len({r["group"] for r in core}) == 109 and
         all(all(r[k] == pool[r["image_id"]][k] for k in ("encoded_sha256", "native_basename", "decoded_rgb_sha256"))
             for r in core), "Exact FIT109 input identity join to detector training")
    head_dir = root/"RFDETR_CENTER5202_FIT109_POINT_MATCHED_ACTUAL_20261120_01"
    fit = read(head_dir/"RESULT_SAFE.json", HEAD_FIT)
    closure = read(head_dir/"INDEPENDENT_SAFE.json", "476b0f3fa748d1d3fd2393bdeb1af2d83e1532995d58b12625d57222e3f2638a")
    pre = read(head_dir/"PRELABEL_SAFE.json", fit["prelabel_sha256"])
    captured = read(public_round/"results/RFDETR_CENTER5202_FIT109_CAPTURE_PUBLIC_SAFE_20261120.json", CAPTURE_AUDIT)
    record, = [h for h in fit["all_initializations"] if h["seed"] == 5217 and h["fold"] is None]
    need(fit["detector_checkpoint_sha256"] == captured["checkpoint_sha256"] == CHECKPOINT and
         captured["independent_result_sha256"] == pre["accepted_capture_sha256"] and
         fit["images"] == captured["images"] == 109 and fit["FIT"] and not fit["DEV"] and not fit["TEST"] and
         closure["fit_result_sha256"] == HEAD_FIT and closure["run_natural_exit"] == 0 and
         record["endpoint"] == "all_fit" and record["source_groups"] == 92 and record["steps"] == 200 and
         record["weights_sha256"] == "3a60f658149f4262451cd9118902bbe52dc83345d9fbcd23973a11dc473fcdf1",
         "Independent source-only occupancy refit and its full FIT109 image capture")
    detail = dict(source_checkpoint_sha256=CHECKPOINT, detector_source_rgb_manifest_sha256=FIT_MANIFEST,
                  detector_fit_index_sha256=FIT_INDEX, detector_source_visits_sha256=visit_sha,
                  detector_epochs=24, detector_visits=24*2940, source_rgb_accepted_readback_sha256=DECODE_AUDIT,
                  head_source_roster_sha256=COHORT, head_fit_result_sha256=HEAD_FIT,
                  head_weight_bundle_sha256=record["weights_sha256"], supervised_head_groups=record["source_groups"],
                  source_original_hw=[800,800])
    return [r["decoded_rgb_sha256"] for r in pool.values()], detail


def verify(a):
    pool, detail = sources(a.private_root, a.public_round)
    roster = list(csv.DictReader(io.StringIO(pin(a.roster, ROSTER).decode())))
    terminal = read(a.intake_terminal, a.intake_terminal_sha)
    intake = read(a.intake/"RESULT_SAFE.json", a.intake_result_sha)
    need(terminal["actual_exit"] == 0 and terminal["output_footer_verified"] is True and
         terminal["worker_sha256"] == intake["worker_sha256"] and terminal["result_sha256"] == a.intake_result_sha,
         "Original observer verified the intake worker and exact result")
    need(intake["status"] == "INSTANCE_RSDD_RESERVE74_IMAGE_INTAKE_COMPLETE_V1" and
         intake["exact_qualified_rf_rgb"] == intake["images"] == 74 and intake["roster_sha256"] == ROSTER and
         intake["historical_test_manifest_sha256"] == TEST_MANIFEST, "Accepted reserve decoded RGB roster")
    new = read(a.intake/"IMAGE_MANIFEST_PRIVATE.json", intake["image_manifest_sha256"])
    old = read(a.historical_test_manifest, TEST_MANIFEST)
    rows = new["items"]
    need(len(roster) == len(rows) == 74 and new["expected_ids"] == [int(r["author_train_image_id"]) for r in roster],
         "Full intended reserve author-ID population")
    for x, r in zip(rows,roster):
        need((x["image_id"],x["source_image_name"],x["sensor"],x["acquisition_group"]) ==
             (int(r["author_train_image_id"]),r["file_name"],r["sensor"],int(r["paper_scene"])), "Reserve identity")
    oldrows = old["items"]
    groups = {r["acquisition_group"] for r in rows}
    groups_before = {r["acquisition_group"] for r in oldrows}
    need(len(groups) == 13 and len(groups_before) == 112 and len(oldrows) == 740 and
         all(r["original_hw"] == [512,512] for r in rows+oldrows), "Source-scene and native canvas counts")
    rgb = [r["rgb_sha256"] for r in rows]
    oldrgb = {r["rgb_sha256"] for r in oldrows}
    fitrgb = set(pool)
    need(all(re.fullmatch("[0-9a-f]{64}", d) for d in rgb+list(oldrgb)+pool), "Valid RGB identity digests")
    prior_hit = sum(d in oldrgb for d in rgb)
    fit_hit = sum(d in fitrgb for d in rgb)
    within = len(rgb)-len(set(rgb))
    need(intake["historical_test_rgb_matching_reserve_images"] == prior_hit and
         intake["within_reserve_duplicate_extra_images"] == within, "Reproduced intake duplicate comparison")
    confirmed = sorted(set(a.confirmed_common_hrsid_scene))
    need(set(confirmed) <= groups, "Identified fitting-source acquisition must belong to the locked reserve")
    passed = not (prior_hit or fit_hit or (groups & groups_before) or confirmed)
    result = dict(status="INSTANCE_RSDD_RESERVE74_SOURCE_IDENTITIES_VERIFIED_V1", passed=passed,
                  worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), reserve_roster_sha256=ROSTER,
                  reserve_manifest_sha256=intake["image_manifest_sha256"], reserve_intake_result_sha256=a.intake_result_sha,
                  reserve_intake_terminal_sha256=a.intake_terminal_sha, historical_test_manifest_sha256=TEST_MANIFEST,
                  source_images=len(pool), source_fit109=109, historical_test_images=len(oldrows), reserve_images=len(rows),
                  reserve_parents=len(groups), reserve_rgb_matches_hrsid_fit=fit_hit,
                  reserve_rgb_matches_historical_test=prior_hit, reserve_parent_matches_historical_test=len(groups&groups_before),
                  reserve_within_duplicate_images=within, confirmed_common_hrsid_fitting_acquisitions=len(confirmed),
                  confirmed_common_hrsid_fitting_reserve_scenes=confirmed,
                  geographic_acquisition_disjointness_verified=False, hrsid_physical_cross_scene_disjointness_verified=False,
                  hrsid_physical_acquisition_crosswalk_available=False,
                  physical_scope="Gaofen-3 absent from HRSID sensor list. HRSID prefixes do not identify TerraSAR-X products across releases. Zero confirmed links means none documented in available metadata.",
                  pixel_identity_scope="Native whole RGB frames. Different crops or resampled imagery can depict common geography.",
                  label_records_read=0, model_forwards=0, optimizer_updates=0, **detail)
    a.output.mkdir(parents=True, exist_ok=False)
    raw = (json.dumps(result,sort_keys=True,indent=2)+"\n").encode()
    (a.output/"RESULT_SAFE.json").write_bytes(raw)
    print(json.dumps(dict(status=result["status"],result_sha256=sha256(raw).hexdigest())),flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    for name in ("private-root","public-round","roster","historical-test-manifest","intake","intake-terminal","output"):
        p.add_argument("--"+name,type=Path,required=True)
    for name in ("intake-result-sha","intake-terminal-sha"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--confirmed-common-hrsid-scene",type=int,action="append",default=[])
    verify(p.parse_args())
