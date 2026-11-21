"""Compare two commit-pinned GitHub tree listings with locally audited extraction bytes."""

from datetime import datetime, timezone
from hashlib import sha1, sha256
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCAL = HERE.parent / "PSEG_SSDD_OFFICIAL_TEST232_INPUT_WORK_20261121_01"
TREES = dict(images="2d9d40c73ee91cd5d27c82ee6d8e9018ec08cf9b", annotations="62b9fdf01fa39781ef237c6ee1d24390a64550eb")
REPO = "DonnieRaymond3/cs231n_ship_detection"
COMMIT = "30cdc0e7e7fa28b67c8a2f200f25ec7bc08dd97b"
AUDIT = "4d8ae373333a9f8d19b3252d86f8a44e5894427ef4c14cdc3fc58c20a46410f2"
INVENTORY = "ace3e722369657d9dd6a552eeac13244b3fa88772701ae0769e676816b46152e"
GT = "021d52858ffa3fda9a2d28ee21f1ca114800554edf08007704f6d06a412d50bf"


def git_id(kind, data):
    return sha1(kind.encode() + b" " + str(len(data)).encode() + b"\x00" + data).hexdigest()


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def table(role):
    obj = json.loads((HERE / ("GIT_TREE_" + role + "_PRIVATE.json")).read_bytes())
    require(obj["truncated"] is False and obj["sha"] == TREES[role], "Exact commit-linked directory")
    entries = obj["tree"]
    require(all(x["type"] == "blob" and x["mode"] in ("100644","100755") and "/" not in x["path"] for x in entries),
            "Direct ordinary files")
    tree = b"".join(row["mode"].encode() + b" " + row["path"].encode() + b"\0" + bytes.fromhex(row["sha"])
                    for row in sorted(entries, key=lambda x:x["path"].encode()))
    require(git_id("tree", tree) == TREES[role], "Cryptographic reconstruction of entire Git tree")
    return {x["path"]: x for x in entries}


def main():
    names = ["%06d.jpg" % i for i in range(1,1161) if i % 10 in (1,9)]
    local = LOCAL / "PRIVATE_INPUTS"
    audit = (LOCAL / "LOCAL_SAFE.json").read_bytes()
    require(sha256(audit).hexdigest() == AUDIT and json.loads(audit)["image_inventory_sha256"] == INVENTORY, "Original local extraction")
    records, anns = table("images"), table("annotations")
    require(set(records) == set(names) and set(p.name for p in local.iterdir()) == set(names + ["test.json"]), "Exact TEST members")
    manifest, matches, size_matches = sha256(), 0, 0
    for name in names:
        data = (local / name).read_bytes()
        record = records[name]
        manifest.update(("%s\t%d\t%s\n" % (name, len(data), sha256(data).hexdigest())).encode())
        size_matches += len(data) == record["size"]
        matches += len(data) == record["size"] and git_id("blob", data) == record["sha"]
    require(manifest.hexdigest() == INVENTORY, "Every local image still matches the independently verified inventory")
    raw = (local / "test.json").read_bytes()
    require(sha256(raw).hexdigest() == GT, "Original local TEST annotation exact bytes")
    gt_git = git_id("blob", raw)
    gt_matches = len(raw) == anns["test.json"]["size"] and gt_git == anns["test.json"]["sha"]
    total = matches + int(gt_matches)
    result = dict(status="PSEG_SSDD_TEST232_THIRD_PARTY_MIRROR_GIT_BLOB_COMPARISON_COMPLETE",
                  observed_utc=datetime.now(timezone.utc).isoformat(), repository=REPO, commit=COMMIT,
                  mirror_root_tree="8a1f1dcd25a8b4162ef3b0cceb4e545aa3373418", commit_linked_directory_tree_ids=TREES,
                  mirror_data_path="SSDD/BBox_RBox_PSeg_SSDD/coco_style",
                  initial_mirror_missing_SSDD_root=dict(repository="sohrabhassibi/CS231N_finalproj",commit="0da6a2c152f4480faa0961893ffa9feabee9e60a"),
                  method="Git SHA-1 blob identifier over the unmodified local bytes with the canonical type-and-length header, plus member byte length",
                  local_audit_sha256=AUDIT, original_images_inventory_sha256=INVENTORY, original_test_json_sha256=GT,
                  original_test_json_bytes=len(raw), original_test_json_git_blob_id=gt_git,
                  image_names_equal=True, image_size_matches=size_matches, image_blob_and_size_matches=matches, image_total=len(names),
                  annotation_blob_and_size_matches=int(gt_matches), annotation_total=1, member_matches=total, member_total=233,
                  all_members_match=(total==233), raw_mirror_payload_bytes_downloaded=0, author_hosted_member_bytes_verified=0,
                  whole_archive_authenticated=False, mirror_is_third_party=True, model_forwards=0, AP_computed=False)
    with (HERE / "MIRROR_CORROBORATION_SAFE.json").open("x", encoding="utf8") as out:
        json.dump(result, out, sort_keys=True, indent=2)
        out.write("\n")
    print("images %d / 232, annotations %d / 1, members %d / 233" % (matches, gt_matches, total))


if __name__ == "__main__":
    main()
