from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

root = Path(__file__).parent
source = root / "admission"
result_bytes = (source / "RESULT_SAFE.json").read_bytes()
result = json.loads(result_bytes)
private_bytes = (source / "TRAIN_ROSTER_PRIVATE.json").read_bytes()
roster = json.loads(private_bytes)
assert sha256(result_bytes).hexdigest() == "e40636ec2cfb4c42718673a553f069f5f4e2833e3e81264c9d44a43ece7b9927"
assert sha256(private_bytes).hexdigest() == result["admitted_roster_private_sha256"]
assert sha256((root / "PRELOCK.md").read_bytes()).hexdigest() == result["protocol_sha256"]
assert sha256((root / "ERRATUM.md").read_bytes()).hexdigest() == result["erratum_sha256"]
assert sha256((root / "admit_training_corrected.py").read_bytes()).hexdigest() == result["worker_sha256"]
admit = roster["admitted_train"]
deny = roster["excluded_train"]
groups = roster["groups_by_first_filename"]
assert len(admit) + len(deny) == 928 and not (admit.keys() & deny.keys())
assert set(sum(groups, [])) == set(admit)
assert sum(map(len, groups)) == len(admit)
assert len({admit[n]["rgb_size_prefixed_sha256"] for n in admit}) == len(groups)
assert set(Counter(map(len, groups)).items()) == {(int(k), n) for k, n in result["train_admitted_decoded_group_sizes"].items()}
assert all(len({admit[n]["rgb_size_prefixed_sha256"] for n in g}) == 1 for g in groups)
assert all(all(admit[n]["decoded_group_representative"] == g[0] and
               admit[n]["decoded_group_size"] == len(g) for n in g) for g in groups)
for members, stats in ((admit, result["admitted_train"]), (deny, result["removed_train"])):
    assert len(members) == stats["images"]
    assert sum(m["pseg_count"] for m in members.values()) == stats["pseg_hulls"]
    assert sum(m["spsd_count"] for m in members.values()) == stats["spsd_hulls"]
    assert sum(len(m["assigned_iou"]) for m in members.values()) == stats["assigned"]
    assert sum(sum(i >= .5 for i in m["assigned_iou"]) for m in members.values()) == stats["iou_ge_50"]
assert all(not (m["cross_test_decoded_overlap"] or m["cross_test_encoded_overlap"] or m["invalid_dual_reference"]) for m in admit.values())
assert all(m["cross_test_decoded_overlap"] or m["cross_test_encoded_overlap"] or m["invalid_dual_reference"] for m in deny.values())
same = [j for j in deny.values() if j["cross_test_encoded_overlap"]]
assert len(same) == result["train_cross_test_both_enc_dec"]
cross = len({m["rgb_size_prefixed_sha256"] for m in same})
assert cross == 10
print(json.dumps({"partition_and_hashes_verified": True, "exact_cross_test_groups": cross, "train": len(admit),
                  "train_decoded_groups": len(groups), "removed": len(deny),
                  "result_sha256": sha256(result_bytes).hexdigest(), "worker_sha256": result["worker_sha256"]}))

