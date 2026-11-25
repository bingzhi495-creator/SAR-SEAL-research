"""Form an image-identity-disjoint development split using an accepted train-only roster."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re

ADMISSION = "e40636ec2cfb4c42718673a553f069f5f4e2833e3e81264c9d44a43ece7b9927"
READBACK = "f6be62a8e860fa5c135bcadef88d1d57317b9bb8a12ae32c7af13aab2140d841"
ROSTER = "f4e23e5eb7f58a881db48e4a89d2902a253aa945e7a92ab7d1674c8b02b3fb17"
SALT = "SPSD-PSEG-SOURCE-DUAL-20261125-V1|"
STATUS = "DUAL_REFERENCE_CLEAN_GROUP_SPLIT_COMPLETE"


def digest(data):
    return sha256(data).hexdigest()


def read_checked(path, expected):
    data = path.read_bytes()
    if digest(data) != expected:
        raise ValueError("Admitted file checksum: " + path.name)
    return data


def write_json(path, obj):
    data = (json.dumps(obj, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(data)
    return digest(data)


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("admission", "readback", "roster", "protocol", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    read_checked(args.admission, ADMISSION)
    accepted = json.loads(read_checked(args.readback, READBACK))
    if accepted["result_sha256"] != ADMISSION or accepted["partition_and_hashes_verified"] is not True:
        raise ValueError("Independent source roster readback")
    roster = json.loads(read_checked(args.roster, ROSTER))["admitted_train"]
    by_image = defaultdict(list)
    for name, item in roster.items():
        group = item["rgb_size_prefixed_sha256"]
        if (not re.fullmatch("[a-f0-9]{64}", group) or not re.fullmatch(r"\d{6}\.jpg", name)
                or name[-5] in "19" or item["cross_test_decoded_overlap"] or item["cross_test_encoded_overlap"]
                or item["invalid_dual_reference"] or not (0 < item["height"] and 0 < item["width"])):
            raise ValueError("All rows must be admitted author TRAIN JPEGs")
        by_image[group].append(name)
    if len(roster) != 915 or len(by_image) != 901:
        raise ValueError("Exact previously admitted image population and groups")
    ordered_groups = sorted(by_image, key=lambda key: (digest((SALT + key).encode("ascii")), key))
    partitions = {"source_gate": ordered_groups[:180], "fit": ordered_groups[180:]}
    items = {key: sorted(name for group in groups for name in by_image[group]) for key, groups in partitions.items()}
    if set(items["source_gate"]).intersection(items["fit"]) or sum(map(len, items.values())) != len(roster):
        raise ValueError("Unique and complete clean partition")
    args.out.mkdir()
    private_sha = write_json(args.out / "SOURCE_PARTITION_PRIVATE.json", dict(
        status=STATUS, parent_roster_sha256=ROSTER, salt=SALT, partition=items,
        decoded_groups={key: [sorted(by_image[group]) for group in groups] for key, groups in partitions.items()},
    ))
    stats = {key: dict(decoded_groups=len(groups), images=len(items[key]),
                       group_size_counts=dict(sorted(Counter(len(by_image[group]) for group in groups).items())))
             for key, groups in partitions.items()}
    data = dict(status=STATUS, clean_source_images=len(roster), clean_source_groups=len(by_image),
                existing_source_admission_sha256=ADMISSION, independently_verified_roster_sha256=ROSTER,
                private_partition_sha256=private_sha, protocol_sha256=digest(args.protocol.read_bytes()),
                worker_sha256=digest(Path(__file__).read_bytes()), partitions=stats,
                fit_gate_group_overlap=len(set(partitions["source_gate"]).intersection(partitions["fit"])),
                label_or_model_used_to_assign=False, historical_test_loaded=False, optimizer_updates=0)
    result_sha = write_json(args.out / "RESULT_SAFE.json", data)
    print(json.dumps({"status": STATUS, "result_sha256": result_sha}))


if __name__ == "__main__":
    main()
