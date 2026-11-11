"""Verify the frozen FIT/DEV role of exact encoded-image duplicates."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path, PureWindowsPath
import subprocess

from audit_pseg_archive_roles import COCO, listed_members, require
from verify_pseg_header_collision_bytes import encoded_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("archive", "unrar", "dev-manifest", "fit-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--dev-manifest-sha256", required=True)
    args = parser.parse_args()
    raw_manifest = args.dev_manifest.read_bytes()
    require(sha256(raw_manifest).hexdigest() == args.dev_manifest_sha256, "Wrong frozen DEV manifest")
    dev = {PureWindowsPath(row["image_ref"]["path"]).name: row["image_ref"]["sha256"]
           for row in json.loads(raw_manifest)["items"]}
    fit = {path.name for path in args.fit_root.iterdir() if path.suffix.lower() == ".jpg"}
    listing = subprocess.run([str(args.unrar), "lt", "-c-", "--", str(args.archive)],
                             check=True, capture_output=True, timeout=30,
                             creationflags=subprocess.CREATE_NO_WINDOW)
    members = list(listed_members(listing.stdout.decode("utf-8", errors="replace")))
    roles = {role: {parts: entry for parts, entry in members
                    if parts[1:-1] == COCO + (role,) and parts[-1].endswith(".jpg")}
             for role in ("train", "test")}
    require(len(dev) == 187 and len(fit) == 741 and not dev.keys() & fit and
            dev.keys() | fit == {parts[-1] for parts in roles["train"]},
            "Frozen FIT and DEV must partition the candidate's training images")
    candidate_keys = set(roles["train"].values()) & set(roles["test"].values())
    train_groups = defaultdict(Counter)
    for parts, entry in roles["train"].items():
        if entry not in candidate_keys:
            continue
        name = parts[-1]
        digest = encoded_digest(args.archive, args.unrar, parts, entry)
        role = "DEV187" if name in dev else "FIT741"
        expected = dev[name] if role == "DEV187" else sha256((args.fit_root / name).read_bytes()).hexdigest()
        require(digest == expected, "Candidate JPEG differs from the assigned frozen population")
        train_groups[digest][role] += 1
    test_groups = Counter(encoded_digest(args.archive, args.unrar, parts, entry)
                          for parts, entry in roles["test"].items() if entry in candidate_keys)
    shared = train_groups.keys() & test_groups.keys()
    matched_test, matched_train = Counter(), Counter()
    for key in shared:
        group = train_groups[key]
        matched_test["+".join(sorted(group))] += test_groups[key]
        matched_train.update(group)
    result = dict(status="PSEG_SSDD_TEST_ENCODED_DUPLICATES_TRAIN_ROLE_ATTRIBUTION_ACTUAL",
                  tested_train_members_verified_against_roles=dict(matched_train),
                  test_images_by_matching_training_role=dict(matched_test),
                  exact_test_images=sum(test_groups[key] for key in shared),
                  official_remote_whole_archive_parity_verified=False,
                  test_annotations_opened=False, test_pixels_decoded=False, model_forward_calls=0)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
