"""Audit PSeg-SSDD split membership from a candidate RAR's headers only."""
import argparse
from collections import Counter
from hashlib import md5, sha256
import json
from pathlib import Path
from pathlib import PureWindowsPath
import re
import subprocess


EXPECTED_BYTES = 1043241197
MASTER = ("PSeg_SSDD", "voc_style", "JPEGImages")
VOC = MASTER[:2]
COCO = ("BBox_RBox_PSeg_SSDD", "coco_style", "images")
TEST = {"test": 232, "test_inshore": 46, "test_offshore": 186}


def require(ok, detail):
    if not ok:
        raise ValueError(detail)


def listed_members(text):
    fields = {}
    for line in text.splitlines() + ["Name:"]:
        key, sep, value = line.strip().partition(":")
        if not sep:
            continue
        if key == "Name":
            if fields.get("Type") == "File" and {"Name", "Size", "CRC32"} <= fields.keys():
                parts = PureWindowsPath(fields["Name"]).parts
                require(re.fullmatch(r"[0-9A-Fa-f]{8}", fields["CRC32"]), "Unexpected file CRC")
                yield parts, (int(fields["Size"]), fields["CRC32"].lower())
            fields = {}
        fields[key] = value.strip()


def index_images(records, directory):
    values = {}
    for parts, record in records:
        if parts[1:-1] != directory or not parts[-1].lower().endswith(".jpg"):
            continue
        name = parts[-1]
        require(re.fullmatch(r"[0-9]{6}\.jpg", name) and name not in values,
                "Unexpected logical image name or duplicate archive member")
        values[name] = record
    return values


def archive_digests(path):
    hashes = (sha256(), md5(usedforsecurity=False))
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            for digest in hashes:
                digest.update(chunk)
    return [digest.hexdigest() for digest in hashes]


def audit(records):
    master = index_images(records, MASTER)
    roles = {r: index_images(records, COCO + (r,)) for r in ("train", *TEST)}
    train, test = roles["train"], roles["test"]
    require(len(master) == 1160 and len(train) == 928 and len(test) == 232,
            "Unexpected master and split census")
    require(not train.keys() & test.keys() and train.keys() | test.keys() == master.keys(),
            "Train/test names do not partition the master archive")
    require(all(n[-5] not in "19" for n in train) and all(n[-5] in "19" for n in test),
            "Archive filenames violate the author's suffix rule")
    require(roles["test_inshore"].keys().isdisjoint(roles["test_offshore"])
            and roles["test_inshore"].keys() | roles["test_offshore"].keys() == test.keys(),
            "Archive subgroups do not partition test")
    for role, members in roles.items():
        alternate = index_images(records, VOC + ("JPEGImages_" + role,))
        require(members == alternate and all(master[n] == f for n, f in members.items()),
                "COCO, VOC and master header identities differ")
        require(len(members) == (928 if role == "train" else TEST[role]),
                "Unexpected declared role size")
    labels = {}
    prefix = ("BBox_RBox_PSeg_SSDD", "coco_style", "annotations")
    for parts, entry in records:
        if parts[1:-1] == prefix and parts[-1].endswith(".json"):
            require(parts[-1] not in labels, "Duplicate annotation member")
            labels[parts[-1]] = {"bytes": entry[0], "crc32": entry[1]}
    require(set(labels) == {f"{r}.json" for r in roles}, "Unexpected annotation member set")
    role_headers = {key: Counter(entries.values()) for key, entries in roles.items()}
    shared = role_headers["train"].keys() & role_headers["test"].keys()
    return {
        "images": {"master": len(master), **{r: len(v) for r, v in roles.items()}},
        "unique_size_crc32_keys": {r: len(v) for r, v in role_headers.items()},
        "cross_role_size_crc32_collision": {
            "distinct_keys": len(shared),
            "train_images": sum(role_headers["train"][k] for k in shared),
            "test_images": sum(role_headers["test"][k] for k in shared),
        },
        "train_test_names_disjoint": True,
        "same_name_coco_voc_master_headers_equal": True,
        "test_suffixes": sorted({name[-5] for name in test}),
        "coco_annotation_member_headers": labels,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("archive", "unrar", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    stat = args.archive.stat()
    require(stat.st_size == EXPECTED_BYTES, "Candidate archive size differs")
    result = subprocess.run([str(args.unrar), "lt", "-c-", "--", str(args.archive)],
                            check=True, capture_output=True, timeout=30,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    members = list(listed_members(result.stdout.decode("utf-8", errors="replace")))
    data = audit(members)
    sha, md5sum = archive_digests(args.archive)
    after = args.archive.stat()
    require((after.st_size, after.st_mtime_ns, after.st_ino) ==
            (stat.st_size, stat.st_mtime_ns, stat.st_ino), "Candidate changed during audit")
    output = dict(status="PSEG_SSDD_CANDIDATE_RAR_ROLE_HEADERS_ACTUAL",
                  archive_bytes=stat.st_size, archive_sha256=sha, archive_md5=md5sum,
                  role_metadata=data, TEST_member_payload_opened=False,
                  model_forward_calls=0, official_remote_byte_identity_verified=False)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(output, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
