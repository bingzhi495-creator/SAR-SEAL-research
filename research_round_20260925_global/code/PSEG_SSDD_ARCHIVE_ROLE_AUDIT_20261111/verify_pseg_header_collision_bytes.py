"""Verify exact encoded-image overlap suggested by PSeg-SSDD RAR headers."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from zlib import crc32

from audit_pseg_archive_roles import COCO, audit, listed_members, require


def encoded_digest(archive, unrar, parts, expected):
    name = "\\".join(parts)
    result = subprocess.run([str(unrar), "p", "-inul", "--", str(archive), name],
                            check=True, capture_output=True, timeout=30,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    data = result.stdout
    require((len(data), f"{crc32(data):08x}") == expected,
            "Extracted JPEG stream disagrees with its archive header")
    require(data[:2] == b"\xff\xd8", "Expected an encoded JPEG stream")
    return sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("archive", "unrar", "role-report", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    before = args.archive.stat()
    reference = json.loads(args.role_report.read_bytes())
    require(before.st_size == reference["archive_bytes"] and
            reference["status"] == "PSEG_SSDD_CANDIDATE_RAR_ROLE_HEADERS_ACTUAL",
            "Expected closed archive header report")
    result = subprocess.run([str(args.unrar), "lt", "-c-", "--", str(args.archive)],
                            check=True, capture_output=True, timeout=30,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    members = list(listed_members(result.stdout.decode("utf-8", errors="replace")))
    require(audit(members) == reference["role_metadata"], "Archive role metadata changed")
    images = {
        role: [(parts, entry) for parts, entry in members
               if parts[1:-1] == COCO + (role,) and parts[-1].endswith(".jpg")]
        for role in ("train", "test")
    }
    candidate_keys = {entry for _, entry in images["train"]} & {
        entry for _, entry in images["test"]
    }
    extracted = {
        role: [encoded_digest(args.archive, args.unrar, parts, entry)
               for parts, entry in selected if entry in candidate_keys]
        for role, selected in images.items()
    }
    digest_counts = {role: Counter(rows) for role, rows in extracted.items()}
    exact = digest_counts["train"].keys() & digest_counts["test"].keys()
    after = args.archive.stat()
    require((after.st_size, after.st_mtime_ns, after.st_ino) ==
            (before.st_size, before.st_mtime_ns, before.st_ino), "Candidate changed during audit")
    out = dict(
        status="PSEG_SSDD_CANDIDATE_RAR_EXACT_TRAIN_TEST_JPEG_OVERLAP_ACTUAL",
        archive_sha256=reference["archive_sha256"],
        role_report_sha256=sha256(args.role_report.read_bytes()).hexdigest(),
        tested_encoded_members={role: len(rows) for role, rows in extracted.items()},
        cross_role_equal_sha256_groups=len(exact),
        cross_role_equal_jpeg_images={role: sum(count[k] for k in exact)
                                     for role, count in digest_counts.items()},
        TEST_image_pixels_decoded=False, TEST_annotation_opened=False, model_forward_calls=0,
        remote_official_whole_byte_identity_verified=False,
    )
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(out, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
