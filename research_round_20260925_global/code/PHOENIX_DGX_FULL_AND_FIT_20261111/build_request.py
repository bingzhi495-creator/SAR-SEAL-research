"""Materialize a frozen two-image Phoenix request from closed HRSID FIT114 inputs."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
from zipfile import ZipFile


STATUS = "PHOENIX_HRSID_FIT114_FIXED_CANDIDATE_REQUEST_V1"
SALT = "PHOENIX-FIT-PILOT-v1"
P192 = "HRSID_FIT_NATIVE_QUERY_RETENTION_20261109/fit_query_export.py"
P192_SHA = "6a2d7e38d69b82ef3df790a5c359c408b1e74778c39672b47a62c4dc86cabbf5"
COHORT_SHA = "70e9a0e8fb30f5ba70b48b9067c1b95d07bfec23d6b248d92c5983727acceb6e"
QUERY_SHA = "23c4ab97a3459f77936a95d0d09965db339e7af54e2a4a1b9d0d9f787f8690f7"
BUNDLE_SHA = "af12a4e86ea36e9d7fe6ac47b17c684a4812baba6aa9be019f889f833e0f577c"
ACCEPT_SHA = "ba1d8a0df0678f7063b73998ddb487941160898a8758d6c02ea6469595768968"
TERM_SHA = "329f4fb71bcaa5fc0d11d684c52d28d0ff605be837d43eb95ffc138a8c5c969c"
REPORT_SHA = "5d6b0b3da74c0c3af1d13da0881f88c26bbaa61a58c983d5fbb36155b3ea2672"
CHECKPOINT_SHA = "7824943afc7856cceb247b281f0c2d71dc087d96906d0aed0f28b6f0e34840e1"
PROTOCOL = "fd55294d80a44bc50747bb4bb5555a6c74f4967f"
WIDTH, SIZE = 164, 800
MIB = 2**20


def need(condition, message):
    if not condition:
        raise ValueError(message)


def choose(rows):
    candidates = [row for row in rows if row["core"] is True and row["stress"] is False]
    ordered = sorted(candidates, key=lambda r: (sha256(SALT.encode() + b"\0" + bytes.fromhex(r["encoded_sha256"])).digest(),
                                                 r["group"], r["image_id"]))
    chosen, groups, contents = [], set(), set()
    for row in ordered:
        group, digest = row["group"], row["encoded_sha256"]
        if group in groups or digest in contents:
            continue
        chosen.append(row)
        groups.add(group)
        contents.add(digest)
        if len(chosen) == 2:
            return chosen
    raise ValueError("Two distinct core FIT114 source groups and PNG byte objects required")


def candidate_rows(block):
    need(set(block) == {"image_id", "native_ids", "native_ranks", "ship_scores", "all_rles"},
         "Exact P192 native candidate block required")
    queries, ranks, scores, masks = (block[key] for key in ("native_ids", "native_ranks", "ship_scores", "all_rles"))
    need(len(queries) == len(ranks) and len(queries) == len(set(queries))
         and ranks == sorted(set(ranks)) and all(type(rank) is int and 1 <= rank <= 100 for rank in ranks)
         and all(type(query) is int and 0 <= query < WIDTH for query in queries)
         and len(scores) == len(masks) == WIDTH and all(type(score) is float and 0. <= score <= 1. for score in scores),
         "Exact native ship rank, query and score required")
    result = []
    for rank, query in zip(ranks, queries, strict=True):
        rle = masks[query]
        need(set(rle) == {"size", "counts"} and rle["size"] == [SIZE, SIZE]
             and isinstance(rle["counts"], str) and rle["counts"].isascii() and bool(rle["counts"]),
             "Exact original-grid native compressed RLE required")
        result.append(dict(native_rank=rank, query_id=query, score=scores[query], category_id=1, segmentation=rle))
    return result


def read_original(code_root, root):
    source = code_root / P192
    raw = source.read_bytes()
    need(sha256(raw).hexdigest() == P192_SHA, "Exact ordinary original P192 source required")
    p192 = ModuleType("_phoenix_exact_p192")
    p192.__file__ = str(source)
    exec(compile(raw, str(source), "exec"), p192.__dict__)
    read = p192.pinned
    accept = json.loads(read(root / "ROOT_CLOSED_EXPORT_PRIVATE.json", ACCEPT_SHA, 8192))
    control = root / "EXPORT_CONTROL"
    terminal = json.loads(read(control / "TERMINAL_PRIVATE.json", TERM_SHA, 8192))
    need(accept["terminal"] == terminal and accept["resource_closed"] is True
         and accept["status"] == "ORIGINAL_GT_FREE_FIT_Q164_EXPORT_ACCEPTED"
         and terminal["original_OS_wait"] is True and type(terminal["exit"]) is int and terminal["exit"] == 0
         and terminal["guard"] is None and terminal["phase"] == "EXPORT_CONTROL",
         "Original P192 export must be independently accepted and closed")

    def read_ref(path, ref, digest, limit):
        need(Path(ref["path"]) == path and ref["sha256"] == digest, "Exact original P192 reference required")
        data = read(path, digest, limit)
        need(len(data) == ref["bytes"], "Exact original P192 reference byte count required")
        return data

    report_path = root / "EXPORT_ACTUAL_PRIVATE" / "EXPORT_SAFE.json"
    report = json.loads(read_ref(report_path, accept["report_ref"], REPORT_SHA, 8192))
    stdout = read_ref(control / "WORKER.stdout", terminal["stdout_ref"],
                      "6784151cb8ba7bcf27c0c009ed51ed3ef3dd0d286701985503856a5a3f3a95d1", 8192)
    read_ref(control / "WORKER.stderr", terminal["stderr_ref"],
             "0d57aa909d870648f717d1f1a385f7d51875ea5c0ef33c5440b5fde3e294585c", 8192)
    need(json.loads(stdout.splitlines()[-1]) == report
         and report["status"] == "HRSID_FIT_Q164_EXPORT_COMPLETE" and report["protocol"] == PROTOCOL
         and report["native_rows_bitwise_verified"] is True and report["images"] == 114
         and report["checkpoint_sha256"] == CHECKPOINT_SHA and report["cohort_sha256"] == COHORT_SHA
         and report["predictions_sha256"] == QUERY_SHA
         and all(report[key] is False for key in ("AP", "DEV", "GT_opened", "TEST")),
         "Original export report and stdout must agree on exact FIT-only identities")
    cohort = read(root / "COHORT_PRIVATE.json", COHORT_SHA, MIB)
    query_path = root / "EXPORT_ACTUAL_PRIVATE" / "QUERIES_PRIVATE.jsonl"
    queries = read_ref(query_path, accept["queries_ref"], QUERY_SHA, 2*MIB)
    return p192, cohort, [json.loads(line) for line in queries.splitlines()], report


def read_images(bundle, cohort, chosen):
    from PIL import Image

    with bundle.open("rb") as stream:
        before = os.fstat(stream.fileno())
        digest = sha256()
        for chunk in iter(lambda: stream.read(MIB), b""):
            digest.update(chunk)
        need(before.st_size == 151_320_868 and digest.hexdigest() == BUNDLE_SHA,
             "Exact closed original FIT114 archive required")
        stream.seek(0)
        with ZipFile(stream) as archive:
            need(archive.getinfo("COHORT_PRIVATE.json").file_size == len(cohort)
                 and archive.read("COHORT_PRIVATE.json") == cohort, "Exact original P192 cohort inside FIT114 archive")
            result = []
            for row in chosen:
                name = row["native_basename"]
                need(re.fullmatch(r"P\d+_[A-Za-z0-9_]+\.png", name) and name.split("_", 1)[0] == row["group"],
                     "Original FIT-only image basename required")
                entry = archive.getinfo("images/" + name)
                need(0 < entry.file_size == row["encoded_bytes"] <= 4*MIB, "Exact bounded FIT PNG length required")
                data = archive.read(entry)
                need(sha256(data).hexdigest() == row["encoded_sha256"], "Original FIT PNG bytes differ")
                with Image.open(BytesIO(data)) as image:
                    need(image.format == "PNG" and image.size == (SIZE, SIZE)
                         and sha256(image.convert("RGB").tobytes()).hexdigest() == row["decoded_rgb_sha256"],
                         "Original FIT PNG decoded RGB differs")
                result.append(data)
        def identity(meta):
            return meta.st_dev, meta.st_ino, meta.st_size, meta.st_mtime_ns
        need(identity(before) == identity(os.fstat(stream.fileno())) == identity(bundle.stat()),
             "Closed FIT archive changed during readback")
    return result


def build(code_root, original, bundle, output):
    for path in (code_root, original, bundle):
        need(path.is_absolute() and path.resolve(strict=True) == path, "Canonical immutable source paths required")
    need(output.is_absolute() and output.parent.resolve(strict=True) == output.parent
         and not output.exists() and not output.is_symlink(), "Canonical fresh output directory required")
    p192, cohort_raw, blocks, report = read_original(code_root, original)
    cohort = json.loads(cohort_raw)
    rows = cohort["images"]
    need(cohort["fit_only"] is True and cohort["protocol"] == PROTOCOL
         and len(rows) == len(blocks) == 114
         and [row["image_id"] for row in rows] == [block["image_id"] for block in blocks]
         and len({row["image_id"] for row in rows}) == 114
         and sum(len(block["native_ids"]) for block in blocks) == report["native_mapped_rows"] == 10572,
         "Exact P192 closed cohort, native row count and image order required")
    chosen = choose(rows)
    by_id = {block["image_id"]: block for block in blocks}
    images = [dict(image_id=row["image_id"], image_file=f"image_{i:03}.png",
                   encoded_sha256=row["encoded_sha256"], decoded_rgb_sha256=row["decoded_rgb_sha256"],
                   candidate_rows=candidate_rows(by_id[row["image_id"]]))
              for i, row in enumerate(chosen, 1)]
    pngs = read_images(bundle, cohort_raw, chosen)
    provenance = dict(selection_salt=SALT, fit_bundle_sha256=BUNDLE_SHA, cohort_sha256=COHORT_SHA,
                      predictions_sha256=QUERY_SHA, checkpoint_sha256=CHECKPOINT_SHA,
                      original_export_protocol=PROTOCOL, original_export_terminal_sha256=TERM_SHA,
                      original_export_acceptance_sha256=ACCEPT_SHA)
    request = dict(status=STATUS, provenance=provenance, images=images)
    aggregate = dict(status="PHOENIX_HRSID_FIT114_REQUEST_BUILT_V1", images=2, distinct_groups=2,
                     native_candidate_rows_per_image=[len(row["candidate_rows"]) for row in images],
                     request_sha256=sha256(p192.encoded(request)).hexdigest(), provenance=provenance,
                     GT_opened=False, DEV=False, TEST=False, training=False, model_forward_calls=0)
    with TemporaryDirectory(prefix=".phoenix_request_", dir=output.parent) as temp:
        staging = Path(temp) / "request"
        staging.mkdir()
        for image, data in zip(images, pngs, strict=True):
            with (staging / image["image_file"]).open("xb") as stream:
                stream.write(data)
        p192.write(staging / "REQUEST_PRIVATE.json", request)
        p192.write(staging / "BUILD_SAFE.json", aggregate)
        staging.rename(output)
    return aggregate


def deny_network(event, _):
    if event.startswith("socket."):
        raise PermissionError("Offline request builder required")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("code-root", "original", "bundle", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    sys.addaudithook(deny_network)
    print(json.dumps(build(args.code_root, args.original, args.bundle, args.output), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
