"""Rebuild the original HRSID DEV705 evaluation inputs from annotations/train2017.json and the author PNG ZIP."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from types import ModuleType
from zipfile import ZipFile


SPLIT_SALT = "public-dev-20260925|"
AUDITOR_SHA = "eef43d0eac869191dac6a30be3a452fe6160de389b81c9f2f4cb645ef7d31389"
OUTPUTS = {
    "DEV_IMAGE_ONLY_MANIFEST_PRIVATE.json": "39a40d089d4a54448fe6efca8ea5cc98ad1eb7c0587815f3dd9a3b097173e78f",
    "DEV_ORIGINAL_GT_PRIVATE.json": "393b3da233991dc5fafc98b4ddeaa341b042d9571434ef8ce53c3f2fdada5628",
    "DEV_EMPTY_RASTERS_PRIVATE.json": "f3f2d644b2e0ce475b4a46b382ec7ec51f2efa4609c8f85b4f8f552ff57d5889",
}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def roles_from_train(images, fit):
    groups = {row["id"]: fit.image_stem(row["file_name"]).split("_", 1)[0] for row in images}
    require(len(groups) == len(images) == 3642, "Expected all 3642 original TRAIN image identities")
    prefixes = sorted(set(groups.values()), key=lambda p: (sha256((SPLIT_SALT + p).encode()).digest(), p))
    require(len(prefixes) == 137, "Original TRAIN prefix census")
    dev = set(prefixes[:(len(prefixes) + 4) // 5])
    roles = [dict(image_id=row["id"], file_name=row["file_name"], group=groups[row["id"]],
                  split="dev" if groups[row["id"]] in dev else "fit") for row in images]
    require(Counter(r["split"] for r in roles) == {"fit":2937, "dev":705}, "Predeclared FIT/DEV group split")
    return roles


def prepare(args):
    source = args.code / "HRSID_OFFICIAL_PNG_DEV705_AUDIT_20261109/hrsid_png_dev_audit.py"
    raw = source.read_bytes()
    require(sha256(raw).hexdigest() == AUDITOR_SHA, "Published DEV input auditor differs")
    audit = ModuleType("_dev705_audit")
    audit.__file__ = str(source)
    exec(compile(raw, str(source), "exec"), audit.__dict__)
    audit.FIT_SOURCE = args.code / "HRSID_OFFICIAL_PNG_FIT_AUDIT_20261109/hrsid_png_fit_audit.py"
    fit = audit.fit_tools()
    runtime = fit.runtime_versions()
    require(runtime == audit.RUNTIME, "Use the frozen HRSID PNG runtime, found " + str(runtime))
    require(not args.output.exists() and args.output.parent.is_dir(), "Use a new output directory under an existing parent")
    identity = fit.archive_identity(args.archive.stat())
    require(args.archive.stat().st_size == fit.ARCHIVE_BYTES, "Official author ZIP size")
    with args.archive.open("rb") as stream:
        digest = sha256()
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
        require(digest.hexdigest() == fit.ARCHIVE_SHA, "Official author ZIP identity")
        stream.seek(fit.CENTRAL_OFFSET)
        require(sha256(stream.read(fit.CENTRAL_BYTES)).hexdigest() == fit.CENTRAL_SHA, "Original ZIP directory")
        with ZipFile(stream) as archive:
            data = fit.checked_member(archive, fit.TRAIN_MEMBER, fit.TRAIN_LIMIT)
            require(sha256(data).hexdigest() == fit.TRAIN_SHA, "Original TRAIN annotation bytes")
            roles = roles_from_train(json.loads(data)["images"], fit)
            # The historical audit supplied precomputed FIT fingerprints. Recreate
            # them from author pixels here so the overlap result is independently usable.
            fit_keys = {"encoded_sha256":set(), "decoded_rgb_sha256":set()}
            for row in roles:
                if row["split"] == "fit":
                    name = "images/" + fit.image_stem(row["file_name"]) + ".png"
                    png = fit.checked_member(archive, name, fit.PNG_LIMIT)
                    fit_keys["encoded_sha256"].add(sha256(png).hexdigest())
                    fit_keys["decoded_rgb_sha256"].add(fit.decode_rgb(png))
            require(all(len(s) == 2937 for s in fit_keys.values()), "Distinct FIT image fingerprints")
            stats, images, gt, findings = audit.audit_dev(archive, roles, train_sha=fit.TRAIN_SHA, fit_fingerprints=fit_keys)
    require(fit.archive_identity(args.archive.stat()) == identity, "Original archive changed during intake")
    require(stats["dev_images_matching_fit_fingerprints"] == {"encoded_sha256":0, "decoded_rgb_sha256":0} and
            all(stat["unique_fingerprints"] == 705 for stat in stats["dev_duplicate_counts"].values()) and
            (len(images), len(gt["annotations"]), len(findings)) == (705, 1782, 4), "DEV, FIT overlap or raster census")
    prefix_counts = Counter(row["group"] for row in roles if row["split"] == "dev")
    name_to_value = dict(zip(OUTPUTS, (images, gt, findings)))
    payloads = {name:(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                      + "\n").encode() for name, value in name_to_value.items()}
    actual = {name:sha256(data).hexdigest() for name, data in payloads.items()}
    require(actual == OUTPUTS, "Ordered original DEV manifest, annotations or raster findings differ")
    report = dict(stats, status="B5_HRSID_DEV705_INPUTS_REBUILT", fit_png_member_opened=True,
                  fit_images_redecoded=2937, images=705, annotations=1782, empty_rasters=4,
                  model_forward_calls=0, optimizer_updates=0, archive_sha256=fit.ARCHIVE_SHA,
                  central_directory_sha256=fit.CENTRAL_SHA, train_annotation_sha256=fit.TRAIN_SHA,
                  image_manifest_sha256=actual["DEV_IMAGE_ONLY_MANIFEST_PRIVATE.json"],
                  original_dev_gt_sha256=actual["DEV_ORIGINAL_GT_PRIVATE.json"],
                  empty_raster_findings_sha256=actual["DEV_EMPTY_RASTERS_PRIVATE.json"], artifacts_sha256=actual,
                  dev_images_by_source_prefix=dict(sorted(prefix_counts.items())), runtime=runtime,
                  ancestor_split_rule="First ceil(number of TRAIN prefixes / 5) ordered by SHA256(public-dev-20260925| + prefix), then prefix",
                  audit_source_sha256=AUDITOR_SHA, worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.mkdir()
    for name, payload in payloads.items():
        path = args.output / name
        with path.open("xb") as stream:
            stream.write(payload)
        require(sha256(path.read_bytes()).hexdigest() == actual[name], "Output readback differs: " + name)
    result_sha = audit.write_json(args.output / "RESULT_SAFE.json", report)
    print(json.dumps(dict(status=report["status"], images=len(images), annotations=len(gt["annotations"]),
                          source_prefixes=len(prefix_counts), result_sha256=result_sha)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--archive", required=True, type=Path, help="Unmodified author HRSID PNG ZIP")
    parser.add_argument("--output", required=True, type=Path, help="Fresh output directory")
    parser.add_argument("--code", type=Path, default=Path(__file__).resolve().parent.parent,
                        help="Published research_round_20260925_global/code")
    prepare(parser.parse_args())

