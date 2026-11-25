"""Check exact decoded native RGB overlap for the closed SRNet FIT and reserve cohorts."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

from PIL import Image, ImageOps, __version__ as pillow_version


ACCEPTANCE = "0e457818ff97eac25348c87dcc96994764d582a0dcb9bdc60ec148dcf91ee2b0"
CHECKPOINT = "8998d0f6c11082250a46af48ec0c5455fb7606de8b6bb6a00ad2956046ea95ea"
FIT_RESULT = "6f8551a15b28d0184e76199cec415b46bcb7a688eaff72640ba6347ccd9ea447"
FIT_TERMINAL = "4153fcf97ea9093ce0b657da0b18838262e202ac54f10d34dff793abeb071a91"
FIT_INDEX = "62966857c78cea1e3b060700c62a7b896942686c2e674d3eb09f516a92250bc1"
FIT_ARCHIVE = "1c8f9b75e79fbc02c983ccd283a70082fcd5ab0d6538c777e8b286dbfdb78100"
FIT_MANIFEST = "5bf021523d1ba3507a5ef7ba6a731da8699c37eb32d6b8ea962c52dcfbaafb47"
FIT_READBACK = "5e6aee82aaa4e7c1480d1140cf3447b5afbc1e11bec56212a42029764d50201e"
FIT_VISITS = "e6d3f01a21107fe2c0bfdbfa18a5924ed913113ba3a54674378c133e376c3aa6"
INTAKE_RESULT = "da9d92ea16e74f988d46d9855d26b530111b4c5e650fef513895c5d2e4441065"
INTAKE_TERMINAL = "b97ba047bf9422346f210b2a513be606cdc648cccdc0e8545affdd510f9ffaeb"
INTAKE_MANIFEST = "154d0d748b96df0d56ac249adcb8b6ae63275a09dfd7a0d7480839005e044338"
INTAKE_ROSTER = "2614e567e9ef1b2c791c1b357b03adfe7302ece53e911c1963c40e623ad405fc"
STATUS = "SRNET_FIT741_RSDD74_DECODED_RGB_EXPOSURE_COMPLETE"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path, expected):
    raw = path.read_bytes()
    require(digest(raw) == expected, "Unexpected bound input: " + path.name)
    return raw


def rgb_key(raw):
    with Image.open(BytesIO(raw)) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        return rgb.height, rgb.width, digest(rgb.tobytes())


def compare(source, target):
    src, dst = Counter(source), Counter(target)
    common = src.keys() & dst.keys()
    return dict(unique_source_rgb=len(src), unique_target_rgb=len(dst),
                matching_rgb_identities=len(common),
                source_images_matching=sum(src[k] for k in common),
                target_images_matching=sum(dst[k] for k in common),
                cross_pair_count=sum(src[k]*dst[k] for k in common))


def load_fit(a):
    acceptance = json.loads(pin(a.acceptance, ACCEPTANCE))
    report = json.loads(pin(a.training / "RESULT/RESULT_SAFE.json", FIT_RESULT))
    terminal = json.loads(pin(a.training / "ORIGINAL_WAIT_PRIVATE.json", FIT_TERMINAL))
    require(acceptance["status"] == "P354_SRNET_E24_ORIGINAL_CLOSURE_VERIFIED"
            and acceptance["checkpoint"]["sha256"] == report["checkpoint"]["sha256"] == CHECKPOINT
            and acceptance["training_report"]["sha256"] == FIT_RESULT
            and acceptance["original_terminal"]["sha256"] == FIT_TERMINAL
            and acceptance["outer_exit"] == acceptance["worker_exit"] == terminal["exit_code"] == 0
            and acceptance["active_after_wait"] == 0 and acceptance["original_os_wait"] is True
            and report["FIT"] is True and report["DEV"] is report["TEST"] is report["AP"] is False
            and (report["fit_images"], report["completed_epochs"], report["optimizer_updates"],
                 report["consumed_visits"]) == (741, 24, 4464, 17856)
            and (report["fit_index_sha256"], report["fit_archive_sha256"]) == (FIT_INDEX, FIT_ARCHIVE),
            "Original closed SRNet FIT741 E24 receipt and data identity required")
    readback = json.loads(pin(a.fit_stage / "READBACK_SAFE.json", FIT_READBACK))
    require(readback["status"] == "SSDD_FIT741_ONLY_LAYOUT_INDEPENDENT_BYTES_VERIFIED_V1"
            and readback["fit_archive_sha256"] == FIT_ARCHIVE and readback["fit_index_sha256"] == FIT_INDEX
            and readback["portable_manifest_sha256"] == FIT_MANIFEST,
            "Exact independently materialized training layout required")
    manifest_raw = pin(a.fit_manifest, FIT_MANIFEST)
    manifest = json.loads(manifest_raw)
    require(manifest["source_byte_refs"]["FIT_COCO_annotations"]["sha256"] == FIT_INDEX,
            "Training COCO index fingerprint must agree with portable image manifest")
    rows = manifest["rows"]
    ids = {r["image_id"] for r in rows}
    require(len(rows) == len(ids) == len({r["original_basename"] for r in rows}) == 741,
            "Exact 741 logical training IDs and unique staged filenames required")
    visits = pin(a.training / "RESULT/FIT_VISITS_PRIVATE.jsonl", FIT_VISITS).splitlines()
    require(report["visit_ledger"]["sha256"] == FIT_VISITS and len(visits) == 4464,
            "Exact closed FIT access ledger required")
    consumed = set()
    for i, line in enumerate(visits, 1):
        visit = json.loads(line)
        require(visit["optimizer_update"] == i and len(visit["consumed_ids"]) == 4
                and set(visit["scheduled_ids"] + visit["consumed_ids"] + visit["rejected_attempt_ids"]) <= ids,
                "Every training access must stay within the bound FIT list")
        consumed.update(visit["consumed_ids"])
    require(consumed == ids, "Every bound FIT image must occur in the original training ledger")

    archive = pin(a.fit_archive, FIT_ARCHIVE)
    layout = (a.fit_stage / "LAYOUT/train").resolve(strict=True)
    source = []
    encoded = set()
    with ZipFile(BytesIO(archive)) as z:
        require(z.read("manifest.json") == manifest_raw, "FIT manifest copy in the archive differs")
        for row in rows:
            basename = Path(row["original_basename"])
            require(len(basename.parts) == 1 and basename.suffix.lower() == ".jpg", "FIT filename scope")
            path = (layout / basename).resolve(strict=True)
            require(path.parent == layout, "FIT JPEG must resolve within original training layout")
            raw = z.read(row["archive_image_entry"])
            expected = row["original_image_ref"]
            require(len(raw) == expected["bytes"] and digest(raw) == expected["sha256"]
                    and path.read_bytes() == raw, "FIT archive and staged image bytes differ")
            encoded.add(expected["sha256"])
            key = rgb_key(raw)
            require(list(key[:2]) == row["original_shape"], "FIT native image dimensions differ")
            source.append(key)
    require(len(encoded) == readback["unique_encoded_images"] == 730, "Expected 730 distinct source JPEGs")
    return source, len(encoded)


def load_reserve(root):
    result = json.loads(pin(root / "intake/RESULT_SAFE.json", INTAKE_RESULT))
    terminal = json.loads(pin(root / "observed_intake/TERMINAL.json", INTAKE_TERMINAL))
    require(result["status"] == "INSTANCE_RSDD_RESERVE74_IMAGE_INTAKE_COMPLETE_V1"
            and result["image_manifest_sha256"] == INTAKE_MANIFEST and result["roster_sha256"] == INTAKE_ROSTER
            and result["images"] == result["exact_qualified_rf_rgb"] == 74
            and result["target_polygon_rows_opened"] == 0 and terminal["actual_exit"] == 0
            and terminal["output_footer_verified"] is True and terminal["result_sha256"] == INTAKE_RESULT
            and terminal["worker_sha256"] == result["worker_sha256"],
            "Exact naturally closed label-free reserve intake required")
    manifest = json.loads(pin(root / "intake/IMAGE_MANIFEST_PRIVATE.json", INTAKE_MANIFEST))
    rows = manifest["items"]
    require(len(rows) == len(set(manifest["expected_ids"])) == 74
            and [r["image_id"] for r in rows] == manifest["expected_ids"]
            and len({r["acquisition_group"] for r in rows}) == 13
            and Counter(r["sensor"] for r in rows) == Counter({"GF-3":42,"TerraSAR-X":32}),
            "Exact ordered reserve roster and sensor counts")
    image_dir = (root / "intake/images").resolve(strict=True)
    keys = []
    for row in rows:
        reference = row["image_ref"]
        name = Path(row["source_image_name"])
        require(len(name.parts) == 1 and name.suffix.lower() == ".jpg"
                and Path(reference["path"]) == Path("images") / name,
                "Reserve path must be the manifest image only")
        path = (image_dir / name).resolve(strict=True)
        require(path.parent == image_dir, "Reserve JPEG must resolve within original intake")
        raw = pin(path, reference["sha256"])
        key = rgb_key(raw)
        require(len(raw) == reference["bytes"] and list(key[:2]) == row["original_hw"] == [512,512]
                and key[2] == row["rgb_sha256"], "Reserve intake exact JPEG and native RGB parity")
        keys.append(key)
    require(len(set(keys)) == result["rgb_unique"] == 74, "Reserve within-set RGB tally changed")
    return keys, [r["sensor"] for r in rows]


def run(a):
    pin(a.protocol, a.protocol_sha256)
    fit, encoded = load_fit(a)
    reserve, sensors = load_reserve(a.intake)
    tally = compare(fit, reserve)
    fit_keys = set(fit)
    by_sensor = Counter(sensor for sensor, key in zip(sensors, reserve) if key in fit_keys)
    result = dict(status=STATUS, worker_sha256=digest(Path(__file__).read_bytes()),
                  protocol_sha256=a.protocol_sha256, source_checkpoint_sha256=CHECKPOINT,
                  source_acceptance_sha256=ACCEPTANCE, source_fit_result_sha256=FIT_RESULT,
                  source_fit_terminal_sha256=FIT_TERMINAL, source_fit_index_sha256=FIT_INDEX,
                  source_fit_archive_sha256=FIT_ARCHIVE, source_fit_portable_manifest_sha256=FIT_MANIFEST,
                  source_fit_layout_readback_sha256=FIT_READBACK, source_fit_visit_ledger_sha256=FIT_VISITS,
                  source_distinct_consumed_ids=len(fit), source_images=len(fit),
                  source_unique_encoded_images=encoded, source_final_epochs=24, source_consumed_visits=17856,
                  target_intake_result_sha256=INTAKE_RESULT, target_intake_terminal_sha256=INTAKE_TERMINAL,
                  target_image_manifest_sha256=INTAKE_MANIFEST, target_roster_sha256=INTAKE_ROSTER,
                  target_images=len(reserve), target_parent_groups=13,
                  target_intake_native_rgb_parity=len(reserve),
                  source_exact_jpeg_stage_archive_parity=len(fit),
                  no_exact_decoded_rgb_overlap=tally["target_images_matching"] == 0,
                  source_within_decoded_duplicate_extra_images=len(fit)-tally["unique_source_rgb"],
                  target_within_decoded_duplicate_extra_images=len(reserve)-tally["unique_target_rgb"],
                  target_rgb_matches_by_sensor={sensor:by_sensor[sensor] for sensor in sorted(set(sensors))},
                  rgb_method="Pillow EXIF transpose then native RGB; (height, width, SHA-256 of RGB bytes)",
                  pillow=pillow_version, product_and_geographic_independence_verified=False,
                  target_polygon_rows_opened=0, training_label_rows_opened=0, prediction_rows_opened=0,
                  model_forwards=0, optimizer_updates=0, scores_computed=0, **tally)
    a.output.mkdir(exist_ok=False)
    raw = (json.dumps(result, sort_keys=True, indent=2) + "\n").encode()
    (a.output / "RESULT_SAFE.json").write_bytes(raw)
    print(json.dumps(dict(status=STATUS, result_sha256=digest(raw))), flush=True)


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("acceptance", "training", "fit-stage", "fit-manifest", "fit-archive",
                 "intake", "protocol", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--protocol-sha256", required=True)
    run(p.parse_args())

