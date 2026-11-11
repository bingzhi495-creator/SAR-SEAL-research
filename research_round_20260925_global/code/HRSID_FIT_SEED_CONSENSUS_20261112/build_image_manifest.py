"""Build a FIT image-only manifest without opening a label or model."""

import argparse
from collections import defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re

from PIL import Image


SALT = "HRSID-seed-area-v1|"
GROUP = re.compile(r"P[0-9]+")
RECTANGLE = re.compile(r"P[0-9]+_([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)\.png")


def group_of(basename):
    group = basename.partition("_")[0]
    if not GROUP.fullmatch(group):
        raise ValueError("Unexpected FIT PNG source prefix")
    return group


def salted_key(value):
    return sha256((SALT + value).encode("utf-8")).digest(), value


def rank_images(basenames, excluded_groups, count):
    grouped = defaultdict(list)
    for name in basenames:
        group = group_of(name)
        if group not in excluded_groups:
            grouped[group].append(name)
    ranked = sorted(grouped, key=salted_key)
    return [(g, min(grouped[g], key=salted_key)) for g in ranked[:count]], len(grouped), sum(map(len, grouped.values()))


def candidate_fields(path, with_identity):
    fields = ("group", "image_id", "encoded_sha256") if with_identity else ("group",)
    with path.open(encoding="utf-8-sig") as file:
        return [tuple(row[k] for k in fields) for line in file if (row := json.loads(line))]


def images_in(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))["images"]


def rectangle(basename):
    match = RECTANGLE.fullmatch(basename)
    if match is None:
        return None
    first, last, left, right = map(int, match.groups())
    if (last - first, right - left) != (800, 800):
        raise ValueError("Unexpected rectangle dimensions in FIT PNG basename")
    return first, last, left, right


def intersects(a, b):
    return min(a[1], b[1]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[2], b[2])


def check_rectangles(basenames, candidate_rows, layout_path, previous_path):
    rows = images_in(layout_path)
    by_name = {x["native_basename"]: x for x in rows}
    by_id = {x["image_id"]: x for x in rows}
    if len(rows) != len(by_name) or len(rows) != len(by_id) or set(by_name) != set(basenames):
        raise ValueError("Native image-only metadata does not match the FIT directory census")

    prior_groups = defaultdict(list)
    prior_encoded = set()
    prior_decoded = set()
    for group, image_id, encoded in candidate_rows:
        row = by_id[image_id]
        if row["encoded_sha256"] != encoded or group_of(row["native_basename"]) != group:
            raise ValueError("Prior captured image did not match native image metadata")
        prior_groups[group].append(rectangle(row["native_basename"]))
        prior_encoded.add(encoded)
        prior_decoded.add(row["decoded_rgb_sha256"])

    previous = images_in(previous_path)
    excluded_groups = {x["group"] for x in previous}
    excluded_decoded = {x["decoded_rgb_sha256"] for x in previous}
    for old in previous:
        matched = by_name[old["native_basename"]]
        if (group_of(old["native_basename"]) != old["group"]
                or matched["image_id"] != old["image_id"]
                or any(matched[k] != old[k] for k in ("encoded_sha256", "decoded_rgb_sha256", "encoded_bytes"))):
            raise ValueError("Previous image-only oracle manifest failed the layout join")

    eligible = []
    certified = []
    for name, row in by_name.items():
        group = group_of(name)
        if (group in excluded_groups or row["encoded_sha256"] in prior_encoded
                or row["decoded_rgb_sha256"] in excluded_decoded):
            continue
        eligible.append(name)
        bounds = rectangle(name)
        old_bounds = prior_groups[group]
        if bounds is not None and all(x is not None and not intersects(bounds, x) for x in old_bounds):
            certified.append(name)
    facts = {
        "prior_p316_linked_rows": len(candidate_rows),
        "prior_oracle_excluded_groups": len(excluded_groups),
        "prior_oracle_excluded_decoded_hashes": len(excluded_decoded),
        "prior_p316_excluded_encoded_hashes": len(prior_encoded),
        "prior_p316_unparsed_rectangles": sum(r is None for rows in prior_groups.values() for r in rows),
        "eligible_before_rectangle_filter_images": len(eligible),
        "eligible_before_rectangle_filter_groups": len({group_of(x) for x in eligible}),
    }
    return certified, by_name, prior_decoded, facts


def describe_image(path, group, ordinal):
    encoded = path.read_bytes()
    with Image.open(BytesIO(encoded)) as image:
        if image.format != "PNG" or image.size != (800, 800):
            raise ValueError("Expected an 800 by 800 PNG")
        rgb = image.convert("RGB").tobytes()
    return {
        "group": group,
        "basename": path.name,
        "encoded_sha256": sha256(encoded).hexdigest(),
        "decoded_rgb_sha256": sha256(rgb).hexdigest(),
        "encoded_bytes": len(encoded),
        "ordinal": ordinal,
    }


def write_new(path, data):
    payload = (json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as file:
        file.write(payload)
    return sha256(payload).hexdigest()


def main(args):
    if (args.native_layout is None) != (args.prior_oracle_manifest is None):
        raise ValueError("Provide both closed image metadata inputs for the rectangle-disjoint variant")
    if args.private_output.exists() or args.safe_output.exists():
        raise FileExistsError("Output paths must be fresh")
    coordinate_variant = args.native_layout is not None
    candidate_rows = candidate_fields(args.p316_candidates, coordinate_variant)
    groups = {row[0] for row in candidate_rows}
    if not all(GROUP.fullmatch(g) for g in groups):
        raise ValueError("Unexpected excluded source prefix")
    names = [p.name for p in args.fit_train.glob("*.png") if p.is_file()]
    native_groups = {group_of(name) for name in names}

    if coordinate_variant:
        certified, layout, prior_decoded, variant_facts = check_rectangles(
            names, candidate_rows, args.native_layout, args.prior_oracle_manifest,
        )
        selected, group_count, image_count = rank_images(certified, set(), args.count)
        variant = "P316_COORDINATE_DISJOINT_ORACLE36_PREFIX_EXCLUDED"
    else:
        selected, group_count, image_count = rank_images(names, groups, args.count)
        variant_facts = {}
        variant = "ALL_P316_PREFIXES_EXCLUDED"
    facts = {
        "variant": variant,
        "requested_images": args.count,
        "native_png_images": len(names),
        "native_source_prefixes": len(native_groups),
        "prior_p316_rows": len(candidate_rows),
        "prior_p316_source_prefixes": len(groups),
        "p316_and_native_prefixes_in_common": len(groups & native_groups),
        "eligible_groups": group_count,
        "eligible_images": image_count,
        "fit_only": True,
        "labels_opened": False,
        "models_executed": False,
        **variant_facts,
    }
    if len(selected) != args.count:
        facts.update(status="STOP_INSUFFICIENT_SOURCE_PREFIXES", selected_images=0, png_contents_opened=0)
        write_new(args.safe_output, facts)
        print(json.dumps(facts, sort_keys=True))
        return 2

    rows = [describe_image(args.fit_train / name, group, index)
            for index, (group, name) in enumerate(selected, start=1)]
    for field in ("encoded_sha256", "decoded_rgb_sha256"):
        if len({row[field] for row in rows}) != args.count:
            raise ValueError("Duplicate selected image content")
    if coordinate_variant:
        for row in rows:
            if any(layout[row["basename"]][k] != row[k] for k in ("encoded_sha256", "decoded_rgb_sha256", "encoded_bytes")):
                raise ValueError("Native image bytes did not match closed metadata")
        facts.update(
            selected_p316_rectangle_disjoint=len(rows),
            selected_prior_p316_decoded_hash_overlap=sum(row["decoded_rgb_sha256"] in prior_decoded for row in rows),
        )
    private = {"status": "FIT_SEED_CONSENSUS_IMAGE_MANIFEST_V1", "variant": variant, "images": rows}
    digest = write_new(args.private_output, private)
    facts.update(
        status="FIT_SEED_CONSENSUS_IMAGE_MANIFEST_READY",
        selected_images=len(rows),
        selected_unique_prefixes=len({row["group"] for row in rows}),
        selected_reused_p316_prefixes=sum(row["group"] in groups for row in rows),
        png_contents_opened=len(rows),
        manifest_sha256=digest,
        encoded_bytes=sum(row["encoded_bytes"] for row in rows),
    )
    write_new(args.safe_output, facts)
    print(json.dumps(facts, sort_keys=True))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for required in ("fit-train", "p316-candidates", "private-output", "safe-output"):
        parser.add_argument("--" + required, type=Path, required=True)
    parser.add_argument("--native-layout", type=Path)
    parser.add_argument("--prior-oracle-manifest", type=Path)
    parser.add_argument("--count", type=int, default=36)
    options = parser.parse_args()
    if options.count < 1:
        parser.error("--count must be positive")
    raise SystemExit(main(options))
