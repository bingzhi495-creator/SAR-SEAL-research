"""Project the closed seam feasibility cohort into isolated image and score rosters."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path


PROTOCOL = "6b0955af80a701175f2a5b1ebdb4f9d82d239df2e18586e5037dc414c850eba2"
ENRICHED_ROSTER = "e8568f25f3a1c04dc8ec5fc99546d502c69c25ed2008ec2a28f484047f20cd00"
ENRICHED_RGB = "94c4b60ca687e60215df336718800dd7ace47fb1cb29ac05718182b2763567cd"
NATIVE = "a7e4d2a7ff9e8ff238adbf94a1d0397553d8746e2558c316db4e417f86f8fec9"
IMAGE_FIELDS = ("encoded_sha256", "decoded_rgb_sha256", "encoded_bytes")


def need(valid, message):
    if not valid:
        raise ValueError(message)


def read(path, expected):
    data = path.read_bytes()
    need(sha256(data).hexdigest() == expected, "Input digest differs: " + path.name)
    return data


def write(path, data):
    raw = (json.dumps(data, sort_keys=True, allow_nan=False, indent=2) + "\n").encode()
    with path.open("xb") as file:
        file.write(raw)
    return sha256(raw).hexdigest()


def shift(pair):
    a, b = (x["window"] for x in pair)
    return [b[0] - a[0], b[2] - a[2]]


def build(selected, native):
    images = {}
    score = []
    pair_keys = set()
    for source in selected:
        group = source["group"]
        near_far = source["near_far"]
        interior = source["matched_interior"]
        if interior is None:
            continue
        need(min(near_far["margins"]) > 0 and min(near_far["margins"]) <= 16
             and max(near_far["margins"]) >= 64 and min(interior["margins"]) >= 64,
             "Fixed target visibility and edge strata")
        need(list(map(abs, shift(near_far["pair"]))) == list(map(abs, shift(interior["pair"]))),
             "Fixed source-matched absolute crop offset")
        rows = {}
        for kind, item in (("near_far", near_far), ("both_interior", interior)):
            pair = item["pair"]
            pair_keys.add(tuple(sorted(r["native_basename"] for r in pair)))
            for member in pair:
                name = member["native_basename"]
                n = native[name]
                need(name.split("_")[0] == group and all(n[k] == member[k] for k in IMAGE_FIELDS),
                     "Exact native identity")
                if name not in images:
                    images[name] = dict(ordinal=len(images) + 1, group=group, basename=name,
                                        **{k: member[k] for k in IMAGE_FIELDS})
            order = sorted(range(2), key=lambda i: item["margins"][i]) if kind == "near_far" else (0, 1)
            rows[kind] = []
            for i in order:
                member = pair[i]
                name = member["native_basename"]
                rows[kind].append(dict(
                    image_id=native[name]["image_id"],
                    annotation_id=item["annotation_ids"][i],
                    basename=name,
                    image_ordinal=images[name]["ordinal"],
                    margin=item["margins"][i],
                    window=member["window"],
                ))
        score.append(dict(ordinal=len(score) + 1, group=group,
                          offset_abs_yx=list(map(abs, shift(near_far["pair"]))),
                          near_far_shift_yx=shift(rows["near_far"]),
                          both_interior_shift_yx=shift(rows["both_interior"]), **rows))

    return list(images.values()), score, pair_keys


def main(args):
    read(args.protocol, PROTOCOL)
    selected = json.loads(read(args.enriched_roster, ENRICHED_ROSTER))
    rgb = json.loads(read(args.enriched_rgb, ENRICHED_RGB))
    need(rgb["gate"] == "FUTURE_FORWARD_DATA_FEASIBLE" and rgb["nonidentical_crop_pairs"] == 0
         and rgb["roster_sha256"] == ENRICHED_ROSTER and rgb["exact_matched_control_source_groups"] == 40,
         "Exact enriched RGB acceptance")
    layout = json.loads(read(args.native_layout, NATIVE))["images"]
    native = {row["native_basename"]: row for row in layout}
    need(len(native) == len(layout) == 2937, "Exact native FIT metadata roster")
    images, score, pairs = build(selected["records"], native)
    need(len(images) == len({r["encoded_sha256"] for r in images})
         == len({r["decoded_rgb_sha256"] for r in images}) == 144 and len(pairs) == 73
         and len(score) == len({r["group"] for r in score}) == 40, "Exact deduplicated prospective scope")
    image_sha = write(args.image_output, dict(
        status="HRSID_FIT_SEAM_CONTEXT_IMAGE_MANIFEST_V1",
        FIT=True, DEV=False, TEST=False, GT_fields_present=False,
        protocol_sha256=PROTOCOL, images=images))
    score_sha = write(args.score_output, dict(
        status="HRSID_FIT_SEAM_CONTEXT_SCORE_ROSTER_V1",
        FIT=True, DEV=False, TEST=False, image_manifest_sha256=image_sha,
        protocol_sha256=PROTOCOL, source_records=score))
    safe = dict(status="HRSID_FIT_SEAM_CONTEXT_MANIFESTS_CLOSED",
                FIT=True, DEV=False, TEST=False, image_manifest_has_GT=False,
                previously_admitted_label_metadata_only=True, FIT_label_index_opened=False,
                PNG_contents_opened=0, source_groups=len(score), target_pairs=2*len(score),
                unique_crop_pairs=len(pairs), unique_images=len(images),
                image_manifest_sha256=image_sha, score_roster_sha256=score_sha, protocol_sha256=PROTOCOL)
    safe_sha = write(args.safe_output, safe)
    print(json.dumps(dict(safe, manifest_safe_sha256=safe_sha), sort_keys=True))


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("protocol", "enriched-roster", "enriched-rgb", "native-layout", "image-output", "score-output", "safe-output"):
        parser.add_argument("--" + name, type=Path, required=True)
    main(parser.parse_args())
