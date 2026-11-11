"""Generate a second, image-only FIT cohort with fresh probe source prefixes."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
from types import ModuleType


NATIVE = "a7e4d2a7ff9e8ff238adbf94a1d0397553d8746e2558c316db4e417f86f8fec9"
ORACLE = "af43530e7f8f118261b5523cdde738cca467ead813d38f1d4c989c8f4102e184"
P316 = "4b5574f3b32dba4c2963c57a4637eaa3695f1118a75cc3b7a36ec92bd0c60319"
CONSENSUS = "0453980c7c4f88b08506080b05ac58ccb7a89b948a055b6b2be7b719394dafac"
BUILDER = "7b6df2988e3846e06f2261646b25f672fa7aeecc53b493bcff5bbb374811673f"
VARIANT = "P316_COORDINATE_DISJOINT_ORACLE36_AND_CONSENSUS36_PREFIX_EXCLUDED"
DIGEST_FIELDS = ("encoded_sha256", "decoded_rgb_sha256", "encoded_bytes")


def pinned(path, expected):
    payload = path.read_bytes()
    if sha256(payload).hexdigest() != expected:
        raise ValueError("Unexpected image-only metadata or builder digest: " + path.name)
    return payload


def load_builder(path):
    source = pinned(path, BUILDER)
    builder = ModuleType("_original_image_manifest_builder")
    exec(compile(source, str(path), "exec"), builder.__dict__)
    return builder


def eligible_confirmation(certified, layout, previous, p316_decoded, group_of):
    groups = {row["group"] for row in previous}
    encoded = {row["encoded_sha256"] for row in previous}
    decoded = {row["decoded_rgb_sha256"] for row in previous} | set(p316_decoded)
    return [
        name for name in certified
        if (group_of(name) not in groups
            and layout[name]["encoded_sha256"] not in encoded
            and layout[name]["decoded_rgb_sha256"] not in decoded)
    ]


def main(args):
    if args.private_output.exists() or args.safe_output.exists():
        raise FileExistsError("Output files must be new")
    builder = load_builder(args.base_builder)
    pinned(args.native_layout, NATIVE)
    pinned(args.prior_oracle, ORACLE)
    pinned(args.p316_candidates, P316)
    prior = json.loads(pinned(args.prior_consensus, CONSENSUS))["images"]
    oracle = builder.images_in(args.prior_oracle)
    p316 = builder.candidate_fields(args.p316_candidates, True)
    names = [path.name for path in args.fit_train.glob("*.png") if path.is_file()]
    certified, layout, p316_decoded, base_facts = builder.check_rectangles(
        names, p316, args.native_layout, args.prior_oracle,
    )
    if (len(prior) != len(oracle) or len(prior) != 36
            or len({x["group"] for x in prior + oracle}) != 72 or len(p316) != 114):
        raise ValueError("Unexpected prior image cohorts")
    for row in prior:
        name = row["basename"]
        if (builder.group_of(name) != row["group"]
                or name not in certified
                or any(layout[name][key] != row[key] for key in DIGEST_FIELDS)):
            raise ValueError("Earlier seed-consensus image did not reproduce the certified roster")

    remaining = eligible_confirmation(certified, layout, prior + oracle, p316_decoded, builder.group_of)
    selected, groups, images = builder.rank_images(remaining, set(), 36)
    facts = {
        "variant": VARIANT,
        "native_png_images": len(names),
        "previous_consensus_excluded_groups": len({x["group"] for x in prior}),
        "previous_oracle_excluded_groups": len({x["group"] for x in oracle}),
        "prior_p316_linked_rows": base_facts["prior_p316_linked_rows"],
        "eligible_images": images,
        "eligible_groups": groups,
        "fit_only": True,
        "labels_opened": False,
        "models_executed": False,
    }
    if len(selected) < 24:
        facts.update(status="STOP_INSUFFICIENT_SOURCE_PREFIXES", selected_images=0, png_contents_opened=0)
        builder.write_new(args.safe_output, facts)
        print(json.dumps(facts, sort_keys=True))
        return 2

    rows = [
        builder.describe_image(args.fit_train / name, group, ordinal)
        for ordinal, (group, name) in enumerate(selected, start=1)
    ]
    for row in rows:
        if any(layout[row["basename"]][key] != row[key] for key in DIGEST_FIELDS):
            raise ValueError("Live image bytes did not match the closed native image manifest")
    for field in DIGEST_FIELDS[:2]:
        if len({x[field] for x in rows}) != len(rows):
            raise ValueError("Duplicate selected image content")
    manifest_sha = builder.write_new(args.private_output, {
        "status": "FIT_SEED_CONSENSUS_IMAGE_MANIFEST_V1",
        "variant": VARIANT,
        "images": rows,
    })
    p316_prefixes = {x[0] for x in p316}
    facts.update(
        status="FIT_SEED_AREA_CONFIRM_IMAGE_MANIFEST_READY",
        selected_images=len(rows),
        selected_unique_prefixes=len({x["group"] for x in rows}),
        selected_p316_rectangle_disjoint=len(rows),
        selected_p316_encoded_hash_overlap=0,
        selected_p316_decoded_hash_overlap=0,
        selected_reused_p316_prefixes=sum(x["group"] in p316_prefixes for x in rows),
        selected_previous_probe_prefix_overlap=0,
        selected_previous_probe_hash_overlap=0,
        png_contents_opened=len(rows),
        encoded_bytes=sum(x["encoded_bytes"] for x in rows),
        manifest_sha256=manifest_sha,
    )
    builder.write_new(args.safe_output, facts)
    print(json.dumps(facts, sort_keys=True))
    return 0


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("fit-train", "p316-candidates", "native-layout", "prior-oracle", "prior-consensus",
                 "base-builder", "private-output", "safe-output"):
        parser.add_argument("--" + name, type=Path, required=True)
    raise SystemExit(main(parser.parse_args()))
