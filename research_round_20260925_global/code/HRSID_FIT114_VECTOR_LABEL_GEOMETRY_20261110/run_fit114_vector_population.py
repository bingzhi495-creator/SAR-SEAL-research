"""Read the complete, frozen HRSID FIT114 polygon-first geometry population."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
from statistics import fmean
import sys
from time import perf_counter
from types import ModuleType
from unittest.mock import patch
from zipfile import ZipFile


PILOT_SHA = "0b9726556a27dcca40ca9571390a8d04bb6c0dea7f4d982c29235283791858d9"
VECTOR_SHA = "4877c48d38a03e58e97163381ba544df3c90a78dde85c9141085fa0b92da3b09"
EXPECTED_PATH_VIEWS = {("core", "direct"): 235, ("core", "crop"): 201,
                       ("stress", "direct"): 13, ("stress", "crop"): 7}
PILOT_WITNESS_CURSORS = (0, 1, 2, 5)
AREAS = ("small", "other")
LAYERS = ("core", "stress")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pinned_module(path, expected, name):
    require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file(),
            "Canonical closed pilot source required")
    raw = path.read_bytes()
    require(0 < len(raw) <= 64 * 1024 and sha256(raw).hexdigest() == expected, "Exact unchanged pilot source required")
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def representation(annotation):
    if "segmentation" not in annotation:
        return "missing"
    value = annotation["segmentation"]
    if type(value) is list:
        return "polygon_list" if value else "empty_polygon_list"
    if type(value) is dict:
        return "RLE"
    return "null" if value is None else "other"


def observed_projection(original, source_rows, prepared, row_id, arm_names):
    """Census all prepared and arm-present rows before the unchanged projection."""
    by_area = {area: defaultdict(Counter) for area in AREAS}
    identities = [("small" if float(row["area"]) < 1024 else "other", representation(row)) for row in source_rows]
    for (area, category), live in zip(identities, prepared["masks"].flatten(1).any(1).tolist(), strict=True):
        counts = by_area[area][category]
        counts["prepared"] += 1
        counts["prepared_empty"] += int(not live)

    calls = []

    def observe(*targets):
        require(not calls and len(targets) == len(arm_names), "Exactly one original common projection per view")
        calls.append(True)
        for name, target in zip(arm_names, targets, strict=True):
            for original_id, live in zip(target[row_id].tolist(), target["masks"].flatten(1).any(1).tolist(), strict=True):
                area, category = identities[original_id]
                counts = by_area[area][category]
                counts[name + "_present"] += 1
                counts[name + "_empty"] += int(not live)
        projected = original(*targets)
        for original_id in projected[0][row_id].tolist():
            area, category = identities[original_id]
            by_area[area][category]["fixed_common"] += 1
        return projected

    return observe, by_area, calls


def finish_census(by_area, calls, records, counts):
    require(len(calls) == 1, "Exactly one observed original projection required")
    seen = skipped = eligible = 0
    for area, categories in by_area.items():
        eligible_rows = [row for row in records if row["small"] == (area == "small")]
        polygon = categories["polygon_list"]
        polygon["vector_eligible"] = len(eligible_rows)
        polygon["vector_final_empty"] = sum(row["vector_empty"] for row in eligible_rows)
        for category in categories.values():
            category["vector_ineligible"] = category["fixed_common"] - category["vector_eligible"]
            require(category["vector_ineligible"] >= 0, "Vector rows escaped fixed polygon eligibility")
            seen += category["fixed_common"]
            skipped += category["vector_ineligible"]
            eligible += category["vector_eligible"]
    require(seen == counts["common"] and skipped == sum(counts["skipped"].values())
            and eligible == counts["polygon_eligible"] == len(records), "Exact per-representation support required")


def footprint_error(before, after):
    require(before.keys() == after.keys() and before["by_arm"].keys() == after["by_arm"].keys(),
            "Exact polygon reference and arm fields required")
    errors = [abs(before["visible_polygon_area"] - after["visible_polygon_area"])]
    for key, row in before["by_arm"].items():
        other = after["by_arm"][key]
        require(row.keys() == other.keys(), "Exact continuous geometry output fields required")
        for metric, value in row.items():
            require((value is None) == (other[metric] is None), "Continuous geometry defined status changed")
            if value is not None:
                errors.append(abs(value - other[metric]))
    return max(errors)


def summarize(pilot, math_api, rows, multiplicity):
    result = dict(eligible_rows=len(rows), vector_final_empty=sum(row["vector_empty"] for _, row in rows))
    for reference in pilot.REFERENCES:
        arms = pilot.ARMS[:-1] if reference == "vector_raster" else pilot.ARMS
        values = {
            arm: [row[reference][arm] for _, row in rows if row[reference][arm] is not None]
            for arm in arms
        }
        by_arm = {arm: dict(mean_iou=fmean(vals) if vals else None, rows=len(vals),
                            undefined_rows=len(rows)-len(vals)) for arm, vals in values.items()}
        effects = {}
        for earlier, later in zip(arms, arms[1:]):
            paired = [(ordinal, row[reference][later] - row[reference][earlier]) for ordinal, row in rows
                      if row[reference][later] is not None and row[reference][earlier] is not None]
            effects[later + "_minus_" + earlier] = math_api.paired_cluster_stats(paired, multiplicity)
        result[reference] = dict(arms=by_arm, paired_effects=effects)
    return result


def run(args):
    started = perf_counter()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and sys.flags.isolated == sys.flags.utf8_mode == 1
            and sys.dont_write_bytecode and sys.version_info[:3] == (3, 10, 13), "Exact isolated Windows CPU scope")
    pilot = pinned_module(args.pilot_dir / "run_fit114_vector_trial.py", PILOT_SHA, "_p251_unchanged_four_view_pilot")
    native, np, torch, shapely = pilot.install_support(args.code, args.shapely_site)
    vector = pinned_module(args.pilot_dir / "vector_label_geometry.py", VECTOR_SHA, "vector_label_geometry")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import common_mask_supervision
    import vector_population_math as math_api

    original_projection = common_mask_supervision.common_targets
    original_vector_area = vector.vector_footprint_metrics
    bundle_path, output = args.bundle, args.output
    require(bundle_path.is_absolute() and bundle_path.resolve(strict=True) == bundle_path
            and output.is_absolute() and output.parent.resolve(strict=True) == output.parent
            and not output.exists() and not output.is_symlink(), "Canonical sealed FIT packet and fresh private output")
    audit = native.NativeInputAudit()
    records = defaultdict(list)
    coverage = defaultdict(lambda: {area: defaultdict(Counter) for area in AREAS})
    source_counts = defaultdict(Counter)
    path_views, skipped = Counter(), defaultdict(Counter)
    witness_rows = 0
    max_run_error = 0.
    ledger_sha = sha256()
    with bundle_path.open("rb") as stream:
        before = pilot.fingerprint(os.fstat(stream.fileno()))
        require(before[2] == pilot.BUNDLE_BYTES, "Exact sealed FIT-only packet length")
        file_sha = sha256()
        for chunk in iter(lambda: stream.read(1024**2), b""):
            file_sha.update(chunk)
        require(file_sha.hexdigest() == pilot.BUNDLE_SHA, "Exact sealed FIT-only packet SHA-256")
        stream.seek(0)
        with ZipFile(stream) as bundle:
            cohort = json.loads(pilot.pinned_member(bundle, "COHORT_PRIVATE.json", pilot.COHORT_SHA, 1024**2))
            index = json.loads(pilot.pinned_member(bundle, "FIT114_COCO_PRIVATE.json", pilot.SUBSET_SHA, 4*1024**2))
            history = [json.loads(line) for line in pilot.pinned_member(
                bundle, "HISTORICAL_VIEWS_PRIVATE.jsonl", pilot.HISTORY_SHA, 1024**2).splitlines()]
            images = {row["id"]: row["file_name"] for row in index["images"]}
            annotations = defaultdict(list)
            for annotation in index["annotations"]:
                require(annotation["image_id"] in images, "Only frozen FIT annotations allowed")
                annotations[annotation["image_id"]].append(annotation)
            rows = cohort["images"]
            members = {"COHORT_PRIVATE.json", "FIT114_COCO_PRIVATE.json", "HISTORICAL_VIEWS_PRIVATE.jsonl",
                       *("images/" + row["native_basename"] for row in rows)}
            require(cohort["fit_only"] is True and len(rows) == len(images) == 114 and len(history) == 456
                    and len(index["annotations"]) == 1109 and len(bundle.namelist()) == len(members)
                    and set(bundle.namelist()) == members, "Exact frozen complete FIT-only population")
            output.mkdir()
            with (output / "VIEW_AUDIT_PRIVATE.jsonl").open("xb") as ledger:
                for ordinal, row in enumerate(rows):
                    image_id, name = row["image_id"], row["native_basename"]
                    require(images[image_id] == name and row["core"] != row["stress"], "Exact historical source image")
                    layer = "core" if row["core"] else "stress"
                    png = pilot.pinned_member(bundle, "images/" + name, row["encoded_sha256"], 4*1024**2)
                    image, scales = native.io_utils.decode_pil_image_bytes(png, draft_size=None)
                    require(len(png) == row["encoded_bytes"] and image.size == (800, 800) and image.mode == "RGB"
                            and scales == (1., 1.) and pilot.digest(np.asarray(image).tobytes()) == row["decoded_rgb_sha256"],
                            "Exact historical decoded PNG")
                    annotations_i = annotations[image_id]
                    image, target = audit.convert(image, dict(image_id=image_id, annotations=annotations_i))
                    native.target_summary(target, 800, image_id, normalized=False)
                    prepared_rows = pilot.source_annotations(annotations_i, target, (800, 800), torch)
                    for item in prepared_rows:
                        source_counts[layer][representation(item)] += 1
                    # Preserve historical raw conversion. If a nonpolygon row lacks the
                    # key, pass an explicit null only to the pilot's polygon eligibility check.
                    labels = [item if "segmentation" in item else dict(item, segmentation=None) for item in annotations_i]
                    for rep in range(pilot.REPEATS):
                        cursor = ordinal*pilot.REPEATS + rep
                        old = history[cursor]
                        seed, path = pilot.BASE_SEED + cursor, old["path"]
                        require(old["ordinal"] == ordinal and old["seed"] == seed and old["layer"] == layer
                                and old["group"] == row["group"] and (layer, path) in EXPECTED_PATH_VIEWS,
                                "Exact original view order, path, seed and proxy group")
                        inspect, by_area, calls = observed_projection(
                            original_projection, prepared_rows, target, common_mask_supervision.ROW_ID, pilot.ARMS[:-1],
                        )

                        def vector_area(rings, hw, masks):
                            nonlocal witness_rows, max_run_error
                            measured = math_api.run_footprint_metrics(rings, hw, masks)
                            if cursor in PILOT_WITNESS_CURSORS:
                                error = footprint_error(original_vector_area(rings, hw, masks), measured)
                                require(error <= 1e-8, "Horizontal-run geometry differs from original unit-cell result")
                                max_run_error = max(max_run_error, error)
                                witness_rows += 1
                            return measured

                        with (patch.object(common_mask_supervision, "common_targets", inspect),
                              patch.object(vector, "vector_footprint_metrics", vector_area)):
                            eligible, counts = pilot.compare_view(native, audit, image, target, labels, old, torch, np)
                        finish_census(by_area, calls, eligible, counts)
                        key = layer, path
                        records[key].extend((ordinal, result) for result in eligible)
                        skipped[key].update(counts["skipped"])
                        path_views[key] += 1
                        for area, categories in by_area.items():
                            for category, tally in categories.items():
                                coverage[key][area][category].update(tally)
                        visit = dict(cursor=cursor, ordinal=ordinal, seed=seed, layer=layer, path=path,
                                     flip=old["flip"], representation=by_area, **counts)
                        line = (json.dumps(visit, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
                        require(ledger.write(line) == len(line), "Incomplete original FIT view evidence")
                        ledger_sha.update(line)
                    if (ordinal+1) % 20 == 0 or ordinal+1 == len(rows):
                        print(json.dumps(dict(event="P251_FIT_VECTOR_POPULATION_PROGRESS", images=ordinal+1,
                                              views=(ordinal+1)*pilot.REPEATS)), flush=True)
                ledger.flush()
                os.fsync(ledger.fileno())
        require(pilot.fingerprint(os.fstat(stream.fileno())) == pilot.fingerprint(bundle_path.stat()) == before,
                "Exact frozen FIT packet unchanged")
    require(dict(path_views) == EXPECTED_PATH_VIEWS and witness_rows == 49 and not torch.cuda.is_initialized()
            and torch.get_num_threads() == torch.get_num_interop_threads() == 1, "Exact complete FIT census and geometry witness")
    multiplicity = math_api.bootstrap_draws(len(rows))
    scopes = {}
    for scope in (*LAYERS, "combined"):
        scope_layers = LAYERS if scope == "combined" else (scope,)
        scopes[scope] = {}
        for path in ("direct", "crop"):
            keys = [(layer, path) for layer in scope_layers]
            data = [row for key in keys for row in records[key]]
            census, skips = {area: defaultdict(Counter) for area in AREAS}, Counter()
            for key in keys:
                skips.update(skipped[key])
                for area, by_representation in coverage[key].items():
                    for category, tally in by_representation.items():
                        census[area][category].update(tally)
            scopes[scope][path] = dict(
                views=sum(path_views[key] for key in keys), representation_by_area=census, vector_skip_reasons=skips,
                all=summarize(pilot, math_api, data, multiplicity),
                small=summarize(pilot, math_api, [(i, row) for i, row in data if row["small"]], multiplicity),
                other=summarize(pilot, math_api, [(i, row) for i, row in data if not row["small"]], multiplicity),
            )
    result = dict(
        status="HRSID_FIT114_VECTOR_FIRST_COMPLETE_POPULATION_GEOMETRY",
        images=114, views=456, polygon_unit_cell_witness_rows=witness_rows, max_polygon_run_area_error=max_run_error,
        fit_bundle_sha256=pilot.BUNDLE_SHA, cohort_sha256=pilot.COHORT_SHA, subset_sha256=pilot.SUBSET_SHA,
        historical_views_sha256=pilot.HISTORY_SHA, actual_view_audit_sha256=ledger_sha.hexdigest(),
        pilot_sha256=PILOT_SHA, vector_helper_sha256=VECTOR_SHA,
        population_math_sha256=pilot.digest(Path(math_api.__file__).read_bytes()),
        population_runner_sha256=pilot.digest(Path(__file__).read_bytes()),
        stock_source_sha256=pilot.SUPPORT_SOURCES, shapely_version=shapely.__version__, geos_version=shapely.geos_version_string,
        source_annotation_representation_counts=source_counts,
        area_threshold=1024, support="fixed ordered nonempty A intersection B intersection B-conditioned C",
        bootstrap=dict(seed=math_api.BOOTSTRAP_SEED, replicates=math_api.BOOTSTRAP_DRAWS,
                       cluster="original source image", total_clusters=len(rows), all_scopes_same_draws=True,
                       tie_tolerance=math_api.TIE_TOLERANCE),
        metrics=scopes, FIT=True, DEV=False, TEST=False, AP=False, model_forward_calls=0, optimizer_updates=0,
        wall_seconds=perf_counter()-started,
    )
    path = output / "RESULT_SAFE.json"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(status=result["status"], result_sha256=pilot.digest(path.read_bytes()),
                          wall_seconds=result["wall_seconds"]), sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("pilot-dir", "bundle", "code", "shapely-site", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
