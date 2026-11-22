"""Paired, acquisition-balanced physical-dimension scoring on the frozen NASTaR cohort."""

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import io
import json
from pathlib import Path
import sys
from time import perf_counter, sleep
from urllib.parse import quote
import zlib

import numpy as np
from PIL import Image
from pycocotools import mask as codec
import requests


ARMS = ("raw", "fixed", "free_matched_count", "free_natural")
PAIRS = (("fixed", "free_matched_count"), ("raw", "free_natural"), ("raw", "fixed"), ("raw", "free_matched_count"))
LABELS = ("Longitude", "Latitude", "Heading", "Length", "Width")
GEOMETRY = "4b4302c3f4eff9d4d8a6e759b62e065ab929cc9a7aeb7348268b40808ff7957c"
PROTOCOL = "54c4094d79ade98f06b7147608eb84c66439e722ef13f59bc0f2c8793f1231f9"
CHECKPOINT = "011bd30ed056c871be1d61e14d71b3b2b258330a9d20cf2f164ca4a7c4f23c11"
POINT_HEAD = "3a60f658149f4262451cd9118902bbe52dc83345d9fbcd23973a11dc473fcdf1"
STATUS = "NASTAR_AIS_PHYSICAL_SCORE_COMPLETE"


def require(value, message):
    if not value:
        raise ValueError(message)


def data_sha(raw):
    return sha256(raw).hexdigest()


def pinned(path, digest):
    raw = path.read_bytes()
    require(data_sha(raw) == digest, "Input digest mismatch: "+path.name)
    return raw


def module(path, digest):
    spec = spec_from_file_location(path.stem, path)
    mod = module_from_spec(spec)
    exec(compile(pinned(path,digest), str(path), "exec"), mod.__dict__)
    return mod


def grid_from_tags(tags):
    keys = tags["geo_key_directory"]
    require(len(keys) == 4+4*keys[3] and keys[0] == 1, "Unrecognized GeoKey directory")
    scalar = {key: value for key, loc, count, value in np.reshape(keys[4:], (-1, 4)) if loc == 0 and count == 1}
    require(scalar.get(1025) == 1, "PixelIsArea georeferencing required")
    model = scalar.get(1024)
    require(model in (1, 2), "Unsupported coordinate system")
    epsg = scalar.get(3072 if model == 1 else 2048)
    require(epsg is not None and 0 < epsg < 32767, "Explicit EPSG horizontal CRS required")
    if tags["model_transformation"] is not None:
        matrix = np.asarray(tags["model_transformation"], dtype=float).reshape(4,4)
        require(np.isfinite(matrix).all() and np.array_equal(matrix[3], [0,0,0,1]), "Invalid affine homogeneous row")
        g = matrix.ravel()[[3,0,1,7,4,5]]
    else:
        dx, dy, _ = tags["model_pixel_scale"]
        ties = np.reshape(tags["model_tiepoint"], (-1, 6))
        require(dx > 0 and dy > 0 and len(ties) > 0, "Invalid pixel scale or tiepoint")
        origins = np.column_stack((ties[:,3]-dx*ties[:,0], ties[:,4]+dy*ties[:,1]))
        require(np.allclose(origins, origins[0], rtol=0, atol=1e-8), "Inconsistent raster tiepoints")
        g = (origins[0,0], dx, 0, origins[0,1], 0, -dy)
    return int(epsg), g


def within_radius(geo, lon, lat, crs, transform):
    """Exact ellipsoidal center-to-reference distance for every original cell."""
    origin, matrix = geo._grid(transform)
    rows, cols = np.indices((512,512), dtype=np.float64)
    x = origin[0]+matrix[0,0]*(cols+.5)+matrix[0,1]*(rows+.5)
    y = origin[1]+matrix[1,0]*(cols+.5)+matrix[1,1]*(rows+.5)
    to_wgs84 = geo.Transformer.from_crs(crs, 4326, always_xy=True)
    longitude, latitude = to_wgs84.transform(x, y, errcheck=True)
    _, _, distance = geo._GEOD.inv(np.full(x.shape,lon), np.full(x.shape,lat), longitude, latitude)
    require(np.isfinite(distance).all(), "Nonfinite cell coordinates")
    return distance <= 18.0


def validity(intake_dir, row):
    nodata = row["geometry"]["gdal_nodata"]
    if nodata is None:
        return None
    raw = (intake_dir / row["image_path"]).read_bytes()
    require(data_sha(raw) == row["png_sha256"], "Native pixel file changed")
    with Image.open(io.BytesIO(raw)) as image:
        pixels = np.asarray(image)
        require(image.mode == "L" and pixels.shape == (512,512) and data_sha(pixels.tobytes()) == row["decoded_sha256"],
                "Native raster values differ from image-only intake")
    return pixels != float(nodata)


def clean_labels(row):
    fields = []
    for key in LABELS:
        try:
            fields.append(float(row[key]))
        except (ValueError, TypeError):
            fields.append(float("nan"))
    lon, lat, heading, length, beam = fields
    valid = dict(position=np.isfinite(lon) and -180 <= lon <= 180 and np.isfinite(lat) and -90 < lat < 90,
                 heading=np.isfinite(heading) and 0 <= heading <= 359,
                 length=np.isfinite(length) and length > 0, beam=np.isfinite(beam) and beam > 0)
    return fields, valid


def physical_measure(geo, intake_dir, row, labels, stock, changes):
    lon, lat, heading, length, beam = labels
    crs, transform = grid_from_tags(row["geometry"])
    col, line = geo.ais_pixel_xy(lon,lat,crs,transform)
    if not (0 <= col < 512 and 0 <= line < 512):
        return {"reason":"reference_outside_chip"}
    near = within_radius(geo,lon,lat,crs,transform)
    anchor = None
    for position, det in enumerate(stock, 1):
        original = codec.decode(det["segmentation"]).astype(bool)
        if original[near].any():
            anchor = (position, original)
            break
    if anchor is None:
        return {"reason":"no_associated_native_mask"}
    position, original = anchor
    edits = changes.get(str(position), {})
    fixed_k = edits.get("fixed", {}).get("swaps", 0)
    natural_k = edits.get("free_natural", {}).get("swaps", 0)
    require(fixed_k == edits.get("free_matched_count", {}).get("swaps", 0), "Matched edit count differs at anchor")
    record = dict(associated=True, fixed_k=fixed_k, natural_k=natural_k)
    valid = validity(intake_dir, row)
    cache, arms, invalid = {}, {}, {}
    for arm in ARMS:
        update = edits.get(arm)
        raster = codec.decode(update["rle"]).astype(bool) if update else original
        cache_key = raster.tobytes()
        if cache_key not in cache:
            if raster[0,:].any() or raster[-1,:].any() or raster[:,0].any() or raster[:,-1].any():
                cache[cache_key] = "edge_contact"
            elif valid is not None and np.any(raster & ~valid):
                cache[cache_key] = "explicit_nodata"
            else:
                cache[cache_key] = geo.heading_extents_m(raster,lon,lat,heading,crs,transform,valid)
        if isinstance(cache[cache_key], str):
            invalid[arm] = cache[cache_key]
        else:
            arms[arm] = cache[cache_key]
    return dict(**record, arms=arms, invalid=invalid, reference=(length, beam))


def summary(rows):
    cov = Counter(selected=len(rows))
    distinct = {k:set() for k in ("declared_length","declared_beam","declared_length_beam","true_heading","reported_vessel")}
    groups = defaultdict(list)
    fail_by_arm = {arm:Counter() for arm in ARMS}
    for row in rows:
        cov.update(row["counts"])
        for field, value in row.get("label_values", {}).items():
            distinct[field].add(value)
        record = row.get("record")
        if record is None:
            continue
        if "reason" in record:
            cov[record["reason"]] += 1
            continue
        cov["associated"] += 1
        for arm, reason in record["invalid"].items():
            fail_by_arm[arm][reason] += 1
        if record["invalid"]:
            cov["any_arm_measurement_failure"] += 1
        else:
            groups[row["pass_id"]].append(record)
            cov["all_arm_measured"] += 1
    arms = {}
    for arm in ARMS:
        bins = [[(r["arms"][arm], r["reference"]) for r in items if arm in r["arms"]] for items in groups.values()]
        bins = [np.asarray(v) for v in bins if v]
        if not bins:
            arms[arm] = dict(ships=0,groups=0)
            continue
        errors = [v[:,0] - v[:,1] for v in bins]
        arms[arm] = dict(ships=sum(map(len, bins)), groups=len(bins),
                         length_beam_mean_m=np.mean([v[:,0].mean(axis=0) for v in bins],axis=0).tolist(),
                         ais_length_beam_mean_m=np.mean([v[:,1].mean(axis=0) for v in bins],axis=0).tolist(),
                         length_beam_bias_m=np.mean([e.mean(axis=0) for e in errors],axis=0).tolist(),
                         length_beam_mae_m=np.mean([abs(e).mean(axis=0) for e in errors],axis=0).tolist(),
                         length_beam_ship_median_ae_m=np.median(abs(np.concatenate(errors)),axis=0).tolist())
    contrasts = {}
    for first, second in PAIRS:
        means, ships, positive_ships, positive_groups = [], 0, 0, 0
        for name in sorted(groups):
            paired = [r for r in groups[name] if first in r["arms"] and second in r["arms"]]
            if not paired:
                continue
            budget = "natural_k" if second == "free_natural" else "fixed_k"
            positive = sum(r[budget]>0 for r in paired)
            ships += len(paired)
            positive_ships += positive
            positive_groups += positive>0
            d = [np.abs(np.subtract(r["arms"][first],r["reference"])) -
                 np.abs(np.subtract(r["arms"][second],r["reference"])) for r in paired]
            means.append(np.mean(d,axis=0))
        contrast = dict(paired_ships=ships, paired_groups=len(means), positive_budget_ships=positive_ships,
                        positive_budget_groups=positive_groups, zero_budget_ships=ships-positive_ships)
        if means:
            values = np.asarray(means)
            draws = np.random.Generator(np.random.PCG64(5217)).integers(len(means),size=(4096,len(means)))
            contrast["length_beam_mae_reduction_m"] = values.mean(axis=0).tolist()
            contrast["length_beam_interval_95_m"] = np.percentile(values[draws].mean(axis=1),[2.5,97.5],axis=0).T.tolist()
        contrasts[first+" minus "+second] = contrast
    primary = contrasts["fixed minus free_matched_count"]
    gates = dict(exposure=primary["positive_budget_ships"]>=20 and primary["positive_budget_groups"]>=8,
                 length_mean_at_least_half_m=bool(primary["paired_ships"] and primary["length_beam_mae_reduction_m"][0]>=.5),
                 length_lower_positive=bool(primary["paired_ships"] and primary["length_beam_interval_95_m"][0][0]>0))
    return dict(coverage=cov, selected_passes=len({r["pass_id"] for r in rows}),
                distinct_valid_label_counts={k:len(v) for k,v in distinct.items()}, failure_by_arm=fail_by_arm,
                arm_measurements=arms, paired_contrasts=contrasts, gates=gates, passes_all_gates=all(gates.values()))


def closed_inputs(args):
    pinned(args.protocol, PROTOCOL)
    terminal = json.loads(args.replay_terminal.read_bytes())
    require(type(terminal["actual_exit"]) is int and terminal["actual_exit"] == 0 and terminal["output_footer_verified"] is True,
            "Observed successful replay exit and checked output footer required before AIS access")
    rr = json.loads(pinned(args.replay/"RESULT_SAFE.json",terminal["result_sha256"]))
    require(rr["status"] == "NASTAR_FROZEN_REPLAY_COMPLETE" and rr["independent_decisions_equal"] is True
            and rr["target_labels_opened"] is False and rr["optimizer_updates"] == 0 and rr["point_head_sha256"] == POINT_HEAD,
            "Only closed, label-free, independently reproduced point edits are eligible")
    cap = json.loads(pinned(args.capture/"RESULT_SAFE.json",rr["capture_result_sha256"]))
    require(cap["status"] == "NASTAR_FROZEN_CAPTURE_COMPLETE" and cap["target_labels_opened"] is False
            and cap["protocol_sha256"] == PROTOCOL and cap["checkpoint_sha256"] == CHECKPOINT
            and all(rr[k] == cap[k] for k in ("protocol_sha256","predictions_sha256","audit_sha256","checkpoint_sha256",
                                             "images","manifest_sha256","lock_sha256")), "Frozen capture and edit lineage mismatch")
    manifest = json.loads(pinned(args.intake/"intake.json",rr["manifest_sha256"]))
    lock = json.loads(pinned(args.intake/"cohort_lock.json",rr["lock_sha256"]))
    require(manifest["lock_sha256"] == rr["lock_sha256"] and manifest["ready"] == manifest["expected"] == cap["images"]
            and manifest["failed"] == 0 and len(lock["selected_passes"]) == 24
            and lock["central_directory_sha256"] == "f84628515acca6c258ef50e5d5a80f398ab63934166aa49f3d6df4a3fedf27cf",
            "Only the locked cohort in its original order is eligible")
    files = ((args.capture/"PREDICTIONS_PRIVATE.jsonl","predictions_sha256"),
             (args.capture/"CANDIDATE_AUDIT_PRIVATE.jsonl","audit_sha256"),(args.replay/"EDITS_PRIVATE.jsonl","edits_sha256"))
    streams = [[json.loads(line) for line in pinned(path,rr[key]).splitlines()] for path,key in files]
    blocks = []
    first = {}
    for index, (item,row,pred,audit,edits) in enumerate(zip(lock["items"],manifest["items"],*streams,strict=True),1):
        require(index == item["image_id"] == row["image_id"] == pred["image_id"] == audit["image_id"] == edits["image_id"]
                and row["status"] == "ready" and all(row[k] == item[k] for k in ("pass_id","product","patch_number"))
                and audit["source_image_name"] == Path(row["image_path"]).name and audit["original_hw"] == [512,512],
                "ROI streams must refer to the same selected chip")
        original = first.setdefault(row["decoded_sha256"],index)
        require(row["duplicate_of_image_id"] == (None if original == index else original), "Decoded duplicate identity mismatch")
        ranks = [c["native_rank"] for c in audit["candidates"]]
        require(ranks == pred["native_ranks"] == audit["native_ranks"] == sorted(set(ranks)) and all(1<=n<=100 for n in ranks),
                "Native compact ranks must retain the top-100 order")
        stock = pred["coco"]
        changes = edits["changes"]
        require(all(k.isdigit() and k == str(int(k)) and 1 <= int(k) <= len(stock) for k in changes), "Unknown edit rank")
        for position,(det,candidate) in enumerate(zip(stock,audit["candidates"],strict=True),1):
            base = det["segmentation"]
            area = int(codec.area(base))
            require(det["image_id"] == index and det["category_id"] == 1 and base["size"] == [512,512]
                    and det["score"] == candidate["score"] and area == candidate["mask_area"] and candidate["position"] == position,
                    "Native mask or candidate identity differs")
            updates = changes.get(str(position),{})
            require(set(updates) <= set(ARMS[1:]) and updates.get("fixed",{}).get("swaps",0) ==
                    updates.get("free_matched_count",{}).get("swaps",0), "Unknown arm or unequal fixed/matched budget")
            for change in updates.values():
                after, count = change["rle"], change["swaps"]
                require(set(change) == {"rle","swaps"} and after["size"] == [512,512] and 32<=area<1024
                        and type(count) is int and 1<=count<=min(8,area//10) and int(codec.area(after)) == area
                        and area-int(codec.area(codec.merge([base,after],intersect=True))) == count,
                        "Actual signed edit count or conserved area differs from replay")
        blocks.append((item,row,stock,changes))
    require(len(blocks) == cap["images"] and len(first) == manifest["unique_decoded_images"], "Incomplete image population")
    return blocks,rr,terminal["result_sha256"]


def load_ais(product, url, member, destination):
    """Hash and CRC-check the selected official product CSV before matching patch names."""
    path = destination/(product+".csv")
    if path.exists():
        raw = path.read_bytes()
    else:
        for attempt in range(3):
            try:
                with requests.get(url,timeout=(12,35),stream=True,allow_redirects=False,headers={"Accept-Encoding":"identity"}) as response:
                    response.raise_for_status()
                    require(response.status_code == 200, "Official CSV fetch redirected or failed")
                    raw = response.raw.read(member.file_size+1,decode_content=False)
                    require(len(raw) == member.file_size and zlib.crc32(raw) == member.CRC, "CSV release CRC or size mismatch")
                break
            except (requests.RequestException,OSError,ValueError):
                if attempt==2:
                    raise
                sleep(attempt+1)
        with path.open("xb") as output:
            output.write(raw)
    require(len(raw) == member.file_size and zlib.crc32(raw) == member.CRC, "Private CSV cache differs from official archive")
    return raw


def selected_labels(blocks, prepare, private):
    directory = {m.filename:m for m in prepare.pinned_directory()}
    by_product = defaultdict(list)
    for item, *_ in blocks:
        by_product[item["product"]].append(item)
    matched = {}
    product_hashes = []
    for product, items in by_product.items():
        require(prepare.PRODUCT.fullmatch(product), "Invalid locked product name")
        expected = prepare.ROOT+"Data/"+quote(product,safe="")+"/AIS.csv"
        require(all(item["ais_url"] == expected for item in items), "Official product CSV URL required")
        member = directory[f"{prepare.ACCESSION}/Data/{product}/AIS.csv"]
        raw = load_ais(product,expected,member,private)
        product_hashes.append((product,data_sha(raw)))
        lookup = defaultdict(list)
        # Author's converter resolves patch basenames with or without an extension.
        # Couple only exact native/display spellings within the selected product.
        aliases = {}
        for item in items:
            stem = item["ais_patch_stem"]
            for name in (stem,stem+".tif",stem+"_uint8",stem+"_uint8.tif"):
                require(name not in aliases, "Ambiguous selected product/patch alias")
                aliases[name] = stem
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
        require(reader.fieldnames and set(("Patch_name","MMSI", *LABELS)) <= set(reader.fieldnames), "Missing official AIS columns")
        for line in reader:
            parts = (line["Patch_name"] or "").strip().replace("\\","/").split("/")
            if any(part.startswith("wake_patches") for part in parts[:-1]):
                continue
            stem = aliases.get(parts[-1])
            if stem is not None:
                lookup[stem].append({key:line[key] for key in (*LABELS,"MMSI")})
        for item in items:
            hits = lookup[item["ais_patch_stem"]]
            matched[item["image_id"]] = (hits[0] if len(hits)==1 else None,len(hits))
    return matched,data_sha(json.dumps(sorted(product_hashes),separators=(",",":")).encode())


def run(args):
    blocks,receipt,replay_sha = closed_inputs(args)
    sys.path.append(str(args.geo_site))
    geo = module(Path(__file__).with_name("ais_geometry.py"),GEOMETRY)
    import pyproj
    require(pyproj.__version__ == "3.7.1" and pyproj.proj_version_str == "9.5.1", "Qualified geodesic implementation required")
    prepare = module(Path(__file__).with_name("prepare_nastar.py"),"ad53d9a20dc11edfd356b9c90e9633ba0affcbbe026deb0d4e38796cd4087731")
    args.output.mkdir(exist_ok=True)
    require(not (args.output/"RESULT_SAFE.json").exists(), "Choose a fresh score directory")
    cache = args.output/"ais_private"
    cache.mkdir(exist_ok=True)
    labels,csv_digest = selected_labels(blocks,prepare,cache)
    rows = []
    for item,row,stock,changes in blocks:
        label,hits = labels[item["image_id"]]
        counts = Counter()
        result = dict(pass_id=item["pass_id"],duplicate=row["duplicate_of_image_id"] is not None,counts=counts)
        rows.append(result)
        if label is None:
            counts["ambiguous_exact_patch_rows" if hits else "no_exact_patch_row"] += 1
            continue
        counts["exactly_one_patch_row"] += 1
        fields,valid = clean_labels(label)
        values = {}
        if valid["length"]:
            values["declared_length"] = fields[3]
        if valid["beam"]:
            values["declared_beam"] = fields[4]
        if valid["length"] and valid["beam"]:
            values["declared_length_beam"] = tuple(fields[3:])
        if valid["heading"]:
            values["true_heading"] = fields[2]
        identifier = str(label.get("MMSI","")).strip()
        if len(identifier) == 9 and identifier.isdecimal() and int(identifier)>0:
            values["reported_vessel"] = identifier
        result["label_values"] = values
        for key,v in valid.items():
            counts["valid_"+key] += int(v)
        if not all(valid.values()):
            counts["incomplete_physical_reference"] += 1
            continue
        counts["complete_physical_reference"] += 1
        result["record"] = physical_measure(geo,args.intake,row,fields,stock,changes)
    primary = summary([r for r in rows if not r["duplicate"]])
    return dict(status=STATUS,units="metres",arm_axes=["true_heading_length","across_heading_beam"],
                source_head_sha256=POINT_HEAD,checkpoint_sha256=CHECKPOINT,protocol_sha256=PROTOCOL,
                lock_sha256=receipt["lock_sha256"],intake_sha256=receipt["manifest_sha256"],
                capture_result_sha256=receipt["capture_result_sha256"],replay_result_sha256=replay_sha,
                replay_terminal_sha256=data_sha(args.replay_terminal.read_bytes()),edits_sha256=receipt["edits_sha256"],
                selected_product_csv_hash_aggregate=csv_digest,geometry_sha256=GEOMETRY,
                row_join="Unique exact original or uint8 ship-patch basename/stem inside the locked product CSV",
                roster_selected=len(rows),exact_decoded_duplicates=sum(r["duplicate"] for r in rows),
                primary_distinct_images=primary,all_selected_sensitivity=summary(rows),
                full_primary_criterion=primary["passes_all_gates"],model_forwards=0,optimizer_updates=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for key in ("code","intake","capture","replay","replay-terminal","protocol","geo-site","output"):
        parser.add_argument("--"+key,required=True,type=Path)
    args = parser.parse_args()
    started = perf_counter()
    result = run(args)
    result.update(elapsed_seconds=perf_counter()-started,worker_sha256=data_sha(Path(__file__).read_bytes()))
    raw = (json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    with (args.output/"RESULT_SAFE.json").open("xb") as stream:
        stream.write(raw)
    print(json.dumps(dict(status=STATUS,result_sha256=data_sha(raw))),flush=True)


if __name__ == "__main__":
    main()
