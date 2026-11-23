"""Inventory immutable author tree metadata for the SAR Ship Gulf length reference."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
from hashlib import sha256
import gzip
from http.client import IncompleteRead, RemoteDisconnected
import json
from pathlib import Path
import re
from time import perf_counter, sleep
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request, urlopen

REPO = "mshaya/sar-ship-gulf"
REVISION = "820ae84a00677d535c4db315adffb129b0d4a835"
TREE_URL = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/chips"
RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/"
PILOT_SCENE = "S1A_IW_GRDH_1SDV_20240811T001831_20240811T001856_055158_06B8DB_B6FF"
PILOT_PASS = "055158-20240811"
PREFIX = "SAR-SHIP-GULF-20261123/v1"
LOCK = "6001a3d9be9cf5cccb71e4156603d91c2dec1858ba46816a8380f2d315272cbb"
STATUS = "GULF_AIS_FROZEN_CENSUS_COMPLETE"
SCENE = re.compile(r"(S1A)_IW_GRDH_1SDV_(\d{8}T\d{6})_(\d{8}T\d{6})_(\d{6})_([A-F0-9]{6})_([A-F0-9]{4})")
FILE = re.compile(r"(\d{9})_(\d{8}T\d{6})\.npz")
HEX64 = re.compile("[a-f0-9]{64}")
HEX40 = re.compile("[a-f0-9]{40}")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def encode(value):
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def digest(data):
    return sha256(data).hexdigest()


def dated(text):
    return datetime.strptime(text, "%Y%m%dT%H%M%S")


def tree_address(url):
    parsed = urlsplit(url)
    base = urlsplit(TREE_URL)
    query = parse_qs(parsed.query, strict_parsing=True)
    require((parsed.scheme, parsed.netloc, parsed.path) == (base.scheme, base.netloc, base.path)
            and not parsed.fragment and set(query) <= {"cursor", "recursive", "limit", "expand"}
            and query.get("recursive") == ["true"] and query.get("limit") == ["1000"]
            and all(len(v) == 1 for v in query.values()), "Unexpected repository, revision or pagination")
    return url


def page(url):
    for attempt in range(6):
        try:
            req = Request(tree_address(url), headers={"User-Agent":"scientific-dataset-metadata/1",
                                                     "Accept":"application/json","Accept-Encoding":"gzip"})
            with urlopen(req, timeout=25) as response:
                tree_address(response.geturl())
                require(response.status == 200, "Repository tree did not return 200")
                body = response.read(5_000_001)
                require(len(body) <= 5_000_000, "Oversize tree page")
                if response.headers.get("Content-Encoding") == "gzip":
                    body = gzip.decompress(body)
                data = json.loads(body)
                require(isinstance(data, list) and 0 < len(data) <= 1000, "Unexpected tree page structure")
                link = response.headers.get("Link", "")
                if link:
                    match = re.fullmatch(r'<([^>]+)>;\s*rel="next"', link)
                    require(match is not None, "Ambiguous repository pagination")
                    return data, tree_address(match.group(1))
                return data, None
        except HTTPError as error:
            if error.code not in (408, 425, 429, 500, 502, 503, 504) or attempt == 5:
                raise
            retry = error.headers.get("Retry-After", "")
            delay = min(60, int(retry)) if retry.isdecimal() else 2**attempt
        except (URLError, TimeoutError, ConnectionError, IncompleteRead, RemoteDisconnected):
            if attempt == 5:
                raise
            delay = 2**attempt
        sleep(delay)
    raise RuntimeError("Unavailable metadata page")


def census():
    url = TREE_URL + "?recursive=true&limit=1000"
    seen_urls, scenes, files, page_counts = set(), {}, {}, []
    invalid_names, seen_paths = [], set()
    while url:
        require(url not in seen_urls and len(seen_urls) < 100, "Cyclic or excessively long metadata listing")
        seen_urls.add(url)
        rows, url = page(url)
        page_counts.append(len(rows))
        for row in rows:
            path = row["path"]
            parts = path.split("/")
            require(path not in seen_paths and parts[0] == "chips" and len(parts) in (2,3) and HEX40.fullmatch(row["oid"]),
                    "Invalid path or Git object ID in author tree")
            seen_paths.add(path)
            scene = parts[1]
            m = SCENE.fullmatch(scene)
            require(m is not None, "Unexpected scene naming in author tree")
            if row["type"] == "directory":
                require(len(parts) == 2 and scene not in scenes, "Duplicate or nested scene directory")
                scenes[scene] = row["oid"]
                continue
            require(row["type"] == "file" and len(parts) == 3 and path not in files, "Duplicate or unexpected asset")
            name = FILE.fullmatch(parts[2])
            ptr = row.get("lfs", {})
            if name is None:
                invalid_names.append(path)
                continue
            try:
                dated(name[2])
            except ValueError:
                invalid_names.append(path)
                continue
            require(type(row["size"]) is int and row["size"] > 0
                    and row["size"] == ptr.get("size") and HEX64.fullmatch(ptr.get("oid","")),
                    "Nonconforming NPZ filename, size or LFS pointer")
            require(m[2] <= name[2] <= m[3], "Filename acquisition second is outside native SAFE product interval")
            files[path] = dict(path=path, scene=scene, file_sha256=ptr["oid"], size_bytes=row["size"],
                               git_blob_oid=row["oid"], mmsi=name[1], acquisition_time=name[2])
        print(json.dumps(dict(metadata_pages=len(page_counts), scene_directories=len(scenes), chip_files=len(files))),
              flush=True)
    require({r["scene"] for r in files.values()} <= set(scenes) and PILOT_SCENE in scenes,
            "Undocumented parent scene or missing pilot")
    return scenes, files, page_counts, sorted(invalid_names)

def group_id(scene):
    parsed = SCENE.fullmatch(scene)
    require(parsed is not None, "Invalid scene for acquisition grouping")
    return parsed[4] + "-" + parsed[2][:8]


def rank_key(domain, value):
    return digest(f"{PREFIX}|{domain}|{value}".encode("ascii")), value


def select(scenes, files):
    groups = defaultdict(list)
    scene_groups = defaultdict(list)
    for scene in sorted(scenes):
        scene_groups[group_id(scene)].append(scene)
    for record in files.values():
        groups[group_id(record["scene"])].append(record)
    available = set(scene_groups) - {PILOT_PASS}
    require(group_id(PILOT_SCENE) == PILOT_PASS and len(available) >= 64, "Pilot or pass eligibility differs")
    chosen = sorted(available, key=lambda v: rank_key("pass", v))[:64]
    used, items, passes = set(), [], []
    for pass_id in chosen:
        pass_used = set()
        skipped = Counter()
        candidates = sorted(groups[pass_id], key=lambda v: rank_key("chip", v["path"]))
        for row in candidates:
            if len(pass_used) == 4:
                break
            vessel = row["mmsi"]
            if vessel in used:
                skipped["duplicate_same_pass" if vessel in pass_used else "duplicate_prior_pass"] += 1
                continue
            used.add(vessel)
            pass_used.add(vessel)
            items.append(dict(image_id=len(items)+1, pass_id=pass_id, scene_id=row["scene"], path=row["path"],
                              url=RESOLVE + row["path"], file_sha256=row["file_sha256"], size_bytes=row["size_bytes"],
                              git_blob_oid=row["git_blob_oid"], mmsi=vessel, acquisition_time=row["acquisition_time"]))
        passes.append(dict(pass_id=pass_id, selected=len(pass_used), eligible_files=len(candidates),
                           scene_count=len(scene_groups[pass_id]), skipped_duplicates=dict(skipped)))
    require(len(set(v["path"] for v in items)) == len(used) == len(items)
            and set(p["pass_id"] for p in passes) == set(chosen)
            and all(0 <= p["selected"] <= 4 for p in passes), "Signed roster policy violated")
    return items, passes


def write(path, data):
    with path.open("xb") as f:
        require(f.write(data) == len(data), "Incomplete private roster or census write")
    return digest(data)


def run(args):
    start = perf_counter()
    require(not args.output.exists() and LOCK == args.protocol_sha256 == digest(args.protocol.read_bytes()),
            "Fresh census output and exact published lock required")
    scenes, files, pages, bad = census()
    groups = set(map(group_id, scenes))
    dates = {g.split("-")[1] for g in groups}
    require((len(scenes),len(dates),len(groups)) == (194,67,78), "Official scene/date/orbit directory census differs")
    items, passes = select(scenes, files)
    args.output.mkdir()
    catalog = [dict(type="scene",path="chips/"+p,git_tree_oid=scenes[p]) for p in sorted(scenes)]
    catalog.extend(dict(type="npz",**files[p]) for p in sorted(files))
    catalog.extend(dict(type="rejected_filename",path=p) for p in bad)
    catalog_sha = write(args.output / "CATALOG_PRIVATE.jsonl", b"".join(map(encode,catalog)))
    roster = dict(repo=REPO, revision=REVISION, protocol_sha256=LOCK, pilot_pass_excluded=PILOT_PASS,
                  catalog_sha256=catalog_sha, passes=passes, items=items)
    roster_sha = write(args.output / "ROSTER_PRIVATE.json", encode(roster))
    selected_pass_ids = {p["pass_id"] for p in passes}
    public = dict(scene_directories=len(scenes), calendar_dates=len(dates), absolute_orbit_date_groups=len(groups),
                  catalog_file_entries=len(files)+len(bad), valid_named_files=len(files), invalid_filenames=len(bad),
                  distinct_filename_mmsis=len({v["mmsi"] for v in files.values()}),
                  total_catalog_bytes=sum(v["size_bytes"] for v in files.values()),
                  pilot_excluded_scenes=sum(group_id(p)==PILOT_PASS for p in scenes),
                  pilot_excluded_chips=sum(group_id(r["scene"])==PILOT_PASS for r in files.values()),
                  candidate_passes=len(groups)-1, selected_passes=len(passes), selected_chips=len(items),
                  selected_mmsis=len({v["mmsi"] for v in items}), represented_passes=len({v["pass_id"] for v in items}),
                  selected_calendar_dates=len({g.split("-")[1] for g in selected_pass_ids}),
                  represented_scenes=len({v["scene_id"] for v in items}),
                  selected_npz_bytes=sum(v["size_bytes"] for v in items),
                  pass_yield_distribution=dict(sorted(Counter(v["selected"] for v in passes).items())),
                  skipped_duplicate_within_pass=sum(v["skipped_duplicates"].get("duplicate_same_pass",0) for v in passes),
                  skipped_duplicate_prior_pass=sum(v["skipped_duplicates"].get("duplicate_prior_pass",0) for v in passes))
    result = dict(status=STATUS, phase="census", protocol_sha256=LOCK, worker_sha256=digest(Path(__file__).read_bytes()),
                  repo=REPO, revision=REVISION, root_tree_api=TREE_URL, counts=public,
                  catalog_sha256=catalog_sha, roster_sha256=roster_sha, tree_pages=len(pages),
                  tree_page_entries=pages, target_labels_opened=False, npz_opened=False,
                  pixel_arrays_opened=False, model_forwards=0, optimizer_updates=0, seconds=perf_counter()-start)
    stamp = write(args.output / "RESULT_SAFE.json", encode(result))
    print(json.dumps(dict(status=STATUS,result_sha256=stamp)),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

