"""Read-only, reproducible audit of ten named Umbra public port directories.

Only S3 ListObjectsV2, small STAC JSON GETs and six asset HEADs are used.
The public S3 listing is authoritative for availability: STAC asset hrefs can
refer to Umbra's private production bucket and are not treated as public URLs.
"""

from __future__ import annotations

import concurrent.futures
import datetime as dt
import hashlib
import itertools
import json
import re
import sys
import threading
import time
import urllib.parse
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import requests


BUCKET = "umbra-open-data-catalog"
S3 = f"https://{BUCKET}.s3.us-west-2.amazonaws.com"
PREFIX = "sar-data/tasks/"
PORTS = (
    "Port of Antwerp, Belgium",
    "Port of Busan, South Korea",
    "Port of Hamburg, Germany",
    "Port of Hong Kong, Hong Kong",
    "Port of Jebel Ali, United Arab Emirates",
    "Port of Long Beach, California, United States",
    "Port of Newark (NY/NJ), United States",
    "Port of Rotterdam, Netherlands",
    "Port of Savannah, Georgia, United States",
    "Port of Singapore, Singapore",
)
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
UUID = re.compile(r"^[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}$")
TLS = threading.local()


def session() -> requests.Session:
    if not hasattr(TLS, "session"):
        TLS.session = requests.Session()
        TLS.session.trust_env = False  # no workstation proxy needed
    return TLS.session


def request(method: str, url: str, **kwargs) -> requests.Response:
    err = None
    for attempt in range(3):
        try:
            r = session().request(method, url, timeout=(12, 35), **kwargs)
            r.raise_for_status()
            return r
        except (requests.RequestException, TimeoutError) as exc:
            err = exc
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"{method} {url}: {err}")


def public_url(key: str) -> str:
    return f"{S3}/{urllib.parse.quote(key, safe='/')}"


def list_port(port: str) -> list[dict]:
    pref = PREFIX + port + "/"
    token = None
    rows = []
    while True:
        params = {"list-type": "2", "prefix": pref, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        root = ET.fromstring(request("GET", S3, params=params).content)
        for obj in root.findall("s:Contents", NS):
            rows.append({
                "key": obj.findtext("s:Key", namespaces=NS),
                "size": int(obj.findtext("s:Size", namespaces=NS) or "0"),
            })
        if root.findtext("s:IsTruncated", namespaces=NS) != "true":
            break
        token = root.findtext("s:NextContinuationToken", namespaces=NS)
        if not token:
            raise RuntimeError(f"Truncated without continuation: {port}")
    return rows


def classify(name: str) -> str:
    if name.endswith(".stac.v2.json"):
        return "STAC"
    for product in ("GEC", "SICD", "SIDD", "CPHD", "CSI", "DI"):
        if re.search(rf"_{product}(?:_|\.)", name):
            return product
    return "OTHER"


def item_for_listing(row: tuple[str, str]) -> dict:
    port, key = row
    raw = request("GET", public_url(key)).content
    item = json.loads(raw)
    p = item.get("properties", {})
    return {
        "port": port,
        "key": key,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "stac_id": item.get("id"),
        "task_id": p.get("umbra:task_id"),
        "collect_id": p.get("umbra:collect_id"),
        "datetime": p.get("datetime"),
        "start_datetime": p.get("start_datetime"),
        "end_datetime": p.get("end_datetime"),
        "platform": p.get("platform"),
        "instrument_mode": p.get("sar:instrument_mode"),
        "product_type": p.get("sar:product_type"),
        "frequency_band": p.get("sar:frequency_band"),
        "polarizations": p.get("sar:polarizations"),
        "looks_azimuth": p.get("sar:looks_azimuth"),
        "looks_range": p.get("sar:looks_range"),
        "resolution_azimuth_m": p.get("sar:resolution_azimuth"),
        "resolution_range_m": p.get("sar:resolution_range"),
        "orbit_state": p.get("sat:orbit_state"),
        "observation_direction": p.get("sar:observation_direction"),
        "bbox": item.get("bbox"),
    }


def head_asset(row: tuple[str, str, str]) -> dict:
    port, product, key = row
    r = request("HEAD", public_url(key), allow_redirects=True)
    return {
        "port": port,
        "product": product,
        "key": key,
        "status": r.status_code,
        "content_length": int(r.headers.get("content-length", "-1")),
        "content_type": r.headers.get("content-type"),
        "accept_ranges": r.headers.get("accept-ranges"),
    }


def main() -> None:
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    out = Path("research_round_20260920/results/umbra_port_catalog_independent_audit_20260925.json")
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        listed = dict(zip(PORTS, pool.map(list_port, PORTS)))
    records = []
    stac_rows = []
    asset_candidates = []
    for port in PORTS:
        for obj in listed[port]:
            key = obj["key"]
            relative = key[len(PREFIX + port + "/"):]
            parts = relative.split("/")
            rec = dict(obj, port=port, relative=relative, product=classify(parts[-1]))
            rec["task_path"] = parts[0] if len(parts) >= 3 else None
            rec["acquisition_path"] = parts[1] if len(parts) >= 3 else None
            records.append(rec)
            if rec["product"] == "STAC":
                stac_rows.append((port, key))
            elif rec["product"] in ("GEC", "SICD"):
                asset_candidates.append((port, rec["product"], key))
    # Every listed small STAC JSON is read. This prevents a task UUID from
    # being silently promoted to a distinct collect without metadata evidence.
    with concurrent.futures.ThreadPoolExecutor(max_workers=14) as pool:
        items = list(pool.map(item_for_listing, stac_rows))
    by_parent = defaultdict(list)
    for r in records:
        by_parent[r["key"].rsplit("/", 1)[0]].append(r)
    for item in items:
        parent = item["key"].rsplit("/", 1)[0]
        item["public_assets"] = {
            r["product"]: {"key": r["key"], "size": r["size"]}
            for r in by_parent[parent]
            if r["product"] in ("GEC", "SICD")
        }
    # One deterministic pair for each of three ports, six HTTP HEADs total.
    checks = []
    for port in (PORTS[0], PORTS[7], PORTS[9]):
        for product in ("GEC", "SICD"):
            choices = sorted(k for p, kind, k in asset_candidates if p == port and kind == product)
            if choices:
                checks.append((port, product, choices[0]))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        head = list(pool.map(head_asset, checks))
    counts = {}
    for port in PORTS:
        rr = [r for r in records if r["port"] == port]
        ii = [i for i in items if i["port"] == port]
        folders = set((r["task_path"], r["acquisition_path"]) for r in rr if r["task_path"])
        by_folder = defaultdict(set)
        for r in rr:
            if r["task_path"]:
                by_folder[(r["task_path"], r["acquisition_path"])].add(r["product"])
        pairs = sum({"GEC", "SICD"} <= s for s in by_folder.values())
        dtimes = sorted(i["datetime"] for i in ii if i["datetime"])
        counts[port] = {
            "listed_keys": len(rr),
            "listed_bytes": sum(r["size"] for r in rr),
            "task_paths": len({r["task_path"] for r in rr if r["task_path"]}),
            "acquisition_paths": len(folders),
            "stac_items": len(ii),
            "distinct_task_ids": len({i["task_id"] for i in ii if i["task_id"]}),
            "distinct_collect_ids": len({i["collect_id"] for i in ii if i["collect_id"]}),
            "distinct_stac_ids": len({i["stac_id"] for i in ii if i["stac_id"]}),
            "gec_sicd_same_folder_pairs": pairs,
            "product_counts": dict(Counter(r["product"] for r in rr)),
            "first_datetime": dtimes[0] if dtimes else None,
            "last_datetime": dtimes[-1] if dtimes else None,
            "unique_utc_dates": len({s[:10] for s in dtimes}),
            "platform_counts": dict(Counter(i["platform"] for i in ii)),
            "mode_counts": dict(Counter(i["instrument_mode"] for i in ii)),
        }
    all_task_ids = [i["task_id"] for i in items if i["task_id"]]
    all_collect_ids = [i["collect_id"] for i in items if i["collect_id"]]
    all_stac_ids = [i["stac_id"] for i in items if i["stac_id"]]
    grouped_collect = defaultdict(set)
    grouped_task = defaultdict(set)
    for i in items:
        grouped_collect[i["collect_id"]].add((i["port"], i["task_id"], i["datetime"]))
        grouped_task[i["task_id"]].add((i["port"], i["collect_id"], i["datetime"]))
    summary = {
        "ports": len(PORTS),
        "listed_keys": len(records),
        "stac_items": len(items),
        "distinct_task_ids": len(set(all_task_ids)),
        "distinct_collect_ids": len(set(all_collect_ids)),
        "distinct_stac_ids": len(set(all_stac_ids)),
        "invalid_uuid_task_paths": sorted({r["task_path"] for r in records if r["task_path"] and not UUID.fullmatch(r["task_path"])}),
        "task_path_mismatches": [i["key"] for i in items if i["task_id"] != i["key"].split("/")[-3]],
        "collect_ids_with_multiple_port_task_time": {k: sorted(v) for k, v in grouped_collect.items() if len(v) > 1},
        "task_ids_with_multiple_port_collect_time": {k: sorted(v) for k, v in grouped_task.items() if len(v) > 1},
        "duplicate_stac_ids": sorted(k for k, n in Counter(all_stac_ids).items() if n > 1),
        "polarization_counts": dict(Counter(tuple(i["polarizations"] or []) for i in items)),
        "frequency_band_counts": dict(Counter(i["frequency_band"] for i in items)),
        "product_type_counts": dict(Counter(i["product_type"] for i in items)),
        "looks_azimuth_counts": dict(Counter(str(i["looks_azimuth"]) for i in items)),
        "looks_range_counts": dict(Counter(str(i["looks_range"]) for i in items)),
        "items_with_both_public_assets": sum({"GEC", "SICD"} <= set(i["public_assets"]) for i in items),
        "same_basename_gec_sicd_pairs": sum(
            i["public_assets"]["GEC"]["key"].removesuffix("_GEC.tif")
            == i["public_assets"]["SICD"]["key"].removesuffix("_SICD.nitf")
            for i in items if {"GEC", "SICD"} <= set(i["public_assets"])
        ),
    }
    # JSON keys cannot be tuples; convert once, explicitly.
    summary["polarization_counts"] = {"/".join(k): v for k, v in summary["polarization_counts"].items()}
    def bbox_intersects(a: list[float], b: list[float]) -> bool:
        return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]

    summary["cross_port_bbox_overlap_pairs"] = sum(
        bbox_intersects(a["bbox"], b["bbox"])
        for a, b in itertools.combinations(items, 2) if a["port"] != b["port"]
    )
    summary["same_port_bbox_overlap_pairs"] = sum(
        bbox_intersects(a["bbox"], b["bbox"])
        for a, b in itertools.combinations(items, 2) if a["port"] == b["port"]
    )
    summary["same_port_bbox_pairs"] = sum(
        a["port"] == b["port"] for a, b in itertools.combinations(items, 2)
    )
    nearest_seconds = []
    for port in PORTS:
        ordered = sorted((i for i in items if i["port"] == port), key=lambda i: i["datetime"])
        times = [dt.datetime.fromisoformat(i["datetime"].replace("Z", "+00:00")) for i in ordered]
        nearest_seconds.extend((b - a).total_seconds() for a, b in zip(times, times[1:]))
    summary["adjacent_same_port_lt_60s"] = sum(x < 60 for x in nearest_seconds)
    summary["adjacent_same_port_lt_1h"] = sum(x < 3600 for x in nearest_seconds)
    summary["adjacent_same_port_lt_24h"] = sum(x < 86400 for x in nearest_seconds)
    payload = {
        "schema": "umbra-port-catalog-independent-audit-v1",
        "started_utc": started,
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "public_bucket": S3,
        "stac_catalog": f"{S3}/stac/catalog.json",
        "fixed_port_directories": list(PORTS),
        "summary": summary,
        "by_port": counts,
        "head_checks": head,
        "items": sorted(items, key=lambda i: (i["port"], i["datetime"] or "", i["key"])),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(out)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(counts, indent=2, ensure_ascii=False))
    print(json.dumps(head, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Audit failed: {exc}", file=sys.stderr)
        raise
