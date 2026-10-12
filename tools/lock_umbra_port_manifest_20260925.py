"""Freeze a metadata-only, port-disjoint Umbra VV collection sample.

This script reads the pinned independent public-catalog audit snapshot. It never
requests images or labels. The fixed sample is an acquisition manifest, not a
validated ship-instance benchmark.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "research_round_20260920/results/umbra_port_catalog_independent_audit_20260925.json"
OUTPUT = ROOT / "research_round_20260920/results/umbra_port_prospective_manifest_20260925.json"
SOURCE_SHA256 = "24aab51c1c9f25eca464af93fa9935f23a989970bb6ec1a02a1e54faf869a183"
PER_PORT = 12

# Port-level allocation was fixed before reading images, labels, or model scores.
GROUPS = {
    "train": [
        "Port of Antwerp, Belgium",
        "Port of Jebel Ali, United Arab Emirates",
        "Port of Long Beach, California, United States",
        "Port of Newark (NY/NJ), United States",
        "Port of Rotterdam, Netherlands",
        "Port of Singapore, Singapore",
    ],
    "dev": [
        "Port of Hamburg, Germany",
        "Port of Hong Kong, Hong Kong",
    ],
    "sealed_test": [
        "Port of Busan, South Korea",
        "Port of Savannah, Georgia, United States",
    ],
}


def parse_utc(value: str) -> dt.datetime:
    timestamp = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError(f"Timestamp has no timezone: {value}")
    return timestamp.astimezone(dt.timezone.utc)


def evenly_spaced_indices(n: int, k: int) -> list[int]:
    """Return k distinct endpoint-inclusive nearest-grid indices, halves up."""
    if n < k or k < 2:
        raise ValueError(f"Need n >= k >= 2, got n={n}, k={k}")
    # round(i * (n - 1) / (k - 1)) with exact integer half-up arithmetic.
    denominator = 2 * (k - 1)
    indices = [(2 * i * (n - 1) + (k - 1)) // denominator for i in range(k)]
    if len(set(indices)) != k or indices[0] != 0 or indices[-1] != n - 1:
        raise AssertionError(f"Non-distinct or non-endpoint indices: {indices}")
    return indices


def build(source: dict) -> dict:
    if source.get("schema") != "umbra-port-catalog-independent-audit-v1":
        raise ValueError("Unexpected source schema")
    if source.get("summary", {}).get("stac_items") != 496:
        raise ValueError("Unexpected source item count")
    all_ports = [port for ports in GROUPS.values() for port in ports]
    if len(all_ports) != len(set(all_ports)) or set(all_ports) != set(source["fixed_port_directories"]):
        raise ValueError("Port allocation must cover ten source ports exactly once")

    items_by_port: dict[str, list[dict]] = defaultdict(list)
    for item in source["items"]:
        if item["polarizations"] == ["VV"]:
            if item["instrument_mode"] != "SPOTLIGHT" or item["frequency_band"] != "X":
                raise ValueError(f"Unexpected VV acquisition mode: {item['collect_id']}")
            for kind in ("GEC", "SICD"):
                asset = item["public_assets"][kind]
                if not asset["key"] or asset["size"] <= 0:
                    raise ValueError(f"Missing public asset listing: {item['collect_id']} {kind}")
            items_by_port[item["port"]].append(item)

    rows = []
    per_port = []
    by_group = {}
    for group, ports in GROUPS.items():
        group_rows = []
        for port in ports:
            candidates = items_by_port[port]
            by_date: dict[str, list[dict]] = defaultdict(list)
            for item in candidates:
                by_date[parse_utc(item["datetime"]).date().isoformat()].append(item)
            dates = sorted(by_date)
            indices = evenly_spaced_indices(len(dates), PER_PORT)
            port_rows = []
            for rank, index in enumerate(indices):
                date = dates[index]
                # A repeated UTC date is resolved before any pixels: earliest UTC
                # timestamp, then collect ID, then STAC ID, all ascending.
                item = min(
                    by_date[date],
                    key=lambda x: (parse_utc(x["datetime"]), x["collect_id"], x["stac_id"]),
                )
                row = {
                    "group": group,
                    "port": port,
                    "port_date_rank": rank,
                    "vv_date_index_0based": index,
                    "utc_date": date,
                    "datetime": item["datetime"],
                    "collect_id": item["collect_id"],
                    "task_id": item["task_id"],
                    "stac_id": item["stac_id"],
                    "stac_key": item["key"],
                    "stac_sha256": item["sha256"],
                    "platform": item["platform"],
                    "polarizations": item["polarizations"],
                    "instrument_mode": item["instrument_mode"],
                    "frequency_band": item["frequency_band"],
                    "orbit_state": item["orbit_state"],
                    "observation_direction": item["observation_direction"],
                    "looks_azimuth": item["looks_azimuth"],
                    "looks_range": item["looks_range"],
                    "resolution_azimuth_m": item["resolution_azimuth_m"],
                    "resolution_range_m": item["resolution_range_m"],
                    "bbox": item["bbox"],
                    "public_assets": item["public_assets"],
                }
                port_rows.append(row)
            if len({row["utc_date"] for row in port_rows}) != PER_PORT:
                raise AssertionError(f"Repeated selected date in {port}")
            rows.extend(port_rows)
            group_rows.extend(port_rows)
            per_port.append({
                "group": group,
                "port": port,
                "candidate_vv_collects": len(candidates),
                "candidate_vv_utc_dates": len(dates),
                "selected_collects": len(port_rows),
                "selected_first_utc_date": port_rows[0]["utc_date"],
                "selected_last_utc_date": port_rows[-1]["utc_date"],
                "listed_gec_bytes": sum(row["public_assets"]["GEC"]["size"] for row in port_rows),
                "listed_sicd_bytes": sum(row["public_assets"]["SICD"]["size"] for row in port_rows),
            })
        by_group[group] = {
            "ports": len(ports),
            "selected_collects": len(group_rows),
            "listed_gec_bytes": sum(row["public_assets"]["GEC"]["size"] for row in group_rows),
            "listed_sicd_bytes": sum(row["public_assets"]["SICD"]["size"] for row in group_rows),
        }

    if len(rows) != 120 or len({row["collect_id"] for row in rows}) != 120:
        raise AssertionError("Expected 120 distinct collect IDs")
    if len({row["task_id"] for row in rows}) != 120:
        raise AssertionError("Unexpected same-task close acquisition in frozen sample")
    for group, expected in (("train", 72), ("dev", 24), ("sealed_test", 24)):
        if by_group[group]["selected_collects"] != expected:
            raise AssertionError(f"Unexpected {group} sample count")

    return {
        "schema": "umbra-port-prospective-metadata-manifest-v1",
        "source_snapshot_path": SOURCE.relative_to(ROOT).as_posix(),
        "source_snapshot_sha256": SOURCE_SHA256,
        "source_snapshot_finished_utc": source["finished_utc"],
        "basis": "Public S3 listing and STAC metadata only; no pixels, masks, or AP inspected",
        "unit": "Unique umbra:collect_id; task ID is retained for related-acquisition audit",
        "port_groups": GROUPS,
        "eligibility": "Exact VV polarization, X band, SPOTLIGHT, positive listed GEC+SICD sizes",
        "date_sampling": "Within each port filter eligible VV; sort distinct UTC dates; choose 12 indices round_half_up(i*(N-1)/11), i=0..11; on each selected date choose earliest UTC datetime, then collect_id, then STAC id ascending",
        "replacement_rule": "No silent substitutions. Log inaccessible/corrupt/missing or annotation failures and assess feasibility against the original 120 IDs; any amended cohort requires a new preregistered manifest before model evaluation.",
        "truth_status": "No ship-instance masks exist in this manifest; sealed-test labels must be made and held by an independent coordinator",
        "per_port": per_port,
        "by_group": by_group,
        "selected": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Compare regenerated bytes with the committed manifest")
    args = parser.parse_args()
    raw = SOURCE.read_bytes()
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != SOURCE_SHA256:
        raise ValueError(f"Source snapshot changed: {actual_sha256} != {SOURCE_SHA256}")
    manifest = build(json.loads(raw))
    encoded = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    if args.check:
        if OUTPUT.read_bytes() != encoded:
            raise AssertionError("Manifest differs from pinned source/rule output")
        print(f"Manifest reproduced: {len(manifest['selected'])} collect IDs, SHA-256 {hashlib.sha256(encoded).hexdigest()}")
    else:
        OUTPUT.write_bytes(encoded)
        print(f"Manifest written: {OUTPUT}, {len(manifest['selected'])} collect IDs, SHA-256 {hashlib.sha256(encoded).hexdigest()}")


if __name__ == "__main__":
    main()
