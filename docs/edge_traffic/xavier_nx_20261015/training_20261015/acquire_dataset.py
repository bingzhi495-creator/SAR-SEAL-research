"""Acquire official public dataset v4 members by HTTP ranges and verify CRC/size.

No archive duplication, no external executable, no signed URL is logged.
"""
import concurrent.futures
import json
import random
from pathlib import Path
import struct
import threading
import time
import zlib
import requests
from inspect_archive import RemoteArchive, ROOT, PUBLIC_URL


def acquire():
    rows = sorted(json.loads((ROOT / "archive_inventory.json").read_text()), key=lambda x: x["offset"])
    remote = RemoteArchive()
    wanted = []
    chosen = set()
    selection = {}
    rng = random.Random(20261015)
    for split, count in [("train", 1024), ("val", 128), ("test", 128)]:
        images = sorted(r["name"] for r in rows if "/images/" + split + "/" in r["name"])
        selected = sorted(rng.sample(images, count))
        selection[split] = selected
        for image in selected:
            chosen.add(image)
            chosen.add(image.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt")
    selection_file = ROOT / "pilot_selection.json"
    if selection_file.exists() and json.loads(selection_file.read_text()) != selection:
        raise RuntimeError("Refusing to change an existing pilot selection")
    selection_file.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    for i, row in enumerate(rows):
        name = row["name"]
        if name in chosen or name.rsplit("/", 1)[-1] in {
            "classes.txt", "intersection.yaml", "README.md", "README_zh.md", "test_coco.json"
        }:
            row["end"] = rows[i + 1]["offset"] - 1 if i + 1 < len(rows) else min(
                remote.size - 1, row["offset"] + row["compressed_bytes"] + len(name.encode("utf-8")) + 1024)
            wanted.append(row)
    groups = []
    for row in wanted:
        if not groups or row["offset"] - groups[-1][-1]["end"] > 4096 or row["end"] - groups[-1][0]["offset"] > 2 * 1024 * 1024:
            groups.append([])
        groups[-1].append(row)
    output = (ROOT / "dataset").resolve()
    output.mkdir(exist_ok=True)
    local = threading.local()
    completed = 0
    transferred = 0
    started = time.monotonic()

    def fetch(group):
        pending = [r for r in group if not (output / r["name"]).is_file() or (output / r["name"]).stat().st_size != r["bytes"]]
        if not pending:
            return len(group), 0
        if not hasattr(local, "session"):
            local.session = requests.Session()
        start, end = group[0]["offset"], group[-1]["end"]
        for attempt in range(4):
            try:
                response = local.session.get(remote.url, headers={"Range": f"bytes={start}-{end}"}, timeout=(20, 90))
                if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{remote.size}":
                    raise RuntimeError(f"Range response {response.status_code}")
                payload = response.content
                if len(payload) != end - start + 1:
                    raise RuntimeError("Truncated range")
                for row in pending:
                    offset = row["offset"] - start
                    if payload[offset:offset + 4] != b"PK\x03\x04":
                        raise RuntimeError("ZIP local header mismatch")
                    name_length, extra_length = struct.unpack_from("<HH", payload, offset + 26)
                    offset += 30 + name_length + extra_length
                    compressed = payload[offset:offset + row["compressed_bytes"]]
                    raw = zlib.decompress(compressed, -15) if row["compression"] == 8 else compressed
                    if len(raw) != row["bytes"] or zlib.crc32(raw) & 0xFFFFFFFF != row["crc32"]:
                        raise RuntimeError("Member CRC/size mismatch: " + row["name"])
                    target = (output / row["name"]).resolve()
                    if output not in target.parents:
                        raise RuntimeError("Archive member escapes destination")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_suffix(target.suffix + ".partial")
                    temporary.write_bytes(raw)
                    temporary.replace(target)
                return len(group), len(payload)
            except Exception as error:
                if attempt == 3:
                    raise RuntimeError(f"Acquisition failed near member {group[0]['name']}: {type(error).__name__}") from None
                print(f"Retry group near {group[0]['name']} attempt={attempt+1} error={type(error).__name__}", flush=True)
                time.sleep(1 + attempt)

    print(json.dumps({"status": "acquiring", "version": 4, "groups": len(groups), "members": len(wanted),
                      "compressed_GB": sum(g[-1]["end"] - g[0]["offset"] + 1 for g in groups) / 1e9}), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=48) as pool:
        futures = [pool.submit(fetch, g) for g in groups]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            count, size = future.result()
            completed += count
            transferred += size
            if index % 25 == 0 or index == len(groups):
                print(f"groups={index}/{len(groups)} members={completed}/{len(wanted)} received_GB={transferred/1e9:.3f} elapsed_s={time.monotonic()-started:.1f}", flush=True)
    summary = {"status": "downloaded_verified_crc", "source": PUBLIC_URL, "dataset_version": 4,
               "archive_bytes": remote.size, "downloaded_members": completed,
               "transferred_bytes": transferred, "elapsed_seconds": time.monotonic() - started,
               "pilot_counts": {key: len(value) for key, value in selection.items()},
               "method": "verified range and per-member ZIP CRC32/size; original image bytes retained"}
    (ROOT / "acquisition_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    acquire()
