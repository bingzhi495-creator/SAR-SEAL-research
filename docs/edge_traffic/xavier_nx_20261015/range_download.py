"""Bounded parallel HTTP-range download, with byte-range response verification."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import requests

parser = argparse.ArgumentParser()
parser.add_argument("url")
parser.add_argument("output")
parser.add_argument("--workers", type=int, default=10)
args = parser.parse_args()
path = Path(args.output)
path.parent.mkdir(parents=True, exist_ok=True)
response = requests.head(args.url, allow_redirects=True, timeout=20)
response.raise_for_status()
size = int(response.headers["Content-Length"])
chunk_size = 1024 * 1024
parts = path.parent / (path.name + ".parts")
parts.mkdir(exist_ok=True)


def download_range(start):
    end = min(start + chunk_size, size) - 1
    part = parts / str(start)
    if part.exists() and part.stat().st_size == end - start + 1:
        return
    for attempt in range(3):
        try:
            result = requests.get(args.url, headers={"Range": f"bytes={start}-{end}"}, timeout=(15, 40))
            expected_range = f"bytes {start}-{end}/{size}"
            if result.status_code != 206 or result.headers.get("Content-Range") != expected_range:
                raise RuntimeError(f"Incorrect range response {result.status_code}: {result.headers.get('Content-Range')}")
            if len(result.content) != end - start + 1:
                raise RuntimeError("Truncated byte range")
            part.write_bytes(result.content)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1)


with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
    futures = [executor.submit(download_range, start) for start in range(0, size, chunk_size)]
    for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
        future.result()
        print(f"Completed {index}/{len(futures)} verified ranges", flush=True)
temporary = path.with_suffix(path.suffix + ".complete")
digest = hashlib.sha256()
with temporary.open("wb") as target:
    for start in range(0, size, chunk_size):
        data = (parts / str(start)).read_bytes()
        target.write(data)
        digest.update(data)
if temporary.stat().st_size != size:
    raise RuntimeError("Assembled size mismatch")
temporary.replace(path)
manifest = {"file": path.name, "source": args.url, "bytes": size,
            "sha256": digest.hexdigest(), "download": "verified HTTP ranges",
            "publication": "local only"}
path.with_suffix(path.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps(manifest), flush=True)
