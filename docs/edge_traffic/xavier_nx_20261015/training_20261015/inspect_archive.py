"""Inspect the official public Kaggle ZIP via verified HTTP ranges.

Signed redirect URLs are held in memory and never printed or published.
"""
import io
import json
from pathlib import Path
import struct
import time
import zipfile
import requests

ROOT = Path(__file__).resolve().parent
PUBLIC_URL = "https://www.kaggle.com/api/v1/datasets/download/starsw/intersection-flow-5k?datasetVersionNumber=4"


class RemoteArchive(io.RawIOBase):
    def __init__(self):
        self.session = requests.Session()
        with self.session.get(PUBLIC_URL, stream=True, timeout=(20, 30)) as response:
            response.raise_for_status()
            if response.headers.get("Content-Type", "").split(";")[0] != "application/zip":
                raise RuntimeError("Dataset endpoint did not return a ZIP archive")
            self.url = response.url
            self.size = int(response.headers["Content-Length"])
        self.position = 0
        self.requests_made = 0
        self.bytes_received = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        return self.position

    def read(self, size=-1):
        if size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        if size == 0:
            return b""
        if size > 32 * 1024 * 1024:
            raise RuntimeError("Unexpected large unbounded archive read")
        start, end = self.position, self.position + size - 1
        for attempt in range(3):
            try:
                r = self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=(20, 60))
                if r.status_code != 206 or r.headers.get("Content-Range") != f"bytes {start}-{end}/{self.size}":
                    raise RuntimeError(f"Incorrect byte-range response: HTTP {r.status_code}")
                data = r.content
                if len(data) != size:
                    raise RuntimeError("Truncated archive range")
                self.position += size
                self.requests_made += 1
                self.bytes_received += size
                return data
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1)


if __name__ == "__main__":
    archive = RemoteArchive()
    with zipfile.ZipFile(archive) as z:
        rows = [{"name": i.filename, "bytes": i.file_size, "compressed_bytes": i.compress_size,
                 "offset": i.header_offset, "crc32": i.CRC, "compression": i.compress_type}
                for i in z.infolist()]
        (ROOT / "archive_inventory.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(json.dumps({"archive_bytes": archive.size, "entries": len(rows),
                          "uncompressed_bytes": sum(r["bytes"] for r in rows)}, indent=2), flush=True)
        for basename in ["README_zh.md", "classes.txt", "intersection.yaml"]:
            matches = [i for i in z.namelist() if i.endswith("/" + basename) or i == basename]
            for name in matches[:1]:
                content = z.read(name).decode("utf-8-sig")
                (ROOT / ("source_" + basename)).write_text(content, encoding="utf-8")
                print(basename, content[:8500], flush=True)
        for split in ["train", "val", "test"]:
            images = [r for r in rows if "/images/" + split + "/" in r["name"]]
            labels = [r for r in rows if "/labels/" + split + "/" in r["name"]]
            print(split, "images", len(images), "labels", len(labels),
                  "examples", [r["name"] for r in images[:4]], flush=True)
