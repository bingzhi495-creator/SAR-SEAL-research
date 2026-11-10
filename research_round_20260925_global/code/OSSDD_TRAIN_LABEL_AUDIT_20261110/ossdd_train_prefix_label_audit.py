from collections import Counter
from hashlib import sha256
from io import BytesIO
from itertools import pairwise
import json
import tarfile

import numpy as np
from PIL import Image, ImageDraw

revision = "46cee3d9e4ff8b4716e2ea5bbe2934e13cb36460"
member_path = "webdataset/train/train-000000.tar"
size = 526489600
length = 16 * 2**20
uri = f"https://huggingface.co/datasets/sylviaHoch/OpenSARShip-Ship-Detection-Dataset/resolve/{revision}/{member_path}?download=true"
import subprocess

download = f"""
$ErrorActionPreference='Stop'
$client=[Net.Http.HttpClient]::new()
$cancel=[Threading.CancellationTokenSource]::new([TimeSpan]::FromSeconds(60))
try {{
    $q=[Net.Http.HttpRequestMessage]::new([Net.Http.HttpMethod]::Get,'{uri}')
    $q.Headers.Range=[Net.Http.Headers.RangeHeaderValue]::new(0,{length-1})
    $r=$client.Send($q,[Net.Http.HttpCompletionOption]::ResponseHeadersRead,$cancel.Token)
    $contentRange=[string]$r.Content.Headers.ContentRange
    if([int]$r.StatusCode -ne 206 -or $contentRange -ne 'bytes 0-{length-1}/{size}' -or $r.Content.Headers.ContentLength -ne {length}) {{
        throw "Exact bounded HTTP range unavailable: $([int]$r.StatusCode), $contentRange"
    }}
    $stream=$r.Content.ReadAsStream($cancel.Token)
    $stdout=[Console]::OpenStandardOutput()
    $remaining={length}
    $buffer=[byte[]]::new(65536)
    while($remaining -gt 0) {{
        $read=$stream.ReadAsync($buffer,0,[Math]::Min($remaining,$buffer.Length),$cancel.Token).GetAwaiter().GetResult()
        if($read -le 0){{throw 'Short response'}}
        $stdout.Write($buffer,0,$read)
        $remaining-=$read
    }}
    if($stream.ReadAsync($buffer,0,1,$cancel.Token).GetAwaiter().GetResult() -ne 0){{throw 'Oversized response'}}
    $stdout.Flush()
    $r.Dispose()
    $q.Dispose()
}} finally {{
    $cancel.Dispose()
    $client.Dispose()
}}
"""
downloaded = subprocess.run(["pwsh", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", download],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=65)
if downloaded.returncode != 0:
    raise RuntimeError(downloaded.stderr.decode("utf-8", errors="replace"))
raw = downloaded.stdout
content_range = f"bytes 0-{length-1}/{size}"
if len(raw) != length:
    raise ValueError("Unexpected prefix size")

suffixes = {"vv.tif", "vh.tif", "mask.tif", "meta.json", "aabb.txt", "rbb.txt"}
groups = {}
with tarfile.open(fileobj=BytesIO(raw), mode="r:") as archive:
    for member in archive:
        if member.offset_data + member.size > len(raw):
            break
        matches = [value for value in suffixes if member.name.endswith("." + value)]
        if not member.isfile() or len(matches) != 1:
            raise ValueError(f"Unexpected train shard entry type={member.type!r}, suffix={member.name[-24:]!r}")
        suffix = matches[0]
        key = member.name[:-(1 + len(suffix))]
        fields = groups.setdefault(key, {})
        if suffix in fields:
            raise ValueError("Duplicate train shard field")
        fields[suffix] = archive.extractfile(member).read()

def parse_boxes(raw_bytes, values):
    lines = raw_bytes.decode("utf-8").strip().splitlines()
    rows = [row.split() for row in lines[1:]]
    if len(rows) != int(lines[0]) or any(len(row) != values+1 for row in rows):
        raise ValueError("Box list count or format")
    result = {int(row[0]): tuple(map(float, row[1:])) for row in rows}
    if len(result) != len(rows) or not all(np.isfinite(coords).all() for coords in result.values()):
        raise ValueError("Duplicate box identity or non-finite coordinates")
    return result

def center_mask(quads, dims, offset):
    yy, xx = np.ogrid[:dims[0], :dims[1]]
    result = np.zeros(dims, dtype=bool)
    for points in quads:
        x, y = np.array(points).T
        x, y = x - x.mean(), y - y.mean()
        cyclic = [points[i] for i in np.argsort(np.arctan2(y, x))]
        cross = [(bx-ax)*(yy+offset-ay) - (by-ay)*(xx+offset-ax)
                 for (ax, ay), (bx, by) in pairwise(cyclic + cyclic[:1])]
        result |= np.logical_and.reduce([v >= -1e-8 for v in cross])
    return result

def fill_mask(quads, dims):
    image = Image.new("1", (dims[1], dims[0]))
    draw = ImageDraw.Draw(image)
    for quad in quads:
        xy = np.array(quad).T
        center = xy.mean(axis=1)[:, None]
        cycle = np.argsort(np.arctan2(*(xy-center)[::-1]))
        draw.polygon([quad[i] for i in cycle], fill=1)
    return np.asarray(image, dtype=bool)

def compare(rebuilt, observed):
    both = int(np.count_nonzero(rebuilt & observed))
    union = int(np.count_nonzero(rebuilt | observed))
    diff = int(np.count_nonzero(rebuilt ^ observed))
    return {"intersection": both, "union": union, "xor": diff,
            "iou": both / union if union else 1.0}

observed_labels = Counter()
replays = {"pixel_center_integer": [], "pixel_center_half": [], "pillow_polygon_fill": []}
count = 0
ships = 0
matching_box_ids = 0
image_label_fields = Counter()
for fields in groups.values():
    if fields.keys() != suffixes:
        continue
    meta = json.loads(fields["meta.json"])
    if meta["split"] != "train":
        raise ValueError("Train only")
    aabb = parse_boxes(fields["aabb.txt"], 6)
    rbb = parse_boxes(fields["rbb.txt"], 10)
    quads = [list(zip(v[0:8:2], v[1:8:2])) for v in rbb.values()]
    with Image.open(BytesIO(fields["mask.tif"])) as handle:
        labels = np.array(handle)
        image_label_fields[str(handle.mode)] += 1
    values = np.unique(labels).tolist()
    if labels.shape != (700, 700) or any(value not in (0, 1, 255) for value in values):
        raise ValueError("Unexpected mask dimensions or values")
    observed = labels != 0
    observed_labels.update(values)
    replays["pixel_center_integer"].append(compare(center_mask(quads, labels.shape, 0.0), observed))
    replays["pixel_center_half"].append(compare(center_mask(quads, labels.shape, 0.5), observed))
    replays["pillow_polygon_fill"].append(compare(fill_mask(quads, labels.shape), observed))
    count += 1
    ships += len(rbb)
    matching_box_ids += aabb.keys() == rbb.keys()
if count == 0:
    raise ValueError("No complete author training samples")

def summary(rows):
    return {"exact": sum(r["xor"] == 0 for r in rows),
            "xor": sum(r["xor"] for r in rows),
            "mean_iou": sum(r["iou"] for r in rows)/len(rows),
            "minimum_iou": min(r["iou"] for r in rows)}
print(json.dumps({"status": "OSSDD_TRAIN_SHARD_PREFIX_LABEL_REPRESENTATION_AUDIT_V1",
                  "revision": revision, "member_path": member_path, "shard_size": size,
                  "http_range": content_range, "downloaded_bytes": len(raw),
                  "prefix_sha256": sha256(raw).hexdigest(), "complete_train_samples": count,
                  "incomplete_sample_prefixes": len(groups)-count, "rotated_box_rows": ships,
                  "samples_same_aabb_rbb_ids": matching_box_ids, "mask_value_sample_counts": dict(observed_labels),
                  "mask_mode_sample_counts": dict(image_label_fields),
                  "reconstructed_box_union": {key: summary(value) for key, value in replays.items()},
                  "decoded_sar_images": 0, "model_executed": False, "official_test_opened": False}, sort_keys=True))
