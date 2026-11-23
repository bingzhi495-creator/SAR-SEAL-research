"""Check whole-mask binary parity against the two closed, label-free Phoenix inferences."""
from collections import Counter
from hashlib import sha256
from io import BytesIO
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as codec


def require(ok, description):
    if not ok:
        raise ValueError(description)


def endpoint_stats(native, refined, coordinates, probabilities):
    coords = np.asarray(coordinates)
    q = np.asarray(probabilities, dtype=float)
    original, observed = (codec.decode(rle).ravel(order="C") for rle in (native, refined))
    require(q.shape == coords.shape and coords.ndim == 2 and coords.shape[1] == 2
            and coords.dtype.kind in "iu" and np.all((0 <= coords) & (coords < original.size))
            and not original[coords[:, 0]].any() and original[coords[:, 1]].all()
            and np.isfinite(q).all() and np.all((0 <= q) & (q <= 1)),
            "Invalid native signed endpoints or probabilities")
    return q.size, int(np.count_nonzero(observed[coords] != (q > .5))), int(np.count_nonzero(q == .5))


def read_pinned(path, expected):
    data = path.read_bytes()
    require(sha256(data).hexdigest() == expected, "Digest mismatch for " + path.name)
    return data


def closed(root, phase, expected):
    end = json.loads((root / ("observed_" + phase) / "TERMINAL.json").read_bytes())
    safe = json.loads(read_pinned(root / phase / "RESULT_SAFE.json", expected))
    require(type(end["actual_exit"]) is int and end["actual_exit"] == 0 and end["output_footer_verified"] is True
            and end["result_sha256"] == expected and end["phase"] == safe["phase"] == phase
            and end["worker_sha256"] == safe["worker_sha256"] and safe["target_labels_opened"] is False,
            "A successfully closed label-free " + phase + " phase is required")
    return safe


def by_position(entries, field):
    result = {e["position"]: e[field] for e in entries}
    require(len(result) == len(entries) and list(result) == sorted(result), "Duplicate or unordered original position")
    return result


def verify(whole_root, matched_root):
    whole_sha = "b5fe207b3e4f3d6c9caf391b6ed60c373abf4a0580256271343856116b3d0e5e"
    matched_sha = "ec061d588abcef68d3ced78296b3be5ca9b3f9ad93b3d1138ed90d5b6e6e7ddc"
    whole = closed(whole_root, "inference", whole_sha)
    matched = closed(matched_root, "inference", matched_sha)
    images = []
    for root, safe, request_sha in ((whole_root, whole, whole["request_result_sha256"]), (matched_root, matched,
            "08ba5d901c7c6d0b17a8b17cd8a3bb5f646139e66e2aae2459554cacb0fabd2a")):
        receipt = closed(root, "request", request_sha)
        require(receipt["request_sha256"] == safe["request_sha256"], "Cross-stage request digest mismatch")
        images.append(json.loads(read_pinned(root/"request/REQUEST_PRIVATE.json", safe["request_sha256"]))["images"])
    def stream(root, name, digest):
        return map(json.loads, read_pinned(root/"inference"/name, digest).splitlines())
    outputs = stream(whole_root, "WHOLE_MASKS_PRIVATE.jsonl", whole["masks_sha256"])
    endpoints = stream(matched_root, "ENDPOINT_PROBABILITIES_PRIVATE.jsonl", matched["endpoint_sha256"])
    counts = Counter()
    for ordinal, (full_image, old_image, new, old) in enumerate(zip(*images, outputs, endpoints, strict=True), 1):
        require(new["ordinal"] == ordinal and old_image["image_id"] == old["image_id"]
                and full_image["decoded_rgb_sha256"] == old_image["decoded_rgb_sha256"],
                "Images or ledger order differ between runs")
        name = f"image_{ordinal:03}.png"
        require(full_image["image_file"] == name, "Unexpected PNG name")
        data = read_pinned(whole_root/"request"/name, full_image["encoded_sha256"])
        with Image.open(BytesIO(data)) as f:
            rgb = np.asarray(f)
            require(f.mode == "RGB" and rgb.dtype == np.uint8 and rgb.shape == (512, 512, 3)
                    and sha256(rgb.tobytes()).hexdigest() == full_image["decoded_rgb_sha256"],
                    "Native decoded RGB changed")
        counts["images"] += 1
        original = by_position(full_image["candidates"], "segmentation")
        lvis, coco = (by_position(new[rule], "segmentation") for rule in ("lvis","coco"))
        sizes = {pos: int(codec.area(rle)) for pos, rle in original.items()}
        require(list(original) == list(range(1,1+len(original))) and list(lvis) == [p for p,a in sizes.items() if a > 0]
                and list(coco) == [p for p,a in sizes.items() if 100*a > 512*512], "Incorrect writer eligibility")
        for pos, mask in coco.items():
            counts["shared_policy_masks"] += 1
            counts["identical_shared_rles"] += int(mask == lvis[pos])
            counts["shared_policy_pixel_differences"] += int(np.count_nonzero(codec.decode(mask) != codec.decode(lvis[pos])))
        for candidate, pred in zip(old_image["candidates"], old["candidates"], strict=True):
            pos = candidate["position"]
            require(pos == pred["position"] and original[pos] == candidate["segmentation"],
                    "Original candidate or mask differs between runs")
            n, diffs, ties = endpoint_stats(original[pos], lvis[pos], candidate["coords"], pred["probabilities"])
            counts.update(matched_candidates=1, endpoint_decisions=n, endpoint_disagreements=diffs,
                          decisions_at_one_half=ties, full_empty_in_old_subset=int(codec.area(lvis[pos]) == 0))
    require(counts["images"] == 74 and counts["matched_candidates"] == 4107 and counts["endpoint_decisions"] == 259242
            and counts["shared_policy_masks"] == counts["identical_shared_rles"] == 172
            and counts["shared_policy_pixel_differences"] == counts["endpoint_disagreements"] == counts["decisions_at_one_half"] == 0
            and counts["full_empty_in_old_subset"] == matched["counts"]["full_output_empty"] == 2,
            "Historical binary parity not established")
    return dict(status="PHOENIX_RSDD74_BINARY_PARITY_VERIFIED", counts=dict(counts),
                whole_inference_result_sha256=whole_sha, matched_inference_result_sha256=matched_sha,
                whole_masks_sha256=whole["masks_sha256"], matched_endpoint_sha256=matched["endpoint_sha256"],
                worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), target_labels_opened=False, model_calls=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("whole-root", "matched-root", "output"):
        parser.add_argument("--"+flag, type=Path, required=True)
    args = parser.parse_args()
    summary = verify(args.whole_root, args.matched_root)
    raw = (json.dumps(summary, sort_keys=True, indent=2)+"\n").encode()
    with args.output.open("xb") as out:
        out.write(raw)
    print(json.dumps({"status":summary["status"], "result_sha256":sha256(raw).hexdigest()}))
