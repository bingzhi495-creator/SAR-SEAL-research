"""Post hoc count of near-duplicate rivals among the unchanged failed associations."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from pycocotools import mask

RESULT_SHA = "582d11542c3ec777fb2309493459045d51b51edc500a888970ccb03d2b57b3ac"


def checked(path, digest):
    data = path.read_bytes()
    if sha256(data).hexdigest() != digest:
        raise ValueError("Changed closed artifact")
    return data


def rival_counts(anchor, donor):
    if not anchor or not donor:
        return Counter()
    a, b = ([row["rle"] for row in block] for block in (anchor, donor))
    overlaps = mask.iou(a,b,[0]*len(b))
    same_a = mask.iou(a,a,[0]*len(a))
    same_b = mask.iou(b,b,[0]*len(b))
    ar = np.argsort(-overlaps, axis=1, kind="stable")
    br = np.argsort(-overlaps, axis=0, kind="stable")
    out = Counter()
    for i, point in enumerate(anchor):
        if not 32 <= point["area"] < 1024:
            continue
        j = ar[i,0]
        best = overlaps[i,j]
        if best < .50:
            continue
        other = br[0,j]
        if other != i:
            out["not_reciprocal"] += 1
            for pct in (85,90):
                if same_a[i,other] >= pct/100:
                    out[f"nonreciprocal_anchor_rival_iou_ge_{pct}"] += 1
            continue
        row_rival = ar[i,1] if len(b) > 1 else None
        col_rival = br[1,j] if len(a) > 1 else None
        row_bad = row_rival is not None and best-overlaps[i,row_rival]+1e-12 < .10
        col_bad = col_rival is not None and best-overlaps[col_rival,j]+1e-12 < .10
        if not (row_bad or col_bad):
            continue
        out["ambiguous"] += 1
        for pct in (85,90):
            if ((row_bad and same_b[j,row_rival] >= pct/100)
                    or (col_bad and same_a[i,col_rival] >= pct/100)):
                out[f"ambiguous_either_rival_iou_ge_{pct}"] += 1
    return out


def main(args):
    result = json.loads(checked(args.closed / "RESULT_SAFE.json",RESULT_SHA))
    by_seed = {a["seed"]: [json.loads(line) for line in checked(
        args.closed / str(a["seed"]) / "CANDIDATES_PRIVATE.jsonl",a["prediction_sha256"]).splitlines()]
        for a in result["arms"]}
    counts = {}
    for seed in (5201,5203):
        summed = Counter()
        for anchor, donor in zip(by_seed[5202],by_seed[seed],strict=True):
            if anchor["rgb_sha256"] != donor["rgb_sha256"]:
                raise ValueError("Different images")
            summed.update(rival_counts(anchor["candidates"],donor["candidates"]))
        expected = result["coverage"]["per_donor"][str(seed)]
        for field in ("not_reciprocal","ambiguous"):
            if summed[field] != expected[field]:
                raise ValueError("Original rejection counts changed")
        counts[seed] = summed
    out = dict(status="FIT_SEED_MATCH_RIVAL_OVERLAP_POSTHOC_ACTUAL", posthoc=True, FIT=True, DEV=False, TEST=False,
               GT_opened=False, association_changed=False, model_forwards=0, result_sha256=RESULT_SHA,
               thresholds=[.85,.90], donors=counts)
    data = (json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    with args.output.open("xb") as f:
        f.write(data)
    print(json.dumps(dict(status=out["status"], sha256=sha256(data).hexdigest(), donors=counts)))

if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("--closed",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    main(p.parse_args())
