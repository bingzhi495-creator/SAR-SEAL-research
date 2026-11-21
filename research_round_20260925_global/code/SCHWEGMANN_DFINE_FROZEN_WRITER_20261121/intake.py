"""Image-only extraction from a pinned public author record array. No reference field is decoded."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha1, sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
from types import ModuleType
from zipfile import ZipFile

import numpy as np
from PIL import Image, __version__ as pil_version

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ["SAR_SEAL_RUNTIME_ROOT"]).resolve(strict=True)
PRIOR = ROOT / "P254_SCHWEGMANN_EXTERNAL_EVIDENCE_PRIVATE_20261110_01"
ORIGINAL = ROOT / "github_release_20261011/research_round_20260925_global/code/SCHWEGMANN_FROZEN_TRANSFER_20261110/qualify_schwegmann.py"
ALLOW = {"id", "fullname", "sensor", "polarisation", "patchfu"}
STRING = re.compile(rb'"(?:[^"\\]|\\.)*"')
KEY = re.compile(rb'\s*"([^"\\]+)"\s*:\s*')
TOKENS = re.compile(rb'"(?:[^"\\]|\\.)*"|[][{}]')
WS = re.compile(rb"\s*")


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def hash256(data):
    return sha256(data).hexdigest()


def pinned(path, sha):
    with path.open("rb") as f:
        before = os.fstat(f.fileno())
        data = f.read()
        after = os.fstat(f.fileno())
    need((before.st_size, before.st_mtime_ns) == (len(data), after.st_mtime_ns)
         and hash256(data) == sha, "Input identity: " + path.name)
    return data


def write(path, obj):
    data = (json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf8")
    with path.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    return hash256(data)


def skip_value(raw, start):
    first = raw[start:start+1]
    if first == b'"':
        return STRING.match(raw, start).end()
    if first not in (b"[", b"{"):
        end = re.compile(rb"[^,}\]]+").match(raw, start)
        need(end, "Scalar boundary")
        return end.end()
    depth = 0
    for token in TOKENS.finditer(raw, start):
        char = raw[token.start():token.start()+1]
        if char in (b"{", b"["):
            depth += 1
        elif char in (b"}", b"]"):
            depth -= 1
            if depth == 0:
                return token.end()
    raise ValueError("Truncated composite")


def project(raw):
    """Identify record fields lexically and deserialize only the five named image fields."""
    match = re.match(rb"\s*\[\s*\[\s*", raw)
    need(match, "Author outer array")
    pos = match.end()
    for ordinal in range(1, 1597):
        need(raw[pos:pos+1] == b"{", "Record start")
        pos += 1
        row = {}
        while True:
            m = KEY.match(raw, pos)
            need(m, "Direct field name")
            key = m[1].decode("utf8")
            start = m.end()
            pos = skip_value(raw, start)
            if key in ALLOW:
                need(key not in row, "Duplicate selected field")
                row[key] = json.loads(raw[start:pos])
            pos = WS.match(raw, pos).end()
            char = raw[pos:pos+1]
            pos += 1
            if char == b"}":
                break
            need(char == b",", "Record delimiter")
        need(set(row) == ALLOW, "Missing original image fields")
        yield ordinal, row
        pos = WS.match(raw, pos).end()
        if ordinal < 1596:
            need(raw[pos:pos+1] == b",", "Record separator")
            pos = WS.match(raw, pos+1).end()
    need(re.fullmatch(rb"\]\s*\]\s*", raw[pos:]), "Original record count/terminator")


def prior_rgbs():
    directory = PRIOR / "COHORT_03/IMAGE_ONLY"
    manifest = json.loads(pinned(directory / "IMAGE_MANIFEST_PRIVATE.json",
                                "21bde94a8e5f1b9f68f69ee8545e12bed5270846420719f74d9bdfbad4be3fdf"))
    packed = pinned(directory / "IMAGES_ONLY_PRIVATE.zip",
                    "428d696ea4a59a417dccdf26bfcc05f55f9ab20de69045e9f1f5b62c43ab11d7")
    out = []
    with ZipFile(BytesIO(packed)) as z:
        need([r["basename"] for r in manifest["images"]] == z.namelist(), "Historical image census")
        for row in manifest["images"]:
            data = z.read(row["basename"])
            need(len(data) == row["bytes"] and hash256(data) == row["sha256"], "Historical PNG")
            with Image.open(BytesIO(data)) as im:
                need(im.mode == "RGB" and im.size == (101,101), "Historical input geometry")
                out.append(hash256(im.tobytes()))
    need(len(out) == len(set(out)) == 32, "32 unique historical inputs")
    return out


def run():
    begin = datetime.now(timezone.utc).isoformat()
    policy = pinned(HERE / "PROTOCOL.md", "32d859a527cf31c75aa9f2d84e7f6a80f3574bf2048746c5236e61193ecff693")
    old_code = pinned(ORIGINAL, "41704d3d3980b6ae6442d17b92625e37ed782e417e06a585806227d881ce7d79")
    old = ModuleType("historical_image_rule")
    exec(compile(old_code, str(ORIGINAL), "exec"), old.__dict__)
    need(np.__version__ == "1.26.4" and pil_version == "12.3.0", "Original image transform runtime")
    previous = prior_rgbs()
    excluded = set(previous)
    data = pinned(PRIOR / "SHIP_POSITIVES_FROZEN_PRIVATE.json", old.SOURCE_SHA256)
    need(len(data) == old.SOURCE_BYTES and sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == old.SOURCE_BLOB,
         "Immutable author accession")
    groups, candidates = defaultdict(set), {}
    aliases, identities, excluded_pols = {}, set(), Counter()
    ord_to_key, eligible_per_group = [], Counter()
    staging = HERE / "IMAGE_ONLY.partial"
    need(not (HERE / "IMAGE_ONLY").exists(), "Fresh image destination required")
    staging.mkdir()
    for ordinal, row in project(data):
        key = old.acquisition(row)
        groups[key].add(row["fullname"])
        alias = row["id"].rsplit("_",2)[0]
        need(aliases.setdefault(alias, row["fullname"]) == row["fullname"]
             and row["id"] not in identities, "Unique author ID and stable product")
        identities.add(row["id"])
        ord_to_key.append(key)
        if row["polarisation"] not in ("HH","VV"):
            continue
        power = np.asarray(row["patchfu"], dtype="<f8")
        need(power.shape == (101,101), "101-square author source")
        if not np.isfinite(power).all() or (power < 0).any():
            excluded_pols[row["polarisation"]] += 1
            continue
        gray = old.gray_levels(power)
        rgb = np.repeat(gray[:,:,None],3,axis=2)
        rgb_sha = hash256(rgb.tobytes())
        eligible_per_group[key] += 1
        if rgb_sha in candidates:
            need(candidates[rgb_sha][1] == key, "Ambiguous duplicate across different acquisition groups")
            continue
        rank = hash256(old.SELECTION + sha256(gray.tobytes()).digest())
        if rgb_sha in excluded:
            ref = None
        else:
            name = f"{ordinal:04d}.png"
            buf = BytesIO()
            Image.fromarray(rgb).save(buf, format="PNG", compress_level=6)
            enc = buf.getvalue()
            with (staging / name).open("xb") as f:
                f.write(enc)
            readback = (staging / name).read_bytes()
            with Image.open(BytesIO(readback)) as decoded:
                need(readback == enc and decoded.mode == "RGB" and decoded.size == (101,101)
                     and hash256(decoded.tobytes()) == rgb_sha, "PNG/RGB readback")
            ref = dict(path=name,bytes=len(enc),sha256=hash256(enc))
        candidates[rgb_sha] = (ordinal,key,rank,ref)
    group_names = {key:f"acq_{i:03d}" for i,key in enumerate(sorted(groups),1)}
    need(len(ord_to_key) == 1596 and len(groups) == 16 and len(aliases) == len(set(aliases.values())) == 25
         and Counter(len(p) for p in groups.values()) == {1:7,2:9} and
         Counter(k[0] for k in groups) == {"S1A":13,"RS2":3}
         and sum(eligible_per_group.values()) == 782, "Closed source and image eligibility census")
    matched_prior = []
    for key in sorted(groups):
        ranked=sorted((v[2],s) for s,v in candidates.items() if v[1] == key)
        matched_prior.extend(s for _,s in ranked[:2])
    need(previous == matched_prior, "Recreated original 32 image selection and acquisition assignment")
    items, counts = [], {}
    group_ledger = [group_names[key] for key in ord_to_key]
    ledger_sha = hash256("".join(f"{i}\t{g}\n" for i,g in enumerate(group_ledger,1)).encode("ascii"))
    for sha,(ordinal,key,_,ref) in sorted(candidates.items(),key=lambda item:item[1][0]):
        if ref is not None:
            items.append(dict(image_id=ordinal, original_hw=[101,101], image_ref=ref,
                              acquisition_group=group_names[key],record_ordinal=ordinal,rgb_sha256=sha))
    need(set(candidates) >= excluded and len(items) == len(candidates)-32 and
         {r["image_ref"]["path"] for r in items} == {p.name for p in staging.iterdir()}, "Complete retained images")
    for key, group in group_names.items():
        unique = sum(v[1] == key for v in candidates.values())
        count = sum(r["acquisition_group"] == group for r in items)
        counts[group]=dict(eligible_records=eligible_per_group[key], unique_rgbs=unique,
                           duplicate_rgbs=eligible_per_group[key]-unique,previously_scored_rgbs=unique-count,images=count)
    signature = hash256(policy)
    staging.rename(HERE / "IMAGE_ONLY")
    manifest_sha = write(HERE/"IMAGE_MANIFEST_PRIVATE.json",dict(schema="SCHWEGMANN_FROZEN_WRITER_IMAGE_ONLY_V1",
        source_json_sha256=old.SOURCE_SHA256, group_index_sha256=ledger_sha, protocol_sha256=signature,
        record_ordinal_base=1, reference_values_deserialized=False, model_forwards=0,
        expected_ids=[r["image_id"] for r in items], items=items))
    summary = dict(status="SCHWEGMANN_FROZEN_WRITER_IMAGESET_ACCEPTED",start_utc=begin,
        completed_utc=datetime.now(timezone.utc).isoformat(),source_commit=old.SOURCE_COMMIT,source_git_blob=old.SOURCE_BLOB,
        source_json_sha256=old.SOURCE_SHA256,protocol_sha256=signature,source_records=1596,products=25,acquisitions=16,
        eligible=sum(eligible_per_group.values()),unique_eligible=len(candidates),excluded_prior_rgbs=len(excluded),
        retained=len(items),invalid_copolar_by_polarisation=dict(excluded_pols),group_counts=counts,
        manifest_sha256=manifest_sha,group_index_sha256=ledger_sha,model_forwards=0,patchgt_deserialized=False,
        old_32_eligibility_and_order_reproduced=True)
    summary_sha = write(HERE/"AGGREGATE_SAFE.json",summary)
    print(json.dumps(dict(images=len(items),manifest_sha256=manifest_sha,summary_sha256=summary_sha,groups=counts)))


if __name__ == "__main__":
    run()
