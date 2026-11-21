"""Compare only the prelocked official PSeg-SSDD test records with the author's public Drive object."""

import argparse
from hashlib import sha256
from http.client import HTTPException
import json
import os
from pathlib import Path
import sys
from time import monotonic
from types import ModuleType
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


PRELOCK_SHA = "3dbc4e540678349bb57b4d80702f993c2fa69ce26dad769fadcada90748b4b89"
PARSER_SHA = "b3dd521147949af46c534be45a517a34e252263c966099cbd7420debf6d1f794"
SIZE = 1_043_241_197
LIMIT = 8 * 1024**2
BASE = "Official-SSDD-OPEN/BBox_RBox_PSeg_SSDD/coco_style/"
LABEL = BASE + "annotations/test.json"
IMAGES = {BASE + f"images/test/{number:06d}.jpg" for number in range(1, 1161) if number % 10 in (1, 9)}
AUTHOR = "https://drive.usercontent.google.com/download?id=1glNJUGotrbEyk43twwB9556AdngJsynZ&export=download&confirm=t"


def require(ok, description):
    if not ok:
        raise ValueError(description)


def published_parser(path):
    source = path.read_bytes()
    require(sha256(source).hexdigest() == PARSER_SHA, "closed public parser changed")
    module = ModuleType("pseg_published_rar_header_reader")
    sys.modules[module.__name__] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    module.role_for = lambda name: name if name == LABEL or name in IMAGES else None
    return module


def make_spans(records):
    spans = []
    for member in records:
        group = "annotation" if member.role == LABEL else "images"
        require(0 < member.end - member.start <= LIMIT, "record exceeds permitted request size")
        if (spans and group == spans[-1]["group"] and spans[-1]["end"] == member.start
                and member.end - spans[-1]["start"] <= LIMIT):
            spans[-1]["end"] = member.end
            spans[-1]["count"] += 1
        else:
            spans.append(dict(start=member.start, end=member.end, group=group, count=1))
    return spans


def verify_span(opener, span, local_digest):
    start, end = span["start"], span["end"]
    expected = end-start
    request = Request(AUTHOR + "&range_version=" + str(start), headers={
        "Range": f"bytes={start}-{end-1}", "Accept-Encoding": "identity",
        "User-Agent": "Python PSeg-SSDD official TEST input provenance",
    })
    evidence = dict(**span, expected_bytes=expected, local_sha256=local_digest, matched=False)
    for attempt in (1, 2):
        try:
            with opener.open(request, timeout=20) as response:
                scope = response.headers.get("Content-Range")
                evidence.update(attempts=attempt, http_status=response.status, content_range=scope)
                if (response.status != 206 or scope != f"bytes {start}-{end-1}/{SIZE}"
                        or response.headers.get("Content-Encoding", "identity").lower() != "identity"):
                    evidence["failure"] = "Remote did not confirm the exact unencoded partial response"
                    return evidence
                remote = response.read(expected + 1)
                digest = sha256(remote).hexdigest()
                evidence.update(remote_bytes=len(remote), remote_sha256=digest,
                                matched=len(remote) == expected and digest == local_digest)
                if not evidence["matched"]:
                    evidence["failure"] = "Local and returned remote record identity differed"
                else:
                    evidence.pop("failure", None)
                return evidence
        except HTTPError as error:
            evidence.update(attempts=attempt, http_status=error.code, failure="HTTPError")
            if error.code not in (429, 500, 502, 503, 504):
                return evidence
        except (OSError, URLError, HTTPException) as error:
            evidence.update(attempts=attempt, failure=type(error).__name__)
    return evidence


def main():
    cli = argparse.ArgumentParser()
    for name in ("prelock", "parser", "archive", "output"):
        cli.add_argument("--" + name, required=True, type=Path)
    args = cli.parse_args()
    started = monotonic()
    require(not args.output.exists(), "fresh output required")
    require(sha256(args.prelock.read_bytes()).hexdigest() == PRELOCK_SHA, "prelock changed")
    parser = published_parser(args.parser)
    identity = parser.source_identity(args.archive.stat())
    require(identity[2] == SIZE and len(IMAGES) == 232, "wrong archive size or expected image census")
    with args.archive.open("rb") as handle:
        require(parser.source_identity(os.fstat(handle.fileno())) == identity, "archive replaced before open")
        records, headers = parser.local_members(handle)
        require(len(records) == 233 and {r.role for r in records} == IMAGES | {LABEL},
                "missing or duplicate exact TEST members")
        annotation, = [r for r in records if r.role == LABEL]
        require((annotation.unpacked, annotation.crc) == (397010, 0xdacc1bcd), "TEST COCO header changed")
        spans = make_spans(records)
        opener = build_opener(ProxyHandler({}))
        evidence = []
        for span in spans:
            handle.seek(span["start"])
            payload = parser.read_exact(handle, span["end"] - span["start"])
            remote = verify_span(opener, span, sha256(payload).hexdigest())
            evidence.append(remote)
            print(json.dumps(dict(span=len(evidence), total=len(spans), record_count=span["count"],
                                  http=remote.get("http_status"), matched=remote["matched"])), flush=True)
        require(parser.source_identity(os.fstat(handle.fileno())) == identity == parser.source_identity(args.archive.stat()),
                "archive changed during verification")
    matched = [entry for entry in evidence if entry["matched"]]
    complete = len(matched) == len(evidence)
    totals = {role: sum(entry["count"] for entry in matched if entry["group"] == role)
              for role in ("annotation", "images")}
    result = dict(status="SELECTED_TEST_RECORDS_REMOTE_MATCHED" if complete else "REMOTE_MATCH_INCOMPLETE",
                  original_worker_pid=os.getpid(), prelock_sha256=PRELOCK_SHA, header_reader_sha256=PARSER_SHA,
                  worker_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), archive_bytes=SIZE,
                  local_headers_crc_verified=headers, expected_annotation_records=1, expected_jpeg_records=232,
                  matched_records=totals, matched_total_record_bytes=sum(x["expected_bytes"] for x in matched),
                  selected_record_spans=evidence, verified_whole_archive=False,
                  http_proxy_used=False, raw_payloads_written=False, model_runs=0, AP=False,
                  elapsed_seconds=round(monotonic()-started, 3))
    with args.output.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(dict(status=result["status"], pid=os.getpid(), matched=totals,
                          result_sha256=sha256(args.output.read_bytes()).hexdigest())), flush=True)
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
