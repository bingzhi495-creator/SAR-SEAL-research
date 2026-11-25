"""Read exact observed metadata and published cohort decisions before native data access."""
from hashlib import sha256
import json
from pathlib import Path
import re
from types import ModuleType, SimpleNamespace
from urllib.parse import urlsplit
from uuid import UUID


METADATA_STATUS = "UMBRA_SUPPLEMENT_METADATA_LOCKED"
COMPLETE = "UMBRA_SUPPLEMENT_NATIVE_WINDOWS_COMPLETE"
PARTIAL = "UMBRA_SUPPLEMENT_NATIVE_WINDOWS_PARTIAL"
READBACK = "UMBRA_SUPPLEMENT_NATIVE_READBACK_COMPLETE"
NATIVE_CORE = "a4dd5571d686a6de92fa5ed9b6af0810671a896c677dc2a83f2124d64db584e6"
READBACK_CORE = "c925569ffe18adfc9b73d412258aab464cbad2aad513b749f71c8008f541b709"
HERE = Path(__file__).resolve().parent


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(data):
    return sha256(data).hexdigest()


def hexadecimal(value, length=64):
    return isinstance(value, str) and re.fullmatch(rf"[a-f0-9]{{{length}}}", value) is not None


def pin(path, expected):
    data = path.read_bytes()
    require(hexadecimal(expected) and digest(data) == expected, f"Exact digest required for {path.name}")
    return data


def load_worker(path, expected):
    module = ModuleType(path.stem)
    module.__file__ = str(path)
    exec(compile(pin(path, expected), str(path), "exec"), module.__dict__)
    return module


def admission_flags(parser):
    for name in ("manifest", "metadata-result", "metadata-terminal", "published-lock"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--published-commit", required=True)


def public_object(row):
    location = urlsplit(row["author_url"])
    return (location.scheme == "https" and location.netloc == "umbra-open-data-catalog.s3.us-west-2.amazonaws.com"
            and location.path.startswith("/sar-data/tasks/ship_detection_testdata/")
            and location.path.endswith("_GEC.tif") and not location.query and not location.fragment
            and isinstance(row["etag"], str) and len(row["etag"]) > 0
            and type(row["object_bytes"]) is int and row["object_bytes"] >= 65536)


def read_admission(args):
    terminal = json.loads(pin(args.metadata_terminal, args.metadata_terminal_sha256))
    result = json.loads(pin(args.metadata_result, args.metadata_result_sha256))
    lock = pin(args.published_lock, args.published_lock_sha256)
    require(terminal["phase"] == "verify" and terminal["actual_exit"] == 0 and
            terminal["output_footer_verified"] is True and terminal["result_sha256"] == args.metadata_result_sha256
            and hexadecimal(terminal["worker_sha256"]) and result["worker_sha256"] == terminal["worker_sha256"],
            "Observed natural metadata worker with matching result footer and source")
    require(result["status"] == METADATA_STATUS and result["cohort_private_sha256"] == args.manifest_sha256
            and result["image_reads"] == result["label_reads"] == result["model_forwards"] == 0
            and hexadecimal(result["protocol_sha256"]), "Closed metadata with zero image or outcome access")
    require(hexadecimal(args.published_commit, 40) and args.manifest_sha256.encode("ascii") in lock,
            "Published exact pre-pixel cohort lock")
    manifest = json.loads(pin(args.manifest, args.manifest_sha256))
    items = manifest["items"]
    require(manifest["schema"] == "umbra_gec_supplement_v1" and
            isinstance(items, list) and 18 <= len(items) <= 24 and len(items) == result["selected_scene_count"]
            and [row["window_id"] for row in items] == [f"UMB-{i:03d}" for i in range(101, 101+len(items))]
            and all(public_object(row) for row in items) and manifest.get("scenes", items) == items,
            "Exact eligible supplemental cohort")
    require(len({row["author_url"] for row in items}) == len(items), "Unique exact author assets")
    for key in ("task_id", "collect_id"):
        require(len({UUID(row[key]) for row in items}) == len(items), "Unique task and acquisition identities")
    binding = dict(manifest_sha256=args.manifest_sha256, metadata_terminal_sha256=args.metadata_terminal_sha256,
                   metadata_result_sha256=args.metadata_result_sha256, metadata_worker_sha256=result["worker_sha256"],
                   metadata_protocol_sha256=result["protocol_sha256"], published_lock_sha256=args.published_lock_sha256,
                   published_commit=args.published_commit, native_protocol_sha256=digest((HERE/"PRELOCK.md").read_bytes()))
    return SimpleNamespace(items=items, binding=binding)


def write_json(path, value):
    data = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(data)
    return digest(data)


def write_result(path, value):
    print(json.dumps(dict(status=value["status"], result_sha256=write_json(path, value))), flush=True)
