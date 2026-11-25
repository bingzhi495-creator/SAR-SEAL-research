"""Exact source and observed receipt bindings for the DEV705 cap experiment."""
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import json
import sys


B5 = "SAR_B5_MAIN_TABLES_20261123"
REPLAY_CODE = "e7d4d06116539e7f2f62486557a22364bc359c3f58e3b9fe5b1b853abe256257"
SCORE_CODE = "f089cc3c692edcfe3aab836dbacb28e2081c18cc5a8f1b3f29859e051424a1a3"
CAPTURE_CODE = "b9eb3d4169653a7e7be634022f89dba81f17375a4ed511b45238a583647aa95b"
CAPTURE = "5feb6ee353443b97a4a360f146ee0a0093747eb92f348d1c3c49462526f3a3fc"
CAP_TERMINAL = "8351b20a3f838f5f7a31453f974a4fe2259f7b246e1a517d4cf0755566adbaf4"
REPLAY = "f29a85ec36134380f4d2fcc4bab860c695f24044bdac8ae5e76541073ac8dade"
REP_TERMINAL = "f172ecbfc3c606b874b6c7839b35f0c0c3c5a8ae69ae20e69a2e4c575d60df9d"
SCORE = "fd13153c34133a509a21a23efd81fabf419bce46757734f1fad04c3b47205d0c"
SCORE_TERMINAL = "1cd9a3aeb2c83f207788e6eb9e1c4d1b3b7c09b5214b8a292565e9b3a1b93675"
MATCH = "99817f192798f52f03efe5e3ac422a453019bf458d33096a03844cb60360436c"
SPARSE = "cb9b8e839de85588fdf116ce5554ed6cadbc20bdbe76e38180561528bfd870d5"
STOCK = "da55f2d3f9bbf08b26d5fae76606b2496a5b482c5fb61f4fab53c23172e5c06e"
OLD_EDITS = "f8c20ce94715e651a1e5cd90e2213a3a2793c2b9454de45de212d07c69335a6b"
MODES = ("N","SF","SU=F","L")
SEEDS = (5217,5218,5219)
TOTAL = {5217:(40316,303285),5218:(40512,307976),5219:(40571,306721)}
RENDER_STATUS = "HRSID_DEV705_LEARNED_BOUNDED_PAIRING_RENDER_COMPLETE"
SCORE_STATUS = "HRSID_DEV705_LEARNED_BOUNDED_PAIRING_SCORE_COMPLETE"


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(data):
    return sha256(data).hexdigest()


def pin(path, expected):
    raw=path.read_bytes()
    require(digest(raw)==expected,"Locked SHA-256 mismatch: "+path.name)
    return raw


def load(path, expected, name):
    raw=pin(path,expected)
    spec=spec_from_file_location(name,str(path))
    mod=module_from_spec(spec)
    sys.modules[name]=mod
    exec(compile(raw,str(path),"exec"),vars(mod))
    return mod


def closed(root,phase,result_sha,terminal_sha,worker_sha):
    doc=json.loads(pin(root/phase/"RESULT_SAFE.json",result_sha))
    term=json.loads(pin(root/f"observed_{phase}"/"TERMINAL.json",terminal_sha))
    require(term["actual_exit"]==0 and term["output_footer_verified"] is True and term["phase"]==phase
            and term["result_sha256"]==result_sha and term["worker_sha256"]==doc["worker_sha256"]==worker_sha,
            "Exact observed successful historical "+phase)
    return doc


def verified_lines(path, expected):
    h=sha256()
    with path.open("rb") as stream:
        for line in stream:
            require(line.endswith(b"\n"),"Complete bound stream record")
            h.update(line)
            yield line
    require(h.hexdigest()==expected,"Complete original stream "+path.name)


def write_result(path, record):
    raw=(json.dumps(record,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    with path.open("xb") as f:
        f.write(raw)
    print(json.dumps(dict(status=record["status"],result_sha256=digest(raw))),flush=True)
