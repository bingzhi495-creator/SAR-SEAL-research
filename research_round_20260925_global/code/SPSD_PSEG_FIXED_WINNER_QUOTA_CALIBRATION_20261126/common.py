"""Shared receipt and configuration validation for the isolated experiment."""
from hashlib import sha256
import json
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace

HERE = Path(__file__).resolve().parent
LOCK = "87b8c75be3b02358f00287fbcfe72a8dccccf7dee84a1a9afe92b77d35573ba8"
QUOTA = "15e66410804d4ba8f64951b3e9d4b54e98958ade9fc23c8c363bd8e0e0a1c5bd"
WRITER = "fb49070f10d064fbe1bcce89e7baf1fb59d3ae81950863437adb95c37c7c4b81"


def need(ok, error):
    if not ok:
        raise ValueError(error)


def digest(raw):
    return sha256(raw).hexdigest()


def pin(path, wanted):
    raw = path.read_bytes()
    need(digest(raw) == wanted, "Exact input hash: " + path.name)
    return raw


def load(path, wanted, mutate=lambda content: content):
    module = ModuleType(path.stem)
    module.__file__ = str(path)
    exec(compile(mutate(pin(path,wanted).decode("utf-8-sig")), str(path), "exec"), vars(module))
    return module


def settings(path, keys):
    doc = json.loads(path.read_bytes())
    need(set(doc) == set(keys), "Exact private path fields")
    return SimpleNamespace(**{key: Path(value).resolve(strict=True) for key,value in doc.items()})


def closed(result_path, terminal_path, worker_path, result_sha=None, term_sha=None, worker_sha=None):
    raw = pin(result_path,result_sha) if result_sha else result_path.read_bytes()
    end = json.loads(pin(terminal_path,term_sha) if term_sha else terminal_path.read_bytes())
    worker = pin(worker_path,worker_sha) if worker_sha else worker_path.read_bytes()
    result = json.loads(raw)
    need(end["actual_exit"] == 0 and end["output_footer_verified"] is True
         and end["result_sha256"] == digest(raw) and end["worker_sha256"] == result["worker_sha256"] == digest(worker),
         "A prior process must have an independently verified successful natural exit")
    return result,digest(raw)


def write(path, data):
    body = (json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    with path.open("xb") as out:
        out.write(body)
        out.flush()
        os.fsync(out.fileno())
    return digest(body)


def closeout(output, worker, status, **result):
    sha = write(output, dict(status=status,protocol_sha256=LOCK,worker_sha256=digest(worker.read_bytes()),**result))
    print(json.dumps(dict(status=status,result_sha256=sha)),flush=True)
