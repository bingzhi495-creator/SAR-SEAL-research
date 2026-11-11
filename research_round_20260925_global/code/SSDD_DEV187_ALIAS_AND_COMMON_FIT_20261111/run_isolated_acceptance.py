"""Run the exact P351 CPU acceptance as one bounded original worker."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import runpy
import sys


P351_SHA256 = "e395a4ef6513869d1d0229d46a58183629201144ed6ce6be57e5b56aa04a79d4"
STATUS = "P351_SSDD_WINDOWS_REPEAT5202_E24_INDEPENDENTLY_ACCEPTED_V1"


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("source", "owner-source", "request", "wait", "psutil-root", "result"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--wait-sha256", required=True)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    if source != args.source or sha256(source.read_bytes()).hexdigest() != P351_SHA256:
        raise ValueError("Original P351 source SHA-256 differs")
    dependencies = args.psutil_root.resolve(strict=True)
    if dependencies != args.psutil_root or not (dependencies / "psutil/__init__.py").is_file():
        raise ValueError("Canonical existing process-inspection package path required")
    if (args.result.name != "RESULT_SAFE.json" or args.result.exists() or args.result.is_symlink()
            or args.result.parent.resolve(strict=True) != args.result.parent):
        raise ValueError("Fresh bounded-owner result required")
    accepted = args.result.parent / "ACCEPTANCE"
    sys.path.append(str(dependencies))
    sys.argv = [
        str(source), "--owner-source", str(args.owner_source), "--request", str(args.request),
        "--request-sha256", args.request_sha256, "--wait", str(args.wait),
        "--wait-sha256", args.wait_sha256, "--output", str(accepted),
    ]
    runpy.run_path(str(source), run_name="__main__")
    raw = (accepted / "INDEPENDENT_ACCEPTANCE_SAFE.json").read_bytes()
    result = json.loads(raw)
    if result["status"] != STATUS or result["training_seed"] != 5202 or result["initialization_seed"] != 5201:
        raise ValueError("Unexpected closed P351 acceptance scope")
    with args.result.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


if __name__ == "__main__":
    main()
