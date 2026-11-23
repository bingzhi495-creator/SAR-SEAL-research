"""Apply a locked image-only column displacement to the accepted Phoenix worker."""

import argparse
from hashlib import sha256
from pathlib import Path
import sys


ORIGINAL_WORKER_SHA256 = "ea98a849ae181f06f3ffc0e41b29d2a91a7f867b4a4bcba6909c957f27db2b99"
DIAGNOSTIC_PROTOCOL_SHA256 = "b5a74bebb9f2ab99c523d57a6074cd978fb23916dda8c9075ea008e51177f2e6"


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("The accepted worker does not contain one exact insertion point")
    return source.replace(old, new)


def shifted_source(source):
    source = replace_once(
        source,
        'STATUS = "PHOENIX_RSDD74_EQUAL_COUNT_INFERENCE_COMPLETE"',
        'STATUS = "RSDD74_IMAGE_ALIGNMENT_PHOENIX_COMPLETE"',
    )
    rgb_check = '    require(digest(rgb.tobytes()) == row["decoded_rgb_sha256"], "Exact source RGB bytes")'
    source = replace_once(
        source, rgb_check,
        rgb_check + '\n    columns = np.clip(np.arange(HW[1]) - SHIFT_COLUMNS, 0, HW[1] - 1)'
                    '\n    rgb = np.ascontiguousarray(rgb[:, columns, :])',
    )
    prefix = '                  source_replay=request["source_replay"],'
    return replace_once(
        source, prefix,
        '                  shift_columns=SHIFT_COLUMNS, diagnostic_protocol_sha256=DIAGNOSTIC_PROTOCOL_SHA256,\n'
        '                  original_worker_sha256=ORIGINAL_WORKER_SHA256,\n' + prefix,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--shift-columns", required=True, type=int, choices=(-1, 1))
    parser.add_argument("--diagnostic-lock", required=True, type=Path)
    parser.add_argument("--diagnostic-lock-sha256", required=True)
    parser.add_argument("--original-worker", required=True, type=Path)
    args, original_args = parser.parse_known_args()
    raw_source = args.original_worker.read_bytes()
    if sha256(raw_source).hexdigest() != ORIGINAL_WORKER_SHA256:
        raise ValueError("Accepted original Phoenix worker identity required")
    if (args.diagnostic_lock_sha256 != DIAGNOSTIC_PROTOCOL_SHA256
            or sha256(args.diagnostic_lock.read_bytes()).hexdigest() != DIAGNOSTIC_PROTOCOL_SHA256):
        raise ValueError("Exact diagnostic protocol required")
    SHIFT_COLUMNS = args.shift_columns
    sys.argv = [sys.argv[0], *original_args]
    worker_source = shifted_source(raw_source.decode("utf-8"))
    exec(compile(worker_source, str(args.original_worker), "exec"), globals())
