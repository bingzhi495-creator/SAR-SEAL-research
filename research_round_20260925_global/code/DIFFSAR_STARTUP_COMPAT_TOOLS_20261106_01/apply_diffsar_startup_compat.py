#!/usr/bin/env python3
"""Preview or apply five verified startup edits to an explicit source copy.

Standard library only. This script imports or executes no author/ML code.
Use an independent source copy; it cannot identify externally frozen originals.
"""

import argparse
import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path


UPSTREAM_COMMIT = "bdc3362372213bc8c3a1f6d5b7b74dc224d1b871"
# Only minimal line fragments are included, never third-party whole files.
EDITS = (
    ("detectron2/modeling/backbone/__init__.py", (
        ("remove_absent_fpn_exports",
         b"from .fpn import FPN, FPN_1p_5p, SCJE_FPN, FPN_1p_4p",
         b"from .fpn import FPN, SCJE_FPN",
         "Remove absent FPN exports; retain the selected SCJE_FPN implementation."),
        ("remove_absent_unused_fan_import", b"from .fan import FAN", None,
         "Remove import of an absent backbone unused by the target R101 configuration."),
        ("remove_absent_unused_bifpn_import",
         b"from detectron2_backbone.backbone.bifpn import build_resnet_bifpn_p1_p5_backbone",
         None, "Remove absent external BiFPN import unused by the target configuration."),
    )),
    ("diffsarshipinst/loss.py", (
        ("remove_unused_poly_focal_import",
         b"from fvcore.nn import sigmoid_focal_loss_jit, poly_focal_loss_jit",
         b"from fvcore.nn import sigmoid_focal_loss_jit",
         "Remove the unused unavailable PolyLoss import; retain the active sigmoid focal loss."),
    )),
    ("detectron2/data/transforms/transform.py", (
        ("replace_removed_pillow_alias",
         b"    def __init__(self, src_rect, output_size, interp=Image.LINEAR, fill=0):",
         b"    def __init__(self, src_rect, output_size, interp=Image.Resampling.BILINEAR, fill=0):",
         "Use Pillow's documented equivalent bilinear alias; preserve interpolation selection."),
    )),
)


class Refused(ValueError):
    """A source precondition does not hold; no source mutation is permitted."""


class IncompleteApply(RuntimeError):
    """An I/O failure occurred after one or more per-file atomic replacements."""

    def __init__(self, paths, error_type):
        self.paths = paths
        self.error_type = error_type


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def line_content(line):
    if line.endswith(b"\r\n"):
        return line[:-2]
    if line.endswith((b"\n", b"\r")):
        return line[:-1]
    return line


def prepare(source_root):
    root = Path(source_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise Refused("source-root must be an existing directory")
    prepared = []
    for relative, edits in EDITS:
        target = root
        for component in Path(relative).parts:
            target = target / component
            attrs = getattr(target.lstat(), "st_file_attributes", 0)
            if target.is_symlink() or attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
                raise Refused(relative + ": symbolic links/junctions are refused")
        try:
            target.resolve(strict=True).relative_to(root)
        except ValueError:
            raise Refused(relative + ": target escapes source-root")
        if not target.is_file():
            raise Refused(relative + ": target must be a regular file inside source-root")
        before = target.read_bytes()
        before.decode("utf-8", errors="strict")
        if b"\x00" in before:
            raise Refused(relative + ": binary content is refused")
        lines = before.splitlines(keepends=True)
        changes = []
        for edit_id, old, new, reason in edits:
            matches = [i for i, line in enumerate(lines) if line_content(line) == old]
            if len(matches) != 1:
                raise Refused(relative + ": " + edit_id +
                              " requires exactly one original line; observed " + str(len(matches)))
            index = matches[0]
            ending = lines[index][len(old):]
            lines[index] = b"" if new is None else new + ending
            changes.append({"id": edit_id, "explanation": reason})
        after = b"".join(lines)
        prepared.append((target, before, after, {
            "path": relative,
            "before_sha256": sha256(before),
            "after_sha256": sha256(after),
            "changes": changes,
        }))
    return prepared


def apply_prepared(prepared):
    # All five line-count checks on all three files have already passed.
    # Stage every replacement before touching a source file.
    staged = []
    applied = []
    try:
        for target, before, after, row in prepared:
            descriptor, name = tempfile.mkstemp(prefix=".diffsar-compat-", suffix=".tmp", dir=target.parent)
            temporary = Path(name)
            staged.append((target, temporary, row))
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(after)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, stat.S_IMODE(target.stat().st_mode))
        for target, before, _, row in prepared:
            if target.read_bytes() != before:
                raise Refused(row["path"] + ": content changed since preflight")
        for target, temporary, row in staged:
            os.replace(temporary, target)
            applied.append(row["path"])
        for target, _, after, row in prepared:
            if target.read_bytes() != after:
                raise OSError("post-write verification failed")
    except OSError as error:
        if applied:
            raise IncompleteApply(applied, type(error).__name__) from error
        raise
    finally:
        cleanup_error = None
        for _, temporary, _ in staged:
            try:
                if temporary.exists():
                    temporary.unlink()
            except OSError as error:
                cleanup_error = error
        if cleanup_error is not None:
            if applied:
                raise IncompleteApply(applied, type(cleanup_error).__name__) from cleanup_error
            raise cleanup_error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True,
                        help="Explicit independent source-copy directory; no implicit default.")
    parser.add_argument("--apply", action="store_true",
                        help="Apply after every precondition passes; default is a read-only preview.")
    args = parser.parse_args(argv)
    try:
        prepared = prepare(args.source_root)
        report = {"schema": "diffsar-startup-compat-v1", "status": "PREVIEW",
                  "upstream_commit": UPSTREAM_COMMIT, "change_count": 5,
                  "files": [row for _, _, _, row in prepared]}
        if args.apply:
            apply_prepared(prepared)
            report["status"] = "APPLIED"
        print(json.dumps(report, ensure_ascii=True, indent=2))
        return 0
    except IncompleteApply as error:
        print(json.dumps({"schema": "diffsar-startup-compat-v1", "status": "PARTIAL_APPLY",
                          "applied_paths": error.paths, "error_type": error.error_type}, ensure_ascii=True))
        return 3
    except (Refused, OSError, UnicodeError) as error:
        # Do not put private absolute paths from OS exceptions in portable reports.
        detail = str(error) if isinstance(error, Refused) else type(error).__name__
        print(json.dumps({"schema": "diffsar-startup-compat-v1", "status": "REFUSED",
                          "error": detail}, ensure_ascii=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
