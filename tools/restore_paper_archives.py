"""Restore listed historical paper PDFs from verified byte chunks; stdlib only.

This utility assembles an archive. It does not train, evaluate, import research
code, validate scientific claims, or remove the preserved input chunks.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile


SCHEMA = "sar-paper-archive-parts-v1"
MAX_PART_BYTES = 3 * 1024 * 1024
DEFAULT_MANIFEST = ".publication_parts/manifest.json"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class ArchiveError(RuntimeError):
    """An archive input, target, or publishing precondition is invalid."""


def require(condition, message):
    if not condition:
        raise ArchiveError(message)


def relative_path(value, root, *, target=False):
    require(isinstance(value, str) and value and "\\" not in value and ":" not in value,
            "Archive paths must be relative POSIX paths")
    raw_parts = value.split("/")
    require(all(part not in ("", ".", "..") for part in raw_parts),
            "Empty or traversal path components are forbidden")
    posix = PurePosixPath(value)
    require(not posix.is_absolute() and len(posix.parts) >= 2,
            "An archive path must be below its allowed directory")
    if target:
        require(posix.parts[0] in ("paper", "paper_tgrs") and posix.suffix == ".pdf",
                "Only paper/ or paper_tgrs/ PDF targets are allowed")
    else:
        require(posix.parts[0] == ".publication_parts" and value != DEFAULT_MANIFEST,
                "Parts must be below .publication_parts/ and cannot be the manifest")
    require(".git" not in posix.parts, "Git metadata paths are forbidden")
    path = root.joinpath(*posix.parts)
    cursor = root
    for part in posix.parts:
        cursor = cursor / part
        require(not cursor.is_symlink(), "Symlink archive paths are forbidden")
    resolved = path.resolve()
    require(root in resolved.parents, "Archive path escapes the workspace")
    return path


def exact_keys(value, keys, label):
    require(isinstance(value, dict) and set(value) == set(keys),
            label + " has an invalid schema")


def byte_reference(value, label, maximum=None):
    require(type(value["bytes"]) is int and value["bytes"] > 0,
            label + " requires a positive integer byte count")
    require(isinstance(value["sha256"], str) and SHA256_RE.fullmatch(value["sha256"]) is not None,
            label + " requires a lowercase SHA-256")
    if maximum is not None:
        require(value["bytes"] <= maximum, label + " exceeds the 3 MiB part limit")


def file_reference(path):
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            count += len(block)
            digest.update(block)
    return count, digest.hexdigest()


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON keys are forbidden")
        result[key] = value
    return result


def read_manifest(root, manifest_path):
    manifest = Path(manifest_path)
    if not manifest.is_absolute():
        manifest = root / manifest
    require(manifest == root / DEFAULT_MANIFEST and not manifest.is_symlink(),
            "Use the workspace .publication_parts/manifest.json")
    require(not manifest.parent.is_symlink() and manifest.resolve().parent == root / ".publication_parts",
            "Manifest directory must remain inside the workspace")
    with manifest.open("r", encoding="utf-8") as stream:
        value = json.load(stream, object_pairs_hook=reject_duplicate_keys)
    exact_keys(value, ("schema", "files"), "Manifest")
    require(value["schema"] == SCHEMA and isinstance(value["files"], list) and value["files"],
            "Manifest schema or file list is invalid")
    plans = []
    targets = set()
    for entry in value["files"]:
        exact_keys(entry, ("path", "bytes", "sha256", "parts"), "PDF entry")
        byte_reference(entry, "PDF entry")
        target = relative_path(entry["path"], root, target=True)
        require(entry["path"] not in targets, "Duplicate PDF targets are forbidden")
        targets.add(entry["path"])
        require(isinstance(entry["parts"], list) and entry["parts"], "PDF entry requires parts")
        parts = []
        for part in entry["parts"]:
            exact_keys(part, ("path", "bytes", "sha256"), "Part entry")
            byte_reference(part, "Part entry", MAX_PART_BYTES)
            path = relative_path(part["path"], root)
            require(path.is_file() and path.stat().st_size == part["bytes"],
                    "A part is missing or has the wrong byte count")
            require(file_reference(path) == (part["bytes"], part["sha256"]),
                    "A preserved part SHA-256 or byte count differs")
            parts.append((path, part))
        require(sum(part["bytes"] for _, part in parts) == entry["bytes"],
                "Part byte counts do not sum to the PDF byte count")
        # Check every existing target before writing any new target.
        if target.exists():
            require(target.is_file() and file_reference(target) == (entry["bytes"], entry["sha256"]),
                    "Existing PDF differs; refusing to overwrite it")
        plans.append((target, entry, parts))
    return plans


def restore_one(root, target, entry, parts):
    if target.exists():
        require(file_reference(target) == (entry["bytes"], entry["sha256"]),
                "Existing PDF changed after validation")
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    relative_path(entry["path"], root, target=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=target.parent,
                prefix="." + target.name + ".restore-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            total_digest = hashlib.sha256()
            total_bytes = 0
            for path, part in parts:
                # Recheck confinement immediately before opening each input.
                relative_path(part["path"], root)
                part_digest = hashlib.sha256()
                part_bytes = 0
                with path.open("rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        part_bytes += len(block)
                        require(part_bytes <= part["bytes"], "Part grew while being restored")
                        part_digest.update(block)
                        total_digest.update(block)
                        total_bytes += len(block)
                        stream.write(block)
                require((part_bytes, part_digest.hexdigest()) == (part["bytes"], part["sha256"]),
                        "Part SHA-256 or byte count differs")
            require((total_bytes, total_digest.hexdigest()) == (entry["bytes"], entry["sha256"]),
                    "Assembled PDF SHA-256 or byte count differs")
            stream.flush()
            os.fsync(stream.fileno())
        relative_path(entry["path"], root, target=True)
        # A same-directory hard link publishes the complete file atomically and
        # refuses to replace a target created after the earlier existence check.
        try:
            os.link(temporary, target)
            changed = True
        except FileExistsError:
            require(target.is_file() and not target.is_symlink() and
                    file_reference(target) == (entry["bytes"], entry["sha256"]),
                    "PDF appeared with different bytes; refusing to overwrite it")
            changed = False
        require(file_reference(target) == (entry["bytes"], entry["sha256"]),
                "Published PDF verification failed")
        return changed
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def restore_archives(repo_root, manifest_path=DEFAULT_MANIFEST):
    """Return restored/unchanged/target POSIX paths after exact byte verification."""
    root = Path(repo_root).resolve(strict=True)
    require(root.is_dir(), "Workspace must be a directory")
    plans = read_manifest(root, manifest_path)
    result = {"restored": [], "unchanged": [], "targets": []}
    for target, entry, parts in plans:
        changed = restore_one(root, target, entry, parts)
        result["restored" if changed else "unchanged"].append(entry["path"])
        result["targets"].append(entry["path"])
    return result


def git_command(root, args, allowed_codes=(0,)):
    result = subprocess.run(["git"] + args, cwd=root, stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, check=False)
    require(result.returncode in allowed_codes,
            "Git operation failed (exit " + str(result.returncode) + ")")
    return result


def commit_and_push(root, targets):
    top = git_command(root, ["rev-parse", "--show-toplevel"]).stdout.strip()
    require(Path(top).resolve() == root, "Only the checked-out workspace repository may be updated")
    require(git_command(root, ["diff", "--cached", "--quiet"], (0, 1)).returncode == 0,
            "Git index already contains changes; refusing an unrelated commit")
    git_command(root, ["add", "--"] + targets)
    changed = git_command(root, ["diff", "--cached", "--quiet", "--"] + targets, (0, 1))
    if changed.returncode == 0:
        return False
    git_command(root, ["-c", "user.name=SAR archive automation", "-c",
                      "user.email=41898282+github-actions[bot]@users.noreply.github.com",
                      "commit", "--only", "-m", "Restore byte-identical historical paper PDFs",
                      "--"] + targets)
    git_command(root, ["push", "origin", "HEAD:main"])
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--commit-and-push", action="store_true")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    try:
        result = restore_archives(root, args.manifest)
        pushed = commit_and_push(root, result["targets"]) if args.commit_and_push else False
        print(json.dumps({"restored_count": len(result["restored"]),
                          "unchanged_count": len(result["unchanged"]), "pushed": pushed}))
        return 0
    except (ArchiveError, OSError, ValueError, TypeError, KeyError) as error:
        # Do not echo captured Git output, remote URLs, environment, or secrets.
        print("Paper archive restoration failed: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
