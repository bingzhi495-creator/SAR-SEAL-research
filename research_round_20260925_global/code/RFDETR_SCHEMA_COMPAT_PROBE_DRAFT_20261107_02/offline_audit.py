"""Unexecuted v02 offline audit candidate; this module grants no run permission.

The sole expected event is still blocked before bind. Its label only explains a
hash-pinned urllib3 IPv6 capability check; it never turns a denied event into an
allowed operation. Root must separately admit the source and worker execution.
"""
import copy
import errno
import hashlib
import socket
import sys
import threading
from pathlib import Path, PurePosixPath, PureWindowsPath


URLLIB3_CONNECTION_BYTES = 4444
URLLIB3_CONNECTION_SHA256 = (
    "2633bbdb69731e5ccb5cf4e4afd65605d86c7979cc5633126f50c92d5ad74a74"
)


def _valid_expected_ref(ref):
    if not isinstance(ref, dict) or set(ref) != {"path", "bytes", "sha256"}:
        return False
    path = ref["path"]
    return (
        isinstance(path, str) and bool(path) and not any(c in path for c in "\x00\r\n")
        and (PurePosixPath(path).is_absolute() or PureWindowsPath(path).is_absolute())
        and ".." not in PurePosixPath(path.replace("\\", "/")).parts
        and type(ref["bytes"]) is int and ref["bytes"] == URLLIB3_CONNECTION_BYTES
        and ref["sha256"] == URLLIB3_CONNECTION_SHA256
    )


def classify_attempt(event, *, family, address, callers, expected_source_ref,
                     capture_error=None):
    """Classify already-normalized evidence, without I/O or socket operations."""
    valid_callers = type(callers) is list
    first = callers[0] if valid_callers and callers else None
    if not valid_callers and capture_error is None:
        capture_error = "invalid callers evidence"
    expected = (
        event == "socket.bind"
        and type(family) is int and family == int(socket.AF_INET6)
        and type(address) is tuple and len(address) == 2
        and address[0] == "::1" and type(address[1]) is int and address[1] == 0
        and isinstance(first, dict)
        and first.get("module") == "urllib3.util.connection"
        and first.get("function") == "_has_ipv6"
        and first.get("code_identity_matches_export") is True
        and first.get("source_ref") == expected_source_ref
        and _valid_expected_ref(expected_source_ref)
        and capture_error is None
    )
    return {
        "event": event,
        "classification": (
            "expected_blocked_capability_probe" if expected else "unexpected_blocked"
        ),
        "allowed": False,
        "family": family,
        "address": list(address) if isinstance(address, tuple) else copy.deepcopy(address),
        "address_type": type(address).__name__,
        "caller": copy.deepcopy(first),
        "callers": copy.deepcopy(callers),
        "capture_error": capture_error,
    }


def assess_events(records, expected_source_ref):
    """Fail closed on unknown/malformed evidence or more than one expected bind."""
    expected_count = unexpected_count = 0
    invalid_count = 0 if type(records) is list else 1
    for record in records if type(records) is list else []:
        if not isinstance(record, dict):
            invalid_count += 1
            continue
        label = record.get("classification")
        if record.get("allowed") is not False or record.get("capture_error") is not None:
            invalid_count += 1
        if label == "unexpected_blocked":
            unexpected_count += 1
        elif label == "expected_blocked_capability_probe":
            expected_count += 1
            address = record.get("address")
            if record.get("address_type") == "tuple" and isinstance(address, list):
                address = tuple(address)
            check = classify_attempt(
                record.get("event"), family=record.get("family"), address=address,
                callers=record.get("callers", []), expected_source_ref=expected_source_ref,
                capture_error=record.get("capture_error"))
            if check["classification"] != label or check["caller"] != record.get("caller"):
                invalid_count += 1
        else:
            invalid_count += 1
    return {
        "schema": "offline-audit-v02", "qualified": False,
        "passed": (_valid_expected_ref(expected_source_ref)
                   and unexpected_count == 0 and invalid_count == 0 and expected_count <= 1),
        "expected_count": expected_count, "unexpected_count": unexpected_count,
        "invalid_count": invalid_count, "events": copy.deepcopy(records),
    }


def _source_ref(filename):
    path = Path(filename).resolve(strict=True)
    if not path.is_file():
        raise ValueError("caller source is not an ordinary file")
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _capture_callers():
    # Capture -> hook -> actual audited Python caller. Never search deeper for
    # a whitelisted frame: classification is restricted to this first caller.
    frame = sys._getframe(2)
    result = []
    try:
        while frame is not None and len(result) < 12:
            code = frame.f_code
            namespace = frame.f_globals
            entry = {
                "module": namespace.get("__name__"), "function": code.co_name,
                "line": frame.f_lineno, "source_ref": None,
                "code_identity_matches_export": (
                    getattr(namespace.get(code.co_name), "__code__", None) is code),
            }
            try:
                entry["source_ref"] = _source_ref(code.co_filename)
            except Exception as exc:
                entry["source_error"] = type(exc).__name__
                entry["code_filename"] = code.co_filename
            result.append(entry)
            frame = frame.f_back
    finally:
        del frame
    return result


def _diagnostic_value(value):
    if value is None or type(value) in (str, int, float, bool):
        return value
    if type(value) in (tuple, list):
        return [_diagnostic_value(item) for item in value]
    return {"type": type(value).__module__ + "." + type(value).__qualname__}


def _denied_event(event):
    # socket.__new__ allocates a local socket object, with no bind/connect/send.
    # It must remain possible for urllib3 to reach the deliberately denied bind.
    return (
        (event.startswith("socket.") and event != "socket.__new__")
        or event.startswith("subprocess.")
        or event in {"os.system", "os.fork", "os.forkpty", "os.posix_spawn", "os.exec",
                     "os.startfile", "os.startfile/2", "pty.spawn"}
    )


class OfflineAudit:
    """One worker-local recorder; construction does not install a global hook."""

    def __init__(self, expected_source_ref):
        if not _valid_expected_ref(expected_source_ref):
            raise ValueError("Root source reference must match the explicit urllib3 pin")
        self._expected_ref = copy.deepcopy(expected_source_ref)
        self._events = []
        self._lock = threading.Lock()
        self._installed = False
        self._marker = object()

    def hook(self, event, args):
        if event == "sar_seal.offline_audit_v02.installed":
            if len(args) == 1 and args[0] is self._marker:
                self._installed = True
            return
        if not _denied_event(event):
            return
        family = address = None
        callers = []
        capture_error = None
        try:
            callers = _capture_callers()
            if event.startswith("socket."):
                if args and isinstance(args[0], socket.socket):
                    family = int(args[0].family)
                if event in {"socket.bind", "socket.connect"} and len(args) >= 2:
                    address = args[1]
                elif event == "socket.sendto" and len(args) >= 2:
                    address = args[-1]
                elif event == "socket.sendmsg" and len(args) >= 2:
                    address = args[1]
                elif event in {"socket.getaddrinfo", "socket.getnameinfo"}:
                    address = tuple(_diagnostic_value(value) for value in args)
                elif event in {"socket.gethostbyname", "socket.gethostbyaddr"} and args:
                    address = args[0]
            record = classify_attempt(
                event, family=family, address=address, callers=callers,
                expected_source_ref=self._expected_ref)
            record["address"] = _diagnostic_value(address)
        except Exception as exc:
            capture_error = type(exc).__name__ + ": audit evidence capture failed"
            record = classify_attempt(
                event, family=None, address=None, callers=callers,
                expected_source_ref=self._expected_ref, capture_error=capture_error)
        with self._lock:
            self._events.append(record)
        # Always raise, including for the sole expected capability check. urllib3
        # catches this OSError and returns False without an actual bind occurring.
        raise OSError(errno.EPERM, "offline audit denied " + event + ": "
                      + record["classification"])

    def receipt(self):
        with self._lock:
            records = list(self._events)
        result = assess_events(records, self._expected_ref)
        result["installed"] = self._installed
        result["passed"] = result["passed"] and self._installed
        result["whitelist_source_ref"] = copy.deepcopy(self._expected_ref)
        return result


def install_offline_audit(expected_source_ref):
    """Root-controlled worker API; no installation occurs at module import.

    The supplied absolute reference must first match the actual source bytes.
    CPython audit hooks are permanent for this process. This bounded diagnostic
    does not claim to constrain malicious native code outside CPython audits.
    """
    observer = OfflineAudit(expected_source_ref)
    if _source_ref(expected_source_ref["path"]) != expected_source_ref:
        raise ValueError("actual urllib3 source does not equal Root source reference")
    sys.addaudithook(observer.hook)
    sys.audit("sar_seal.offline_audit_v02.installed", observer._marker)
    if not observer._installed:
        raise RuntimeError("audit hook installation could not be verified")
    return observer
