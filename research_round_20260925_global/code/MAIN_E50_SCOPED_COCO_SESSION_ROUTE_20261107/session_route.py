"""Apply a scoped module path to one audited direct SSH exec."""
from pathlib import PurePosixPath
import re
import shlex


def canonical_overlay(value):
    if not isinstance(value, str) or any(mark in value for mark in (":", "\\", "\0", "\r", "\n")):
        raise ValueError("Invalid module search path")
    path = PurePosixPath(value)
    if path.anchor != "/" or path == PurePosixPath("/") or str(path) != value or ".." in path.parts:
        raise ValueError("Use one canonical absolute module directory")
    return value


def build_launch_argv(original_argv, logical, overlay):
    """Keep the SSH argv and direct exec, adding only an explicit environment prelude."""
    if (
        len(logical) != 8 or logical[0] != "exec" or logical[2] != "-u"
        or logical[4] != "--spec" or logical[6] != "--spec-sha"
        or not re.fullmatch(r"[0-9a-f]{64}", logical[7])
        or not original_argv or original_argv[-1] != shlex.join(logical)
    ):
        raise ValueError("Only the original direct owner exec may receive this environment")
    prelude = "export PYTHONPATH=" + shlex.quote(canonical_overlay(overlay))
    return [*original_argv[:-1], prelude + " && " + original_argv[-1]]


def launch_with_overlay(u80, request, overlay):
    """Run U80 once and persist full SSH argv separately from the owner's exec."""
    from pathlib import Path

    overlay = canonical_overlay(overlay)
    spec, config, _ = u80.validate_request(request)
    if request["mode"] != "U74_GPU":
        raise ValueError("Overlay applies to fixed GPU prediction only")
    owner = spec["owner_source"]["path"]
    reference = request["spec_ref"]
    logical = [
        "exec", config["remote_python"], "-u", owner, "--spec", reference["path"],
        "--spec-sha", reference["sha256"],
    ]
    destination = Path(request["persistence_directory"])

    class ScopedTransport(u80.SSHTransport):
        def __init__(self):
            super().__init__(config)
            self.child = None
            self.plan = None

        def command(self, remote_command):
            argv = super().command(remote_command)
            if remote_command == logical:
                return build_launch_argv(argv, logical, overlay)
            return argv

        def popen(self, remote_command, stdout, stderr):
            if remote_command != logical or self.child is not None:
                raise ValueError("One original owner exec only")
            self.plan = self.command(remote_command)
            u80.write_new(destination / "SESSION_ROUTE_PLANNED.json", dict(
                schema="Root-U80-session-overlay-argv-v1", ssh_argv=self.plan,
                logical_command=logical, module_path=overlay, spec_ref=reference,
                transport_config_ref=request["transport_config_ref"],
            ))
            self.child = super().popen(remote_command, stdout, stderr)
            return self.child

    transport = ScopedTransport()
    result = u80.launch_saved_spec(request, transport=transport)
    child = transport.child
    if (
        child is None or child.args != transport.plan or child.pid != result["local_SSH_pid"]
        or child.returncode != result["SSH_returncode"] or result["local_SSH_wait_returned"] is not True
    ):
        raise ValueError("Actual terminal does not match the scoped SSH process")
    observed = u80.write_new(destination / "SESSION_ROUTE_OBSERVED.json", dict(
        schema="Root-U80-session-overlay-observed-v1", ssh_argv=child.args,
        logical_command=logical, module_path=overlay, local_SSH_pid=child.pid,
        SSH_returncode=child.returncode, launch_report_ref=result["report_ref"],
        attempt_ref=result["attempt_ref"], spec_ref=reference, stdout_ref=result["stdout_ref"],
        stderr_ref=result["stderr_ref"],
    ))
    return result, observed
