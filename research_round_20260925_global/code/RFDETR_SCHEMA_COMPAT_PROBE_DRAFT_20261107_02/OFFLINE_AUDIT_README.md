# v02 offline audit helper: candidate, not executed

This independent helper does not modify v01. The original v01 failure remains a
failure: its blocked bind had no recorded stack, so urllib3 is only a strong
static explanation for that historical event.

Root must separately admit the v02 source, environment, operation, and resources.
The supplied whitelist is exactly `{path, bytes, sha256}`. `path` must name the
actual absolute urllib3 `util/connection.py` source file; the helper requires
4444 bytes and SHA-256
`2633bbdb69731e5ccb5cf4e4afd65605d86c7979cc5633126f50c92d5ad74a74`.
The installer recomputes this reference before installing any hook. Importing
this helper alone does not install a hook.

Worker integration, after Root admission and before third-party imports:

```python
audit = offline_audit.install_offline_audit(root_verified_urllib3_source_ref)
# Perform only the separately admitted worker operation.
audit_receipt = audit.receipt()
# Require installed=True, passed=True, unexpected_count=0, invalid_count=0,
# expected_count <= 1; preserve the full receipt on every success/failure path.
```

Check the receipt after dependency imports as well as at final worker closure.
A successful audit is not model compatibility, successful forward, or research
qualification. Every receipt has `qualified=False`. Any additional worker
qualification conditions remain necessary.

All audited socket effects and process launch events raise `OSError` before the
operation. The sole expected label requires the direct Python caller to be
`urllib3.util.connection._has_ipv6`, its exported function to have that exact code
object, its actual source reference to equal the Root pin, the socket family to
be AF_INET6, and the address to be exactly the two-tuple `('::1', 0)` with an
integer port. That bind is still blocked. No loopback bind/connect/send is allowed.
`socket.__new__` remains permitted solely as local object allocation, so urllib3
can reach the denied bind and handle `OSError` as a negative capability probe.

The diagnostic records the direct caller and up to 12 Python frames with module,
function, line, actual source reference or source-read error, plus family/address.
Unrelated frames deeper in the stack cannot satisfy the expected classification.
Unknown evidence, repeated expected probes, all other socket effects, and any
process launch attempt make the audit fail. CPython audit hooks are permanent in
the worker and are not a security boundary against arbitrary native code.

Five stdlib-only synthetic tests exercise exact/near-match classification,
caller/source mismatches, repeat/unknown rejection, and manual hook denial.
They do not install a global hook, create a socket/child, import Torch, read a
checkpoint, or execute any scientific task. No v02 worker has been run here.
