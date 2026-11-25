# Umbra supplemental blind reviewer packets

Build two separate role-assigned packages using the existing offline editor and native PNG inventory checks. The supplement keeps the original acquired identifiers and uses independent export names. An admitted partial intake preserves unavailable records in the cohort flow. Every verified image belongs to both role packages, in the original sequence.

Wait for a closed intake and a separately observed independent readback. The adapter requires their exact worker, result and terminal identities, a payload-free admission file and its SHA-256, and the exact sealed supplemental acquisition manifest. The coordinator binds these values before any image accesses. A generator should link the accepted hashes in this admission shape.

~~~json
{
  "manifest_sha256": "<accepted manifest SHA-256>",
  "intake": {
    "phase": "native-window-intake",
    "terminal_sha256": "<accepted terminal SHA-256>",
    "result_sha256": "<accepted result SHA-256>",
    "worker_sha256": "<accepted worker SHA-256>"
  },
  "readback": {
    "phase": "native-window-readback",
    "terminal_sha256": "<accepted terminal SHA-256>",
    "result_sha256": "<accepted result SHA-256>",
    "worker_sha256": "<accepted worker SHA-256>"
  }
}
~~~

Use absolute paths and a fresh output directory. The checker expects the acquisition worker to have written the exact safe receipt, original ordered private provenance and anonymous PNG directory under the intake root. Pass **--base** to the directory of the accepted original annotation tools and **--prelock** to the exact lock in this directory. **--help** shows all paths. Run via **observe_packet.py** with **--phase packet-build** and the resulting **RESULT_SAFE.json** path. Preserve the observed terminal and results unchanged.

The two ZIPs contain the role-specific offline interface, short instructions, anonymous roster and lossless source PNGs. The coordinator's standalone roster is byte-identical to the archived copies. Keep the two review returns separate under **SAR_ship_supplement_review_A.json** and **SAR_ship_supplement_review_B.json**. Keep all ZIPs, PNGs, individual inventory and private references outside the release repository.

Run the generated-data tests as follows.

~~~text
python -I -B -X utf8 test_build_supplement_packets.py --base ORIGINAL_ANNOTATION_CODE
~~~

