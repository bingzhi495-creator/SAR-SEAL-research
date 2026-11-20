# HRSID official PNG TEST input audit: source and synthetic qualification only

This entry does not load models, generate candidates, train or calculate AP. It
can support a frozen center baseline independently of the pending point-refiner
development decision. No original TEST data have been opened by this source
preparation or its tests. Actual execution requires Root's endpoint freeze,
separate admission and original process/independent readback ownership.

## Reused official-input tools

Supply the ordinary published
`code/HRSID_OFFICIAL_PNG_FIT_AUDIT_20261109/hrsid_png_fit_audit.py` explicitly with
its SHA-256. The entry reuses `checked_member`, `image_stem`, `decode_rgb`,
`raster_instance` and `fingerprint_counts`. It does not call the old FIT or DEV
audit entry, read its configured payloads or substitute TEST into their receipts.
There are no new dependencies: standard Python plus the helper's existing
NumPy, Pillow and pycocotools. Actual library versions must match the admitted
runtime exactly. Synthetic validation in another runtime does not fix the
original TEST empty-mask count or qualify that future runtime.
NumPy and Pillow versions are read from the imported modules. An actual
synthetic-runtime check caught a Pillow package-metadata/version mismatch;
the entry does not silently use package metadata as the decoder version.

## Hard production contract

The command entry has no fixture/count/archive override. It fixes the complete
official PNG archive to 5,698,302,315 bytes and SHA-256
`f089fe3417a6c58f3e71f4bc35f8895ac6b136f29c58191b4a90442ab630d159`,
the original 672,944-byte central directory at offset 5,697,629,273 and its SHA,
and the member census of 5,604 PNGs, five JSONs and three directories. It opens
only `annotations/test2017.json` and its 1,962 selected PNG members as members.
The TEST annotation census is 1,962 images and 5,922 annotations, unique original
integer IDs, one ship category and original 800-square canvases. Whole-archive
hashing reads the ZIP bytes; it does not decode unrelated members. Selected ZIP
member reads check the original CRC. Exact archive identity authenticates the
annotation member even when a separate prior member SHA is unavailable.

The small `Contract` parameter on the Python audit function exists to exercise
real tiny synthetic archives; the command entry always passes `OFFICIAL`.

## Required admission before any payload lookup

Both metadata files are supplied by explicit path and exact SHA. They must be
genuine Root-issued records; this program validates their immutable bindings,
not a signature or the truth of an arbitrary caller's statement.

Frozen endpoint metadata must have:

```json
{
  "status": "HRSID_TEST_ENDPOINTS_FROZEN_FOR_INPUT_AUDIT_V1",
  "phase": "HRSID_OFFICIAL_PNG_TEST_INPUT_AUDIT_ONLY_V1",
  "endpoint_freeze_complete": true,
  "endpoints_sha256": {"center_e24": "<exact SHA256>"}
}
```

Root admission must have:

```json
{
  "status": "ROOT_ADMITTED_HRSID_TEST_INPUT_AUDIT_V1",
  "phase": "HRSID_OFFICIAL_PNG_TEST_INPUT_AUDIT_ONLY_V1",
  "endpoint_freeze_sha256": "<exact frozen metadata SHA256>",
  "input_audit_only": true,
  "expected_runtime_versions": {
    "python": "<exact version>", "numpy": "<exact version>",
    "Pillow": "<exact version>", "pycocotools": "<exact version>"
  }
}
```

Only already frozen endpoints belong in the map; no refiner is selected here.
The bridge metadata is independently pinned JSON with `FIT` and `DEV` lists of
2,937 and 705 original records respectively, each containing `image_id`,
`encoded_sha256` and `decoded_rgb_sha256`. It uses existing closed fingerprints;
the audit does not open FIT or DEV images or annotations. Exact image-byte and
decoded-pixel identities are separate. Matches are reported, never excluded.

## Output and result interpretation

`RESULT_SAFE.json` separates `qualification.whole_archive`, `.member_list` and
`.original_annotations` from image decode and raster diagnostics. It includes
runtime/source/input digests, full census, within-TEST duplicate counts and
cross-split identity-bridge aggregates. It does not include filesystem paths,
per-image fingerprints or operational process claims.

Fresh private local outputs are:

- `TEST_IMAGE_ONLY_MANIFEST_PRIVATE.json`: ordered image identity, member name,
  actual encoded/RGB hashes, dimensions and CRC.
- `TEST_ORIGINAL_ANNOTATIONS_PRIVATE.json`: exact original annotation-member
  bytes, not a filtered or reserialized label view.
- `TEST_RASTER_FINDINGS_PRIVATE.json`: empty/invalid/image failure identities.
- `TEST_IDENTITY_BRIDGES_PRIVATE.json`: exact per-image identity bridge links.

Empty rasters and invalid annotations are counted separately. Empty originals
remain in the annotation member, and do not alone prevent structural input
qualification. Image failures or invalid annotations return a complete rejection
report and command exit 2. Whole identity/member/count/scope failures reject by
exception before a completed result is written. No output is created for a wrong
phase. Neither a zero exit nor the audit establishes geometric annotation
accuracy, scene independence or an untouched test population. Historical HRSID
TEST selection exposure remains disclosed. Root must independently read back
the eventual original output before any model/evaluation consumer is admitted.

## Commands after genuine admission only

All operands below are placeholders; this document resolves no private paths.

```text
python -I -B -X utf8 test_input_audit.py
  --freeze FROZEN_ENDPOINTS --freeze-sha256 FREEZE_SHA
  --admission ROOT_ADMISSION --admission-sha256 ADMISSION_SHA
  --archive ORIGINAL_OFFICIAL_PNG_ZIP
  --fit-helper ORDINARY_PUBLISHED_FIT_HELPER --fit-helper-sha256 HELPER_SHA
  --bridges CLOSED_FIT_DEV_FINGERPRINTS --bridges-sha256 BRIDGES_SHA
  --output NEW_ABSOLUTE_OUTPUT_DIRECTORY
```

For synthetic tests, set `HRSID_AUDIT_HELPER` to the ordinary published FIT helper
and run `python -I -B -X utf8 -m unittest discover -s THIS_DIRECTORY -p 'test_*.py'
-v`. Tests build two generated RGB PNGs and artificial polygon records in memory.
The subprocess phase test writes only generated metadata into a temporary
directory and proves unavailable payload paths are not opened. No dataset path,
credential, model, checkpoint or original annotation is present in the fixture.
