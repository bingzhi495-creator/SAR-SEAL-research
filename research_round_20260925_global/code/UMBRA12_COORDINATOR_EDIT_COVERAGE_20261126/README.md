# Umbra coordinator coverage check

The [ahead-of-read lock](PRELOCK.md) specifies access and counts. The checker takes only the original source receipts, the accepted frozen experiment code and receipt, and the original closed edit ledger. It compares canonical RLE strings in memory. It uses the Python standard library.

Run **test_verify_coverage.py** with Python to check generated ledger records and byte integrity.

Run the exact accepted Windows process observer at **../UEIKAP_RESERVED_HULL_FROZEN_20261122/observe.py**. Pass phase **verify-edits**, the checker as worker, a fresh result file, a separate observer directory, and the worker command. Provide these named inputs to **verify_coverage.py**:

- **--prelock** is this directory's original **PRELOCK.md**.
- **--release** is the absolute path to the research release root containing **code** and **results**.
- **--carrier-run** is the absolute path to the closed private capture and edit run.
- **--intake-terminal**, **--intake-result**, **--intake-readback** and **--packet-receipt** are the exact original safe intake and preparation receipts.
- **--result** is the path to a new result file outside the release and original run roots.

Keep the aggregate to the coordinator until both reviewer exports are sealed. The checker requires the prepared edit ledger and independent verification to agree by digest. Keep the original disk layout and all original files unchanged.

