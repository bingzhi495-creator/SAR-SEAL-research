# N20 repeat202 CPU endpoint tail v03: correct nested metadata routing

Status: UNBOUND / SCIENTIFICALLY UNEXECUTED / synthetic nested-call regression passed. Use03 for future assembly;02 is unsuitable for valid real-format checkpoints. Files01/02 and independent body01 remain unchanged. This preparation does not establish original GPU181 success, new pure182 success, independent CPU qualification or scientific performance.

## Exact artifacts

- New tail03: `tmp/N20_REPEAT202_PAIRED_TRAIN_CPU_RUN_TAIL_UNBOUND_DRAFT_20261016_03.py`; 18930 bytes; SHA256 `6a625affa2dc8038f503ec18ff90349d5dab48d2f4e1d99aac9142212cc99dfc`.
- Portable pure-stdlib test03: `tmp/N20_REPEAT202_PAIRED_TRAIN_CPU_METADATA_TEST_UNBOUND_20261016_03.py`; 4887 bytes; SHA256 `1e3dc98faea207a7d21e5dc208859ed266fe122a3115709b05ea86686fdd68bc`.
- Preserved incorrect-routing tail02: `tmp/N20_REPEAT202_PAIRED_TRAIN_CPU_RUN_TAIL_UNBOUND_DRAFT_20261016_02.py`; SHA256 `6e027aab02eb0f31435ed39c168de3edee298c06a696d1a23b48f38b4158e55c`. Do not activate02.
- Unchanged independent body01: `tmp/N20_REPEAT202_PAIRED_FEATURE_ERROR_ENDPOINT_CPU_QUALIFIER_BODY_UNBOUND_DRAFT_20261016_01.py`; SHA256 `55f52434ce6b263e5fc9c4356c28db705797a4e209e21edc5444c399d17d03f2`.

## Root cause and bounded change

The reviewed unbound GPU202 tail (SHA256 bb8be48150b391d78362998f5b6b9353ce7b1245f36f50b6b6c80cdddeeedc44) checkpoint callback line295 stores binding under snapshot['metadata'], with endpoint_epoch. Existing independent CPU body line102 already reads that nested metadata for the baseline20 fields. The03 helper call therefore passes snapshot['metadata'] rather than the whole snapshot. Its local parameter is renamed metadata. Missing or unequal references still raise AssertionError; no prior20-field check is removed.

Precisely three added keys are checked against REPORT: N20_first201_ROOT_qualified_result, N20_repeat202_initialization_qualified_result, N20_repeat202_cleanup_probe_qualified_result. The call remains at03 line202, immediately after torch.load and before body.validate_snapshot_report. Front owns full23-field REPORT/START/FIT-wrapper consistency, strictref shape/provenance and actual terminal joins; metadata equality alone cannot authenticate provenance.

Text/AST inspection confirms03 differs from02 only in the metadata argument, matching helper-local variable rename and version comment. Input loader and saved-update trace-reader function ASTs are unchanged. Body01 bytes are unchanged. Formula, five tensors/18851 parameters, initialization202610150202, order202610140160001, FIT741,24 epochs/E8/E24,5928/17784 AdamW steps,72843264 point exposure, resource limits and absolute1e-4/rtol0 tolerances are unchanged.

## Actual regression tests through real call expression

The synthetic fixture has23 REPORT binding keys and a snapshot shaped as {'metadata': valid23binding plus endpoint_epoch, 'completed_epoch':8, 'models': synthetic sentinels, 'optimizers': synthetic sentinels}. The three new references are absent at snapshot top level. Test code extracts the actual run() helper call AST and the actual helper definition from the selected tail, then compiles/executes only those nodes. Thus the real argument expression is tested, including the nested lookup. Full run(), Torch imports, tensor loads and scientific computation are never executed.

02's prior25 flat-helper synthetic cases did not exercise this call wiring and could not establish real-format compatibility.03 replaces that insufficient test scope with:

-1 legal nested23-binding case;
-6 missing cases (three refs removed separately from metadata or REPORT);
-18 unequal cases (three refs x metadata/REPORT x path/bytes/sha256 changes).

Actual RED:03 first copied byte-for-byte from02 (6e027aab...), and the valid nested fixture failed with 'missing Root ref: N20_first201_ROOT_qualified_result'. Initial receipt chunk252ffb, exit1. After test portability adaptation it still failed identically: chunkc42344, exit1,1 test,1 failure.

Minimal GREEN: metadata argument/local rename only, valid nested case passed (chunke0213a, exit0). Final expanded suite actual output, chunk542a8f, exit0:

```
test_each_missing_nested_or_report_ref_is_rejected ... ok
test_each_tampered_nested_or_report_ref_is_rejected ... ok
test_valid_nested_23_binding_is_accepted ... ok
Ran 3 tests in 0.013s
OK
```

The25 cases verify only metadata behavior and selected call wiring. They do not replay optimizer trajectories, GPU pointwise logits, masks, RLE, selected indices or AP. No real snapshot/cache/data/result/log was read; no Torch/NumPy/project imports or Git operations occurred.

## Protection and portable execution

Before all local source/test reads and writes, the wrapper excluded original and resolved paths using `tmp/N20_REPEAT202_TRAIN_GPU181_ACTIVE_BOUND_PATHS_20261016_01.json`,2525189B, SHA256 a23a8217bc9c3e5447e4d9534852191fe65479a0b58335da2fcad8ba646119d0. RED/GREEN command outputs include HOST_WRAPPER_GUARD_OK. This private snapshot is intentionally not a test runtime dependency. Test03 defaults to Path(__file__).with_name('N20_REPEAT202_PAIRED_TRAIN_CPU_RUN_TAIL_UNBOUND_DRAFT_20261016_03.py'); optional --source supports isolated regression comparisons. It can run from a directory containing the two public artifact filenames, using Python stdlib only.

ROOT must still review/freeze front+tail03+body01 after original GPU181 actual0 plus new pure182 actual0, perform authorized independent CPU endpoint qualification and externally verify its held terminal. Pending/qualified=False flags remain. STOP EDIT after hashes are reported.
