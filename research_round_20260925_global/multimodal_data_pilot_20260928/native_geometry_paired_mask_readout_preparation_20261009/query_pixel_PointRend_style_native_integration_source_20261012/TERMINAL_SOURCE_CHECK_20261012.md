# Terminal SOURCE-only check (2026-10-12)

Limited to the four new PointRend integration terminal functions. No complete project SOURCE module, Torch, COCO, model, fixture or scientific computation was imported or executed. No actual receipt, SPEC, qualification, GT, checkpoint, results or AUTH was read or hashed.

Before the patch, each isolated terminal accepted artificial records with empty/string identity, boolean/float held exit code, and missing phase. After the patch, each requires a nonempty exact dict original_identity, exact int held actual_exit_code equal to 0, and an explicitly present matching phase. Existing held/reap/birth/error/resource checks are unchanged.

Artificial metadata checks only: 1 valid control accepted and 17 invalid variants rejected per file (72 checks total), with input dictionaries unchanged. Cases include missing/null/empty/string/list/bool identity; bool/float/string/missing held code; bool/float returncode; missing/null/mismatched/empty phase. Only terminal and its stdlib-only require/expect/finite dependencies were AST-extracted into manual namespaces. This is not a real RAW terminal qualification.

All four files passed Python 3.8 grammar AST parsing and static compilation without execution. Prefix/suffix bytes outside terminal and all non-terminal top-level AST nodes are unchanged. Training math, interfaces, budgets, seeds, outputs and adapter are untouched.

| File | Bytes | Lines | SHA256 |
| --- | ---: | ---: | --- |
| train.py | 55271 | 617 | `182189dfac1fc7e6d8094bc9b0a53e5372a4c8317b443ce80c8588a0bfc2c680` |
| qualify_cpu.py | 59936 | 668 | `b46aad55b9cf47d5965e2e669b5db0488ac890b86e9f89ebd21214d0d5a6101f` |
| evaluate.py | 41122 | 483 | `4d5307df6d5f6fdd134e5da9e3ec6e978f40b745b7caab349b41204044a8407a` |
| qualify_evaluation_cpu.py | 46253 | 537 | `66491c16cd394182bca5bf7ae99faa8ff01e1f811527d057088f7be3f4fafefa` |

Previous SOURCE SHA256 values (for this narrow delta only):
- train.py: `32a41c77fece24399fd38d9b6e7842d0b249439596c963398005e17fe0ffc3b4`
- qualify_cpu.py: `02dbc81b775a622c4a296e6680f7e92d9815b21fb79a29c4fab0b559f1bfece1`
- evaluate.py: `802957c7af3cc79d002f58ba2fd2901bd23eae68352584c48de6bd9a996bbdbe`
- qualify_evaluation_cpu.py: `044c8800dcf4e28614ce77a83ec0c8b73082c5adec6f32f7a52bad9a2b0ae593`

Status: SOURCE_IMPLEMENTED_NUMERICAL_VALIDATION_PENDING. Actual RAW/CPU/GPU integration and scientific numerical validation are not established by these artificial metadata checks. ROOT owns final version/status acceptance.
