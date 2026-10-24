# Fixed-candidate adapter draft: unbound, no real SAM execution

This small standard-library module validates an indexed backend contract for exactly 100 original candidates. It is not a runnable SAMRefiner reproduction. No SAM package, checkpoint, image, annotation, tensor, GPU or remote host has been used by this draft.

The author function returns arrays without query IDs. This adapter requires a future bridge to return `BackendResult(query_ids, masks)`. Checking those declared IDs detects declared reordering, omissions and duplicates; it cannot detect a backend that permutes masks but reports unchanged IDs. A real bridge must independently qualify actual SAM batch ordering. No such bridge is implemented here.

## Interface

`refine_candidates(candidates, image_size, backend, *, empty_policy='reject')`

- Supply 100 `Candidate(query_id, category_id, score, mask)` objects with unique IDs. Their order, category and raw S0 objects are preserved. Mask output is immutable nested tuples.
- `image_size` is the declared original `(height, width)`. Input and returned masks must be rectangular nested Python lists/tuples with that exact size and builtin bool/int values 0 or 1. There is no resize, threshold, clipping, 0/255 conversion, GT input or score calibration. Actual image dimensions remain the future caller/bridge's responsibility.
- The backend receives keyword arguments `query_ids`, `masks`, `image_size`, and `iters=5` in one batch, and must return an indexed `BackendResult`. No retry or fallback occurs after backend failure. Five is the configured author-default iteration argument, not evidence that any model iterations were executed.
- Default `reject` fails before the backend if any input mask is empty. It does not discard the candidate or silently evaluate a smaller population.
- Explicit `passthrough` is an **adaptation strategy**, marked by `is_adaptation=True`. Only nonempty masks are submitted in their original relative order, returned masks are checked and refilled at original indices, and empty masks retain their original zero geometry and identity/S0. An entirely empty population invokes no backend and still returns 100 candidates. This differs from the author's unguarded empty-mask path and requires prospective benchmark disclosure.
- `backend_calls` counts Python callable invocations only. It does not measure image encodings, actual SAM iterations, memory, latency or inference success.

## Validation

Finite tests use only 2x3 synthetic masks and a handwritten indexed backend. They check bit-preserved float S0 (including negative zero), identities, order, population, geometry, binary values, default empty rejection, explicit passthrough refilling, all-empty handling and no retry. They are neither scientific data validation nor real SAM, ABI, CUDA, FastGeodis or AP evidence. See `VALIDATION.json` for observed RED/GREEN tool records and file hashes.

Run without site packages or bytecode output:

```text
python -B -S -X utf8 -m unittest -v test_fixed_candidate_adapter
```

Source contract checked at author commit `4bb7f95738c7a2e805f189210fb09e02a2197557`: [sam_refiner.py](https://github.com/linyq2117/SAMRefiner/blob/4bb7f95738c7a2e805f189210fb09e02a2197557/sam_refiner.py) and [utils.py](https://github.com/linyq2117/SAMRefiner/blob/4bb7f95738c7a2e805f189210fb09e02a2197557/utils.py). The real backend requires the author's `forward_with_image_embeddings` interface and FastGeodis; dependencies, model weights, actual ordering and empty-mask behavior remain unexecuted. This draft does not establish an external comparison or a result.
