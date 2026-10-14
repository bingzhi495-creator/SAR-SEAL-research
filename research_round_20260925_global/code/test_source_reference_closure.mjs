import assert from 'node:assert/strict';
import { validateSourceReferenceClosure } from './source_reference_closure.mjs';

const ref = (name, digit) => ({ path: `/synthetic/${name}`, bytes: 1, sha256: digit.repeat(64) });
const adapter = ref('adapter.py', 'a');
const model = ref('model.py', 'b');
const valid = { CPU_adapter_source: adapter, parent: { checkpoints: [model] }, sources: [adapter, model] };
assert.equal(validateSourceReferenceClosure(valid), true);
assert.throws(() => validateSourceReferenceClosure({ ...valid, sources: [model] }), /adapter.py/);
assert.throws(() => validateSourceReferenceClosure({ ...valid, sources: [adapter] }), /model.py/);
assert.throws(() => validateSourceReferenceClosure({ ...valid, sources: [adapter, model, adapter] }), /Duplicate/);
assert.throws(() => validateSourceReferenceClosure({ ...valid, CPU_adapter_source: { ...adapter, bytes: 2 } }), /adapter.py/);
console.log('Synthetic source closure regression passed; no files or scientific data read.');
