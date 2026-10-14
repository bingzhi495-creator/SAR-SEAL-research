// Structural preflight only. No filesystem access, hashing, or scientific qualification.
export function validateSourceReferenceClosure(spec) {
  if (!spec || typeof spec !== 'object' || !Array.isArray(spec.sources)) {
    throw new Error('Source list required');
  }
  const isRef = value => value && typeof value === 'object' && !Array.isArray(value)
    && Object.keys(value).length === 3 && typeof value.path === 'string'
    && Number.isSafeInteger(value.bytes) && value.bytes >= 0
    && typeof value.sha256 === 'string' && /^[a-f0-9]{64}$/.test(value.sha256);
  const closure = new Map();
  for (const ref of spec.sources) {
    if (!isRef(ref)) throw new Error('Malformed source reference');
    if (closure.has(ref.path)) throw new Error(`Duplicate source reference: ${ref.path}`);
    closure.set(ref.path, ref);
  }
  const visit = value => {
    if (isRef(value)) {
      const stored = closure.get(value.path);
      if (!stored || stored.bytes !== value.bytes || stored.sha256 !== value.sha256) {
        throw new Error(`Missing exact source reference: ${value.path}`);
      }
      return;
    }
    if (Array.isArray(value)) value.forEach(visit);
    else if (value && typeof value === 'object') Object.values(value).forEach(visit);
  };
  for (const [key, value] of Object.entries(spec)) if (key !== 'sources') visit(value);
  return true;
}
