// UNBOUND_DRAFT: pure post-actual JSON integer/flag closure; no IO or numeric replay.
// ROOT first verifies original actualCPU0/RAW and reads SHA-pinned JSONL/rows/Q.
// This function does not establish Source authenticity, terminal status or new F values.
const TRUE_FLAGS = ['neighbor_coordinates_exact', 'extra_union_exact',
  'center_features_byte_exact', 'original_sample_order_exact',
  'input_index_digests_exact', 'identity_teacher_seeds_exact',
  'ROOT_external_source_and_terminal_checks_required'];
const FALSE_FLAGS = ['qualified', 'extra_feature_values_independently_rerendered',
  'models_loaded', 'GT_files_opened', 'GT_used_for_coordinates'];
const PROOF_KEYS = ['dataset_index', 'image_id', 'original_unique_count', 'extra_count',
  'sampled_points', 'raw_tensor_payload_bytes', ...TRUE_FLAGS, ...FALSE_FLAGS].sort();
function need(ok, message) { if (!ok) throw new Error(message); }
function object(value) { return value !== null && typeof value === 'object' && !Array.isArray(value); }
function keys(value, expected, label) {
  need(object(value) && Object.keys(value).sort().join('|') === [...expected].sort().join('|'), label + ': exact keys');
}
function integer(value, label) {
  need(Number.isSafeInteger(value) && value >= 0, label + ': nonnegative safe integer');
  return value;
}
function same(a, b) {
  if (a === b) return true;
  if (Array.isArray(a) || Array.isArray(b)) return Array.isArray(a) && Array.isArray(b) &&
    a.length === b.length && a.every((v, i) => same(v, b[i]));
  if (!object(a) || !object(b)) return false;
  const ka = Object.keys(a).sort(), kb = Object.keys(b).sort();
  return ka.length === kb.length && ka.every((k, i) => k === kb[i] && same(a[k], b[k]));
}
function reference(value, label) {
  keys(value, ['path', 'bytes', 'sha256'], label);
  need(typeof value.path === 'string' && /^(?:[A-Za-z]:[\\/]|\\\\|\/)/.test(value.path), label + ': absolute path locator');
  integer(value.bytes, label + '.bytes');
  need(typeof value.sha256 === 'string' && /^[0-9a-f]{64}$/.test(value.sha256), label + ': lowercase SHA256');
  return value;
}

export function verifyProofPopulation(rows, oldRows, gpuRows, cpuQ) {
  need([rows, oldRows, gpuRows].every(a => Array.isArray(a) && a.length === 741), 'Exactly 741 proof/old/GPU rows');
  need(object(cpuQ) && cpuQ.schema === 'N20-fullFIT-feature-neighbor-cache-CPU-qualification-v1' &&
    cpuQ.status === 'N20_FULLFIT_FEATURE_NEIGHBOR_CACHE_CPU_COMPLETE_PENDING_ROOT_ACTUAL0', 'Original pending CPU-Q schema/status');
  const images = new Set(), oldPaths = new Set(), newPaths = new Set();
  const totals = {images: 0, sampled_points: 0, unique_centers: 0, extra_count: 0,
    extra_feature_bytes: 0, raw_tensor_payload_bytes: 0};
  for (let i = 0; i < 741; ++i) {
    const record = rows[i], old = oldRows[i], gpu = gpuRows[i];
    keys(record, ['dataset_index', 'image_id', 'parent_packet', 'packet', 'proof'], 'proof row ' + i);
    need(object(old) && object(gpu), 'Decoded report rows ' + i);
    need(record.dataset_index === i && old.dataset_index === i && gpu.dataset_index === i, 'Exact canonical ordinal ' + i);
    const image = integer(old.image_id, 'image_id ' + i);
    need(record.image_id === image && gpu.image_id === image && !images.has(image), 'Unique exact image identity ' + i);
    need(object(old.identity) && same(old.identity, gpu.identity) && old.identity.image_id === image, 'Full original/new identity ' + i);
    images.add(image);
    const oldRef = reference(old.packet, 'old packet ' + i), newRef = reference(gpu.packet, 'new packet ' + i);
    reference(record.parent_packet, 'proof parent ' + i); reference(record.packet, 'proof packet ' + i);
    reference(gpu.parent_packet, 'GPU parent ' + i);
    need(same(record.parent_packet, oldRef) && same(gpu.parent_packet, oldRef) && same(record.packet, newRef) &&
      oldRef.path !== newRef.path && !oldPaths.has(oldRef.path) && !newPaths.has(newRef.path), 'Exact physical packet refs, no duplicate/self packet ' + i);
    oldPaths.add(oldRef.path); newPaths.add(newRef.path);
    const count = integer(old.points, 'old points ' + i), U = integer(old.unique_sampled_F_points, 'old U ' + i);
    const E = integer(gpu.extra_count, 'GPU E ' + i);
    need(count === 4096 && U <= count && integer(gpu.points, 'GPU points ' + i) === count &&
      integer(gpu.unique_centers, 'GPU U ' + i) === U && integer(gpu.original_unique_count, 'GPU original U ' + i) === U &&
      old.natural_queries === 100 && gpu.natural_queries === 100, 'Full per-image original points/U/natural queries ' + i);
    need(Number.isSafeInteger(old.matching_loss_seed) && old.matching_loss_seed === gpu.matching_loss_seed &&
      Number.isSafeInteger(old.point_seed) && old.point_seed === gpu.point_seed, 'Original matching/point seeds ' + i);
    const featureBytes = E * 256 * 4, rawBytes = featureBytes + E * 8 + U * 9 * 8;
    integer(featureBytes, 'feature bytes ' + i); integer(rawBytes, 'raw bytes ' + i);
    need(integer(gpu.extra_feature_bytes, 'GPU F bytes ' + i) === featureBytes &&
      integer(gpu.raw_tensor_payload_bytes, 'GPU raw bytes ' + i) === rawBytes, 'Exact GPU-row tensor bytes ' + i);
    const p = record.proof; keys(p, PROOF_KEYS, 'numeric proof ' + i);
    need(p.dataset_index === i && p.image_id === image && integer(p.sampled_points, 'proof points ' + i) === count &&
      integer(p.original_unique_count, 'proof U ' + i) === U && integer(p.extra_count, 'proof E ' + i) === E &&
      integer(p.raw_tensor_payload_bytes, 'proof bytes ' + i) === rawBytes, 'Numeric proof identity/counts/bytes ' + i);
    for (const key of TRUE_FLAGS) need(p[key] === true, 'Missing actual numeric/index check ' + key + ' at ' + i);
    for (const key of FALSE_FLAGS) need(p[key] === false, 'Proof scope/qualification flag ' + key + ' at ' + i);
    totals.images++; totals.sampled_points += count; totals.unique_centers += U;
    totals.extra_count += E; totals.extra_feature_bytes += featureBytes; totals.raw_tensor_payload_bytes += rawBytes;
    for (const [key, value] of Object.entries(totals)) integer(value, 'total ' + key);
  }
  need([...newPaths].every(path => !oldPaths.has(path)), 'New packets must not alias any original packet locator');
  need(totals.images === 741 && totals.sampled_points === 3035136 && totals.unique_centers === 1786820 &&
    totals.extra_feature_bytes === totals.extra_count * 1024 &&
    totals.raw_tensor_payload_bytes === totals.extra_count * 1032 + totals.unique_centers * 72, 'Complete fixed population/tensor totals');
  for (const key of ['images', 'sampled_points', 'unique_centers', 'extra_count', 'extra_feature_bytes'])
    need(integer(cpuQ[key], 'CPU Q ' + key) === totals[key], 'CPU Q aggregate differs: ' + key);
  for (const key of ['qualified', 'ROOT_actual_CPU0_verified', 'original_owned_terminal_self_certified',
    'CPU_renderer_equivalence_claimed', 'CUDA_initialized', 'teacher_or_PR_loaded',
    'training_executed', 'AP_evaluated', 'TEST_evaluated']) need(cpuQ[key] === false, 'Original pending CPU Q scope: ' + key);
  return {...totals, proof_population_exact: true, all_numeric_index_flags_exact: true,
    scope: 'POST_ACTUAL_JSON_INTEGER_AND_FLAG_CLOSURE_ONLY', qualified: false,
    actual_CPU_terminal_verified_by_this_module: false, numerical_values_replayed: false,
    source_refs_read_or_hashed: false};
}
