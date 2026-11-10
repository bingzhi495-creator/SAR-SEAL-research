import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request
import zipfile


FILES = {
    'opencv4': ('opencv-python', '4.5.5.64', 'cp36-abi3-win_amd64', 35383167,
                '408d5332550287aa797fd06bef47b2dfed163c6787668cc82ef9123a9484b56a'),
    'opencv5': ('opencv-python-headless', '5.0.0.93', 'cp37-abi3-win_amd64', 43825962,
                '829717b6a95554f273e49e357cee3b3a2a26b6f4842fbc1bed2b45bdd8f87e0e'),
    'numpy2': ('numpy', '2.2.6', 'cp310-cp310-win_amd64', 12904620,
               'f0fd6321b839904e15c46e0d257fdd101dd7f530fe03fd6359c1ea63738703f3'),
}

REALISTIC = ((800, 576), (800, 512), (800, 640), (640, 512), (640, 576),
             (576, 640), (512, 576), (513, 576), (457, 576), (576, 457),
             (768, 576), (1024, 512), (599, 640), (640, 599), (601, 512),
             (10, 1920), (101, 576), (575, 576), (577, 576))
SHAPES = (((7, 11), (4, 8)), ((16, 16), (8, 8)), ((13, 18), (17, 5)),
          ((800, 800), (576, 576)), ((800, 800), (512, 512)),
          ((800, 800), (640, 640)), ((640, 640), (512, 512)),
          ((512, 640), (576, 576)), ((457, 601), (576, 576)),
          ((575, 577), (576, 576)), ((1, 5), (1, 2)), ((5, 1), (2, 1)))


def acquire(entry, temp):
    key, (package, version, abi, length, digest) = entry
    with urllib.request.urlopen(f'https://pypi.org/pypi/{package}/{version}/json', timeout=20) as reply:
        metadata = json.load(reply)
    [wheel] = [item for item in metadata['urls'] if abi in item['filename']]
    if wheel['digests']['sha256'] != digest or wheel['size'] != length:
        raise ValueError(f'Wheel metadata mismatch for {package}')
    if not wheel['url'].startswith('https://files.pythonhosted.org/'):
        raise ValueError(f'Unexpected wheel host for {package}')
    local_wheel = temp / wheel['filename']
    hasher = hashlib.sha256()
    received = 0
    with urllib.request.urlopen(wheel['url'], timeout=25) as reply, local_wheel.open('xb') as out:
        while True:
            block = reply.read(1 << 20)
            if not block:
                break
            received += len(block)
            if received > length:
                raise ValueError(f'Wheel exceeded pinned size for {package}')
            hasher.update(block)
            out.write(block)
    if received != length or hasher.hexdigest() != digest:
        raise ValueError(f'Wheel integrity mismatch for {package}')
    target = temp / key
    with zipfile.ZipFile(local_wheel) as archive:
        archive.extractall(target)
    return key, target


def worker(paths):
    sys.path[:0] = paths
    import cv2
    import numpy as np

    if not Path(cv2.__file__).is_relative_to(paths[0]):
        raise RuntimeError('cv2 is outside the isolated wheel root')
    if len(paths) > 1 and not Path(np.__file__).is_relative_to(paths[1]):
        raise RuntimeError('NumPy is outside the isolated wheel root')
    cv2.ocl.setUseOpenCL(False)
    cv2.setNumThreads(1)

    def indices(n, m):
        numerators = (2 * np.arange(m, dtype=np.int64) + 1) * n
        return numerators // (2 * m), numerators % (2 * m) == 0

    report = {'opencv': cv2.__version__, 'numpy': np.__version__, 'runs': {}}
    for enabled in (True, False):
        cv2.setUseOptimized(enabled)
        for mode_name, mode in [('stock', cv2.INTER_NEAREST), ('exact', cv2.INTER_NEAREST_EXACT)]:
            shapes_different = positions = differences = ties = non_ties = 0
            old_diff = 0
            hashes = hashlib.sha256()
            details = {}
            for n, m in tuple((n, m) for n in range(1, 65) for m in range(1, 65)) + REALISTIC:
                expected, on_tie = indices(n, m)
                actual = cv2.resize(np.arange(n, dtype=np.uint16).reshape(1, n), (m, 1), interpolation=mode)[0]
                mismatches = actual != expected
                count = int(mismatches.sum())
                tie_count = int((mismatches & on_tie).sum())
                shapes_different += count > 0
                positions += m
                differences += count
                ties += tie_count
                non_ties += count - tie_count
                old_diff += int((actual != np.arange(m, dtype=np.int64) * n // m).sum())
                hashes.update(actual.tobytes())
                if (n, m) in REALISTIC or (n, m) in ((2, 1), (4, 2), (5, 2), (2, 5), (6, 5), (8, 7)):
                    item = {'different': count, 'tie': tie_count, 'first': int(actual[0]), 'last': int(actual[-1])}
                    if n < 10:
                        item['actual'] = actual.tolist()
                    details[f'{n}->{m}'] = item

            mask_count = pixel_diffs = 0
            mask_hash = hashlib.sha256()
            per_shape = {}
            for (h, w), (out_h, out_w) in SHAPES:
                y, x = np.ogrid[:h, :w]
                mask = np.ascontiguousarray(((x + 2 * y) % 7 == 0) | (x == w - 1) | (y == h - 1), dtype=np.uint8)
                yi = indices(h, out_h)[0]
                xi = indices(w, out_w)[0]
                expected = mask[np.ix_(yi, xi)]
                actual = cv2.resize(mask, (out_w, out_h), interpolation=mode)
                bad = int((actual != expected).sum())
                mask_count += bad > 0
                pixel_diffs += bad
                mask_hash.update(actual.tobytes())
                per_shape[f'{h}x{w}->{out_h}x{out_w}'] = bad

            report['runs'][f'{mode_name}_optimized_{enabled}'] = {
                'grid_shapes': 64**2 + len(REALISTIC), 'grid_shapes_different': shapes_different,
                'grid_positions': positions, 'grid_positions_different': differences,
                'tie_differences': ties, 'non_tie_differences': non_ties,
                'grid_positions_different_from_floor_jN_div_M': old_diff,
                'grid_outputs_sha256': hashes.hexdigest(), 'selected_grids': details,
                'binary_shapes': len(SHAPES), 'binary_shapes_different': mask_count,
                'binary_pixels_different': pixel_diffs, 'binary_outputs_sha256': mask_hash.hexdigest(),
                'binary_pixels_different_by_shape': per_shape,
            }
    return report


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        print(json.dumps(worker(sys.argv[2:])))
        return
    source = Path(__file__).resolve()
    results = {}
    with tempfile.TemporaryDirectory(prefix='cv_wheel_payload_', dir=source.parent) as scratch:
        temp = Path(scratch)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            isolated = dict(executor.map(lambda entry: acquire(entry, temp), FILES.items()))
        for key, version, numpy_roots in [('opencv4', '4.5.5', []), ('opencv5', '5.0.0', [isolated['numpy2']])]:
            command = [sys.executable, '-B', '-I', str(source), '--worker', str(isolated[key]), *map(str, numpy_roots)]
            child = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30,
                                   env={**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'},
                                   creationflags=subprocess.CREATE_NO_WINDOW)
            outcome = json.loads(child.stdout)
            if outcome['opencv'] != version or outcome['numpy'] != ('1.26.4' if key == 'opencv4' else '2.2.6'):
                raise RuntimeError(f'Unexpected wheel runtime versions for {key}: {outcome}')
            results[key] = outcome
    result = {'schema': 'opencv_nearest_synthetic_v1', 'wheels': FILES,
              'reference': 'output center, exact integer arithmetic, tie to larger index', 'results': results}
    with source.with_name('SUMMARY.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write('\n')
    print(str(source.parent))
    for key, outcome in results.items():
        for name, run in outcome['runs'].items():
            print(key, name, run['grid_shapes_different'], run['grid_positions_different'],
                  run['tie_differences'], run['non_tie_differences'], run['binary_shapes_different'],
                  run['binary_pixels_different'], run['grid_outputs_sha256'], run['binary_outputs_sha256'])


if __name__ == '__main__':
    main()
