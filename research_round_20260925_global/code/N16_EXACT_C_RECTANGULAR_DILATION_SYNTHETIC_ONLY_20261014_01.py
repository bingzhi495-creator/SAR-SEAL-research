"""SYNTHETIC_ONLY ordinary C-engine candidate; no actual scientific execution.

No file/GT/cache/state/weights/source/spec access. Never patches an existing
implementation or retries a scientific run. ROOT must qualify/bind a NEW
source version before use. FP32 threshold expressions remain exactly original.
"""


def domains_from_native_cv2(np, cv2, native_L):
    """Return private P/Omega/C BOOL, sorted C indices/Omega positions INT64.

    A centered odd17 rectangle and constant-zero exterior implement precisely
    the physical Chebyshev8 neighborhood. CV2 only sees a contiguous uint8
    Boolean mask; native FP32 L is neither converted nor rendered again.
    """
    if not isinstance(native_L,np.ndarray) or np.ma.isMaskedArray(native_L):
        raise TypeError('Unmasked NumPy native L required')
    if native_L.dtype != np.float32 or native_L.ndim != 2 or not bool(np.isfinite(native_L).all()):
        raise ValueError('Original finite FP32 native L[H,W] required')
    P, Omega = native_L > 0, np.abs(native_L) <= 2
    if not bool(P.any()):
        # Also handles zero-size arrays without invoking CV2 on empty input.
        C = Omega.copy()
    else:
        source = np.ascontiguousarray(P,dtype=np.uint8)
        dilation = cv2.dilate(source,np.ones((17,17),dtype=np.uint8),anchor=(8,8),
                              iterations=1,borderType=cv2.BORDER_CONSTANT,borderValue=0)
        C = Omega | P | (dilation != 0)
    idx = np.flatnonzero(C).astype(np.int64)
    positions = np.flatnonzero(Omega.ravel()[idx]).astype(np.int64)
    if len(positions) != int(np.count_nonzero(Omega)):
        raise RuntimeError('Physical Omega-to-C mapping differs')
    return P, Omega, C, idx, positions


def _scipy_reference(np, ndimage, native_L):
    if native_L.dtype != np.float32 or native_L.ndim != 2 or not bool(np.isfinite(native_L).all()):
        raise ValueError('Original finite FP32 native L[H,W] required')
    P, Omega = native_L > 0, np.abs(native_L) <= 2
    dilation = ndimage.binary_dilation(P, structure=np.ones((17,17),dtype=np.bool_),
                                     iterations=1,border_value=0)
    C = Omega | P | dilation
    idx = np.flatnonzero(C).astype(np.int64)
    positions = np.flatnonzero(Omega.ravel()[idx]).astype(np.int64)
    return P, Omega, C, idx, positions


def synthetic_tests(np, cv2, ndimage):
    """Private synthetic arrays only; all five returns and K must be exact."""
    rng = np.random.default_rng(20261014016)
    cases = []
    for shape in ((0,0),(0,7),(7,0),(1,1),(1,7),(7,1),(2,2),(3,27),(17,17),(19,23),(31,41)):
        cases += [np.full(shape,-3.,dtype=np.float32),np.full(shape,3.,dtype=np.float32),
                  np.zeros(shape,dtype=np.float32)]
        if np.prod(shape):
            border = np.full(shape,-3.,dtype=np.float32)
            border[0,0] = 3.; border[-1,-1] = 3.
            cases.append(border)
    values = np.array([-np.finfo(np.float32).max,-3.,
        np.nextafter(np.float32(-2),np.float32(-np.inf)),np.float32(-2),
        np.nextafter(np.float32(-2),np.float32(np.inf)),-0.,0.,
        np.nextafter(np.float32(2),np.float32(-np.inf)),np.float32(2),
        np.nextafter(np.float32(2),np.float32(np.inf)),3.,np.finfo(np.float32).max],dtype=np.float32)
    cases.append(values.reshape(3,4))
    for i in range(80):
        shape = (int(rng.integers(1,40)),int(rng.integers(1,45)))
        L = rng.uniform(-8.,8.,size=shape).astype(np.float32)
        if i % 4 == 0:
            L[:] = -3.; L[rng.random(shape)<.015] = 3.
        if i % 4 == 1:
            L[rng.random(shape)<.1] = 0.
        cases.extend((L,L.T,L[::-1,::2]))
    for case_number,L in enumerate(cases):
        before = L.copy()
        expected,actual = _scipy_reference(np,ndimage,L),domains_from_native_cv2(np,cv2,L)
        assert len(actual)==5
        for index,(want,got) in enumerate(zip(expected,actual)):
            assert got.dtype == (np.bool_ if index < 3 else np.int64)
            assert got.shape==want.shape and np.array_equal(got,want), (case_number,index,L.shape)
        P,Omega,C,idx,positions = actual
        assert len(positions)==np.count_nonzero(Omega)
        assert np.array_equal(idx[positions],np.flatnonzero(Omega))
        assert np.all(idx[1:]>idx[:-1]) and not np.any(Omega&~C) and not np.any(P&~C)
        assert np.array_equal(L,before)
    for invalid in (np.array([[0.]],dtype=np.float64),np.array([0.],dtype=np.float32),
                    np.array([[np.nan]],dtype=np.float32),np.array([[np.inf]],dtype=np.float32),
                    np.ma.array([[0.]],mask=[[True]],dtype=np.float32)):
        try:
            domains_from_native_cv2(np,cv2,invalid)
        except (TypeError,ValueError):
            pass
        else:
            raise AssertionError('Invalid/hidden FP32 L accepted')
    return dict(cases=len(cases),invalid_cases=5,all_five_arrays_bitexact=True,
                K_and_Omega_positions_exact=True,input_unchanged=True,
                empty_all_border_sparse_random_noncontiguous_covered=True)


def synthetic_benchmark(np,cv2,ndimage):
    from time import perf_counter
    from statistics import median
    rng = np.random.default_rng(20261014017)
    fixtures = []
    typical = np.full((355,501),-3.,dtype=np.float32)
    typical[rng.random(typical.shape)<.025] = 0.
    typical[rng.random(typical.shape)<.0005] = 3.
    fixtures.append(('355x501_sparse_P',typical))
    maximum = np.full((500,600),-3.,dtype=np.float32)
    maximum[rng.random(maximum.shape)<.5] = 3.
    maximum[rng.random(maximum.shape)<.025] = 0.
    fixtures.append(('500x600_dense_P',maximum))
    empty = np.full((500,600),-3.,dtype=np.float32)
    empty[rng.random(empty.shape)<.025] = 0.
    fixtures.append(('500x600_empty_P',empty))
    rows = []
    for name,L in fixtures:
        expected,actual = _scipy_reference(np,ndimage,L),domains_from_native_cv2(np,cv2,L)
        assert all(np.array_equal(a,b) for a,b in zip(expected,actual))
        samples = {'scipy_ms':[],'cv2_ms':[]}
        for _ in range(5):
            for key,fn in (('scipy_ms',lambda:_scipy_reference(np,ndimage,L)),
                           ('cv2_ms',lambda:domains_from_native_cv2(np,cv2,L))):
                start = perf_counter(); result=fn(); elapsed=(perf_counter()-start)*1000
                assert all(np.array_equal(a,b) for a,b in zip(expected,result))
                samples[key].append(elapsed)
        rows.append(dict(name=name,shape=list(L.shape),P=int(np.count_nonzero(actual[0])),
                         K=len(actual[4]),C=len(actual[3]),all_five_arrays_bitexact=True,
                         scipy_ms_median=median(samples['scipy_ms']),cv2_ms_median=median(samples['cv2_ms']),
                         scipy_ms_range=[min(samples['scipy_ms']),max(samples['scipy_ms'])],
                         cv2_ms_range=[min(samples['cv2_ms']),max(samples['cv2_ms'])],repeats=5))
    return rows


if __name__ == '__main__':
    import json
    from time import perf_counter
    import numpy as np
    import cv2
    import scipy
    from scipy import ndimage
    cv2.setNumThreads(1)
    started = perf_counter()
    tests = synthetic_tests(np,cv2,ndimage)
    rows = synthetic_benchmark(np,cv2,ndimage)
    duration = perf_counter()-started
    assert duration < 120., 'Synthetic test/benchmark budget exceeded'
    print(json.dumps(dict(status='SYNTHETIC_ONLY_PASS',tests=tests,benchmark=rows,
        numpy=np.__version__,scipy=scipy.__version__,cv2=cv2.__version__,
        cv2_threads=cv2.getNumThreads(),elapsed_seconds=duration,
        affinity_changed=False,scientific_speed_claim=False,
        actual_qualification=False,current_run_retry=False),sort_keys=True))
