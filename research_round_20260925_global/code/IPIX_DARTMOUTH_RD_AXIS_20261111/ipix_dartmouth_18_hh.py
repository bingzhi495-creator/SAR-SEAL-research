"""Check range and Doppler axes for the first HH window of IPIX Dartmouth #18."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
from scipy.io import netcdf_file


CDF_SHA = "f296aa6fdccc0be79e4c548bb21e06fb94ca1723e9c07191375c57c9461eda40"
MATLAB_SHA = "40f5499eeb0d1b1d7e158658bdcfddb31a18ef01eea4023e7da8ca0ba21bd517"
WINDOW = 512
PRIMARY = 9


def require(valid, description):
    if not valid:
        raise ValueError(description)


def fixed_file(path, expected_bytes, expected_sha):
    require(path.is_file() and not path.is_symlink() and path.stat().st_size == expected_bytes,
            "Exact acquired McMaster file size required")
    with path.open("rb") as stream:
        observed = sha256()
        while block := stream.read(1024**2):
            observed.update(block)
    require(observed.hexdigest() == expected_sha, "Exact acquired McMaster file SHA-256 required")


def load_hh(path):
    with netcdf_file(path, mode="r", mmap=False) as source:
        require(source.Organization == b"McMaster University" and source.TX_polarization == b"A",
                "Expected Dartmouth alternating-polarization source")
        adc = source.variables["adc_data"]
        require(adc.dimensions == ("nsweep", "ntxpol", "nrange", "nadc")
                and adc.data.shape == (131072, 2, 14, 4) and adc.data.dtype == np.int8,
                "Exact source axes and signed-byte container required")
        i_index, q_index = (int(source.variables[name].data) for name in ("adc_like_I", "adc_like_Q"))
        prf_variable = source.variables["PRF"]
        prf = float(prf_variable.data)
        decimation = int(source.variables["decimation"].data)
        require((i_index, q_index, prf, decimation) == (0, 1, 1000.0, 1)
                and prf_variable.units == b"Hertz"
                and prf_variable.long_name == b"Radar pulse repetition frequency per polarization",
                "Exact author HH indices and per-polarization PRF metadata required")
        ranges = source.variables["range"]
        require(ranges.units == b"metres" and ranges.long_name == b"Range to near edge of radar resolution cell",
                "Ranges must be the original cell near edges in metres")
        near_edges = ranges.data.astype(np.float64)
        require(np.max(np.abs(np.diff(near_edges) - 15.0)) < 0.001, "Exact nominal 15 m spacing required")
        # ipixload.m's HH branch selects horizontal TX index 1 in MATLAB,
        # then reinterprets signed file bytes as unsigned ADC counts.
        signed = adc.data[:WINDOW, 0, :, :][:, :, [i_index, q_index]]
        unsigned = signed.view(np.uint8)
        require(np.array_equal(unsigned, signed.astype(np.int16) % 256),
                "Unsigned byte reinterpretation must agree with arithmetic conversion")
        hh = unsigned[:, :, 0].astype(np.float64) + 1j * unsigned[:, :, 1]
        require(np.isfinite(hh).all(), "Finite HH IQ data required")
        return hh, near_edges, dict(
            cdf_prf_hz_per_polarization=prf, cdf_decimation=decimation,
            cdf_carrier_frequency_ghz=float(source.variables["RF_frequency"].data),
            cdf_pulse_width_ns=float(source.variables["Pulse_length"].data),
            cdf_unambiguous_radial_speed_m_per_s=float(source.variables["Unambig_velocity"].data),
            unsigned_conversion_arithmetic_verified=True,
            raw_signed_container_values_below_zero=int(np.count_nonzero(signed < 0)),
        )


def transform(window, prf):
    freq = np.fft.fftshift(np.fft.fftfreq(len(window), d=1 / prf))
    spectrum = np.fft.fftshift(np.fft.fft(window, axis=0, norm="ortho"), axes=0)
    power = np.abs(spectrum) ** 2
    temporal_energy = np.sum(np.abs(window) ** 2, axis=0)
    spectral_energy = np.sum(power, axis=0)
    restored = np.fft.ifft(np.fft.ifftshift(spectrum, axes=0), axis=0, norm="ortho")
    parseval = float(np.max(np.abs(spectral_energy - temporal_energy) / temporal_energy))
    inverse = float(np.linalg.norm(restored - window) / np.linalg.norm(window))
    require(np.isfinite(power).all() and parseval < 1e-12 and inverse < 1e-12, "FFT invariants failed")
    peak = int(np.argmax(power[:, PRIMARY - 1]))
    return freq, dict(temporal_energy=float(temporal_energy.sum()), spectral_energy=float(spectral_energy.sum()),
                      max_per_gate_parseval_relative_error=parseval, inverse_relative_l2_error=inverse,
                      primary_gate_peak_frequency_hz=float(freq[peak]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("cdf", "official-loader", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    start = perf_counter()
    fixed_file(args.cdf, 15732284, CDF_SHA)
    fixed_file(args.official_loader, 4051, MATLAB_SHA)
    hh, near_edges, acquisition = load_hh(args.cdf)
    prf = acquisition["cdf_prf_hz_per_polarization"]
    frequency, raw = transform(hh, prf)
    centered = hh - hh.mean(axis=0, keepdims=True)
    same_axis, hann = transform(centered * np.hanning(WINDOW)[:, None], prf)
    require(np.array_equal(frequency, same_axis) and frequency[WINDOW // 2] == 0.0,
            "Windowing must preserve the per-polarization FFT axis")
    report = dict(
        status="IPIX_DARTMOUTH_18_HH_RD_AXIS_ENGINEERING_PASS_V1",
        source_id="19931107_141630_starea.cdf", source_bytes=15732284, source_sha256=CDF_SHA,
        official_ipixload_sha256=MATLAB_SHA, numpy=np.__version__, scipy=scipy.__version__,
        polarization="HH", sweeps_hh=WINDOW, hh_start_zero_based=0, cpi_samples_over_prf_seconds=WINDOW/prf,
        rd_shape_frequency_by_range=[WINDOW, len(near_edges)], cdf_adc_shape=[131072, 2, 14, 4],
        frequency_first_hz=float(frequency[0]), frequency_last_hz=float(frequency[-1]),
        frequency_bin_spacing_hz=float(frequency[1] - frequency[0]), zero_doppler_index_zero_based=WINDOW // 2,
        range_cell_near_edges_m=near_edges.tolist(), nominal_range_gate_spacing_m=15.0,
        catalog_range_resolution_m=30.0, catalog_primary_range_gate_one_based=PRIMARY,
        catalog_primary_range_cell_near_edge_m=float(near_edges[PRIMARY-1]),
        catalog_target_neighborhood_gates_one_based=[8, 9, 10, 11],
        catalog_target_type="one_metre_wire_mesh_wrapped_styrofoam_sphere",
        rectangular_unsigned_raw_iq=raw, sample_demeaned_symmetric_hann_iq=hann,
        radial_velocity_sign_calibrated=False, physical_amplitude_calibrated=False,
        independent_velocity_truth=False, instance_masks=False, trained_or_evaluated_model=False,
        raw_data_redistributed=False, runtime_seconds=perf_counter() - start, **acquisition,
    )
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
