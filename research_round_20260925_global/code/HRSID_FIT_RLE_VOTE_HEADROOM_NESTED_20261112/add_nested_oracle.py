"""Add the one-or-two-donor oracle to the exact closed FIT headroom runner."""
import argparse
from hashlib import sha256
import json
from math import isclose
from pathlib import Path
import sys
from types import ModuleType

PROTOCOL = "3886058af6aae8d15217fb29921eacf2ffcacedc0af146f09602ebf226760936"
PRIOR_SOURCE = "fd89f4808138f98c2103e6aa89ec10ae30ca8ac4114dee4cc11954e97c008278"
PRIOR_RESULT = "c2d06ecb71569b8358363233da73e85cfb82a8c0cc652322209028759f271945"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def donor_domain(anchor, donor1, donor2, band):
    return band & ((donor1 != anchor) | (donor2 != anchor))


def add_witness(report, old):
    kept = ("matches", "groups", "edited_matches", "edited_groups", "supported_matches", "adds", "drops",
            "cap_bound_matches", "coreless_supported_matches", "support_totals", "bootstrap_draws",
            "bootstrap_seed", "AP", "DEV", "TEST", "model_forwards", "optimizer_updates")
    require(all(report[name] == old[name] for name in kept) and
            all(report["modes"][name] == value for name, value in old["modes"].items()) and
            all(report["contrasts"][name] == value for name, value in old["contrasts"].items()),
            "Exact reproduction of all closed prior modes and fixed scope")
    contrast = report["contrasts"]
    added = contrast["oracle_any_donor_minus_oracle_disagreement"]["group_equal_delta_points"]
    remaining = contrast["oracle_full_band_minus_oracle_any_donor"]["group_equal_delta_points"]
    total = contrast["oracle_full_band_minus_oracle_disagreement"]["group_equal_delta_points"]
    require(min(added, remaining) >= 0 and total > 0 and
            isclose(added + remaining, total, rel_tol=0, abs_tol=1e-12), "Exact nested decomposition")
    report.update(status="HRSID_FIT_RLE_VOTE_NESTED_SUPPORT_ACTUAL",
                  prior_result_sha256=PRIOR_RESULT, prior_modes_verified=True,
                  source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  nested_gap=dict(any_donor_increment_points=added, remaining_points=remaining,
                                  total_points=total, fraction_any_donor=added / total))


def adapt(text):
    changes = (
        ('PROTOCOL = "6fca35ef4045b080293fe02eb79946ec92f741d774f1644aaadd78647da629ea"',
         f'PROTOCOL = "{PROTOCOL}"'),
        ('MODES = ("raw", "vote", "oracle_disagreement", "oracle_full_band")',
         'MODES = ("raw", "vote", "oracle_disagreement", "oracle_any_donor", "oracle_full_band")'),
        ('("oracle_full_band", "oracle_disagreement"), ("oracle_full_band", "raw"))',
         '("oracle_full_band", "oracle_disagreement"), ("oracle_full_band", "raw"),\n'
         '             ("oracle_any_donor", "oracle_disagreement"), ("oracle_full_band", "oracle_any_donor"))'),
        ('    cap = 2 * min(8, area // 10)',
         '    any_donor = donor_domain(a, d1, d2, band)\n'
         '    require(not np.any(disagreement & ~any_donor) and not np.any(any_donor & ~band),'
         ' "Exact nested support")\n    cap = 2 * min(8, area // 10)'),
        ('                 oracle_full_band=changed(*oracle_counts(a, y, band, adds, drops)))',
         '                 oracle_any_donor=changed(*oracle_counts(a, y, any_donor, adds, drops)),\n'
         '                 oracle_full_band=changed(*oracle_counts(a, y, band, adds, drops)))'),
        ('modes["vote"]["iou"] <= modes["oracle_disagreement"]["iou"] <= modes["oracle_full_band"]["iou"]',
         'modes["vote"]["iou"] <= modes["oracle_disagreement"]["iou"] <= modes["oracle_any_donor"]["iou"]\n'
         '            <= modes["oracle_full_band"]["iou"]'),
        ('    area.write_new(args.output / "RESULT_SAFE.json", report)',
         '    add_witness(report, prior_result)\n    area.write_new(args.output / "RESULT_SAFE.json", report)'),
    )
    for old, new in changes:
        require(text.count(old) == 1, "Exact source adaptation anchor")
        text = text.replace(old, new)
    return text


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--prior-source", required=True, type=Path)
    parser.add_argument("--prior-result", required=True, type=Path)
    params, rest = parser.parse_known_args()
    raw, prior = params.prior_source.read_bytes(), params.prior_result.read_bytes()
    require(sha256(raw).hexdigest() == PRIOR_SOURCE and sha256(prior).hexdigest() == PRIOR_RESULT,
            "Exact source and closed prior result")
    worker = ModuleType("_fit_nested_worker")
    worker.__file__ = str(params.prior_source)
    worker.__dict__.update(donor_domain=donor_domain, add_witness=add_witness, prior_result=json.loads(prior))
    exec(compile(adapt(raw.decode()), str(params.prior_source), "exec"), worker.__dict__)
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("protocol", "bridge", "capture", "witness", "code", "fit-index", "shapely-site", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    def offline(event, _args):
        if event.startswith(("socket.", "subprocess.")):
            raise RuntimeError("Offline CPU diagnostic")
    sys.addaudithook(offline)
    worker.run(parser.parse_args(rest))


if __name__ == "__main__":
    main()
