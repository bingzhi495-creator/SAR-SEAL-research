"""Add one nested support to the exact independently closed DEV187 headroom readout."""
import argparse
from hashlib import sha256
import json
from math import isclose
from pathlib import Path
import sys
from types import ModuleType

PROTOCOL = "03d7b82dafe253d0703f9ff1ff47290d506db75b4afec0ec66ab6f2dcfd23813"
PRIOR_CODE = "0e5f4e922baa9b2e0766af0329bde05a0cd084c5e4fb1d3b7579101521ed963b"
PRIOR_RESULT = "a57dc12f3e5e835f8b3acaee0fd4959e5176d76fcc0d25aa6cb9b7e324dc7748"


def donor_domain(anchor, first, second, band):
    return band & ((first != anchor) | (second != anchor))


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def exact_subset(old, new):
    if isinstance(old, dict):
        return old.keys() <= new.keys() and all(exact_subset(v, new[k]) for k, v in old.items())
    if isinstance(old, list):
        return len(old) == len(new) and all(exact_subset(a, b) for a, b in zip(old, new, strict=True))
    if type(old) is float:
        return isclose(old, new, rel_tol=0, abs_tol=1e-12)
    return old == new


def add_witness(result, prior):
    unchanged = (
        "AP", "DEV", "TEST", "model_forwards", "optimizer_updates", "original_matches", "edited_matches",
        "untouched_matches", "jpeg_groups", "paired_bootstrap_draws", "paired_bootstrap_seed", "precision",
        "editable_support", "group_equal_iou_points", "match_equal_iou_points", "effects", "fixed_thresholds",
    )
    require(all(exact_subset(prior[key], result[key]) for key in unchanged), "Every original mode result reproduced")
    effects = result["effects"]
    captured = effects["any_donor_oracle_minus_majority_oracle"]["group_mean_points"]
    remaining = effects["full_band_oracle_minus_any_donor_oracle"]["group_mean_points"]
    total = effects["full_band_oracle_minus_majority_oracle"]["group_mean_points"]
    require(captured >= 0 and remaining >= 0 and total > 0 and
            isclose(captured + remaining, total, rel_tol=0, abs_tol=1e-12), "Exact nested gap composition")
    result.update(status="HRSID_TO_SSDD_DEV187_RLE_NESTED_SUPPORT_ACTUAL",
                  prior_headroom_sha256=PRIOR_RESULT, prior_mode_witness=True,
                  source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  full_minus_both_gap=dict(any_donor_increment_points=captured, remaining_points=remaining,
                                           total_points=total, fraction_any_donor=captured/total,
                                           fraction_both_agree=remaining/total))


def adapt(raw):
    substitutions = (
        ('PROTOCOL_SHA = "3eca9965da9dc8d8a84fdc537fa737737443f99bba1df4b0d35df8454b349098"',
         f'PROTOCOL_SHA = "{PROTOCOL}"'),
        ('MODES = ("raw", "vote", "majority_oracle", "full_band_oracle")',
         'MODES = ("raw", "vote", "majority_oracle", "any_donor_oracle", "full_band_oracle")'),
        ('            voted = codec.decode(change["vote"]).astype(bool)',
         '            any_donor = donor_domain(anchor, donors[0], donors[1], band)\n'
         '            require(not np.any(majority & ~any_donor) and not np.any(any_donor & ~band),\n'
         '                    "Exact nested native pixel domains")\n'
         '            voted = codec.decode(change["vote"]).astype(bool)'),
        ('                         full_band_oracle=oracle_mask(anchor, truth, band, a, d))',
         '                         any_donor_oracle=oracle_mask(anchor, truth, any_donor, a, d),\n'
         '                         full_band_oracle=oracle_mask(anchor, truth, band, a, d))'),
        ('            support["full_band_oracle"].update(support_counts(band, anchor, truth, a, d))',
         '            support["any_donor_oracle"].update(support_counts(any_donor, anchor, truth, a, d))\n'
         '            support["full_band_oracle"].update(support_counts(band, anchor, truth, a, d))'),
        ('scores[3] >= scores[2] >= scores[1]', 'scores[4] >= scores[3] >= scores[2] >= scores[1]'),
        ('comparisons = ((1, 0), (2, 0), (3, 0), (2, 1), (3, 1), (3, 2))',
         'comparisons = ((1, 0), (2, 0), (4, 0), (2, 1), (4, 1), (4, 2), (3, 0), (3, 1), (3, 2), (4, 3))'),
        ('    digest = worker.write_new(args.output, worker.stable_json(result))',
         '    add_witness(result, previous_result)\n    digest = worker.write_new(args.output, worker.stable_json(result))'),
    )
    for old, new in substitutions:
        require(raw.count(old) == 1, "Exact source adaptation anchor")
        raw = raw.replace(old, new)
    return raw


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--prior-source", required=True, type=Path)
    parser.add_argument("--prior-result", required=True, type=Path)
    args, rest = parser.parse_known_args()
    raw = args.prior_source.read_bytes()
    result = args.prior_result.read_bytes()
    require(sha256(raw).hexdigest() == PRIOR_CODE and sha256(result).hexdigest() == PRIOR_RESULT,
            "Exact closed prior source and result")
    namespace = ModuleType("_nested_oracle")
    namespace.__file__ = str(args.prior_source)
    namespace.__dict__.update(donor_domain=donor_domain, add_witness=add_witness, previous_result=json.loads(result))
    exec(compile(adapt(raw.decode()), str(args.prior_source), "exec"), namespace.__dict__)
    sys.argv[1:] = rest
    namespace.main()


if __name__ == "__main__":
    main()