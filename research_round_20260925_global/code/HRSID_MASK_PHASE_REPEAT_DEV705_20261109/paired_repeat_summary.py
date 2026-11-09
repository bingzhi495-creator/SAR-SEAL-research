"""Summarize the P216 three-seed HRSID DEV705 original-GT AP result."""

from decimal import Decimal, localcontext
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
import re


SEEDS = (5201, 5202, 5203)
SCREENING_SEED = SEEDS[0]
CONFIRMATION_SEEDS = SEEDS[1:]
ROW_FIELDS = {"seed", "native_ap_fraction", "center_ap_fraction"}
DECISION_SOURCE = (
    Path(__file__).resolve().parent.parent / "HRSID_MASK_PHASE_PAIRED_GEOMETRY_20261109" / "paired_decision.py"
)
DECISION_SHA = "240be2455669afc59bb38f127454fba9388ba651f1ddd74b99f1467523b921d8"


def load_fixed_reference():
    raw = DECISION_SOURCE.read_bytes()
    if sha256(raw).hexdigest() != DECISION_SHA:
        raise ValueError("Exact immutable first-pair benchmark source required")
    source = {"__name__": "_three_seed_fixed_ap_reference"}
    exec(compile(raw, str(DECISION_SOURCE), "exec"), source)
    return source


_FIXED = load_fixed_reference()
HISTORICAL_AP = _FIXED["HISTORICAL_AP"]
HISTORICAL_REFERENCE_SHA256 = _FIXED["HISTORICAL_REFERENCE_SHA256"]
GATE = _FIXED["GATE"]


def parse_fraction(value):
    if type(value) is str:
        if re.fullmatch(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?", value) is None:
            raise ValueError("Expected a finite numeric AP fraction in [0, 1]")
        value = Decimal(value)
    return _FIXED["ap_fraction"](value)


def decimal_fraction(value):
    return Decimal(value.numerator) / Decimal(value.denominator)


def fraction_text(value):
    return format(decimal_fraction(value), "f")


def summarize(rows):
    """Summarize all three independently accepted AP pairs on the fixed DEV GT.

    Gate arithmetic uses exact rational numbers from unrounded input decimals.
    Repeating means and square roots are rendered as decimal approximations.
    The caller provides only original-GT AP from independently accepted runs.
    """
    ap = {}
    for row in rows:
        if type(row) is not dict or row.keys() != ROW_FIELDS:
            raise ValueError("Exact seed, native AP and center AP fields required")
        seed = row["seed"]
        if type(seed) is not int or seed not in SEEDS or seed in ap:
            raise ValueError("Exactly one pair for each prelocked seed 5201, 5202 and 5203 required")
        ap[seed] = parse_fraction(row["native_ap_fraction"]), parse_fraction(row["center_ap_fraction"])
    if ap.keys() != set(SEEDS):
        raise ValueError("Exactly one pair for each prelocked seed 5201, 5202 and 5203 required")

    rational = {seed: (Fraction(a), Fraction(b)) for seed, (a, b) in ap.items()}
    history = Fraction(HISTORICAL_AP)
    differences = {seed: b - a for seed, (a, b) in rational.items()}
    mean_native = sum(a for a, _ in rational.values()) / len(SEEDS)
    mean_center = sum(b for _, b in rational.values()) / len(SEEDS)
    mean_difference = mean_center - mean_native
    benchmark = max(mean_native, history)
    if mean_native == history:
        stronger_reference = "both"
    else:
        stronger_reference = "native_mean" if mean_native > history else "historical"
    gap = mean_center - benchmark
    each_above = all(delta > 0 for delta in differences.values())
    margin_pass = gap >= Fraction(GATE)
    variance = sum((delta - mean_difference) ** 2 for delta in differences.values()) / (len(SEEDS) - 1)

    with localcontext() as context:
        context.prec = max(40, 4 - min(value.as_tuple().exponent for pair in ap.values() for value in pair))
        confirmation_a = sum(rational[seed][0] for seed in CONFIRMATION_SEEDS) / len(CONFIRMATION_SEEDS)
        confirmation_b = sum(rational[seed][1] for seed in CONFIRMATION_SEEDS) / len(CONFIRMATION_SEEDS)
        return dict(
            status="HRSID_MASK_PHASE_E24_THREE_SEED_SUMMARY_V1",
            scope="fixed_HRSID_DEV705_original_GT_E24_shared_initialization",
            historical_ap_fraction=format(HISTORICAL_AP, "f"),
            historical_reference_sha256=HISTORICAL_REFERENCE_SHA256,
            gate_ap_points=fraction_text(100 * Fraction(GATE)),
            paired_results=[
                dict(
                    seed=seed, role="screening" if seed == SCREENING_SEED else "confirmation",
                    native_ap_fraction=format(ap[seed][0], "f"), center_ap_fraction=format(ap[seed][1], "f"),
                    center_minus_native_ap_points=fraction_text(100 * differences[seed]),
                )
                for seed in SEEDS
            ],
            all_seeds=dict(
                seeds=list(SEEDS), mean_native_ap_fraction=fraction_text(mean_native),
                mean_center_ap_fraction=fraction_text(mean_center),
                mean_paired_difference_ap_points=fraction_text(100 * mean_difference),
                sample_sd_paired_difference_ap_points=format(100 * decimal_fraction(variance).sqrt(), "f"),
                stronger_reference=stronger_reference, stronger_reference_ap_fraction=fraction_text(benchmark),
                mean_center_minus_historical_ap_points=fraction_text(100 * (mean_center - history)),
                mean_center_minus_stronger_reference_ap_points=fraction_text(100 * gap),
                each_center_above_native=each_above, passes_ap_margin=margin_pass,
            ),
            confirmation_only=dict(
                seeds=list(CONFIRMATION_SEEDS), mean_native_ap_fraction=fraction_text(confirmation_a),
                mean_center_ap_fraction=fraction_text(confirmation_b),
                mean_paired_difference_ap_points=fraction_text(100 * (confirmation_b - confirmation_a)),
            ),
            decision=("MEETS_PRELOCKED_DEV_UTILITY_GATE" if each_above and margin_pass
                      else "DOES_NOT_MEET_PRELOCKED_DEV_UTILITY_GATE"),
        )
