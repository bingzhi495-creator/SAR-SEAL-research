"""Apply the fixed first-pair HRSID DEV705 raw-GT mask AP decision rule."""

from decimal import Decimal, localcontext


HISTORICAL_AP = Decimal("0.5042744492075145")
HISTORICAL_REFERENCE_SHA256 = "91e7af64d6401e6a6635e94375cf8d524fa572e8a4c0a43c45979ddac77ee42d"
GATE = Decimal("0.01")


def ap_fraction(raw):
    if type(raw) not in (int, float, Decimal):
        raise ValueError("Expected a finite numeric AP fraction in [0, 1]")
    value = Decimal(str(raw))
    if not value.is_finite() or not 0 <= value <= 1:
        raise ValueError("Expected a finite numeric AP fraction in [0, 1]")
    return value if value else Decimal(0)


def decide(native_ap, center_ap):
    """Use original-GT AP fractions and return exact decimal strings."""
    native = ap_fraction(native_ap)
    center = ap_fraction(center_ap)
    historical = HISTORICAL_AP
    benchmark = max(native, historical)
    stronger = "both" if native == historical else "native" if native > historical else "historical"

    # Preserve the exact decimal difference even for a positive AP near zero.
    with localcontext() as context:
        context.prec = max(28, 4 - min(value.as_tuple().exponent for value in (native, center, historical)))
        gap = center - benchmark
        if gap >= GATE:
            decision = "ADVANCE_PRELOCKED_REPEATS"
        elif gap > 0:
            decision = "ALLOW_ONE_PRELOCKED_PAIR"
        else:
            decision = "STOP_CURRENT_RECIPE"
        return dict(
            native_ap_fraction=format(native, "f"),
            center_ap_fraction=format(center, "f"),
            historical_ap_fraction=format(historical, "f"),
            historical_reference_sha256=HISTORICAL_REFERENCE_SHA256,
            stronger_reference=stronger,
            stronger_reference_ap_fraction=format(benchmark, "f"),
            center_minus_native_ap_points=format(100 * (center - native), "f"),
            center_minus_historical_ap_points=format(100 * (center - historical), "f"),
            center_minus_stronger_reference_ap_points=format(100 * gap, "f"),
            gate_ap_points="1",
            decision=decision,
        )
