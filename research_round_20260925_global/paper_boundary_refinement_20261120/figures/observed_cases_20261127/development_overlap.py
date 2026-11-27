"""Draw the two verified HRSID DEV705 overlap decompositions at final print size."""
from pathlib import Path
import argparse
import json
import math

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

WIDTH, HEIGHT = 265, 118
INK = "#362E29"
MUTED = "#62554B"
RULE = "#C7BBAE"
PAPER = "#FFFEFB"
FILL = "#F4EFE6"
GAIN = "#285E4E"
LOSS = "#9B442F"

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "svg.fonttype": "none",
    "svg.hashsalt": "nora-dev705-matched-cells",
    "pdf.fonttype": 42,
    "pdf.compression": 8,
})


def aggregate_source():
    here = Path(__file__).resolve().parent
    for file in (here / "dev705_overlap_cases.json",
                 here.parent / "dev705_overlap_cases.json",
                 here.parent.parent / "paper_public" / "figures" / "dev705_overlap_cases.json"):
        if file.is_file():
            return file
    raise FileNotFoundError("Pass --input pointing to the published dev705_overlap_cases.json.")


def signed(value, precision=1):
    if value == 0:
        return f"{0:.{precision}f}"
    sign = "+" if value > 0 else "\N{MINUS SIGN}"
    return sign + f"{abs(value):.{precision}f}"


def draw(source, destination):
    record = json.loads(source.read_text(encoding="utf-8"))
    assert record["seed"] == 5217
    cases = record["cases"]
    assert [case["sign"] for case in cases] == ["positive", "negative"]
    fig = plt.figure(figsize=(WIDTH / 72, HEIGHT / 72), facecolor=PAPER)
    ax = fig.add_axes((0, 0, 1, 1), xlim=(0, WIDTH), ylim=(HEIGHT, 0))
    ax.axis("off")

    def put(x, y, msg, size=8.0, color=INK, weight="normal", align="left"):
        return ax.text(x, y, msg, fontsize=size, color=color, fontweight=weight,
                       ha=align, va="center", clip_on=False)

    put(5, 8, "Intersection (cell area)", 8.5, weight="bold")
    put(260, 8, "8 additions + 8 removals each", 7.5, MUTED, align="right")
    ax.plot([5, 260], [15, 15], lw=0.65, color=RULE)
    number_x = (153, 207, 252)
    put(7, 23, "Placement rule", 7.7, MUTED)
    for x, name in zip(number_x, ("Added", "Removed", "Net")):
        put(x, 23, name, 7.7, MUTED, align="right")

    for case, label, accent, y in zip(cases, ("(a) Gain", "(b) Loss"),
                                       (GAIN, LOSS), (35.5, 81.5)):
        fixed, independent = (case["arms"][k] for k in ("SF", "SU=F"))
        delta = 100 * (independent["iou"] - fixed["iou"])
        assert math.isclose(delta, 100 * case["delta_iou"], abs_tol=1e-10)
        ax.add_patch(Rectangle((5, y - 6.1), 255, 12.2, fc=FILL, ec="none"))
        ax.add_patch(Rectangle((5, y - 6.1), 2, 12.2, fc=accent, ec="none"))
        put(11, y, label, 8.2, accent, "bold")
        put(255, y, "ΔIoU (I − F)  " + signed(delta, 3) + " pt", 8.1, accent, "bold", "right")
        for arm, title, row_y in zip((fixed, independent), ("Distance-paired (F)", "Independent (I)"),
                                      (y + 14.3, y + 26.7)):
            assert (arm["adds"], arm["drops"]) == (8, 8)
            gained = arm["add_overlap"]
            lost = -arm["drop_overlap"]
            net = gained + lost
            assert math.isclose(net, arm["I"] - case["arms"]["N"]["I"], abs_tol=1e-10)
            put(10, row_y, title, 8.0)
            for j, (x, amount) in enumerate(zip(number_x, (gained, lost, net))):
                put(x, row_y, signed(amount), 8.3, INK, "bold" if j == 2 else "normal", "right")
    destination.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination / "development_overlap.pdf", metadata={"Creator": "Matplotlib", "CreationDate": None})
    fig.savefig(destination / "development_overlap.svg", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Published aggregate case JSON.")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    draw(args.input or aggregate_source(), args.output)

