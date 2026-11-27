"""Build the scalar and dual ROH diagram in the historical output directory."""
from pathlib import Path
import subprocess
import sys


def build(output):
    figures = Path(__file__).resolve().parent.parent
    generator = figures / "revision_20261126_02" / "draw_core_revision.py"
    subprocess.run([sys.executable, str(generator), "--output", str(output), "--figure", "roh"], check=True)


if __name__ == "__main__":
    build(Path(__file__).resolve().parent / "public")
