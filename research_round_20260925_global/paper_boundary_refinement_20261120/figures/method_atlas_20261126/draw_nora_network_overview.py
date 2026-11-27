"""Build the current shared schematic in the requested review locations."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    atlas = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=atlas)
    parser.add_argument("--author-scene", type=Path,
                        help="Also build the schematic in the author-review output directory.")
    args = parser.parse_args()
    generator = atlas.parent / "revision_20261126_02" / "draw_core_revision.py"
    destinations = [args.output / "public"]
    if args.author_scene:
        destinations.append(args.output / "local")
    for dest in destinations:
        subprocess.run([sys.executable, str(generator), "--output", str(dest), "--figure", "overview"], check=True)


if __name__ == "__main__":
    main()
