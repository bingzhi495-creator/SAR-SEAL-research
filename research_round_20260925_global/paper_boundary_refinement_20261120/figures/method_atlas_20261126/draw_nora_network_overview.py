"""Rebuild Fig. 1 with the public SAR aggregate and optional local author SAR input."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    atlas = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=atlas)
    parser.add_argument("--author-scene", type=Path, help="Local admitted crop PNG, or the admitted source PDF.")
    parser.add_argument("--author-image-xref", type=int, help="Image object for a local source PDF.")
    args = parser.parse_args()
    generator = atlas.parent / "revision_20261126_02" / "draw_core_revision.py"
    command = [sys.executable, str(generator), "--figure", "overview"]
    subprocess.run([*command, "--output", str(args.output / "public")], check=True)
    if args.author_scene:
        native = [*command, "--output", str(args.output / "local"), "--author-scene", str(args.author_scene)]
        if args.author_image_xref is not None:
            native += ["--author-image-xref", str(args.author_image_xref)]
        subprocess.run(native, check=True)


if __name__ == "__main__":
    main()
