"""Audit the acquired fixed pilot, then write explicit lists and a local YAML."""
from collections import Counter
import json
from pathlib import Path
import shutil
import cv2
import yaml
from training_integrity import audit_split_leakage

ROOT = Path(__file__).resolve().parent
NAMES = ["vehicle", "bus", "bicycle", "pedestrian", "engine", "truck", "tricycle", "obstacle"]


def main():
    acquisition = json.loads((ROOT / "acquisition_summary.json").read_text())
    if acquisition["status"] != "downloaded_verified_crc":
        raise RuntimeError("Dataset acquisition has not completed")
    selection = json.loads((ROOT / "pilot_selection.json").read_text())
    paths = {split: [ROOT / "dataset" / name for name in names] for split, names in selection.items()}
    hashes = audit_split_leakage(paths)
    prepared, split_summary, exclusions = {}, {}, []
    for split, images in paths.items():
        objects, empty = Counter(), 0
        entries = []
        for image in images:
            frame = cv2.imread(str(image))
            if frame is None:
                raise RuntimeError("Unreadable acquired image " + image.name)
            height, width = frame.shape[:2]
            label = Path(str(image).replace("\\images\\", "\\labels\\")).with_suffix(".txt")
            if not label.is_file():
                raise RuntimeError("Missing label for " + image.name)
            lines = [line for line in label.read_text().splitlines() if line.strip()]
            invalid = False
            for line in lines:
                fields = line.split()
                if len(fields) != 5:
                    raise RuntimeError("Unexpected label format: " + image.name)
                cls, x, y, w, h = map(float, fields)
                if cls != int(cls) or not 0 <= cls < 8 or not all(0 <= v <= 1 for v in [x, y, w, h]) or w <= 0 or h <= 0:
                    invalid = True
            if invalid:
                if split != "train":
                    raise RuntimeError("Invalid evaluation label: " + image.name)
                exclusions.append({"split": split, "image": image.name,
                                   "reason": "source normalized bbox exceeds [0,1]; original data retained; entire training image excluded before metrics"})
                continue
            empty += not lines
            for line in lines:
                cls = float(line.split()[0])
                objects[NAMES[int(cls)]] += 1
            entries.append({"path": str(image), "label": str(label), "width": width, "height": height,
                            "sha256": hashes[str(image)]})
        prepared[split] = entries
        (ROOT / (split + "_pilot.txt")).write_text("\n".join(d['path'].replace("\\", "/") for d in entries) + "\n", encoding="utf-8")
        split_summary[split] = {"images": len(entries), "empty_images": empty, "objects": dict(objects),
                                "coarse_filename_prefixes": dict(Counter(p.stem[:3] for p in images))}
    config = {"path": str(ROOT).replace("\\", "/"), "train": "train_pilot.txt", "val": "val_pilot.txt",
              "test": "test_pilot.txt", "nc": 8, "names": NAMES}
    (ROOT / "pilot.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (ROOT / "prepared_dataset.json").write_text(json.dumps(prepared, indent=2), encoding="utf-8")
    summary = {"status": "audited_ready", "source": acquisition["source"], "source_version": 4,
               "selection_seed": 20261015, "splits": split_summary, "class_names": NAMES,
               "exact_cross_split_duplicates": 0, "exclusions": exclusions,
               "scope": "fixed pilot from official split; camera independence not established; coarse filename groups overlap across splits",
               "ebike_separately_defined": False}
    (ROOT / "dataset_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # Ultralytics 8.3.43 AMP check resolves this exact filename in the working directory.
    shutil.copy2(ROOT.parent / "weights" / "yolo11n.pt", ROOT / "yolo11n.pt")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
