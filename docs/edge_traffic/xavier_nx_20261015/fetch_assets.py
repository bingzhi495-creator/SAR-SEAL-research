"""Download official model assets and an official Roboflow example video locally."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent
ASSETS = {
    "weights/yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
    "weights/yolov8s.pt": "https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8s.pt",
    "weights/yolo11n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
    "input/intersection_pexels_5675618.mp4": "https://videos.pexels.com/video-files/5675618/5675618-hd_1920_1080_30fps.mp4",
}


def fetch(item):
    name, url = item
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with requests.get(url, stream=True, timeout=(20, 90)) as response:
            response.raise_for_status()
            if "text/html" in response.headers.get("Content-Type", ""):
                raise RuntimeError(f"Expected binary, received HTML: {name}")
            temporary = path.with_suffix(path.suffix + ".partial")
            with temporary.open("wb") as target:
                for block in response.iter_content(1024 * 1024):
                    target.write(block)
            temporary.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"file": name, "source": url, "bytes": path.stat().st_size,
            "sha256": digest, "publication": "local only; not uploaded to research repository"}


if __name__ == "__main__":
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(fetch, item): item[0] for item in ASSETS.items()}
        for future in concurrent.futures.as_completed(futures):
            try:
                row = future.result()
            except Exception as error:
                row = {"file": futures[future], "error": str(error)}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    (ROOT / "asset_manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
