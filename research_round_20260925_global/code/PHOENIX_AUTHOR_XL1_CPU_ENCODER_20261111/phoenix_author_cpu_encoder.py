"""Run the fixed Phoenix author XL1 image encoder on its official public demo."""
from argparse import ArgumentParser
from hashlib import sha1, sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter

REVISION = "13ee3996ef3d0292fab144316cc8e03b69c0ecfa"
MANIFEST_SHA = "92d5084f8c7d7bceb949fd099489edc5cd835b939dae49959c1df4de026fb853"
WEIGHT_SHA = "9b6ba211e0d6d434085d3d5e379eb527d8b56ef413628f42cb587153cc51af7d"
WEIGHT_BYTES = 814868059
IMAGE_SHA = "ef4645c13e20856c1933d766168540461b9dc55b8db496d95d2cd4a9eb897319"
MASK_SHA = "b0717a0b6d2cdb5b943ecc00a15dc44debdc6a9c2fdce383375008ddc37cbefc"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_checkpoint(path):
    require(path.stat().st_size == WEIGHT_BYTES, "Exact official checkpoint size required")
    digest = sha256()
    with path.open("rb") as source:
        while block := source.read(1 << 20):
            digest.update(block)
    require(digest.hexdigest() == WEIGHT_SHA, "Exact official checkpoint bytes required")


def forbid_network(event, _):
    if event.startswith("socket."):
        raise PermissionError("Offline author forward")


def main():
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("source-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    root = args.source_root
    require(root.is_absolute() and root.resolve(strict=True) == root and args.output.is_absolute()
            and not args.output.exists(), "Canonical source and fresh output required")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "CPU device visibility required")
    sys.addaudithook(forbid_network)
    os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"
    start = perf_counter()

    manifest_bytes = (root / "SOURCE_ACQUISITION_SAFE.json").read_bytes()
    require(sha256(manifest_bytes).hexdigest() == MANIFEST_SHA, "Exact source manifest bytes required")
    manifest = json.loads(manifest_bytes)
    require(manifest["revision"] == REVISION and len(manifest["files"]) == 46,
            "Exact author source revision required")
    for relative, wanted in manifest["files"].items():
        path = root / relative
        require(path.resolve(strict=True).is_relative_to(root), "Author source path escapes root")
        value = path.read_bytes()
        prefix = f"blob {len(value)}\x00".encode("ascii")
        require(sha1(prefix + value).hexdigest() == wanted, "Exact author source blob differs")

    image_path = root / "assets/examples/instance/00_image.jpg"
    mask_path = root / "assets/examples/instance/00_noisy.png"
    checkpoint = root / "phoenix_efficientvit_xl1.pt"
    require(sha256(image_path.read_bytes()).hexdigest() == IMAGE_SHA
            and sha256(mask_path.read_bytes()).hexdigest() == MASK_SHA, "Exact author demo assets required")
    check_checkpoint(checkpoint)

    import numpy as np
    from PIL import Image
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(root))
    import phoenix
    require(Path(phoenix.__file__).resolve() == root / "phoenix/__init__.py",
            "Import must use the verified author module")
    require(not torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint),
            "Tensor-only load of author checkpoint is required")

    load_start = perf_counter()
    model = phoenix.build_phoenix(checkpoint=str(checkpoint), encoder="efficientvit_xl1", device="cpu", strict=True)
    load_seconds = perf_counter() - load_start
    refiner = phoenix.PhoenixRefiner(model)
    with Image.open(image_path) as source:
        image = np.array(source.convert("RGB"))
    with Image.open(mask_path) as source:
        mask_shape = (source.height, source.width)
    require(image.shape[:2] == mask_shape, "Original author image and mask geometry differ")

    encode_start = perf_counter()
    with torch.inference_mode():
        refiner.set_image(image)
    encode_seconds = perf_counter() - encode_start
    features = refiner.predictor.get_image_embedding()
    require(tuple(features.shape) == (1, 256, 64, 64) and features.device.type == "cpu"
            and features.dtype == torch.float32 and bool(torch.isfinite(features).all())
            and not torch.cuda.is_initialized(), "Expected finite author CPU embedding required")
    report = dict(status="PHOENIX_AUTHOR_XL1_OFFICIAL_DEMO_CPU_ENCODER_PASS", author_commit=REVISION,
                  source_manifest_sha256=MANIFEST_SHA, checkpoint_sha256=WEIGHT_SHA,
                  checkpoint_bytes=WEIGHT_BYTES, official_demo_image_sha256=IMAGE_SHA,
                  official_demo_mask_sha256=MASK_SHA, image_shape=list(image.shape),
                  embedding_shape=list(features.shape), embedding_dtype=str(features.dtype),
                  embedding_sha256=sha256(features.detach().contiguous().numpy().tobytes()).hexdigest(),
                  state_keys=len(model.state_dict()), parameters=sum(value.numel() for value in model.parameters()),
                  pytorch=str(torch.__version__), numpy=np.__version__,
                  load_seconds=load_seconds, encode_seconds=encode_seconds, wall_seconds=perf_counter()-start,
                  device="cpu", cuda_initialized=False, fastgeodis_loaded="FastGeodis" in sys.modules,
                  decoder_called=False, sar_images=0, official_TEST_opened=False, AP=False)
    raw = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with args.output.open("xb") as output:
        output.write(raw)
    print(json.dumps(dict(status=report["status"], report_sha256=sha256(raw).hexdigest())), flush=True)


if __name__ == "__main__":
    main()
