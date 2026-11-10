"""Verify the full Phoenix XL1 CPU inference path on the fixed author demo."""
from argparse import ArgumentParser
from hashlib import file_digest, sha1, sha256
import json
import os
from pathlib import Path
import random
import resource
import sys
from time import perf_counter


REVISION = "13ee3996ef3d0292fab144316cc8e03b69c0ecfa"
WEIGHT_BYTES = 814868059
PINS = {
    "SOURCE_ACQUISITION_SAFE.json": "92d5084f8c7d7bceb949fd099489edc5cd835b939dae49959c1df4de026fb853",
    "phoenix_efficientvit_xl1.pt": "9b6ba211e0d6d434085d3d5e379eb527d8b56ef413628f42cb587153cc51af7d",
    "assets/examples/instance/00_image.jpg": "ef4645c13e20856c1933d766168540461b9dc55b8db496d95d2cd4a9eb897319",
    "assets/examples/instance/00_noisy.png": "b0717a0b6d2cdb5b943ecc00a15dc44debdc6a9c2fdce383375008ddc37cbefc",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def verify_inputs(root):
    require((root / "phoenix_efficientvit_xl1.pt").stat().st_size == WEIGHT_BYTES,
            "Official checkpoint length differs")
    for relative, expected in PINS.items():
        path = root / relative
        require(path.resolve(strict=True).is_relative_to(root) and sha256_file(path) == expected,
                "Fixed author input differs")
    manifest = json.loads((root / "SOURCE_ACQUISITION_SAFE.json").read_bytes())
    require(manifest["revision"] == REVISION and len(manifest["files"]) == 46,
            "Author source revision differs")
    for relative, expected in manifest["files"].items():
        path = root / relative
        require(path.resolve(strict=True).is_relative_to(root), "Author source path escapes root")
        value = path.read_bytes()
        git_blob = sha1(f"blob {len(value)}\x00".encode("ascii") + value).hexdigest()
        require(git_blob == expected, "Exact author source blob differs")


def deny_network(event, _):
    if event.startswith("socket."):
        raise PermissionError("Official offline inference")


def main():
    parser = ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.source_root
    require(root.is_absolute() and root.resolve(strict=True) == root and args.output.is_absolute()
            and args.output.parent.resolve(strict=True) == args.output.parent and not args.output.exists(),
            "Canonical input and fresh output required")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "CPU device visibility required")
    start = perf_counter()
    sys.addaudithook(deny_network)
    os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "1"
    verify_inputs(root)

    import numpy as np
    from PIL import Image
    import torch
    from importlib.metadata import version
    import FastGeodis

    require(version("FastGeodis") == "1.0.5", "Exact FastGeodis release required")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(root))
    import phoenix

    require(Path(phoenix.__file__).resolve() == root / "phoenix/__init__.py", "Wrong author source import")
    with Image.open(root / "assets/examples/instance/00_image.jpg") as image_file:
        image = np.asarray(image_file.convert("RGB"))
    with Image.open(root / "assets/examples/instance/00_noisy.png") as mask_file:
        mask = np.array(mask_file.convert("L"))
    require(mask.shape == image.shape[:2] and (mask > .5).any(), "Author demo image/mask geometry differs")
    checkpoint = root / "phoenix_efficientvit_xl1.pt"
    require(not torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint), "Tensor-only checkpoint required")
    tick = perf_counter()
    model = phoenix.build_phoenix(checkpoint=str(checkpoint), encoder="efficientvit_xl1", device="cpu", strict=True)
    load_seconds = perf_counter() - tick
    refiner = phoenix.PhoenixRefiner(model)

    tick = perf_counter()
    with torch.inference_mode():
        refiner.set_image(image)
    encode_seconds = perf_counter() - tick
    features = refiner.predictor.get_image_embedding()
    require(tuple(features.shape) == (1, 256, 64, 64) and features.device.type == "cpu"
            and features.dtype == torch.float32 and bool(torch.isfinite(features).all()),
            "Expected finite author CPU image embedding")

    original_geodis = FastGeodis.generalised_geodesic2d
    original_decoder = refiner.predictor.predict_torch
    original_mask = mask > .5

    def run(operation):
        observations = {"geodesic": 0, "decoder": 0}
        random.seed(5202)
        np.random.seed(5202)
        torch.random.default_generator.manual_seed(5202)

        def observed_geodis(image_tensor, softmask, *values, **kwargs):
            require(image_tensor.device.type == softmask.device.type == "cpu", "Expected native CPU prompt tensors")
            result = original_geodis(image_tensor, softmask, *values, **kwargs)
            require(result.shape == softmask.shape and result.device.type == "cpu"
                    and bool(torch.isfinite(result).all()), "Native CPU geodesic output differs")
            observations["geodesic"] += 1
            return result

        def observed_decoder(**prompts):
            result = original_decoder(**prompts)
            require(result["logits"].device.type == "cpu" and bool(torch.isfinite(result["logits"]).all())
                    and bool(torch.isfinite(result["iou_predictions"]).all()), "Author CPU decoder output differs")
            observations["decoder"] += 1
            return result

        FastGeodis.generalised_geodesic2d = observed_geodis
        refiner.predictor.predict_torch = observed_decoder
        tick = perf_counter()
        try:
            with torch.inference_mode():
                refined = operation()
        finally:
            FastGeodis.generalised_geodesic2d = original_geodis
            refiner.predictor.predict_torch = original_decoder
        require(refined.shape == mask.shape and refined.dtype == np.bool_
                and observations["decoder"] == 5 and observations["geodesic"] >= 5,
                "Exact original-grid mask, five decoder calls and native prompts required")
        return refined, dict(decoder_calls=observations["decoder"], geodesic_calls=observations["geodesic"],
                             foreground_pixels=int(refined.sum()), changed_pixels=int((refined != original_mask).sum()),
                             mask_sha256=sha256(refined.tobytes()).hexdigest(), seconds=perf_counter()-tick)

    current, single = run(lambda: refiner.refine_current(mask, refine_iters=5))
    batch, batched = run(lambda: refiner.refine_batch([mask], refine_iters=5, max_batch=1)[0])
    require(not torch.cuda.is_initialized(), "CUDA state changed during author CPU inference")
    result = dict(status="PHOENIX_AUTHOR_XL1_CPU_FULL_DEMO_COMPLETE", author_commit=REVISION,
                  inputs=PINS, image_shape=list(image.shape), embedding_shape=list(features.shape),
                  embedding_sha256=sha256(features.detach().contiguous().numpy().tobytes()).hexdigest(),
                  pytorch=str(torch.__version__), fastgeodis=version("FastGeodis"), numpy=np.__version__,
                  input_foreground_pixels=int(original_mask.sum()), source_state_keys=len(model.state_dict()),
                  refine_current=single, refine_batch=batched, single_batch_differing_pixels=int((current != batch).sum()),
                  encode_seconds=encode_seconds, load_seconds=load_seconds, wall_seconds=perf_counter()-start,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  cpu_threads=2, device="cpu", cuda_initialized=False, sar_images=0, AP=False)
    raw = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with args.output.open("xb") as target:
        target.write(raw)
    print(json.dumps(dict(status=result["status"], result_sha256=sha256(raw).hexdigest())), flush=True)


if __name__ == "__main__":
    main()
