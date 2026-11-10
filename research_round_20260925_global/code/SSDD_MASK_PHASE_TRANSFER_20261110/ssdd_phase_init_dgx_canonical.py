"""Reconstruct the exact reference Q100 initialization on DGX using one position tensor."""

from hashlib import sha256
from io import BytesIO
import os
from pathlib import Path
import re

from ssdd_phase_init import load_api, require, token


ORIGINAL_INITIALIZER_SHA256 = "444e91009bce7a56e65a9bd2110c3479a86609d29b5ccdc925b06ec013b7894a"
POSITION_KEY = "backbone.0.encoder.encoder.embeddings.position_embeddings"
POSITION_SHAPE = (1, 2305, 384)
POSITION_FILE_BYTES = 3_542_121
NATIVE_DGX_SHA256 = "7b270a9889ac56b815442309e42ef9a7f7937c9ab28df5042cda7f73e384b9cd"
NATIVE_DGX_POSITION_SHA256 = "f5bef45a496862ad91550217654f25e83a76a62a7f42a8dc200010ca9b7966ed"
CANONICAL_Q100_SHA256 = "40524fc63797d2fcab7b50b6ee1f041f9429df73e505cb9e647a5a8cdf6f2e2f"
CANONICAL_POSITION_SHA256 = "5ccc0fd712422fb88f720506835142abc77b98ae97252473cc17e48711ca7aae"


def _load_original():
    path = Path(__file__).resolve().with_name("ssdd_phase_init.py")
    return load_api(path, ORIGINAL_INITIALIZER_SHA256)


def _position_tensor(torch, value):
    return (isinstance(value, torch.Tensor) and value.device.type == "cpu"
            and value.layout == torch.strided and value.dtype == torch.float32
            and tuple(value.shape) == POSITION_SHAPE and value.is_contiguous()
            and bool(torch.isfinite(value).all()))


def _load_position(path, expected_sha256, torch, digest):
    require(path.is_absolute() and path.is_file() and path.resolve(strict=True) == path,
            "Canonical private Q100 position file required")
    with path.open("rb") as stream:
        before = token(os.fstat(stream.fileno()))
        require(before[2] == POSITION_FILE_BYTES, "Exact Q100 position file size required")
        raw = stream.read(POSITION_FILE_BYTES + 1)
        after = token(os.fstat(stream.fileno()))
    require(before == after == token(path.stat()) and len(raw) == POSITION_FILE_BYTES
            and sha256(raw).hexdigest() == expected_sha256, "Exact Q100 position file SHA-256 required")

    state = torch.load(BytesIO(raw), map_location="cpu", weights_only=True)
    require(type(state) is dict and set(state) == {POSITION_KEY},
            "Exactly one registered Q100 position tensor required in the file")
    position = state[POSITION_KEY]
    require(type(position) is torch.Tensor and not position.requires_grad
            and _position_tensor(torch, position)
            and digest({POSITION_KEY: position}) == CANONICAL_POSITION_SHA256,
            "Exact finite canonical Q100 position tensor required")
    return position


def make_canonical_initializer(position_path, expected_file_sha256):
    """Return a drop-in initializer for the original worker with an owner-pinned position file."""
    require(type(expected_file_sha256) is str and re.fullmatch(r"[0-9a-f]{64}", expected_file_sha256) is not None,
            "Independent Q100 position file SHA-256 required")
    path = Path(position_path)

    def initialize(fit_layout, output, coco_path):
        import torch

        original = _load_original()
        describe = original.describe_q100

        def describe_canonical(module, digest):
            native = describe(module, digest)
            require(native["online_tensor_sha256"] == NATIVE_DGX_SHA256,
                    "Exact original DGX Q100 initialization required before position replacement")
            state = module.model.state_dict()
            prior = state.get(POSITION_KEY)
            require(_position_tensor(torch, prior) and digest({POSITION_KEY: prior}) == NATIVE_DGX_POSITION_SHA256,
                    "Exact native DGX Q100 position tensor required")
            position = _load_position(path, expected_file_sha256, torch, digest)
            state[POSITION_KEY] = position
            require(digest(state) == CANONICAL_Q100_SHA256, "Exact canonical Q100 reconstruction required")
            parameter = module.model.get_parameter(POSITION_KEY)
            require(parameter.data_ptr() == prior.data_ptr(), "Registered DGX Q100 position parameter required")
            with torch.no_grad():
                parameter.copy_(position)
            result = describe(module, digest)
            require(result["online_tensor_sha256"] == CANONICAL_Q100_SHA256,
                    "Exact original reference Q100 digest required after replacement")
            return result

        original.describe_q100 = describe_canonical
        return original.initialize_q100(fit_layout, output, coco_path)

    return initialize
