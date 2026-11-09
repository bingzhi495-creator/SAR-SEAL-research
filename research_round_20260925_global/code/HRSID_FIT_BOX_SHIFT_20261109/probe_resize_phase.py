"""Measure pixel-center offsets of the frozen CPU resize operators, with no data."""
import json

import torch
from torch.nn import functional
from torchvision import tv_tensors
from torchvision.transforms.v2.functional import resize


def main():
    torch.set_num_threads(2)
    ramp = torch.arange(800, dtype=torch.float32).reshape(1, 1, 1, 800)
    positions = torch.arange(576, dtype=torch.int64)
    legacy = functional.interpolate(ramp, size=(1, 576), mode="nearest").reshape(-1)
    exact = functional.interpolate(ramp, size=(1, 576), mode="nearest-exact").reshape(-1)
    ideal = (positions.double() + .5) * 800 / 576 - .5
    floor_model = (positions * 800 // 576).float()
    legacy_deviations = legacy != floor_model
    assert ((positions[legacy_deviations] * 800) % 576 == 0).all()
    assert (legacy[legacy_deviations] == floor_model[legacy_deviations] - 1).all()
    assert torch.equal(exact, ((2 * positions + 1) * 800 // 1152).float())

    rectangle = torch.zeros(1, 800, 800, dtype=torch.bool)
    rectangle[0, 257:278, 411:428] = True
    stock = resize(tv_tensors.Mask(rectangle), [576, 576]).as_subclass(torch.Tensor)
    origin = rectangle[0].nonzero().double().mean(0)
    observations = {}
    for mode in ("nearest", "nearest-exact"):
        down = functional.interpolate(rectangle[None].float(), size=(576, 576), mode=mode)
        restored = functional.interpolate(down, size=(800, 800), mode="bilinear", align_corners=False)[0, 0] > .5
        observations[mode] = dict(original_pixels_centroid_shift_yx=(restored.nonzero().double().mean(0) - origin).tolist(),
                                  native_TV_match=bool(torch.equal(down[0].bool(), stock)))
    print(json.dumps(dict(runtime=torch.__version__, axes=576, source_pixels_differ=int((legacy != exact).sum()),
                          legacy_float_boundary_deviations=int(legacy_deviations.sum()),
                          source_coordinate_legacy_bias=float((legacy - ideal).mean()),
                          source_coordinate_exact_bias=float((exact - ideal).mean()),
                          single_synthetic_rectangle=observations), sort_keys=True))


if __name__ == "__main__":
    main()
