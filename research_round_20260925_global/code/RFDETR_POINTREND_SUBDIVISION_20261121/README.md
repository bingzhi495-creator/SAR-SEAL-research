# PointRend adaptation for a frozen RF-DETR carrier

This directory provides the binary point head, RoI sampling, iterative subdivision, and matched-budget mask edits for a within-carrier comparison. The intended input is one frozen RF-DETR image-level spatial tensor plus each candidate's original box and raw query logit plane. The comparison uses the same original candidates and scores. [Pinned Detectron2 PointRend](https://github.com/facebookresearch/detectron2/tree/247691a1c292654867deb35cabb27f18d75a1363/projects/PointRend)

- `roi_point_head.py` samples a 7×7 coarse logit grid, samples the shared image feature map in RoI coordinates, and applies a three-layer binary point head with coarse reinjection. Its training sampler follows the 588 → 147 uncertain + 49 fresh random design and uses bilinear bitmask targets.
- `pointrend_subdivision.py` implements dense initialization and iterative uncertainty-driven scatter. For the 7→224 schedule with cap 784, the working stages are 28, 56, 112 and 224. Each RoI receives 3,136 point predictions. [Subdivision parity and conventions](SUBDIVISION.md)
- `matched_budget.py` compares separate add/remove rankings and a joint signed-gain ranking given identical foreground posteriors and a prelocked per-candidate swap count. Both arms edit exactly twice that count. Natural PointRend masks are obtained by sampling the rendered probability on original pixel centers, so their realized number of changed native pixels is counted separately.

Run `python -B -m unittest discover -s . -p "test_*.py" -v` with PyTorch installed. All 14 CPU tests passed. These cover the subdivision, fine features, original coarse conditioning, a point-head backward pass, final projection, and equal per-candidate edit counts.

The empirical comparison still requires frozen RF-DETR image features, exact candidate alignment and training within fixed FIT source folds. This adaptation uses RF-DETR features and boxes. The full authors' R-CNN framework uses its own FPN, region proposals and learned coarse head. No model performance measurements have been made with this adapter.
