# Binary PointRend subdivision

The entry point in `pointrend_subdivision.py` takes a fixed coarse logit map for each query and an inference callback for its trained point head. Query order and spatial extent must agree across the coarse map, callback and output grid. The caller selects the coordinate frame. For RoI-relative masks, the callback maps normalized RoI centers into the image feature maps using the query box, feature stride and padding geometry. For whole-image masks, it uses the image coordinate frame. The caller also owns mask placement, scoring and thresholding.

```python
def predict_points(xy, coarse_samples):
    fine_samples = sample_query_features(xy)  # caller-defined, [N,C,P]
    return point_head(fine_samples, coarse_samples)  # absolute [N,1,P] logits

result = subdivide_binary_mask(
    coarse_logits, predict_points,
    initial_size=(7, 7), native_size=(224, 224), points_per_step=784,
)
```

The dense first call initializes the working mask. Later stages rank the bilinearly upsampled working logits by proximity to zero. Every callback separately receives samples from the original coarse map with bilinear interpolation, zero padding and `align_corners=False`. The selected pixels take the fresh head predictions. Tied uncertainty scores follow the running PyTorch backend's top-k ordering. Grid dimensions refer to height then width. Point coordinates refer to x then y.

## Reference and adaptations

The numerical reference is Detectron2 commit `247691a1c292654867deb35cabb27f18d75a1363`. Its instance head uses square RoIs, dense initialization, twofold subdivision, promotion of fully covered initial stages and per-stage top-k selection. The binary specialization handles one foreground logit per query. It returns raw logits and realized evaluation counts. Rectangular targets extend the square schedule by doubling each axis up to its specified native length. Clipping the last step preserves the caller's final grid centers. Empty batches return the requested native shape immediately. The interface accepts float32 and float64 coarse maps. The numerical comparison below used float32. [Author's mask inference](https://github.com/facebookresearch/detectron2/blob/247691a1c292654867deb35cabb27f18d75a1363/projects/PointRend/point_rend/mask_head.py) · [Author's point coordinates and sampling](https://github.com/facebookresearch/detectron2/blob/247691a1c292654867deb35cabb27f18d75a1363/projects/PointRend/point_rend/point_features.py)

## CPU verification

Run `python -B -m unittest -v test_pointrend_subdivision` in this directory. Fixtures verify exact sparse replacement and feedback into the next stage, immutable coarse conditioning including a zero-padded boundary sample, regular-grid floating-point rounding, independent per-query cell mapping on an odd rectangular grid, dense-stage promotion, realized counts and empty inputs.

An additional comparison ran the pinned author's subdivision, point-logit, uncertainty and coordinate methods with a shared synthetic point head on PyTorch 2.2.2+cpu. Each case used two masks, RNG seed 117, independent random coarse and fine maps, and a head computing `0.7 * fine + 0.3 * coarse`. Fine maps covered the full mask extent. An adapter captured raw logits before mask postprocessing. Every point coordinate at every executed stage and every output logit compared exactly equal.

| Initial → native side | k per subdivision | Executed sides | Evaluations per query | Evaluations for both queries |
| --- | ---: | --- | ---: | ---: |
| 2 → 16 | 16 | 4, 8, 16 | 48 | 96 |
| 7 → 224 | 784 | 28, 56, 112, 224 | 3,136 | 6,272 |

These counts include dense initialization and repeated head calls at successive scales. They measure point evaluations. The comparison covers the subdivision algorithm on these fixtures. Detector integration, learned-head behavior and segmentation metrics require a separate evaluation.
