"""Vector-first COCO labeling on a fixed resize, crop and flip trace."""

import numpy as np
from pycocotools import mask as coco_mask
import shapely
from shapely.geometry import Polygon, box


class UnsupportedPolygon(ValueError):
    """Original label lacks a simple vector interpretation for this audit."""


def polygon_rings(segmentation):
    """Keep the author's finite, valid, nonempty polygon components."""
    if type(segmentation) is not list or not segmentation:
        raise UnsupportedPolygon("not a polygon-list annotation")
    rings = []
    for part in segmentation:
        if type(part) is not list or len(part) < 6 or len(part) % 2:
            raise UnsupportedPolygon("malformed polygon component")
        coords = np.asarray(part, dtype=np.float64).reshape(-1, 2)
        if not np.isfinite(coords).all():
            raise UnsupportedPolygon("nonfinite polygon coordinates")
        polygon = Polygon(coords)
        if not polygon.is_valid or polygon.area <= 0:
            raise UnsupportedPolygon("invalid or zero-area polygon component")
        rings.append(coords)
    return rings


def _exteriors(geometry):
    if geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        if geometry.interiors:
            raise UnsupportedPolygon("clipping produced a hole absent from COCO union semantics")
        return [np.asarray(geometry.exterior.coords[:-1], dtype=np.float64)]
    if hasattr(geometry, "geoms"):
        return [part for geometry_part in geometry.geoms for part in _exteriors(geometry_part)]
    # A point or line touching the crop boundary has no polygon area.
    return []


def final_polygons(segmentation, source_hw, steps):
    """Apply continuous pixel-edge coordinates, clipping polygons at crop."""
    rings = polygon_rings(segmentation)
    current = tuple(source_hw)
    if len(current) != 2 or min(current) < 1:
        raise ValueError("Positive source canvas required")
    for step in steps:
        before, after = tuple(step["before"]), tuple(step["after"])
        if before != current or len(after) != 2 or min(after) < 1:
            raise ValueError("Polygon trace canvas continuity required")
        height, width = before
        out_height, out_width = after
        kind = step["kind"]
        if kind == "resize":
            scale = (out_width / width, out_height / height)
            rings = [ring * scale for ring in rings]
        elif kind == "crop":
            left, top = step["left"], step["top"]
            if not (0 <= left <= width - out_width and 0 <= top <= height - out_height):
                raise ValueError("Crop must be inside the intermediate canvas")
            window = box(left, top, left + out_width, top + out_height)
            rings = [
                part - (left, top)
                for ring in rings
                for part in _exteriors(Polygon(ring).buffer(0).intersection(window))
            ]
        elif kind == "hflip":
            if before != after:
                raise ValueError("Horizontal flip must preserve the canvas")
            rings = [np.column_stack((width - ring[:, 0], ring[:, 1])) for ring in rings]
        else:
            raise ValueError("Unexpected polygon transform operation")
        current = after
    return rings, current


def polygon_raster(rings, hw):
    """Use COCO's polygon union rasterizer on the final canvas once."""
    if not rings:
        return np.zeros(hw, dtype=np.bool_)
    rles = coco_mask.frPyObjects([ring.reshape(-1).tolist() for ring in rings], *hw)
    return coco_mask.decode(coco_mask.merge(rles)).astype(np.bool_)


def binary_iou(left, right):
    intersection = np.count_nonzero(left & right)
    union = np.count_nonzero(left | right)
    return float(intersection / union) if union else None


def vector_footprint_metrics(rings, hw, candidates):
    """Compare final-grid cell footprints to transformed continuous polygon area.

    This uses GEOS intersections of unit squares, independently of COCO
    discretization. The vector raster and this reference share original labels.
    """
    if not candidates or any(mask.shape != tuple(hw) or mask.dtype != np.bool_ for mask in candidates.values()):
        raise ValueError("Boolean masks on the same final grid required")
    height, width = hw
    reference = shapely.union_all([Polygon(ring) for ring in rings]).intersection(box(0, 0, width, height))
    visible = float(reference.area)
    rows, cols = np.nonzero(np.logical_or.reduce(list(candidates.values())))
    cells = shapely.box(cols, rows, cols + 1, rows + 1)
    overlap = np.asarray(shapely.area(shapely.intersection(cells, reference)), dtype=np.float64)
    if not np.isfinite(overlap).all() or np.any(overlap < -1e-10) or np.any(overlap > 1 + 1e-10):
        raise ArithmeticError("Invalid continuous polygon-cell intersection area")
    result = {}
    for name, candidate in candidates.items():
        chosen = candidate[rows, cols]
        area = int(np.count_nonzero(chosen))
        intersect = float(overlap[chosen].sum(dtype=np.float64))
        if intersect > min(area, visible) + 1e-8:
            raise ArithmeticError("Polygon intersection exceeds the corresponding support")
        intersect = min(intersect, area, visible)
        union = area + visible - intersect
        result[name] = dict(
            iou=intersect / union if union else None, fp_area=area - intersect, fn_area=visible - intersect,
        )
    return dict(visible_polygon_area=visible, by_arm=result)
