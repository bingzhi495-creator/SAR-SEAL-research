"""Bounded decoded grayscale appearance screen using OpenCV 4.x.

A strong result is image evidence within this screen's resize and affine range.
The caller owns split/scene identity, exact image digests and positive controls.
"""

from dataclasses import dataclass
from threading import Lock
from typing import NamedTuple, Sequence

import cv2
import numpy as np


MAX_SIDE = 384
MAX_FEATURES = 128
MAX_CANDIDATES = 32
NEIGHBORS = 8
RATIO = .80
RANSAC_SEED = 1729
GRID_CELLS = 4
_MIN_MATCHES = 12
_LOCK = Lock()


class Features(NamedTuple):
    gray: np.ndarray
    points: np.ndarray
    descriptors: np.ndarray


@dataclass(frozen=True)
class Index:
    descriptors: np.ndarray
    source_rows: np.ndarray
    flann: object


def features(gray: np.ndarray) -> Features:
    """Decode-independent uint8 input; reduce long side to 384 if needed."""
    if gray.ndim != 2 or gray.dtype != np.uint8 or min(gray.shape) == 0:
        raise ValueError("expected a nonempty two-dimensional uint8 image")
    height, width = gray.shape
    if max(height, width) > MAX_SIDE:
        scale = MAX_SIDE / max(height, width)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    gray = np.ascontiguousarray(gray)
    if min(gray.shape) >= 8:
        keypoints, descriptors = cv2.SIFT_create(nfeatures=MAX_FEATURES).detectAndCompute(gray, None)
        if descriptors is not None and keypoints:
            order = sorted(
                range(len(keypoints)),
                key=lambda i: (-keypoints[i].response, *keypoints[i].pt, keypoints[i].size, keypoints[i].angle, i),
            )[:MAX_FEATURES]
            points = np.array([keypoints[i].pt for i in order], dtype=np.float32)
            return Features(gray, points, np.ascontiguousarray(descriptors[order], dtype=np.float32))
    return Features(gray, np.empty((0, 2), np.float32), np.empty((0, 128), np.float32))


def build_index(sources: Sequence[Features]) -> Index:
    """One 4-tree FLANN KD index; maintain row-to-gallery-image identity."""
    nonempty = [(i, source.descriptors) for i, source in enumerate(sources) if len(source.descriptors)]
    if not nonempty:
        return Index(np.empty((0, 128), np.float32), np.empty(0, np.int32), None)
    rows = np.concatenate([np.full(len(desc), i, np.int32) for i, desc in nonempty])
    descriptors = np.concatenate([desc for _, desc in nonempty])
    with _LOCK:
        cv2.setRNGSeed(RANSAC_SEED)
        flann = cv2.flann_Index(descriptors, {"algorithm": 1, "trees": 4})
    return Index(descriptors, rows, flann)


def retrieval(index: Index, query: Features) -> list[int]:
    """Vote from up to eight neighbors per descriptor; cap at 32 images."""
    if index.flann is None or not len(query.descriptors):
        return []
    rows, distances = index.flann.knnSearch(
        query.descriptors, min(NEIGHBORS, len(index.descriptors)), params={"checks": 64},
    )
    votes = {}
    for neighbor_rows, neighbor_distances in zip(rows, distances):
        unique = {}
        for row, distance in zip(neighbor_rows, neighbor_distances):
            if row >= 0 and np.isfinite(distance):
                source = int(index.source_rows[row])
                unique[source] = min(unique.get(source, float("inf")), float(distance))
        for source, distance in unique.items():
            count, distance_sum = votes.get(source, (0, 0.0))
            votes[source] = (count + 1, distance_sum + distance)
    return sorted(votes, key=lambda i: (-votes[i][0], votes[i][1], i))[:MAX_CANDIDATES]


def _ratio_matches(matcher, left, right):
    return {
        candidates[0].queryIdx: candidates[0].trainIdx
        for candidates in matcher.knnMatch(left, right, k=2)
        if len(candidates) == 2 and candidates[0].distance < RATIO * candidates[1].distance
    }


def _corners(shape):
    h, w = shape
    return np.array([[-.5, -.5], [w - .5, -.5], [w - .5, h - .5], [-.5, h - .5]], np.float32)


def _occupied(cells):
    return len(cells) >= 4 and len({x for x, _ in cells}) >= 2 and len({y for _, y in cells}) >= 2


def _rank_correlation(left, right):
    def ranks(values):
        counts = np.bincount(values, minlength=256)
        result = (np.cumsum(counts) - .5 * counts)[values]
        return result - result.mean()

    left_ranks, right_ranks = ranks(left), ranks(right)
    denominator = np.linalg.norm(left_ranks) * np.linalg.norm(right_ranks)
    return float(np.dot(left_ranks, right_ranks) / denominator) if denominator > 0 else 0.


def _texture(source, query, affine, low, high):
    """Median local Spearman correlation on at least four textured cells."""
    height, width = query.shape
    aligned = cv2.warpAffine(source, affine, (width, height), flags=cv2.INTER_LINEAR)
    mask = cv2.warpAffine(np.ones_like(source), affine, (width, height), flags=cv2.INTER_NEAREST)
    valid = cv2.erode(mask, np.ones((7, 7), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    x_edges = np.clip(np.ceil(np.linspace(low[0], high[0], GRID_CELLS + 1)).astype(int), 0, width)
    y_edges = np.clip(np.ceil(np.linspace(low[1], high[1], GRID_CELLS + 1)).astype(int), 0, height)
    correlations = {}
    for y, (y1, y2) in enumerate(zip(y_edges[:-1], y_edges[1:])):
        for x, (x1, x2) in enumerate(zip(x_edges[:-1], x_edges[1:])):
            tile = np.s_[y1:y2, x1:x2]
            keep = valid[tile]
            if int(keep.sum()) < 128:
                continue
            left = aligned[tile][keep]
            right = query[tile][keep]
            if min(np.diff(np.percentile(left, [10, 90]))[0],
                   np.diff(np.percentile(right, [10, 90]))[0]) < 8:
                continue
            correlations[x, y] = _rank_correlation(left, right)
    return correlations


def verify(source: Features, query: Features) -> dict:
    """Verify one gallery-to-query affine with descriptor and texture evidence.

    Geometry is in the working query grid. Reflections, scale outside [0.2,5]
    or singular-value ratio above 2.5 are outside the verified transform class.
    Weak means a coherent fit fails spatial coverage or aligned texture checks.
    """
    result = {
        "status": "none", "reason": "insufficient_descriptors", "mutual_matches": 0,
        "inliers": 0, "inlier_fraction": None, "median_residual_px": None,
        "p90_residual_px": None, "overlap_large_fraction": None, "overlap_small_fraction": None,
        "inlier_cells": None, "inlier_hull_fraction": None, "textured_cells": None,
        "texture_correlation": None, "affine": None,
    }
    if min(len(source.descriptors), len(query.descriptors)) < _MIN_MATCHES:
        return result
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = _ratio_matches(matcher, source.descriptors, query.descriptors)
    backward = _ratio_matches(matcher, query.descriptors, source.descriptors)
    pairs = [(i, j) for i, j in forward.items() if backward.get(j) == i]
    result.update(mutual_matches=len(pairs), reason="insufficient_mutual_matches")
    if len(pairs) < _MIN_MATCHES:
        return result
    left_indices, right_indices = zip(*pairs)
    left, right = source.points[list(left_indices)], query.points[list(right_indices)]
    with _LOCK:
        cv2.setRNGSeed(RANSAC_SEED)
        affine, _ = cv2.estimateAffine2D(
            left, right, method=cv2.RANSAC, ransacReprojThreshold=3.,
            maxIters=4096, confidence=.999, refineIters=10,
        )
    result["reason"] = "invalid_affine"
    if affine is None or not np.isfinite(affine).all():
        return result
    linear, offset = affine[:, :2], affine[:, 2]
    det = np.linalg.det(linear)
    scales = np.linalg.svd(linear, compute_uv=False)
    if det <= 0 or scales[0] > 5 or scales[1] < .2 or scales[0] / scales[1] > 2.5:
        return result
    result["affine"] = affine.tolist()
    residuals = np.linalg.norm(left @ linear.T + offset - right, axis=1)
    # Recompute against the final affine, after OpenCV's inlier refinement.
    inlier_mask = residuals <= 3.
    count = int(inlier_mask.sum())
    fraction = count / len(pairs)
    result.update(inliers=count, inlier_fraction=fraction, reason="insufficient_inliers")
    if count < _MIN_MATCHES or fraction < .35:
        return result
    median, p90 = np.percentile(residuals[inlier_mask], [50, 90])
    result.update(median_residual_px=float(median), p90_residual_px=float(p90), reason="residual")
    if median > 1.5 or p90 > 3:
        return result
    result.update(status="weak", reason="overlap")
    source_corners = (_corners(source.gray.shape) @ linear.T + offset).astype(np.float32)
    overlap_area, overlap = cv2.intersectConvexConvex(source_corners, _corners(query.gray.shape))
    if overlap_area <= 0 or overlap is None:
        return result
    areas = (source.gray.size * det, query.gray.size)
    large, small = (min(1., float(overlap_area / area)) for area in (max(areas), min(areas)))
    result.update(overlap_large_fraction=large, overlap_small_fraction=small)
    if large < .15 or small < .35:
        return result
    polygon = overlap.reshape(-1, 2)
    low, high = polygon.min(axis=0), polygon.max(axis=0)
    inlier_points = right[inlier_mask]
    cell_points = np.floor(GRID_CELLS * (inlier_points - low) / (high - low))
    occupied = {tuple(p) for p in np.clip(cell_points, 0, GRID_CELLS - 1).astype(int)}
    hull = cv2.convexHull(inlier_points)
    hull_fraction = min(1., float(cv2.contourArea(hull) / overlap_area))
    result.update(inlier_cells=len(occupied), inlier_hull_fraction=hull_fraction, reason="inlier_support")
    if not _occupied(occupied) or hull_fraction < .05:
        return result
    correlations = _texture(source.gray, query.gray, affine, low, high)
    result.update(textured_cells=len(correlations), reason="texture_support")
    if not _occupied(correlations):
        return result
    correlation = float(np.median(list(correlations.values())))
    result.update(texture_correlation=correlation, reason="texture_correlation")
    if correlation >= .80:
        result.update(status="strong", reason="verified")
    return result
