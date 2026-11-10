"""Continuous COCO polygon reference and an integer-exact raster support witness."""

import numpy as np
import shapely
from shapely.affinity import affine_transform, scale, translate
from shapely.geometry import Polygon, box

from vector_label_geometry import UnsupportedPolygon, polygon_rings
from vector_population_math import foreground_runs


def continuous_polygon(segmentation, source_hw, trace):
    """Map validated polygon unions through the actual edge-coordinate transforms."""
    rings = polygon_rings(segmentation)
    h,w = source_hw
    geometry = shapely.union_all([Polygon(ring) for ring in rings]).intersection(box(0,0,w,h))
    state = source_hw
    pre_resize = None
    for kind, before, after, args in trace:
        if state != before or min(after) <= 0 or pre_resize is not None:
            raise ValueError("Discontinuous stock geometry trace")
        h,w = before
        out_h,out_w = after
        if kind == "pad":
            x,y = args
            if x < 0 or y < 0 or x+w > out_w or y+h > out_h:
                raise ValueError("Stock pad cannot move content outside the canvas")
            geometry = translate(geometry, xoff=x, yoff=y)
        elif kind == "crop":
            x0,y0,x1,y1 = args
            if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h and (y1-y0,x1-x0) == after):
                raise ValueError("Invalid accepted crop")
            geometry = translate(geometry.intersection(box(x0,y0,x1,y1)), xoff=-x0, yoff=-y0)
        elif kind == "flip":
            if before != after:
                raise ValueError("Flip cannot change shape")
            geometry = affine_transform(geometry, [-1,0,0,1,w,0])
        elif kind == "resize":
            sx,sy = args
            if not np.allclose((sx,sy),(out_w/w,out_h/h),atol=1e-13,rtol=1e-13):
                raise ValueError("Stock resize scale and observed dimensions disagree")
            pre_resize = before
            geometry = scale(geometry,xfact=sx,yfact=sy,origin=(0,0))
        else:
            raise ValueError("Unexpected stock geometry operation")
        state = after
    if pre_resize is None:
        raise ValueError("The stock trace must include its single resize")
    return geometry.intersection(box(0,0,state[1],state[0])),pre_resize,state


def polygon_scores(reference, pre_hw, candidates):
    """Use original label geometry and exact GEOS area of output foreground cells."""
    area = float(reference.area)
    if area <= 0:
        raise ValueError("A continuous reference must be visible")
    center = reference.centroid
    scores = {}
    for name,candidate in candidates.items():
        out_h,out_w = candidate.shape
        mask = np.asarray(candidate, dtype=bool)
        y, left, right = foreground_runs(mask).T
        n = int((right-left).sum())
        areas = shapely.area(shapely.intersection(shapely.box(left,y,right,y+1),reference))
        overlap = float(np.asarray(areas).sum())
        if not -1e-10 <= overlap <= min(n,area)+1e-8:
            raise ArithmeticError("Invalid intersection against reference")
        overlap = min(max(overlap,0.),n,area)
        yy,xx = np.nonzero(mask)
        scores[name] = dict(
            iou=overlap/(area+n-overlap), fp_area=n-overlap, fn_area=area-overlap,
            signed_y_px=float((yy.mean()+.5-center.y)*pre_hw[0]/out_h) if n else None,
            signed_x_px=float((xx.mean()+.5-center.x)*pre_hw[1]/out_w) if n else None,
        )
    return scores


def _cell_overlap(first, end, n, m):
    lo = first*m//n
    hi = (end*m+n-1)//n
    dst = np.arange(lo,hi,dtype=np.int64)*n
    edges = np.arange(first,end+1,dtype=np.int64)*m
    overlaps = np.minimum(dst[:,None]+n,edges[None,1:])-np.maximum(dst[:,None],edges[None,:-1])
    return lo,np.maximum(overlaps,0)


def support_cell(pre_mask,out_hw):
    """One output cell maximizing coverage of the positive pre-resize raster."""
    source = np.asarray(pre_mask,dtype=bool)
    yy,xx = np.nonzero(source)
    if not yy.size:
        raise ValueError("A rescue can only use existing positive raster support")
    h,w = source.shape
    oh,ow = out_hw
    y0,y1,x0,x1 = int(yy.min()),int(yy.max())+1,int(xx.min()),int(xx.max())+1
    dy,wy = _cell_overlap(y0,y1,h,oh)
    dx,wx = _cell_overlap(x0,x1,w,ow)
    coverage = wy @ source[y0:y1,x0:x1].astype(np.int64) @ wx.T
    if int(coverage.sum()) != yy.size*oh*ow:
        raise ArithmeticError("The raster coverage mass must be conserved exactly")
    row,col = np.unravel_index(np.argmax(coverage),coverage.shape)
    output = np.zeros(out_hw,dtype=bool)
    output[dy+row,dx+col] = True
    return output,dict(area=float(coverage.sum()/(h*w)),
                       highest_cell_coverage=float(coverage[row,col]/(h*w)),
                       positive_overlap_cells=int(np.count_nonzero(coverage)),
                       contains_half_cell=bool(2*coverage[row,col]>=h*w))


def polygon_cell_summary(reference,out_hw):
    oh,ow = out_hw
    x0,y0,x1,y1 = reference.bounds
    yy,xx = np.mgrid[max(0,int(np.floor(y0))):min(oh,int(np.ceil(y1))),
                     max(0,int(np.floor(x0))):min(ow,int(np.ceil(x1)))]
    cells = shapely.box(xx,yy,xx+1,yy+1)
    overlap = shapely.area(shapely.intersection(cells,reference))
    return dict(area=float(reference.area),highest_cell_coverage=float(np.max(overlap)),
                positive_overlap_cells=int(np.count_nonzero(overlap>1e-12)),
                contains_half_cell=bool(np.any(overlap>=.5)),
                strict_polygon_center_cells=int(np.count_nonzero(shapely.contains_xy(reference,xx+.5,yy+.5))))

