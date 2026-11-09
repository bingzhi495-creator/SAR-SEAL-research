"""Add same-query normalized box coordinates to the unchanged P192 export."""
import math
import sys

import fit_query_export as fit


def attach_native_boxes(result, raw_boxes):
    fit.need(tuple(raw_boxes.shape) == (1, fit.WIDTH, 4), "Exact native normalized box tensor")
    ids = result["native_ids"]
    fit.need(len(ids) == len(set(ids)) and all(type(i) is int and 0 <= i < fit.WIDTH for i in ids),
             "Exact unique mapped query IDs")
    values = raw_boxes[0, ids].detach().cpu().tolist()
    fit.need(all(all(map(math.isfinite, row)) and row[2] >= 0 and row[3] >= 0 for row in values),
             "Finite unclipped normalized cxcywh and nonnegative extents")
    return dict(result, native_box_cxcywh=values)


def main():
    fit.need(sys.argv[1:2] == ["export"], "GT-free export only")
    original = fit.aligned_query_rows

    def export_with_boxes(raw, hw, **kwargs):
        return attach_native_boxes(original(raw, hw, **kwargs), raw["pred_boxes"])

    fit.aligned_query_rows = export_with_boxes
    try:
        fit.main()
    finally:
        fit.aligned_query_rows = original


if __name__ == "__main__":
    main()
