"""Verify the best attainable binary IoU for the fixed center-empty cases."""
import argparse
from hashlib import sha256
import itertools
import json
from pathlib import Path
import sys


ORIGINAL_POLYGON_SHA = "df88810589aa9ecaf6bce481b970a5f331966f500424290ed0393491671a3fa3"


def best_prefix(area, overlaps):
    import numpy as np
    positive = np.sort(np.asarray(overlaps,dtype=np.float64).reshape(-1))
    positive = positive[positive>0][::-1]
    if len(positive)==0 or not np.isclose(positive.sum(),area,rtol=1e-10,atol=1e-10):
        raise ValueError("Coverage must account for the full continuous area")
    intersections=np.minimum(np.cumsum(positive),area)
    sizes=np.arange(1,len(positive)+1)
    ious=intersections/(area+sizes-intersections)
    winner=int(np.argmax(ious))
    return dict(best_iou=float(ious[winner]),best_foreground_cells=int(sizes[winner]),
                first_cell_iou=float(ious[0]),positive_overlap_cells=len(positive))


def attainable(reference,out_hw):
    import numpy as np
    import shapely
    oh,ow=out_hw
    x0,y0,x1,y1=reference.bounds
    yy,xx=np.mgrid[max(0,int(np.floor(y0))):min(oh,int(np.ceil(y1))),
                   max(0,int(np.floor(x0))):min(ow,int(np.ceil(x1)))]
    areas=shapely.area(shapely.intersection(shapely.box(xx,yy,xx+1,yy+1),reference))
    result=best_prefix(float(reference.area),areas)
    result["centers_on_boundary"]=int(np.count_nonzero(shapely.intersects_xy(reference,xx+.5,yy+.5)
                                                    & ~shapely.contains_xy(reference,xx+.5,yy+.5)))
    return result


def test_prefix_against_exhaustive():
    for values in ([.3,.3,.3,.3],[.520,.490,.003,.018],[.42,.33,.13],[.5,.5],[.82,.33,.02]):
        area=sum(values)
        all_iou=[]
        for n in range(1,len(values)+1):
            for subset in itertools.combinations(values,n):
                p=sum(subset)
                all_iou.append(p/(area+n-p))
        if abs(max(all_iou)-best_prefix(area,values)["best_iou"])>1e-12:
            raise AssertionError("Exact IoU optimization disagrees with enumeration")


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ("paddle-code","vector-code","upstream","bundle","plan","original","replayed",
                 "output","original-polygon","ceiling-output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--protocol-commit",required=True)
    args=p.parse_args()
    test_prefix_against_exhaustive()
    original=args.original_polygon.read_bytes()
    if sha256(original).hexdigest()!=ORIGINAL_POLYGON_SHA:
        raise ValueError("Exact closed original polygon analysis required")
    sys.path[:0]=[str(Path(__file__).resolve().parent),str(args.paddle_code),str(args.vector_code)]
    import polygon_replay
    import polygon_support
    from scope import canonical

    observed=[]
    source=polygon_support.polygon_cell_summary
    def capture(reference,out_hw):
        base=source(reference,out_hw)
        result=attainable(reference,out_hw)
        if result["positive_overlap_cells"]!=base["positive_overlap_cells"]:
            raise ValueError("Geometry support differs from the original three cases")
        observed.append(result)
        return base
    polygon_support.polygon_cell_summary=capture
    try:
        polygon_replay.run(args)
    finally:
        polygon_support.polygon_cell_summary=source
    if args.output.read_bytes()!=original or len(observed)!=3:
        raise ValueError("Original polygon analysis must reproduce byte for byte")
    result=dict(status="MASK_RTDETR_HRSID_THREE_EMPTY_MASK_BINARY_GRID_IOU_CEILING_COMPLETE_V1",
                original_polygon_report_sha256=ORIGINAL_POLYGON_SHA,
                case_count=len(observed), main_polygon_report_reproduced_exactly=True,
                official_DEV_opened=False,official_TEST_opened=False,model_executed=False,
                geometry="same continuous COCO polygon versus any union of final output pixel cells",
                results=observed,
                cases_attaining_half_iou=sum(row["best_iou"]>=.5 for row in observed))
    raw=canonical(result)
    with args.ceiling_output.open("xb") as f:
        f.write(raw)
    print(json.dumps(dict(event=result["status"],sha256=sha256(raw).hexdigest()),sort_keys=True),flush=True)


if __name__=="__main__":
    main()

