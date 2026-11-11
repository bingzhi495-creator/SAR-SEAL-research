"""Qualify nominal SLC row membership from fixed OpenSAR, CDSE and ASF metadata."""
import argparse
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from shapely.geometry import Point, shape


PRODUCT = "S1A_IW_SLC__1SDV_20200407T052956_20200407T053026_032017_03B2E2_18FA"
PRODUCT_ID = "884ad94b-1595-5b26-a50f-7e97afd39ab9"
PINS = {
    "label": "9457786b4d7a15aa40807453d3d802b641c6dc16d5e6334175185f9faedc06ce",
    "cdse": "ea300307067cc979667c6dfa82a86a0aab872cb3ed2f38f76cb5ccb7378f2805",
    "asf5": "79f4b03123f20485e453d14195fd50c9001c8912a638ad255d2bedfc71e42407",
    "asf9": "b863388ed4e1dde73c6a9be6ca6c0a45bdc81cdefe52d4e6056ac6bdd4cd649f",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_pinned(path, key):
    require(path.stat().st_size <= 512000, "Expected bounded metadata")
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == PINS[key], "Metadata bytes differ")
    return raw


def project_inclusive(absolute_first1, first1, last1):
    require(absolute_first1 >= 1 and 1 <= first1 <= last1, "Expected inclusive one-based indices")
    return absolute_first1 + first1 - 2, absolute_first1 + last1 - 2


def containing(rows, span):
    return [i for i, (start, length) in enumerate(rows)
            if start <= span[0] <= span[1] < start + length]


def integers(node, field):
    return tuple(map(int, node.findtext(field).split()))


def polygon(value):
    geom = shape(value)
    require(geom.geom_type == "Polygon" and geom.is_valid and geom.area > 0, "Invalid catalogue footprint")
    return geom


def coverage(geom, centroid, corners):
    return dict(centroid=geom.covers(Point(*centroid)),
                corners=sum(geom.covers(Point(*corner)) for corner in corners))


def asf_check(path, index, cdse, centroid, corners):
    features = json.loads(read_pinned(path, "asf"+str(index)))["features"]
    feature, = features
    props = feature["properties"]
    burst = props["burst"]
    require(props["url"] == f"https://sentinel1-burst.asf.alaska.edu/{PRODUCT}/IW2/VH/{index}.tiff"
            and props["polarization"] == "VH" and props["beamModeType"] == "IW"
            and props["orbit"] == 32017 and burst["subswath"] == "IW2"
            and burst["burstIndex"] == index
            and burst["relativeBurstID"] == cdse["BurstId"] and burst["absoluteBurstID"] == cdse["AbsoluteBurstId"]
            and burst["samplesPerBurst"] == cdse["SamplesPerBurst"], "Cross-catalogue source identity differs")
    delta = abs((datetime.fromisoformat(burst["azimuthTime"].replace("Z", "+00:00"))
                 - datetime.fromisoformat(cdse["AzimuthTime"].replace("Z", "+00:00"))).total_seconds())
    require(delta < 1, "ASF and CDSE times disagree at reported precision")
    return dict(index=index, times_differ_seconds=delta, asf=coverage(polygon(feature["geometry"]),centroid,corners),
                cdse=coverage(polygon(cdse["GeoFootprint"]),centroid,corners))


def run(args):
    xml = ET.fromstring(read_pinned(args.label, "label"))
    require(xml.findtext("./SARData/SARProduct") == PRODUCT
            and xml.findtext("./SARData/SLCSwath") == "2", "Author product differs")
    ship, = xml.findall("./ProcessingData/List_of_ships/Ship")
    require(ship.findtext("Is_Vessel") == "True" and ship.findtext("Confidence") == "HIGH",
            "Pinned positive training box required")
    published_l1 = int(xml.findtext("./SARData/L1_burst_id"))
    corner = xml.find("./ProcessingData/Corner_Coord")
    ax, ay = (integers(corner, f"SARData_{axis}") for axis in ("Sample", "Line"))
    sx, sy = (integers(corner, f"Scene_{axis}") for axis in ("Sample", "Line"))
    require(sx == (1,512,512,1) and sy == (1,1,512,512), "Author local corner convention differs")
    require(all(a == min(ax) + s - 1 for a,s in zip(ax,sx,strict=True))
            and all(a == min(ay) + s - 1 for a,s in zip(ay,sy,strict=True)), "Author corner origins disagree")
    box = ship.find("BoundingBox")
    top,left,bottom,right = (int(box.findtext(k)) for k in ("Top","Left","Bottom","Right"))
    require(1 <= top <= bottom <= 512 and 1 <= left <= right <= 512, "Ship box outside author crop")
    tile_rows = project_inclusive(min(ay), 1, 512)
    tile_cols = project_inclusive(min(ax), 1, 512)
    ship_rows = project_inclusive(min(ay), top, bottom)
    ship_cols = project_inclusive(min(ax), left, right)
    position = ship.find("Centroid_Position")
    cy,cx = (int(position.findtext(f"Scene_{axis}")) for axis in ("Line","Sample"))
    abs_y,abs_x = (int(position.findtext(f"SARData_{axis}")) for axis in ("Line","Sample"))
    require((abs_y-1,abs_x-1) == (project_inclusive(min(ay),cy,cy)[0],project_inclusive(min(ax),cx,cx)[0])
            and ship_rows[0] <= abs_y-1 <= ship_rows[1] and ship_cols[0] <= abs_x-1 <= ship_cols[1],
            "Author centroid has inconsistent origins")
    centroid = tuple(float(position.findtext(k)) for k in ("Longitude","Latitude"))
    corners = list(zip(*(map(float,corner.findtext(k).split()) for k in ("Longitude","Latitude")),strict=True))

    catalogue = json.loads(read_pinned(args.cdse, "cdse"))
    rows = sorted(catalogue["value"], key=lambda row:row["Lines"])
    require(not catalogue.get("@odata.nextLink") and len(rows) == 10
            and all(row["ParentProductId"] == PRODUCT_ID and row["ParentProductName"] == PRODUCT+".SAFE"
                    and row["SwathIdentifier"] == "IW2" and row["PolarisationChannels"] == "VH"
                    and row["Lines"] == 1499*i and row["LinesPerBurst"] == 1499 and row["SamplesPerBurst"] == 24497
                    and row["BurstId"] == 202942+i for i,row in enumerate(rows)), "Unexpected CDSE sequence")
    intervals = [(row["Lines"],row["LinesPerBurst"]) for row in rows]
    selected, = containing(intervals,tile_rows)
    require(containing(intervals,ship_rows) == [selected] and 0 <= tile_cols[0] <= tile_cols[1] < 24497,
            "Author crop or box outside nominal SLC grid")
    alternates = containing(intervals, tuple(row+1 for row in tile_rows))
    cross = [asf_check(path,i,rows[i],centroid,corners) for path,i in ((args.asf5,5),(args.asf9,9))]
    require(selected == 5 and published_l1 == 9 and alternates == [5]
            and cross[0]["asf"]["centroid"] and cross[0]["cdse"]["centroid"]
            and not cross[1]["asf"]["centroid"] and not cross[1]["cdse"]["centroid"], "Catalogue control failed")
    first = intervals[selected][0]
    return dict(status="OPEN_SAR_TRAIN395_DUAL_BURST_NOMINAL_GEOMETRY_ACTUAL",
                exact_metadata_sha256=PINS, cdse_rows=len(rows), cross_catalogue=cross,
                author_l1_field=published_l1, selected_measurement_index0=selected,
                same_index_under_one_row_origin_sensitivity=alternates == [selected],
                rows_per_burst=1499, nominal_columns=24497, selected_start_row0=first,
                tile_rows0=list(tile_rows), tile_columns0=list(tile_cols),
                ship_rows0=list(ship_rows), ship_columns0=list(ship_cols),
                ship_rows_within_burst0=[r-first for r in ship_rows],
                official_per_row_validity_read=False, independent_geolocation_grid_read=False,
                raw_prf_calibrated=False, model_forward_calls=0, AP=False, DEV=False, TEST=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("label","cdse","asf5","asf9","output"):
        parser.add_argument("--"+key, type=Path, required=True)
    args = parser.parse_args()
    result = (json.dumps(run(args), sort_keys=True, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as handle:
        handle.write(result)
    print(json.dumps(dict(status="CLOSED_NOMINAL_GEOMETRY_RESULT",sha256=sha256(result).hexdigest())), flush=True)


if __name__ == "__main__":
    main()
