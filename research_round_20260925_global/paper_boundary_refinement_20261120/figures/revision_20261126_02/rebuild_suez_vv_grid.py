"""Rebuild the 28-column by 32-row display grid using a single public Sentinel-1A scene."""
from argparse import ArgumentParser
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from tifffile import TiffFile

ITEM = "S1A_IW_GRDH_1SDV_20210325T154823_20210325T154848_037157_046001_rtc"
COLLECTION = "https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-1-rtc"
WINDOW = (
    "https://planetarycomputer.microsoft.com/api/data/v1/item/bbox/"
    "32.573,30.011,32.588,30.024/144x144.tif"
    f"?collection=sentinel-1-rtc&item={ITEM}&assets=vv&resampling=nearest"
)

def build(tiff_bytes):
    with TiffFile(BytesIO(tiff_bytes)) as source:
        bands = source.asarray()
        geo = source.geotiff_metadata
    if bands.shape != (144, 144, 2) or bands.dtype != np.float32 or int(geo["GeographicTypeGeoKey"]) != 4326:
        raise ValueError("Expected the 144 x 144 EPSG:4326 float32 display window with an alpha band")
    if not np.all(bands[:, :, 1] == 255):
        raise ValueError("Invalid coverage in the requested window")
    # Row indices increase southward and column indices increase eastward.
    crop = bands[42:106, 32:88, 0].astype(np.float64)
    if not np.all(np.isfinite(crop) & (crop > 0)):
        raise ValueError("The fixed ROI must have finite positive gamma0")
    db = 10 * np.log10(crop)
    display = np.clip((db + 8) / 18, 0, 1) * 255
    grid = np.rint(display.reshape(32, 2, 28, 2).mean(axis=(1, 3))).astype(np.uint8)
    return {
        "source_item": ITEM, "source_metadata": f"{COLLECTION}/items/{ITEM}", "source_window": WINDOW,
        "credit": "Contains modified Copernicus Sentinel data (2021). RTC processing by Catalyst, via Microsoft Planetary Computer (CC BY 4.0).",
        "source_data_license": "https://ewds.climate.copernicus.eu/licences/ec-sentinel",
        "rtc_collection_license": "https://creativecommons.org/licenses/by/4.0/",
        "type": "historical illustrative SAR display brightness", "shape_rows_cols": list(grid.shape),
        "roi_xyxy_end_exclusive": [32, 42, 88, 106], "window_shape_rows_cols": [144, 144],
        "display": "VV gamma0 to 10*log10, clip [-8,+10] dB, linearly scale to [0,255], mean each disjoint 2x2 of displayed values, round nearest/even, uint8",
        "grid_order": "rows top to bottom (north to south); columns left to right (west to east)",
        "gray_u8_sha256": sha256(grid.tobytes()).hexdigest(),
        "gray_u8": grid.tolist(),
    }

def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input-tif", type=Path, help="Optional previously downloaded window")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.input_tif:
        asset = args.input_tif.read_bytes()
    else:
        with urlopen(WINDOW, timeout=45) as result:
            asset = result.read()
    record = build(asset)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(record["gray_u8_sha256"])

if __name__ == "__main__":
    main()