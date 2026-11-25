"""Probe public TerraSAR-X metadata services for the three flagged reserve parents."""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

SERVICES = {
    "esa_eocat": "https://eocat.esa.int/eo-catalogue/collections/TerraSAR-X/items",
    "esa_maap": "https://catalog.maap.eo.esa.int/catalogue/collections/TerraSAR-X/items",
}
TARGETS = {43, 46, 70}
APPENDIX_ROW = re.compile(
    r"^\s*(43|46|70)\s+TerraSAR-X\s+([EW]\d+(?:\.\d+)?)\s+"
    r"([NS]\d+(?:\.\d+)?)\s+(\d{8})\s+", re.MULTILINE
)
MAX_METADATA_BYTES = 2_000_000


def coordinate(token):
    degrees = float(token[1:])
    return -degrees if token[0] in "WS" else degrees


def search_filters(appendix):
    targets = {}
    for parent, longitude, latitude, day in APPENDIX_ROW.findall(appendix):
        center_x, center_y = coordinate(longitude), coordinate(latitude)
        date = datetime.strptime(day, "%Y%m%d").replace(tzinfo=timezone.utc)
        bounds = [center_x - .10, center_y - .10, center_x + .10, center_y + .10]
        # The appendix reports a date without a timezone. Include the days
        # immediately before and after it for this candidate discovery probe.
        start = (date - timedelta(days=1)).isoformat().replace("+00:00", "Z")
        stop = (date + timedelta(days=2) - timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
        targets[int(parent)] = {"bbox": ",".join(f"{value:.2f}" for value in bounds),
                               "datetime": f"{start}/{stop}"}
    if targets.keys() != TARGETS:
        raise ValueError("The three TerraSAR-X appendix records are required")
    return targets


def probe(task):
    service, url, parent, filters = task
    result = {"service": service, "author_parent": parent}
    params = {"limit": 1 if parent is None else 10, **filters}
    if service == "esa_eocat":
        params["httpAccept"] = "application/geo+json"
    request = Request(url + "?" + urlencode(params),
                      headers={"Accept": "application/geo+json, application/json"})
    try:
        with urlopen(request, timeout=14) as response:
            result["http_status"] = response.status
            result["content_type"] = response.headers.get_content_type()
            if result["content_type"] not in {"application/geo+json", "application/json"}:
                result["status"] = "non_json_response"
                return result
            raw = response.read(MAX_METADATA_BYTES + 1)
        if len(raw) > MAX_METADATA_BYTES:
            result["status"] = "metadata_size_bound_exceeded"
            return result
        data = json.loads(raw)
        if data.get("type") != "FeatureCollection" or not isinstance(data.get("features"), list):
            result["status"] = "non_feature_response"
            return result
        result["status"] = "metadata_response"
        result["features_returned"] = len(data["features"])
        result["number_matched"] = data.get("numberMatched", data.get("context", {}).get("matched"))
        result["additional_page"] = any(link.get("rel") == "next" for link in data.get("links", []))
    except HTTPError as error:
        result["status"] = "http_error"
        result["http_status"] = error.code
    except (URLError, TimeoutError, OSError) as error:
        result["status"] = "network_error"
        result["error_type"] = type(getattr(error, "reason", error)).__name__
    except (ValueError, TypeError, AttributeError):
        result["status"] = "invalid_metadata_response"
    return result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--appendix-text", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    targets = search_filters(args.appendix_text.read_text(encoding="utf-8"))
    tasks = [(service, url, parent, filters)
             for service, url in SERVICES.items()
             for parent, filters in [(None, {}), *sorted(targets.items())]]
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(probe, tasks))
    safe = {"status": "complete", "linked_provider_products": 0,
            "qualified_acquisition_groups": 0, "new_image_downloads": 0,
            "rounded_center_window_half_width_degrees": .10,
            "date_window": "reported calendar day plus the preceding and following day",
            "responses": responses}
    args.output.write_text(json.dumps(safe, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(safe))


if __name__ == "__main__":
    main()

