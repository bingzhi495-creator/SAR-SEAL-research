"""Reconstitute the originally scored Gulf masks, components and common length cohort."""

from hashlib import sha256
import json
from types import ModuleType
from zipfile import BadZipFile

import numpy as np


SCORE_WORKER = "868cff4c73c472c131511b0a3bfe8fbedaed584096eabc6a1c1aaf09f3244933"
SCORE_RESULT = "9cc9e5a8a8f1abd8085221422bb21d0f165402731027a316ad7c1801442dbf83"
SCORE_TERMINAL = "846a642b3136c86584999fe5b596ed2593986ca44acfdbe59da873bb298d2df1"


def pinned(path, expected):
    data = path.read_bytes()
    if sha256(data).hexdigest() != expected:
        raise ValueError("Historical Gulf digest differs for " + path.name)
    return data


def checked_score(a):
    path = a.code / "GULF_AIS_FROZEN_20261123/score.py"
    old = ModuleType("_original_gulf_score")
    old.__file__ = str(path)
    exec(compile(pinned(path, SCORE_WORKER), str(path), "exec"), old.__dict__)
    term = json.loads(pinned(a.score_terminal, SCORE_TERMINAL))
    summary = json.loads(pinned(a.score / "RESULT_SAFE.json", SCORE_RESULT))
    old.require(term["phase"] == summary["phase"] == "score" and type(term["actual_exit"]) is int
                and term["actual_exit"] == 0 and term["output_footer_verified"] is True
                and term["result_sha256"] == SCORE_RESULT and term["worker_sha256"] == summary["worker_sha256"] == SCORE_WORKER
                and summary["status"] == old.STATUS and summary["protocol_sha256"] == old.PROTOCOL == a.protocol_sha256
                and summary["heading_opened"] is False and summary["length_members_opened_only_after_geometry"] is True,
                "Historical score receipt differs")
    return old, summary


def restore_components(old, stock, changes, expected):
    for position, candidate in enumerate(stock, 1):
        raw = old.codec.decode(candidate["segmentation"]).astype(bool)
        native = old.native_component(raw)
        if native is not None:
            break
    else:
        raise ValueError("Accepted original native component is missing")
    updates = changes.get(str(position), {})
    axis = old.axis_for(native)
    spans, whole, components, masks = {}, {}, {}, {}
    for arm in old.ARMS:
        mask, selected = raw, native
        if arm in updates:
            mask = old.codec.decode(updates[arm]["rle"]).astype(bool)
            selected = old.overlap_component(mask, native)
        old.require(old.issue(selected) is None, "Original common component became unmeasurable")
        spans[arm] = old.span(selected, axis)
        indices = np.flatnonzero(mask)
        if old.issue(indices) is None:
            whole[arm] = old.span(indices, axis)
        masks[arm] = mask
        components[arm] = selected
    old.require(expected["spans"] == spans and expected["whole"] == whole
                and expected["fixed_k"] == updates.get("fixed", {}).get("swaps", 0),
                "Recovered component extent or quota differs")
    return dict(position=position, local_axis=axis, components=components, masks=masks,
                spans=spans, original_spans=spans, whole=whole, fixed_k=expected["fixed_k"])


def optional_numeric(old, archive, key):
    try:
        value = float(old.numeric(archive, key))
    except (KeyError, ValueError, TypeError, OverflowError, OSError, EOFError, BadZipFile):
        return None
    return value if np.isfinite(value) else None


def load_frozen(a):
    """Return the old SAFE result, scorer, original intake manifest and 25 native anchor records.

    All paths in a are pathlib.Path. a.protocol and a.protocol_sha256 describe the *historical*
    Gulf study. Set a.include_heading=True to open that additional scalar on the locked 25 only.
    Components use raw 512-square C-order pixel indices. local_axis is in column,row order.
    Invalid geographic/heading values are returned as None, preserving each frozen anchor.
    """
    old, summary = checked_score(a)
    manifest, blocks, lineage = old.closed_sources(a)
    old.require(all(summary[k] == v for k, v in lineage.items()), "Historical source lineage differs")
    reproduced = old.analyze(a.intake, manifest, blocks)
    old.require(all(summary[k] == v for k, v in reproduced.items()),
                "Historical endpoints, intervals, gates, coverage or auxiliary results differ")

    # Recover components separately from the scalar summary, replaying the original selection order.
    seen, anchors = set(), []
    for row in manifest["items"]:
        identity = row["gray_sha256"]
        duplicate = identity is not None and identity in seen
        if identity is not None:
            seen.add(identity)
        if row["status"] != "ready" or duplicate:
            continue
        with old.original_npz(a.intake, row) as archive:
            failures = old.scalar_failures(archive, row, manifest["scene_geometry"][row["scene_id"]])
        if failures:
            continue
        stock, changes = blocks[row["image_id"]]
        result = old.measurements(stock, changes)
        if result is None or result["failures"]:
            continue
        anchor = restore_components(old, stock, changes, result)
        anchors.append(dict(row=row, pass_id=row["pass_id"], **anchor))

    old.require(len(anchors) == summary["coverage"]["all_arm_geometry_fixed"] == 25,
                "Geometry-only anchor count differs")
    for anchor in anchors:
        with old.original_npz(a.intake, anchor["row"]) as archive:
            length = old.numeric(archive, "length_m")
            old.require(np.isfinite(length) and length >= 50, "Original common length missing")
            anchor["length"] = length

    primary = old.contrast(anchors)
    old.require(primary == summary["primary"]
                and (primary["paired_ships"], primary["paired_groups"], primary["positive_fixed_ships"],
                     primary["positive_fixed_groups"]) == (25, 19, 23, 18)
                and old.arm_summary(anchors, "spans") == summary["arm_measurements"]
                and old.arm_summary(anchors, "whole") == summary["whole_mask_diagnostic"]["arm_measurements"]
                and old.contrast(anchors, field="whole") == summary["whole_mask_diagnostic"]["paired_all_four"],
                "Frozen per-anchor records must reconstruct the common population and measurements")

    # Read diagnostic coordinates only after the complete frozen cohort itself passes parity.
    for anchor in anchors:
        with old.original_npz(a.intake, anchor["row"]) as archive:
            for key in ("pixel_col", "pixel_row"):
                anchor[key] = old.integer(archive, key)
            for key in ("dt_sec", "sog"):
                anchor[key] = old.numeric(archive, key)
            lon, lat = (optional_numeric(old, archive, key) for key in ("lon", "lat"))
            anchor["lon"] = lon if lon is not None and -180 <= lon <= 180 else None
            anchor["lat"] = lat if lat is not None and -90 < lat < 90 else None
            heading = optional_numeric(old, archive, "heading") if getattr(a, "include_heading", False) else None
            anchor["heading"] = heading if heading is not None and 0 < heading < 360 else None
    return summary, old, manifest, anchors
