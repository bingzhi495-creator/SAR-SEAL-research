"""Exact two-donor common coverage for a fixed set of mask-IoU edges."""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow


def common_flow(left_edges, right_edges):
    """Find a maximum-cardinality assignment without donor reuse.

    The Boolean arrays share the same anchor axis. A true entry means the
    anchor-donor edge is admissible. Return each selected anchor index and its
    two donor indices. The witness follows sorted CSR order and the specified
    SciPy Dinic traversal. No optimization over edge weights is performed.
    """
    left_edges = np.asarray(left_edges)
    right_edges = np.asarray(right_edges)
    if (left_edges.ndim != 2 or right_edges.ndim != 2
            or left_edges.dtype != np.bool_ or right_edges.dtype != np.bool_
            or left_edges.shape[0] != right_edges.shape[0]):
        raise ValueError("Expected two Boolean matrices with the same anchor axis")

    anchor_count, left_count = left_edges.shape
    right_count = right_edges.shape[1]
    if not left_edges.any() or not right_edges.any():
        return {}

    source = 0
    left_start = 1
    anchor_in_start = left_start + left_count
    anchor_out_start = anchor_in_start + anchor_count
    right_start = anchor_out_start + anchor_count
    sink = right_start + right_count

    left_nodes = left_start + np.arange(left_count)
    anchor_in_nodes = anchor_in_start + np.arange(anchor_count)
    anchor_out_nodes = anchor_out_start + np.arange(anchor_count)
    right_nodes = right_start + np.arange(right_count)
    left_anchor_idx, left_donor_idx = np.nonzero(left_edges)
    right_anchor_idx, right_donor_idx = np.nonzero(right_edges)
    origins = np.concatenate((
        np.full(left_count, source),
        left_nodes[left_donor_idx],
        anchor_in_nodes,
        anchor_out_nodes[right_anchor_idx],
        right_nodes,
    ))
    destinations = np.concatenate((
        left_nodes,
        anchor_in_nodes[left_anchor_idx],
        anchor_out_nodes,
        right_nodes[right_donor_idx],
        np.full(right_count, sink),
    ))
    capacity = csr_matrix(
        (np.ones(len(origins), dtype=np.int32), (origins, destinations)),
        shape=(sink + 1, sink + 1),
    )
    capacity.sort_indices()
    result = maximum_flow(capacity, source, sink, method="dinic")

    left_witness = result.flow[left_start:anchor_in_start, anchor_in_start:anchor_out_start].tocoo()
    right_witness = result.flow[anchor_out_start:right_start, right_start:sink].tocoo()
    left_by_anchor = {
        int(anchor): int(donor)
        for donor, anchor, value in zip(left_witness.row, left_witness.col, left_witness.data, strict=True)
        if value > 0
    }
    right_by_anchor = {
        int(anchor): int(donor)
        for anchor, donor, value in zip(right_witness.row, right_witness.col, right_witness.data, strict=True)
        if value > 0
    }
    if left_by_anchor.keys() != right_by_anchor.keys() or len(left_by_anchor) != result.flow_value:
        raise RuntimeError("Integral flow did not produce the expected anchor witness")
    return {anchor: (left_by_anchor[anchor], right_by_anchor[anchor]) for anchor in sorted(left_by_anchor)}
