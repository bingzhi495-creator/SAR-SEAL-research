"""Decode the original stock training visit array with the locked JSON rules."""
import json


def decode_stock_visits(raw, unique, invalid_constant):
    visits = json.loads(raw, object_pairs_hook=unique, parse_constant=invalid_constant)
    if type(visits) is not list or len(visits) != 32 or any(type(row) is not dict for row in visits):
        raise ValueError("Original 32-row qualification visit array required")
    return visits
