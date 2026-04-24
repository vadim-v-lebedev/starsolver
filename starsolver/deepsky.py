"""Messier deep-sky object catalog and plate matching."""
import os
import re
from typing import List, Dict, Optional, Tuple

_CATALOG: Optional[List[Dict]] = None


def _parse_ra(s: str) -> Optional[float]:
    m = re.match(r'(\d+)h\s+(\d+\.?\d*)m(?:\s+(\d+\.?\d*)s)?', s.strip())
    if m:
        h, mn = float(m.group(1)), float(m.group(2))
        sec = float(m.group(3)) if m.group(3) else 0.0
        return (h + mn / 60 + sec / 3600) * 15.0
    return None


def _parse_dec(s: str) -> Optional[float]:
    m = re.match(r'([+-]?)(\d+)deg\s+(\d+\.?\d*)\'(?:\s+(\d+\.?\d*)")?', s.strip())
    if m:
        sign = -1 if m.group(1) == '-' else 1
        d, arcmin = float(m.group(2)), float(m.group(3))
        arcsec = float(m.group(4)) if m.group(4) else 0.0
        return sign * (d + arcmin / 60 + arcsec / 3600)
    return None


def _load_catalog() -> List[Dict]:
    path = os.path.join(os.path.dirname(__file__), 'messier.tsv')
    catalog = []
    with open(path, encoding='utf-8') as f:
        next(f)  # skip header
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 8:
                continue
            messier = parts[0].strip()   # "M1", "M2", …
            ra      = _parse_ra(parts[6])
            dec     = _parse_dec(parts[7])
            if ra is None or dec is None:
                continue
            catalog.append({
                'messier': messier,
                'name':    messier,
                'ra':      ra,
                'dec':     dec,
            })
    return catalog


def get_catalog() -> List[Dict]:
    global _CATALOG
    if _CATALOG is None:
        _CATALOG = _load_catalog()
    return _CATALOG


def match_deepsky(plate, unknown_dets: List[Dict],
                  threshold: float) -> List[Dict]:
    """
    Match unknown detections to Messier catalog objects.

    Mutates unknown_dets in-place (removes matched entries).
    Returns list of {name, messier, ra, dec, x, y} where both name and
    messier are the M-number (e.g. "M31").
    """
    if not unknown_dets:
        return []

    thr2    = threshold * threshold
    matched = []
    taken   = set()

    for obj in get_catalog():
        px = plate.radec_to_pixel(obj['ra'], obj['dec'])
        if px is None:
            continue
        cx, cy = float(px[0]), float(px[1])

        best_i, best_d2 = None, thr2
        for i, det in enumerate(unknown_dets):
            if i in taken:
                continue
            d2 = (det['x'] - cx) ** 2 + (det['y'] - cy) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_i  = i

        if best_i is not None:
            taken.add(best_i)
            det = unknown_dets[best_i]
            matched.append({
                'name':    obj['name'],
                'messier': obj['messier'],
                'ra':      round(obj['ra'],  4),
                'dec':     round(obj['dec'], 4),
                'x':       det['x'],
                'y':       det['y'],
            })

    for i in sorted(taken, reverse=True):
        unknown_dets.pop(i)

    return matched
