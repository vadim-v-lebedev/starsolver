"""
2000-2050 osculating elements from JPL Horizons compressed w Chebyshev polynomials 
"""

import numpy as np
from numpy.polynomial import Chebyshev
import datetime
import pickle
from pathlib import Path
import os
from typing import List, Dict, Tuple


def jd(dt: datetime.datetime) -> float:
    a   = (14 - dt.month) // 12
    y   = dt.year + 4800 - a
    m   = dt.month + 12 * a - 3
    jdn = dt.day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    return jdn - 0.5 + (dt.hour + dt.minute / 60.0 + dt.second / 3600.0) / 24.0


def straighten(x):
    add = np.cumsum(360 * ((x[1:] - x[:-1]) < 0))
    add = np.concatenate([[0], add])
    return x + add


def solve_kepler(M, e):
    """Return eccentric anomaly E in radians for mean anomaly M (degrees)."""
    M, e = np.array(M), np.array(e)
    E = np.where(e < 0.8, M + e * np.sin(M), np.pi)
    
    for _ in range(50):
        # f(E) = E - e*sin(E) - M
        # f'(E) = 1 - e*cos(E)
        dE = (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
        E -= dE
        if np.max(np.abs(dE)) < 1e-12:
            break
    return E


def el2xyz(el):

    a = el['a']
    e = el['e']
    incl = np.radians(el['incl'])
    Omega = np.radians(el['Omega'])
    w = np.radians(el['wp']) - Omega
    
    M = np.radians((el['L'] - el['wp']) % 360)
    
    E = solve_kepler(M, e)

    xp = a * (np.cos(E) - e)
    yp = a * np.sqrt(np.maximum(0.0, 1.0 - e**2)) * np.sin(E)

    cosO, sinO   = np.cos(Omega), np.sin(Omega)
    cosom, sinom = np.cos(w),  np.sin(w)
    cosI, sinI   = np.cos(incl),   np.sin(incl)

    x = (cosO*cosom - sinO*sinom*cosI)*xp + (-cosO*sinom - sinO*cosom*cosI)*yp
    y = (sinO*cosom + cosO*sinom*cosI)*xp + (-sinO*sinom + cosO*cosom*cosI)*yp
    z = (sinom*sinI)*xp + (cosom*sinI)*yp
    
    return x, y, z


def heliocentric2radec(x, y, z, xe, ye, ze):

    eps = np.radians(23.43927944)
    coseps = np.cos(eps)
    sineps = np.sin(eps)
    
    xg, yg, zg = x - xe, y - ye, z - ze

    xq = xg
    yq = coseps * yg - sineps * zg
    zq = sineps * yg + coseps * zg
    r  = np.sqrt(xq**2 + yq**2 + zq**2)
    ra_deg  = np.degrees(np.arctan2(yq, xq)) % 360
    dec_deg = np.degrees(np.arcsin(np.clip(zq / r, -1.0, 1.0)))
    
    return ra_deg, dec_deg


NBLOCKS = 50
POLYDEG = 5

def blockchebyshev(t, coeffs, limits):
    t = np.array(t)
    x = np.zeros_like(t)
    was_found = np.zeros_like(t, dtype=bool)
    for i in range(NBLOCKS):
        mask = (t >= limits[i]) & (t <= limits[i+1])
        t_chunk = t[mask]
        t_scaled = 2 * (t_chunk - limits[i]) / (limits[i + 1] - limits[i]) - 1
        x[mask] = Chebyshev(coeffs[i])(t_scaled)
        was_found[mask] = True
    if np.any(~was_found):
        raise Exception(f"some values out of temporal range")
    return x


def elem_to_date(el, t):
    result = {}
    for key in ["a", "e", "Omega", "wp", "incl", "L"]:
        coeffs, limits = el[key]
        result[key] = blockchebyshev(t, coeffs, limits)
    return result

ASTEROIDS_DATA = Path(os.path.dirname(os.path.abspath(__file__))) / "asteroids_short.pkl"
PLANETS_DATA = Path(os.path.dirname(os.path.abspath(__file__))) / "planets.pkl"

def _get_positions(timestamp_iso: str) -> List[Tuple[str, float, float]]:
    dt = datetime.datetime.fromisoformat(timestamp_iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)

    t = jd(dt)
    
    asteroids = pickle.load(open(ASTEROIDS_DATA, "rb"))
    planets = pickle.load(open(PLANETS_DATA, "rb"))

    xe, ye, ze = el2xyz(elem_to_date(planets["Earth"]["elements"], t))

    result = []
    
    for name, el in planets.items():
        if name == 'Earth':
            continue
        x, y, z = el2xyz(elem_to_date(el["elements"], t))
        ra, dec = heliocentric2radec(x, y, z, xe, ye, ze)
        result.append((name, ra, dec))
        
    for asteroid_i, el in asteroids.items():
        name = f"{asteroid_i} {el['name']}"
        x, y, z = el2xyz(elem_to_date(el["elements"], t))
        ra, dec = heliocentric2radec(x, y, z, xe, ye, ze)
        result.append((name, ra, dec))

    return result


def match_planets(plate, timestamp_iso: str,
                  unknown_dets: List[Dict],
                  threshold: float) -> List[Dict]:
    """
    Match unknown detections to planets.

    Mutates unknown_dets in-place (removes matched entries).
    Returns list of {name, ra, dec, x, y} for matched planets.
    """
    if not timestamp_iso or not unknown_dets:
        return []

    positions = _get_positions(timestamp_iso)

    thr2    = threshold * threshold
    matched = []
    taken   = set()

    for name, ra_deg, dec_deg in positions:
        px = plate.radec_to_pixel(ra_deg, dec_deg)
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
                'name': name,
                'ra':   round(ra_deg,  4),
                'dec':  round(dec_deg, 4),
                'x':    det['x'],
                'y':    det['y'],
            })

    for i in sorted(taken, reverse=True):
        unknown_dets.pop(i)

    return matched