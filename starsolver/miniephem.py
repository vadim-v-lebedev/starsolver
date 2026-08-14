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

import functools

@functools.cache
def _load_planets():
    return pickle.load(open(PLANETS_DATA, "rb"))

@functools.cache
def _load_asteroids():
    return pickle.load(open(ASTEROIDS_DATA, "rb"))


# Saturn's rotation pole in ecliptic J2000 (from IAU pole RA=40.589°, Dec=83.537°)
_SATURN_POLE_ECL = np.array([0.08560, 0.46257, 0.88242])


def _planet_magnitude(name: str, r: float, delta: float, ph: float,
                      gcl_vec: np.ndarray = None, year: float = 2000.0) -> float:
    """Apparent V magnitude using Mallama & Hilton (2018) coefficients.
    r, delta: helio/geocentric distances in AU. ph: phase angle in degrees.
    gcl_vec: unit vector from planet toward Earth in ecliptic J2000 (Saturn only)."""
    log_rd = 5.0 * np.log10(r * delta)

    if name == 'Mercury':
        phase = (6.3280e-2*ph - 1.6336e-3*ph**2 + 3.3644e-5*ph**3
                 - 3.4265e-7*ph**4 + 1.6893e-9*ph**5 - 3.0334e-12*ph**6)
        return -0.613 + log_rd + phase

    if name == 'Venus':
        if ph <= 163.7:
            phase = -1.044e-3*ph + 3.687e-4*ph**2 - 2.814e-6*ph**3 + 8.938e-9*ph**4
        else:
            phase = 236.45828 - 2.81914*ph + 8.39034e-3*ph**2
        return -4.384 + log_rd + phase

    if name == 'Mars':
        if ph <= 50.0:
            return -1.601 + log_rd + 2.267e-2*ph - 1.302e-4*ph**2
        return -0.367 + log_rd - 2.573e-2*ph + 3.445e-4*ph**2

    if name == 'Jupiter':
        if ph <= 12.0:
            return -9.395 + log_rd + (6.16e-4*ph - 3.7e-4)*ph
        x = ph / 180.0
        phase = -2.5 * np.log10(
            1.0 - 1.507*x - 0.363*x**2 - 0.062*x**3 + 2.809*x**4 - 1.876*x**5)
        return -9.428 + log_rd + phase

    if name == 'Saturn':
        B = 0.0
        if gcl_vec is not None:
            B = float(np.degrees(np.arcsin(
                np.clip(np.dot(_SATURN_POLE_ECL, gcl_vec), -1.0, 1.0))))
        B_rad = np.radians(abs(B))
        phase = (-1.825 * np.sin(B_rad) + 0.026 * ph
                 - 0.378 * np.sin(B_rad) * np.exp(-2.25 * ph))
        return -8.914 + log_rd + phase

    if name == 'Uranus':
        # Sub-latitude term omitted (±0.3 mag effect, requires pole geometry)
        phase = 0.0 if ph <= 3.1 else (1.045e-4*ph + 6.587e-3)*ph
        return -7.110 + log_rd + phase

    if name == 'Neptune':
        V0 = float(np.clip(-6.89 - 0.0054 * (year - 1980.0), -7.00, -6.89))
        return V0 + log_rd + 7.944e-3*ph + 9.617e-5*ph**2

    return float('nan')


def _asteroid_magnitude(H: float, G: float, r: float, delta: float, ph: float) -> float:
    """Apparent V magnitude using the H-G phase function (Bowell et al. 1989)."""
    t = np.tan(np.radians(ph / 2.0))
    phi1 = np.exp(-3.33 * t ** 0.63)
    phi2 = np.exp(-1.87 * t ** 1.22)
    return H + 5.0 * np.log10(r * delta) - 2.5 * np.log10((1.0 - G) * phi1 + G * phi2)


def _get_positions(timestamp_iso: str) -> List[Tuple[str, float, float, float]]:
    dt = datetime.datetime.fromisoformat(timestamp_iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)

    t    = jd(dt)
    year = dt.year + (t - jd(datetime.datetime(dt.year, 1, 1,
                                               tzinfo=datetime.timezone.utc))) / 365.25

    asteroids = _load_asteroids()
    planets   = _load_planets()

    xe, ye, ze = (float(v) for v in el2xyz(elem_to_date(planets["Earth"]["elements"], t)))
    d_e2 = xe**2 + ye**2 + ze**2

    result = []

    for name, el in planets.items():
        if name == 'Earth':
            continue
        x, y, z = (float(v) for v in el2xyz(elem_to_date(el["elements"], t)))
        ra, dec  = heliocentric2radec(x, y, z, xe, ye, ze)
        xg, yg, zg = x - xe, y - ye, z - ze
        r     = np.sqrt(x**2  + y**2  + z**2)
        delta = np.sqrt(xg**2 + yg**2 + zg**2)
        cos_ph = np.clip((r**2 + delta**2 - d_e2) / (2.0 * r * delta), -1.0, 1.0)
        ph     = float(np.degrees(np.arccos(cos_ph)))
        gcl_vec = np.array([xg, yg, zg]) / delta if name == 'Saturn' else None
        mag = _planet_magnitude(name, float(r), float(delta), ph, gcl_vec, year)
        result.append((name, float(ra), float(dec), round(float(mag), 1)))

    for asteroid_i, el in asteroids.items():
        name = f"{asteroid_i} {el['name']}"
        x, y, z = (float(v) for v in el2xyz(elem_to_date(el["elements"], t)))
        ra, dec  = heliocentric2radec(x, y, z, xe, ye, ze)
        xg, yg, zg = x - xe, y - ye, z - ze
        r     = float(np.sqrt(x**2  + y**2  + z**2))
        delta = float(np.sqrt(xg**2 + yg**2 + zg**2))
        cos_ph = float(np.clip((r**2 + delta**2 - d_e2) / (2.0 * r * delta), -1.0, 1.0))
        ph     = float(np.degrees(np.arccos(cos_ph)))
        H, G   = el['mag_params']
        mag    = _asteroid_magnitude(float(H), float(G), r, delta, ph)
        result.append((name, float(ra), float(dec), round(float(mag), 1)))

    return result


def match_planets(plate, timestamp_iso: str,
                  unknown_dets: List[Dict],
                  threshold: float) -> List[Dict]:
    """
    Match unknown detections to planets and asteroids.

    Mutates unknown_dets in-place (removes matched entries).
    Returns list of {name, ra, dec, mag, x, y} for matched objects.
    """
    if not timestamp_iso or not unknown_dets:
        return []

    positions = _get_positions(timestamp_iso)

    thr2    = threshold * threshold
    matched = []
    taken   = set()

    for name, ra_deg, dec_deg, mag in positions:
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
                'name':       name,
                'ra':         round(ra_deg,  4),
                'dec':        round(dec_deg, 4),
                'mag':        mag,
                'x':          det['x'],
                'y':          det['y'],
                'brightness': det.get('brightness'),
            })

    for i in sorted(taken, reverse=True):
        unknown_dets.pop(i)

    return matched