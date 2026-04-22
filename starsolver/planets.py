"""
Geocentric planetary positions using Keplerian orbital elements.

Pure-numpy implementation of the JPL "Approximate Positions of Major Planets"
algorithm (Standish 1992), Table 1: valid 1800–2050 AD.

Accuracy (geocentric GCRS vs DE432):
  Mercury, Venus, Mars, Uranus, Neptune: < 2 arcminutes
  Jupiter, Saturn: < 15 arcminutes (great-inequality oscillation ~20 yr period)

No external downloads or third-party libraries required.
"""
import numpy as np
import datetime as _dt
from typing import List, Dict, Tuple


# Orbital elements at J2000.0 + rates per Julian century.
# Format: [a0, e0, I0, L0, w_bar0, O0,  da, de, dI, dL, dw_bar, dO]
# Units:  [au,  ,  deg, deg, deg, deg,  au/cy, /cy, deg/cy, deg/cy, deg/cy, deg/cy]
_EL = {
    'Mercury': [0.38709927, 0.20563593,  7.00497902, 252.25032350,  77.45779628,  48.33076593,
                0.00000037, 0.00001906, -0.00594749, 149472.67411175,  0.16047689,  -0.12534081],
    'Venus':   [0.72333566, 0.00677672,  3.39467605, 181.97909950, 131.60246718,  76.67984255,
                0.00000390,-0.00004107, -0.00078890,  58517.81538729,  0.00268329,  -0.27769418],
    '_Earth':  [1.00000261, 0.01671123, -0.00001531, 100.46457166, 102.93768193,   0.00000000,
                0.00000562,-0.00004392, -0.01294668,  35999.37244981,  0.32327364,   0.00000000],
    'Mars':    [1.52371034, 0.09339410,  1.84969142,  -4.55343205, -23.94362959,  49.55953891,
                0.00001847, 0.00007882, -0.00813131,  19140.30268499,  0.44441088,  -0.29257343],
    'Jupiter': [5.20288700, 0.04838624,  1.30439695,  34.39644051,  14.72847983, 100.47390909,
               -0.00011607,-0.00013253, -0.00183714,   3034.74612775,  0.21252668,   0.20469106],
    'Saturn':  [9.53667594, 0.05386179,  2.48599187,  49.95424423,  92.59887831, 113.66242448,
               -0.00125060,-0.00050991,  0.00193609,   1222.49362201, -0.41897216,  -0.28867794],
    'Uranus':  [19.18916464,0.04725744,  0.77263783, 313.23810451, 170.95427630,  74.01692503,
               -0.00196176,-0.00004397, -0.00242939,    428.48202785,  0.40805281,   0.04240589],
    'Neptune': [30.06992276,0.00859048,  1.77004347, -55.12002969,  44.96476227, 131.78422574,
                0.00026291, 0.00005105,  0.00035372,    218.45945325, -0.32241464,  -0.00508664],
}

PLANET_NAMES = [k for k in _EL if not k.startswith('_')]


def is_available() -> bool:
    return True   # only needs numpy


def _jd(dt: _dt.datetime) -> float:
    a   = (14 - dt.month) // 12
    y   = dt.year + 4800 - a
    m   = dt.month + 12 * a - 3
    jdn = dt.day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    return jdn - 0.5 + (dt.hour + dt.minute / 60.0 + dt.second / 3600.0) / 24.0


def _solve_kepler(M_deg: float, e: float) -> float:
    """Return eccentric anomaly E in degrees for mean anomaly M (degrees)."""
    M = M_deg % 360.0
    E = M + np.degrees(e * np.sin(np.radians(M))) * (1.0 + e * np.cos(np.radians(M)))
    for _ in range(50):
        dE = (M - E + np.degrees(e * np.sin(np.radians(E)))) / (1.0 - e * np.cos(np.radians(E)))
        E += dE
        if abs(dE) < 1e-6:
            break
    return E


def _helio_xyz(el: list, T: float) -> Tuple[float, float, float]:
    a  = el[0] + el[6]  * T
    e  = el[1] + el[7]  * T
    I  = np.radians(el[2]  + el[8]  * T)
    L  = el[3] + el[9]  * T
    wp = el[4] + el[10] * T   # longitude of perihelion ω̄
    O  = el[5] + el[11] * T   # longitude of ascending node Ω

    M = (L - wp) % 360.0

    om  = np.radians(wp - O)   # argument of perihelion ω
    O_r = np.radians(O)
    E   = np.radians(_solve_kepler(M, e))

    xp = a * (np.cos(E) - e)
    yp = a * np.sqrt(max(0.0, 1.0 - e * e)) * np.sin(E)

    cosO, sinO   = np.cos(O_r), np.sin(O_r)
    cosom, sinom = np.cos(om),  np.sin(om)
    cosI, sinI   = np.cos(I),   np.sin(I)

    x = (cosO*cosom - sinO*sinom*cosI)*xp + (-cosO*sinom - sinO*cosom*cosI)*yp
    y = (sinO*cosom + cosO*sinom*cosI)*xp + (-sinO*sinom + cosO*cosom*cosI)*yp
    z = (sinom*sinI)*xp + (cosom*sinI)*yp
    return x, y, z


def _get_positions(timestamp_iso: str) -> List[Tuple[str, float, float]]:
    dt = _dt.datetime.fromisoformat(timestamp_iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_dt.timezone.utc)
    T = (_jd(dt) - 2451545.0) / 36525.0

    ex, ey, ez = _helio_xyz(_EL['_Earth'], T)

    eps    = np.radians(23.43929111 - 0.01300417 * T)
    coseps = np.cos(eps)
    sineps = np.sin(eps)

    result = []
    for name, el in _EL.items():
        if name.startswith('_'):
            continue
        px, py, pz = _helio_xyz(el, T)
        gx, gy, gz = px - ex, py - ey, pz - ez
        # ecliptic → equatorial
        xq = gx
        yq = coseps * gy - sineps * gz
        zq = sineps * gy + coseps * gz
        r  = np.sqrt(xq*xq + yq*yq + zq*zq)
        ra_deg  = float(np.degrees(np.arctan2(yq, xq)) % 360.0)
        dec_deg = float(np.degrees(np.arcsin(np.clip(zq / r, -1.0, 1.0))))
        result.append((name, ra_deg, dec_deg))
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

    try:
        positions = _get_positions(timestamp_iso)
    except Exception:
        return []

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
