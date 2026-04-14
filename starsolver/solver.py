import os
from pathlib import Path
import numpy as np
import tetra3
from typing import List, Dict, Optional


_t3 = None

# Database lives next to this file (works on desktop and inside Chaquopy on Android)
_DEFAULT_DB = Path(os.path.dirname(os.path.abspath(__file__))) / 'tetra3_database'


def _get_tetra3(database_path=None) -> tetra3.Tetra3:
    global _t3
    if _t3 is None:
        path = Path(database_path) if database_path else _DEFAULT_DB
        _t3 = tetra3.Tetra3(path)
    return _t3


def plate_solve(stars: List[Dict], image_width: int, image_height: int,
                fov_estimate: Optional[float] = None,
                fov_max_error: Optional[float] = None,
                solve_timeout: Optional[float] = 10000,
                return_matches: bool = False,
                database_path: Optional[str] = None) -> Optional[Dict]:
    """
    Attempt blind plate solving on detected star positions.

    return_matches: if True, forces solve_timeout=None and includes matched_stars /
                    matched_centroids in the result (needed for intrinsics fitting).
    """
    if len(stars) < 4:
        return None

    top = stars[:50]
    centroids = np.array([[s['y'], s['x']] for s in top], dtype=np.float64)

    t3 = _get_tetra3(database_path)
    result = t3.solve_from_centroids(
        centroids,
        size=(image_height, image_width),
        fov_estimate=fov_estimate,
        fov_max_error=fov_max_error,
        pattern_checking_stars=len(top),
        solve_timeout=solve_timeout,
        return_matches=return_matches,
    )

    if result.get('RA') is None:
        return None

    fov = float(result['FOV'])
    if fov < 30 or fov > 100:
        return None

    out = {
        'RA':      round(float(result['RA']),   4),
        'Dec':     round(float(result['Dec']),  4),
        'Roll':    round(float(result['Roll']), 4),
        'FOV':     round(float(result['FOV']),  4),
        'RMSE':    round(float(result['RMSE']), 2),
        'matches': int(result.get('Matches', 0)),
        'T_solve': round(float(result.get('T_solve', 0)), 1),
        'w':       image_width,
        'h':       image_height,
    }

    if return_matches:
        out['matched_stars']     = result.get('matched_stars', [])
        out['matched_centroids'] = result.get('matched_centroids', [])

        ms = result.get('matched_stars')
        if ms is not None and len(ms) > 0:
            import catalog as _cat
            from catalog import _CONS_NAMES
            ms = np.asarray(ms, dtype=np.float64)   # (M, 2): [[ra_deg, dec_deg], ...]
            ra_rad_cat, dec_rad_cat, _, _ = _cat._get_hip_catalog()

            # Vectorised nearest-neighbour lookup into the full catalog
            ra_q  = np.radians(ms[:, 0])[:, None]
            dec_q = np.radians(ms[:, 1])[:, None]
            dra   = ra_rad_cat[None, :] - ra_q
            ddec  = dec_rad_cat[None, :] - dec_q
            dist2 = (dra * np.cos(dec_q)) ** 2 + ddec ** 2
            cat_idx = np.argmin(dist2, axis=1)

            cons_set = set(_cat._hip_cons[cat_idx].tolist())
            out['constellations'] = sorted(_CONS_NAMES[c] for c in cons_set)
            out['dec_min'] = round(float(ms[:, 1].min()), 2)
            out['dec_max'] = round(float(ms[:, 1].max()), 2)

    return out


