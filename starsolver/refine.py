"""
ICP-based refinement of a tetra3 plate solution.

Each outer iteration:
  1. Projects catalog stars with current rotation + intrinsics.
  2. Assigns each detection to its nearest projected catalog star
     (spatial gate + photometric filter + one-to-one dedup).
  3. Takes a single Gauss-Newton step that jointly updates rotation
     and intrinsics in a single lstsq solve per iteration.

Distortion model selected by detection count:
  < 15 detections  → pinhole  (f, cx, cy only)
  15–50 detections → k1       (+ one radial term)
  > 50 detections  → k1 + k2  (two radial terms)
"""
import numpy as np
from typing import List, Dict, Optional

from minicv import project_points
from plate import Plate
from catalog import _get_hip_catalog, _STAR_NAMES, _CONS_NAMES, _get_hip_id_to_cons


PHOT_SLOPE = -0.29   # log10(brightness) per catalog magnitude, empirical


# ── ICP sub-steps ─────────────────────────────────────────────────────────────

def _project_visible(plate: Plate, v_cel: np.ndarray, near_idx: np.ndarray,
                     threshold: float, w: int, h: int):
    """Project near catalog stars; return those falling within the image margin."""
    px, py = plate.project(v_cel[near_idx])
    margin = threshold + 100.0
    local = np.where(
        (px >= -margin) & (px < w + margin) &
        (py >= -margin) & (py < h + margin)
    )[0]
    return px[local], py[local], near_idx[local]


def _find_candidates(di, px_vis, py_vis, mag_vis, det_x, det_y, det_logb,
                     thr2, phot_b, phot_sig, have_phot):
    """Return ranked (vis_local_idx, dist²) pairs for detection di, or None."""
    ddx = px_vis - det_x[di]
    ddy = py_vis - det_y[di]
    dist2 = ddx * ddx + ddy * ddy
    within = np.where(dist2 < thr2)[0]
    if len(within) == 0:
        return None
    d2_w = dist2[within]
    if have_phot:
        pred_mag = (det_logb[di] - phot_b) / PHOT_SLOPE
        resids = pred_mag - mag_vis[within]
        lims = np.where(resids > 0, 5.0 * phot_sig, 3.0 * phot_sig)
        ok = np.abs(resids) <= lims
        within, d2_w = within[ok], d2_w[ok]
        if len(within) == 0:
            return None
    order = np.argsort(mag_vis[within])
    return list(zip(within[order].tolist(), d2_w[order].tolist()))


def _assign_matches(n_det, det_x, det_y, det_logb,
                    px_vis, py_vis, mag_vis, vis_idx,
                    threshold, phot_b, phot_sig):
    """Spatial gate + photometric filter + greedy one-to-one dedup.
    Returns (det_indices, cat_indices) int32 arrays."""
    thr2 = threshold * threshold
    have_phot = phot_sig < 1e8

    det_cands = [
        _find_candidates(di, px_vis, py_vis, mag_vis, det_x, det_y, det_logb,
                         thr2, phot_b, phot_sig, have_phot)
        for di in range(n_det)
    ]

    det_choice = {di: cands[0] for di, cands in enumerate(det_cands) if cands}

    for _pass in range(5):
        cat_claimants: Dict = {}
        for di, (j, d2) in det_choice.items():
            cat_claimants.setdefault(j, []).append((di, d2))
        displaced = []
        for j, claimants in cat_claimants.items():
            if len(claimants) <= 1:
                continue
            claimants.sort(key=lambda x: x[1])
            for di, _ in claimants[1:]:
                displaced.append(di)
                del det_choice[di]
        if not displaced:
            break
        taken = set(j for j, _ in det_choice.values())
        for di in displaced:
            if det_cands[di] is None:
                continue
            for j, d2 in det_cands[di]:
                if j not in taken:
                    det_choice[di] = (j, d2)
                    taken.add(j)
                    break

    det_list = sorted(det_choice.keys())
    cat_list  = [vis_idx[det_choice[di][0]] for di in det_list]
    return np.array(det_list, dtype=np.int32), np.array(cat_list, dtype=np.int32)


def _fit_photometry(mdi, mci, det_logb, mag):
    """Fit photometric zero-point (fixed slope) with sigma-clipping.
    Returns (phot_b, phot_sig) in magnitude units."""
    offsets = det_logb[mdi] - PHOT_SLOPE * mag[mci]
    b   = float(np.median(offsets))
    res = offsets - b
    sig = float(np.sqrt(np.mean(res ** 2)))
    keep = np.abs(res) < 2.5 * max(sig, 0.1)
    if np.sum(keep) >= 4:
        b   = float(np.median(offsets[keep]))
        sig = float(np.sqrt(np.mean((offsets[keep] - b) ** 2)))
    return b, max(sig / abs(PHOT_SLOPE), 0.1)


def _gauss_newton_step(plate: Plate, v_cel_m: np.ndarray,
                       det_x, det_y, mdi,
                       fit_k1: bool, fit_k2: bool) -> Plate:
    """One Gauss-Newton step; returns an updated Plate."""
    rv = plate.rvec
    f, cx, cy, k1, k2 = plate.f, plate.cx, plate.cy, plate.k1, plate.k2

    px, py, J = project_points(v_cel_m, rv, f, cx, cy, k1, k2, jacobian=True)
    res      = np.concatenate([px - det_x[mdi], py - det_y[mdi]])
    n_params = 6 + fit_k1 + fit_k2
    delta    = np.linalg.lstsq(J[:, :n_params], -res, rcond=None)[0]

    i = 0
    rv = rv + delta[i:i+3];   i += 3
    f  += float(delta[i]);     i += 1
    cx += float(delta[i]);     i += 1
    cy += float(delta[i]);     i += 1
    if fit_k1: k1 = float(np.clip(k1 + delta[i], -0.5, 0.5)); i += 1
    if fit_k2: k2 = float(np.clip(k2 + delta[i], -0.5, 0.5)); i += 1
    if f <= 0:
        f = plate.f

    return Plate(rv, f, cx, cy, k1, k2, plate.w, plate.h)


def _build_result(plate: Plate, v_cel: np.ndarray,
                  ra_rad, dec_rad, mag, hip_ids_arr,
                  mdi, mci, det_x, det_y, stars, phot_b) -> Dict:
    """Compute final residuals and assemble the return dict."""
    w, h = plate.w, plate.h

    px_f, py_f = plate.project(v_cel[mci])
    res_f = np.sqrt((px_f - det_x[mdi]) ** 2 + (py_f - det_y[mdi]) ** 2)

    fov = float(np.degrees(2.0 * np.arctan(w / (2.0 * plate.f))))
    rmse = float(np.sqrt(np.mean(res_f ** 2))) * fov * 3600.0 / w

    ra_deg, dec_deg, roll_deg = plate.radec_roll

    id_to_cons = _get_hip_id_to_cons()
    cons_set   = set()
    dec_vals   = []

    matched_stars = []
    matched_yx    = []
    for di, ci in zip(mdi, mci):
        hip_id  = int(hip_ids_arr[ci])
        dec_val = float(np.degrees(dec_rad[ci]))
        matched_stars.append({
            'hip_id': hip_id,
            'name':   _STAR_NAMES.get(hip_id, ''),
            'ra':     float(np.degrees(ra_rad[ci])),
            'dec':    dec_val,
            'mag':    float(mag[ci]),
            'x':      float(det_x[di]),
            'y':      float(det_y[di]),
        })
        matched_yx.append([float(det_y[di]), float(det_x[di])])
        c = id_to_cons.get(hip_id)
        if c is not None:
            cons_set.add(c)
        dec_vals.append(dec_val)

    constellations = sorted(_CONS_NAMES[c] for c in cons_set)
    dec_min = round(min(dec_vals), 2) if dec_vals else 0.0
    dec_max = round(max(dec_vals), 2) if dec_vals else 0.0

    matched_set = set(mdi.tolist())
    unknown_dets = [
        {'x': float(det_x[di]), 'y': float(det_y[di]),
         'brightness': float(stars[di]['brightness'])}
        for di in range(len(det_x))
        if di not in matched_set
    ]

    return {
        'status':             'refined',
        'RA':                 round(ra_deg,    4),
        'Dec':                round(dec_deg,   4),
        'Roll':               round(roll_deg,  4),
        'FOV':                round(fov,       4),
        'f':                  round(plate.f,   2),
        'cx':                 round(plate.cx,  2),
        'cy':                 round(plate.cy,  2),
        'k1':                 round(plate.k1,  6),
        'k2':                 round(plate.k2,  6),
        'w':                  w,
        'h':                  h,
        'RMSE':               round(rmse, 2),
        'matches':            int(len(mdi)),
        'matched_stars':      matched_stars,
        'matched_centroids':  matched_yx,
        'unknown_detections': unknown_dets,
        'phot_b':             round(phot_b, 4),
        'constellations':     constellations,
        'dec_min':            dec_min,
        'dec_max':            dec_max,
    }


# ── main entry point ──────────────────────────────────────────────────────────

def refine(plate: Plate, stars: List[Dict],
           catalog_path: Optional[str] = None,
           n_iter: int = 15) -> Dict:
    """
    Refine a plate solution using ICP with per-iteration Gauss-Newton steps.

    Returns a dict:
      status: 'refined' | 'failed'
      When 'refined':
        RA, Dec, Roll, FOV, f, cx, cy, k1, k2, RMSE, matches, w, h
        matched_stars:      list of {hip_id, name, ra, dec, mag, x, y}
        matched_centroids:  list of [y, x]
        unknown_detections: list of {x, y, brightness}
    """
    import catalog as _cat
    if _cat._hip_ids is None:
        _cat._load_catalog(catalog_path)
    hip_ids_full = _cat._hip_ids

    w, h = plate.w, plate.h
    n_det = len(stars)
    if n_det < 4:
        return {'status': 'failed'}

    ra_rad_full, dec_rad_full, mag_full, v_cel_full = _get_hip_catalog(catalog_path)

    mag_limit = 7.0

    def _apply_mag_limit(lim):
        sel = mag_full <= lim
        return (ra_rad_full[sel], dec_rad_full[sel], mag_full[sel],
                hip_ids_full[sel], v_cel_full[sel])

    ra_rad, dec_rad, mag, hip_ids_arr, v_cel = _apply_mag_limit(mag_limit)

    fit_k1 = n_det >= 15
    fit_k2 = n_det >= 50

    det_x     = np.array([s['x']          for s in stars], dtype=np.float64)
    det_y     = np.array([s['y']          for s in stars], dtype=np.float64)
    det_bright = np.array([s['brightness'] for s in stars], dtype=np.float64)
    det_logb  = np.log10(np.maximum(det_bright, 1e-12))

    arcsec_per_px = plate.fov_deg * 3600.0 / w
    threshold = max(50.0, 5.0 * (60.0 / arcsec_per_px))

    phot_b   = 0.0
    phot_sig = 1e9

    _fov_half_cos = np.cos(np.radians(min(plate.fov_deg, 90.0) / 2.0) + 0.3)
    near_idx = np.where((v_cel @ plate.R[0]) > _fov_half_cos)[0]

    match_det_idx  = None
    match_star_idx = None

    for it in range(n_iter):
        px_vis, py_vis, vis_idx = _project_visible(
            plate, v_cel, near_idx, threshold, w, h)
        if len(vis_idx) == 0:
            break

        mdi, mci = _assign_matches(
            n_det, det_x, det_y, det_logb,
            px_vis, py_vis, mag[vis_idx], vis_idx,
            threshold, phot_b, phot_sig)
        if len(mdi) < 4:
            break

        match_det_idx  = mdi
        match_star_idx = mci

        phot_b, phot_sig = _fit_photometry(mdi, mci, det_logb, mag)
        plate = _gauss_newton_step(plate, v_cel[mci], det_x, det_y, mdi, fit_k1, fit_k2)

        threshold = max(12.0, threshold * 0.75)

        if it >= 3 and len(mci) >= 10:
            faintest = float(np.max(mag[mci]))
            new_limit = min(faintest + 1.5, 12.0)
            if new_limit > mag_limit + 0.5:
                mag_limit = new_limit
                ra_rad, dec_rad, mag, hip_ids_arr, v_cel = _apply_mag_limit(mag_limit)
                near_idx = np.where((v_cel @ plate.R[0]) > _fov_half_cos)[0]

    if match_det_idx is None or len(match_det_idx) < 4:
        return {'status': 'failed'}

    return _build_result(plate, v_cel, ra_rad, dec_rad, mag, hip_ids_arr,
                         match_det_idx, match_star_idx, det_x, det_y, stars, phot_b)
