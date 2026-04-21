"""
Coordinate transforms between celestial (RA/Dec) and image-plane (pixel) coordinates.

Convention: boresight = X axis (tetra3 / ICP convention).
"""
import numpy as np
from typing import Dict, Optional, Tuple

from minicv import rodrigues, mat_to_rvec, project_points


def _build_rotation_matrix(ra_c: float, dec_c: float, roll: float) -> np.ndarray:
    """Build the 3×3 rotation matrix from (RA, Dec, Roll) in radians.

    Maps celestial unit vectors to the ICP image-frame basis:
      row 0 = boresight, row 1 = image-left, row 2 = image-up.
    """
    boresight = np.array([
        np.cos(dec_c) * np.cos(ra_c),
        np.cos(dec_c) * np.sin(ra_c),
        np.sin(dec_c),
    ])
    north = np.array([
        -np.sin(dec_c) * np.cos(ra_c),
        -np.sin(dec_c) * np.sin(ra_c),
        np.cos(dec_c),
    ])
    east = np.array([-np.sin(ra_c), np.cos(ra_c), 0.0])
    r1 = np.sin(roll) * north + np.cos(roll) * east
    r2 = np.cos(roll) * north - np.sin(roll) * east
    return np.array([boresight, r1, r2])


class Plate:
    """Camera plate solution: rotation + intrinsics + image dimensions."""

    __slots__ = ('rvec', 'f', 'cx', 'cy', 'k1', 'k2', 'w', 'h', 'timestamp')

    def __init__(self, rvec, f: float, cx: float, cy: float,
                 k1: float, k2: float, w: int, h: int,
                 timestamp: Optional[str] = None):
        self.rvec      = np.asarray(rvec, dtype=np.float64).ravel()
        self.f         = float(f)
        self.cx        = float(cx)
        self.cy        = float(cy)
        self.k1        = float(k1)
        self.k2        = float(k2)
        self.w         = int(w)
        self.h         = int(h)
        self.timestamp = timestamp  # ISO 8601 string or None

    @property
    def R(self) -> np.ndarray:
        """3×3 rotation matrix (boresight = X axis convention)."""
        return rodrigues(self.rvec)

    @property
    def fov_deg(self) -> float:
        """Field of view along the longest image dimension, in degrees."""
        return float(np.degrees(2.0 * np.arctan(max(self.w, self.h) / (2.0 * self.f))))

    @property
    def radec_roll(self) -> Tuple[float, float, float]:
        """(RA_deg, Dec_deg, Roll_deg) derived from the rotation matrix."""
        R         = self.R
        boresight = R[0]
        dec = float(np.arcsin(np.clip(boresight[2], -1.0, 1.0)))
        ra  = float(np.arctan2(boresight[1], boresight[0])) % (2.0 * np.pi)
        north = np.array([-np.sin(dec) * np.cos(ra),
                          -np.sin(dec) * np.sin(ra),
                           np.cos(dec)])
        east  = np.array([-np.sin(ra), np.cos(ra), 0.0])
        roll  = float(np.arctan2(float(R[1] @ north), float(R[1] @ east)))
        return float(np.degrees(ra)), float(np.degrees(dec)), float(np.degrees(roll))

    @classmethod
    def from_dict(cls, d: Dict) -> 'Plate':
        """Build from a solver or refine result dict.

        Required keys: RA, Dec, Roll, w, h and either FOV or f.
        Optional keys: f, cx, cy, k1, k2 (defaults to pinhole centred on image).
        """
        w = int(d['w'])
        h = int(d['h'])
        R = _build_rotation_matrix(
            np.radians(d['RA']),
            np.radians(d['Dec']),
            np.radians(d['Roll']),
        )
        rvec = mat_to_rvec(R)
        if 'f' in d:
            f  = float(d['f'])
            cx = float(d.get('cx', w / 2))
            cy = float(d.get('cy', h / 2))
            k1 = float(d.get('k1', 0.0))
            k2 = float(d.get('k2', 0.0))
        else:
            fov = np.radians(d['FOV'])
            f   = w / 2.0 / np.tan(fov / 2.0)  # FOV is always horizontal (width axis)
            cx  = w / 2.0
            cy  = h / 2.0
            k1  = 0.0
            k2  = 0.0
        return cls(rvec, f, cx, cy, k1, k2, w, h,
                   timestamp=d.get('timestamp'))

    def project_with_mask(self, v_cel: np.ndarray):
        """Project (N,3) celestial unit vectors to pixel coordinates with validity mask.

        Returns (px, py, valid) where valid[i] is True when the vector is in
        front of the camera.  px/py are defined for all i but meaningful only
        where valid is True.
        """
        v_cam    = (self.R @ v_cel.T).T
        in_front = v_cam[:, 0] > 0
        safe     = np.where(in_front, v_cam[:, 0], 1.0)
        xn       = np.where(in_front, -v_cam[:, 1] / safe, 0.0)
        yn       = np.where(in_front, -v_cam[:, 2] / safe, 0.0)
        r2       = xn ** 2 + yn ** 2
        d        = 1.0 + self.k1 * r2 + self.k2 * r2 ** 2
        return self.f * xn * d + self.cx, self.f * yn * d + self.cy, in_front

    def project(self, v_cel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project celestial unit vectors to pixel (px, py) arrays."""
        return project_points(v_cel, self.rvec, self.f, self.cx, self.cy, self.k1, self.k2)

    def radec_to_pixel(self, ra_deg: float, dec_deg: float) -> Optional[Tuple[int, int]]:
        """Project (RA, Dec) degrees to pixel (x, y), or None if behind camera or too far off-axis."""
        ra_s, dec_s = np.radians(ra_deg), np.radians(dec_deg)
        v_cel = np.array([[np.cos(dec_s) * np.cos(ra_s),
                           np.cos(dec_s) * np.sin(ra_s),
                           np.sin(dec_s)]])
        # Reject stars behind camera or beyond 1.5× the image half-diagonal in
        # normalised coords.  Checked before distortion to avoid artefacts from
        # applying the distortion model far outside the calibrated FOV.
        v_cam = self.R @ v_cel[0]
        if v_cam[0] <= 0:
            return None
        r2 = (v_cam[1] ** 2 + v_cam[2] ** 2) / v_cam[0] ** 2
        r_max_sq = 2.25 * ((self.w / (2 * self.f)) ** 2 + (self.h / (2 * self.f)) ** 2)
        if r2 > r_max_sq:
            return None
        px, py = self.project(v_cel)
        return int(round(float(px[0]))), int(round(float(py[0])))

    @property
    def datetime(self):
        """Parse timestamp to a datetime object, or None if absent/invalid."""
        if not self.timestamp:
            return None
        from datetime import datetime
        try:
            return datetime.fromisoformat(self.timestamp)
        except (ValueError, TypeError):
            return None

    def to_dict(self) -> Dict:
        """Serialise to a dict compatible with from_dict."""
        ra_deg, dec_deg, roll_deg = self.radec_roll
        d = {
            'RA':   round(ra_deg,   4),
            'Dec':  round(dec_deg,  4),
            'Roll': round(roll_deg, 4),
            'FOV':  round(self.fov_deg, 4),
            'f':    round(self.f,   2),
            'cx':   round(self.cx,  2),
            'cy':   round(self.cy,  2),
            'k1':   round(self.k1,  6),
            'k2':   round(self.k2,  6),
            'w':    self.w,
            'h':    self.h,
        }
        if self.timestamp:
            d['timestamp'] = self.timestamp
        return d
