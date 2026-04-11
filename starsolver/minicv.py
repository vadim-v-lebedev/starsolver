"""
minicv – NumPy reimplementations of OpenCV primitives.

  rodrigues(rvec)            cv2.Rodrigues(rvec)[0]
  mat_to_rvec(R)             cv2.Rodrigues(R)[0]
  gaussian_blur(src, k, σ)   cv2.GaussianBlur(src, (k, k), σ)
  dilate(src, k)             cv2.dilate(src, np.ones((k, k)))
  resize_area(src, f)        cv2.resize(src, (w//f, h//f), interpolation=cv2.INTER_AREA)
  sep_filter2d(src, kern)    cv2.sepFilter2D(src, -1, kern, kern)
  project_points(…)          cv2.projectPoints(…)  [no tvec, simplified camera model]
"""
import numpy as np


# ── rotation ──────────────────────────────────────────────────────────────────

def rodrigues(rvec: np.ndarray) -> np.ndarray:
    """Rotation vector (3,) → rotation matrix (3, 3)."""
    rvec = np.asarray(rvec, dtype=np.float64).ravel()
    theta = np.linalg.norm(rvec)
    if theta < 1e-12:
        return np.eye(3)
    k = rvec / theta
    K = np.array([[    0, -k[2],  k[1]],
                  [ k[2],     0, -k[0]],
                  [-k[1],  k[0],     0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def mat_to_rvec(R: np.ndarray) -> np.ndarray:
    """Rotation matrix (3, 3) → rotation vector (3,)."""
    trace = np.trace(R)
    theta = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    if theta < 1e-10:
        return np.zeros(3)
    k = np.array([R[2, 1] - R[1, 2],
                  R[0, 2] - R[2, 0],
                  R[1, 0] - R[0, 1]]) / (2.0 * np.sin(theta))
    return k * theta


# ── image processing ──────────────────────────────────────────────────────────

def sep_filter2d(src: np.ndarray, kernel_1d: np.ndarray) -> np.ndarray:
    """Apply an isotropic separable 2D filter via two 1D sliding-window passes."""
    from numpy.lib.stride_tricks import sliding_window_view
    size = len(kernel_1d)
    pad  = size // 2
    a = np.pad(src, ((0, 0), (pad, pad)), mode='edge')
    a = (sliding_window_view(a, size, axis=1) * kernel_1d).sum(axis=-1)
    a = np.pad(a, ((pad, pad), (0, 0)), mode='edge')
    return (sliding_window_view(a, size, axis=0) * kernel_1d).sum(axis=-1)


def gaussian_blur(src: np.ndarray, ksize: int, sigma: float) -> np.ndarray:
    """2D Gaussian blur (area-normalised)."""
    x = np.arange(ksize) - ksize // 2
    kernel = np.exp(-x**2 / (2 * sigma**2)).astype(np.float32)
    kernel /= kernel.sum()
    return sep_filter2d(src, kernel)


def dilate(src: np.ndarray, ksize: int) -> np.ndarray:
    """2D max filter with square footprint."""
    from numpy.lib.stride_tricks import sliding_window_view
    pad = ksize // 2
    a = np.pad(src, ((0, 0), (pad, pad)), mode='edge')
    a = sliding_window_view(a, ksize, axis=1).max(axis=-1)
    a = np.pad(a, ((pad, pad), (0, 0)), mode='edge')
    return sliding_window_view(a, ksize, axis=0).max(axis=-1)


def resize_area(src: np.ndarray, factor: int) -> np.ndarray:
    """Downsample uint8 grayscale by integer factor via block averaging → float32 [0, 1]."""
    h, w = src.shape
    h2, w2 = h - h % factor, w - w % factor
    return (src[:h2, :w2]
            .reshape(h2 // factor, factor, w2 // factor, factor)
            .mean(axis=(1, 3))
            .astype(np.float32) / 255.0)


# ── projection ────────────────────────────────────────────────────────────────

def project_points(v_cel: np.ndarray, rvec: np.ndarray,
                   f: float, cx: float, cy: float,
                   k1: float = 0.0, k2: float = 0.0,
                   *, jacobian: bool = False):
    """
    Project celestial unit vectors to pixel coordinates.

    Simplified cv2.projectPoints: no translation vector, single focal length,
    radial distortion only (k1, k2).  Stars behind the camera are mapped to
    (1e9, 1e9) when jacobian=False.

    Parameters
    ----------
    v_cel    : (N, 3) unit vectors in the celestial frame
    rvec     : (3,) rotation vector (Rodrigues)
    f        : focal length in pixels
    cx, cy   : principal point
    k1, k2   : radial distortion coefficients
    jacobian : if True, also return J of shape (2N, 8)
               column order: rvec[0:3], f, cx, cy, k1, k2

    Returns
    -------
    px, py   : (N,) pixel coordinate arrays
    J        : (2N, 8)  [only when jacobian=True]
    """
    rvec = np.asarray(rvec, dtype=np.float64).ravel()
    N    = len(v_cel)

    if not jacobian:
        R         = rodrigues(rvec)
        v_cam     = (R @ v_cel.T).T
        in_front  = v_cam[:, 0] > 0.0
        safe      = np.where(in_front, v_cam[:, 0], 1.0)
        xn        = np.where(in_front, -v_cam[:, 1] / safe, 1e9)
        yn        = np.where(in_front, -v_cam[:, 2] / safe, 1e9)
        r2        = xn**2 + yn**2
        d         = 1.0 + k1*r2 + k2*r2**2
        return f*xn*d + cx, f*yn*d + cy

    # ── Jacobian path: analytical Rodrigues derivative ────────────────────────
    θ = float(np.linalg.norm(rvec))

    # eⱼ × v_cel for j = 0, 1, 2
    ejxv = np.stack([
        np.stack([ np.zeros(N), -v_cel[:, 2],  v_cel[:, 1]], axis=1),
        np.stack([ v_cel[:, 2],  np.zeros(N), -v_cel[:, 0]], axis=1),
        np.stack([-v_cel[:, 1],  v_cel[:, 0],  np.zeros(N)], axis=1),
    ], axis=0)  # (3, N, 3)

    if θ < 1e-9:                       # small-angle limit: R → I
        v_cam = v_cel.copy()
        Jv    = ejxv
    else:
        n = rvec / θ
        c, s = np.cos(θ), np.sin(θ)
        α = s / θ
        β = (1.0 - c) / (θ * θ)

        ωxv      = np.cross(rvec, v_cel)          # (N, 3)
        ω_dot_v  = v_cel @ rvec                   # (N,)
        v_cam    = c * v_cel + α * ωxv + β * np.outer(ω_dot_v, rvec)

        dα        = (c / θ - α / θ) * n           # d(α)/d(rvec[j])
        dβ        = (α / θ - 2.0 * β / θ) * n     # d(β)/d(rvec[j])
        ω_outer_v = np.outer(ω_dot_v, rvec)

        Jv = np.empty((3, N, 3))
        for j in range(3):
            ej = np.zeros(3); ej[j] = 1.0
            Jv[j] = (
                -s * n[j] * v_cel +
                dα[j] * ωxv +
                α * ejxv[j] +
                dβ[j] * ω_outer_v +
                β * (ω_dot_v[:, None] * ej + np.outer(v_cel[:, j], rvec))
            )

    v0 = v_cam[:, 0]
    xn = -v_cam[:, 1] / v0
    yn = -v_cam[:, 2] / v0
    r2 =  xn**2 + yn**2
    d  =  1.0 + k1*r2 + k2*r2**2
    λ  =  k1 + 2.0*k2*r2

    px = f*xn*d + cx
    py = f*yn*d + cy

    # Jacobian of (px, py) w.r.t. (xn, yn)
    dpx_dxn = f * (d + 2.0*xn**2 * λ)
    dpx_dyn = 2.0*f*xn*yn * λ            # == dpy_dxn  (symmetric)
    dpy_dyn = f * (d + 2.0*yn**2 * λ)

    J = np.empty((2 * N, 8))

    for j in range(3):
        dv  = Jv[j]
        dxn = (-xn * dv[:, 0] - dv[:, 1]) / v0
        dyn = (-yn * dv[:, 0] - dv[:, 2]) / v0
        J[:N, j] = dpx_dxn * dxn + dpx_dyn * dyn
        J[N:, j] = dpx_dyn * dxn + dpy_dyn * dyn

    J[:N, 3] = xn*d;        J[N:, 3] = yn*d
    J[:N, 4] = 1.0;         J[N:, 4] = 0.0
    J[:N, 5] = 0.0;         J[N:, 5] = 1.0
    J[:N, 6] = f*xn*r2;     J[N:, 6] = f*yn*r2
    J[:N, 7] = f*xn*r2**2;  J[N:, 7] = f*yn*r2**2

    return px, py, J
