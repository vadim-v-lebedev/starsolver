import numpy as np
from typing import List, Dict

from minicv import gaussian_blur, dilate, resize_area, sep_filter2d


def sky_mask(img: np.ndarray) -> np.ndarray:
    """Placeholder — returns a mask of all ones (full image)."""
    h, w = img.shape[:2]
    return np.ones((h, w), dtype=np.uint8) * 255


def find_stars(gray: np.ndarray, mask: np.ndarray = None,
               ratio_threshold: float = 20.0,
               downsample: int = 4,
               r1: int = 5, r2: int = 15) -> List[Dict]:
    """
    Detect stars using inner/outer variance ratio on a downsampled image.

    A star is a bright compact blob (high inner variance) surrounded by
    uniform dark sky (low annular variance). The ratio var_inner/var_ring
    is high for stars and low for textured foreground or flat background.

    ratio_threshold: minimum var1/var2 to be considered a star candidate
    downsample:      factor to reduce image before processing (speeds up ~16x)
    r1:              inner disk kernel size in pixels (on downsampled image)
    r2:              outer disk kernel size in pixels (on downsampled image)
    """
    small = resize_area(gray, downsample)

    x1 = np.arange(r1) - r1 // 2
    g1 = np.exp(-x1**2 / (2 * (r1 / 2)**2)).astype(np.float32)  # peak-normalised
    x2 = np.arange(r2) - r2 // 2
    g2 = np.exp(-x2**2 / (2 * (r2 / 2)**2)).astype(np.float32)

    sq = small ** 2
    f1 = sep_filter2d(small, g1)
    f2 = sep_filter2d(small, g2)
    p1 = sep_filter2d(sq,    g1)
    p2 = sep_filter2d(sq,    g2)

    n1 = float(g1.sum()) ** 2   # effective 2D kernel sum = (1D sum)²
    n2 = float(g2.sum()) ** 2
    m  = n2 - n1

    var1 = p1 / n1 - (f1 / n1) ** 2
    var2 = np.maximum((p2 - p1) / m - ((f2 - f1) / m) ** 2, 0)

    ratio = var1 / (var2 + 1e-5)

    candidates = ratio > ratio_threshold
    # Gaussian blur before local-max detection so that saturated plateaus
    # (pixel value clipped at 255 over a large area) get a single peak
    # instead of many spurious maxima.
    small_blurred = gaussian_blur(small, 5, 1.0)
    local_max = (small_blurred >= dilate(small_blurred, 5) - 1e-6) & candidates

    ys, xs = np.nonzero(local_max)

    # Background-subtracted aperture flux from existing filter responses.
    # f1[y,x] = sum of pixel values in the inner disk (on downsampled image).
    # Annular background mean = (f2 - f1) / m, so:
    #   flux = f1 - n1 * annular_mean = f1 - n1 * (f2 - f1) / m
    flux_map = f1 - n1 * (f2 - f1) / m

    stars = []
    for xi, yi in zip(xs, ys):
        flux = float(flux_map[yi, xi])
        if flux <= 0:
            continue
        stars.append({
            "x": round(float(xi * downsample), 1),
            "y": round(float(yi * downsample), 1),
            "radius": float(r1 * downsample),
            "brightness": round(flux, 4),
        })

    stars.sort(key=lambda s: s["brightness"], reverse=True)

    return stars


def find_stars_multiscale(gray: np.ndarray, mask: np.ndarray = None,
                          ratio_threshold: float = 20.0,
                          downsample_factors: tuple = (4, 8, 16),
                          r1: int = 5, r2: int = 15) -> List[Dict]:
    """Run find_stars at multiple downsample factors and merge cross-scale duplicates.

    Coarser scales catch large/saturated/defocused stars that the fine scale misses.
    Duplicates (same star detected at several scales) are removed greedily: the
    brightest detection wins and any other detection whose centre falls within its
    radius is discarded.
    """
    all_stars: List[Dict] = []
    for ds in downsample_factors:
        all_stars.extend(find_stars(gray, mask, ratio_threshold, ds, r1, r2))

    # Sort brightest-first so the best detection of each star wins
    all_stars.sort(key=lambda s: s['brightness'], reverse=True)

    merged: List[Dict] = []
    used = bytearray(len(all_stars))
    for i, s in enumerate(all_stars):
        if used[i]:
            continue
        merged.append(s)
        r = s['radius']
        sx, sy = s['x'], s['y']
        for j in range(i + 1, len(all_stars)):
            if used[j]:
                continue
            t = all_stars[j]
            # merge radius = larger of the two stars' radii
            merge_r = max(r, t['radius'])
            dx = sx - t['x']
            dy = sy - t['y']
            if dx * dx + dy * dy < merge_r * merge_r:
                used[j] = True

    return merged
