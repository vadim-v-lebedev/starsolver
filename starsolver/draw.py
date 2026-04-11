"""
Drawing functions for star field overlays.

All functions modify the passed-in numpy image (RGB) in-place and return it.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from typing import Dict, List, Optional, Tuple

from plate import Plate
from catalog import (_LINES_RAW, _STAR_NAMES,
                     _get_hip_coords, _get_hip_catalog, _hip_id_for_radec)


# ── image loading ────────────────────────────────────────────────────────────

def load_image(path: str, mode: str = 'RGB') -> np.ndarray:
    """Load image from path, undoing EXIF rotation, converted to mode."""
    return np.array(ImageOps.exif_transpose(Image.open(path)).convert(mode))


# ── font helper ───────────────────────────────────────────────────────────────

def _get_label_font(size: int = 32):
    """Return a PIL ImageFont with Greek support, falling back to default."""
    from PIL import ImageFont
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/system/fonts/NotoSans-Regular.ttf",   # Android
        "/system/fonts/DroidSans.ttf",           # older Android
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            pass
    return ImageFont.load_default()


# ── constellation lines ───────────────────────────────────────────────────────

def draw_constellations(img: np.ndarray, plate: Plate,
                        catalog_path: Optional[str] = None,
                        color: Tuple = (255, 180, 0),
                        thickness: int = 4,
                        star_radius: int = 25) -> np.ndarray:
    """Draw constellation lines onto img (RGB) using a plate solution."""
    h, w = img.shape[:2]
    try:
        hip_coords = _get_hip_coords(catalog_path)
    except FileNotFoundError as e:
        print(f"draw_constellations: {e}")
        return img

    pil  = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)

    for row in _LINES_RAW:
        hip_list = row[2:]
        for i in range(0, len(hip_list) - 1, 2):
            hip_a, hip_b = hip_list[i], hip_list[i + 1]
            if hip_a not in hip_coords or hip_b not in hip_coords:
                continue
            ra_a, dec_a = hip_coords[hip_a]
            ra_b, dec_b = hip_coords[hip_b]
            pa = plate.radec_to_pixel(ra_a, dec_a)
            pb = plate.radec_to_pixel(ra_b, dec_b)
            if pa is None or pb is None:
                continue
            def inside(p):
                return 0 <= p[0] < w and 0 <= p[1] < h
            if inside(pa) or inside(pb):
                ax, ay = float(pa[0]), float(pa[1])
                bx, by = float(pb[0]), float(pb[1])
                dx, dy = bx - ax, by - ay
                length = (dx*dx + dy*dy) ** 0.5
                if length > 2 * star_radius:
                    ux, uy = dx / length, dy / length
                    pa = (int(round(ax + ux * star_radius)),
                          int(round(ay + uy * star_radius)))
                    pb = (int(round(bx - ux * star_radius)),
                          int(round(by - uy * star_radius)))
                draw.line([pa, pb], fill=color, width=thickness)

    img[:] = np.array(pil)
    return img


# ── catalog star overlay (dev tool) ──────────────────────────────────────────

def draw_catalog_stars(img: np.ndarray, plate: Plate,
                       catalog_path: Optional[str] = None,
                       mag_limit: float = 7.0,
                       opacity: float = 0.25,
                       color: Tuple = (255, 200, 0),
                       thickness: int = 2) -> np.ndarray:
    """Overlay Hipparcos catalog stars (mag <= mag_limit) on img."""
    h, w = img.shape[:2]
    ra_rad, dec_rad, mag, _ = _get_hip_catalog(catalog_path)
    R  = plate.R
    f  = plate.f;  cx = plate.cx;  cy = plate.cy
    k1 = plate.k1; k2 = plate.k2

    v_cel = np.column_stack([np.cos(dec_rad) * np.cos(ra_rad),
                             np.cos(dec_rad) * np.sin(ra_rad),
                             np.sin(dec_rad)])
    v_cam = (R @ v_cel.T).T
    in_front = v_cam[:, 0] > 0
    v_cam = v_cam[in_front]
    mag_f = mag[in_front]

    xn = -v_cam[:, 1] / v_cam[:, 0]
    yn = -v_cam[:, 2] / v_cam[:, 0]
    r2 = xn**2 + yn**2
    d  = 1.0 + k1 * r2 + k2 * r2 ** 2
    px = f * xn * d + cx
    py = f * yn * d + cy

    margin = 50
    visible = ((px >= -margin) & (px < w + margin) &
               (py >= -margin) & (py < h + margin) &
               (mag_f <= mag_limit))

    layer = img.copy()
    pil_layer = Image.fromarray(layer)
    draw = ImageDraw.Draw(pil_layer)
    r = 25
    for x, y in zip(px[visible], py[visible]):
        ix, iy = int(round(x)), int(round(y))
        draw.ellipse([ix - r, iy - r, ix + r, iy + r], outline=color, width=thickness)
    layer = np.array(pil_layer)
    img[:] = np.clip(opacity * layer.astype(np.float32) +
                     (1 - opacity) * img.astype(np.float32), 0, 255).astype(np.uint8)
    return img


# ── star name labels ──────────────────────────────────────────────────────────

def draw_star_names(img: np.ndarray, matched_stars, matched_centroids,
                    star_radius: int = 25,
                    color: Tuple = (255, 180, 0)) -> np.ndarray:
    """Draw Bayer designations (with real Greek letters) next to matched star circles."""
    if len(matched_stars) == 0:
        return img

    matched_stars     = np.asarray(matched_stars,     dtype=np.float64)
    matched_centroids = np.asarray(matched_centroids, dtype=np.float64)
    h, w = img.shape[:2]

    font    = _get_label_font(60)
    offset  = star_radius + 6
    pil_img = Image.fromarray(img)
    draw    = ImageDraw.Draw(pil_img)

    for i in range(len(matched_stars)):
        ra_deg, dec_deg = float(matched_stars[i, 0]), float(matched_stars[i, 1])
        hip_id = _hip_id_for_radec(ra_deg, dec_deg)
        name   = _STAR_NAMES.get(hip_id)
        if name is None:
            continue
        cy_px = int(round(float(matched_centroids[i, 0])))
        cx_px = int(round(float(matched_centroids[i, 1])))
        tx, ty = cx_px + offset, cy_px - 10
        bbox = draw.textbbox((tx, ty), name, font=font)
        tw = bbox[2] - bbox[0]
        if tx + tw > w:
            tx = cx_px - offset - tw
        if ty < 0:
            ty = cy_px + offset
        draw.text((tx, ty), name, font=font, fill=color)

    img[:] = np.array(pil_img)
    return img


# ── detection overlay ────────────────────────────────────────────────────────

def draw_detections(img: np.ndarray, stars: List[Dict],
                    color: tuple = (0, 255, 0), thickness: int = 5,
                    max_highlight: int = 50, max_draw: int = 2000) -> np.ndarray:
    """Draw circles around detected stars on img (in-place)."""
    h, w = img.shape[:2]
    subset = stars[:max_highlight]

    items = [(s['x'], s['y'], 1.0) for s in subset]
    radius = 25
    _draw_circles_with_alpha(img, items, color, radius, thickness)

    if len(stars) < max_highlight:
        
        return img
    
    else:

        subset = stars[max_highlight:max_draw]
        items = [(s['x'], s['y'], 0.25) for s in subset]
        _draw_circles_with_alpha(img, items, color, radius, thickness)

        return img


# ── pipeline drawing helpers ──────────────────────────────────────────────────

def _mag_alpha(mag: float, mag_bright: float = 0.5, mag_faint: float = 7.0,
               alpha_bright: float = 1.0, alpha_faint: float = 0.15) -> float:
    """Map catalog magnitude to a [alpha_faint, alpha_bright] opacity factor."""
    t = max(0.0, min(1.0, (mag - mag_bright) / (mag_faint - mag_bright)))
    return alpha_bright - t * (alpha_bright - alpha_faint)


def _draw_circles_with_alpha(img: np.ndarray, items, color, radius, thickness):
    """Draw circles with per-circle opacity (alpha blend).

    items: iterable of (cx, cy, alpha) tuples.
    Groups by rounded alpha to minimise addWeighted calls.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for cx, cy, a in items:
        groups[round(a, 1)].append((cx, cy))

    for alpha, pts in sorted(groups.items()):
        overlay = img.copy()
        pil_ov  = Image.fromarray(overlay)
        draw    = ImageDraw.Draw(pil_ov)
        for cx, cy in pts:
            draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                         outline=color, width=thickness)
        overlay = np.array(pil_ov)
        img[:] = np.clip(alpha * overlay.astype(np.float32) +
                         (1 - alpha) * img.astype(np.float32), 0, 255).astype(np.uint8)


def _draw_refine_labels(img: np.ndarray, matched_stars: list,
                        star_radius: int = 25) -> None:
    """Draw Bayer designations (gold, PIL for Greek support) next to matched stars."""
    named = [s for s in matched_stars if s['name']]
    if not named:
        return

    h, w = img.shape[:2]
    font   = _get_label_font(60)
    offset = star_radius + 6
    pil_img = Image.fromarray(img)
    draw    = ImageDraw.Draw(pil_img)

    for star in named:
        name   = star['name']
        px, py = int(round(star['x'])), int(round(star['y']))
        tx, ty = px + offset, py - 10
        bbox = draw.textbbox((tx, ty), name, font=font)
        tw = bbox[2] - bbox[0]
        if tx + tw > w:
            tx = px - offset - tw
        if ty < 0:
            ty = py + offset
        a = _mag_alpha(star.get('mag', 3.0))
        draw.text((tx, ty), name, font=font, fill=(255, 180, 0, int(255 * a)))

    img[:] = np.array(pil_img)


def _draw_unknown_labels(img: np.ndarray, unknowns: list,
                         plate: Plate, phot_b: float = 0.0,
                         star_radius: int = 25) -> None:
    """Label unknown detections with nearest catalog distance and mag diff."""
    if not unknowns:
        return

    h, w = img.shape[:2]

    ra_rad, dec_rad, mag_cat, _ = _get_hip_catalog()
    f  = plate.f;  cx = plate.cx;  cy = plate.cy
    k1 = plate.k1; k2 = plate.k2

    R = plate.R
    v_cel = np.column_stack([np.cos(dec_rad) * np.cos(ra_rad),
                             np.cos(dec_rad) * np.sin(ra_rad),
                             np.sin(dec_rad)])
    v_cam = (R @ v_cel.T).T
    in_front = v_cam[:, 0] > 0
    safe_d = np.where(in_front, v_cam[:, 0], 1.0)
    xn = np.where(in_front, -v_cam[:, 1] / safe_d, 0.0)
    yn = np.where(in_front, -v_cam[:, 2] / safe_d, 0.0)
    r2 = xn ** 2 + yn ** 2
    d  = 1.0 + k1 * r2 + k2 * r2 ** 2
    cat_px = f * xn * d + cx
    cat_py = f * yn * d + cy
    margin = 100
    vis = in_front & (cat_px >= -margin) & (cat_px < w + margin) & \
                     (cat_py >= -margin) & (cat_py < h + margin)
    vis_px  = cat_px[vis]
    vis_py  = cat_py[vis]
    vis_mag = mag_cat[vis]

    PHOT_SLOPE = -0.29
    font    = _get_label_font(40)
    offset  = star_radius + 6
    pil_img = Image.fromarray(img)
    draw    = ImageDraw.Draw(pil_img)

    for u in unknowns:
        ux, uy = u['x'], u['y']
        dists      = np.sqrt((vis_px - ux) ** 2 + (vis_py - uy) ** 2)
        nearest_i  = int(np.argmin(dists))
        nearest_dist = float(dists[nearest_i])
        nearest_mag  = float(vis_mag[nearest_i])

        bright = u.get('brightness', 0)
        if bright > 0:
            pred_mag = (np.log10(bright) - phot_b) / PHOT_SLOPE
            line2 = f"{pred_mag - nearest_mag:+.1f}m"
        else:
            line2 = ""

        line1 = f"{nearest_dist:.1f}px"
        px, py = int(round(ux)), int(round(uy))
        tx, ty = px + offset, py - 10
        bbox1 = draw.textbbox((tx, ty), line1, font=font)
        tw = bbox1[2] - bbox1[0]
        if tx + tw > w:
            tx = px - offset - tw
        if ty < 0:
            ty = py + offset

        draw.text((tx, ty), line1, font=font, fill=(200, 50, 50))
        if line2:
            ty2 = ty + (bbox1[3] - bbox1[1]) + 2
            draw.text((tx, ty2), line2, font=font, fill=(200, 50, 50))

    img[:] = np.array(pil_img)
