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
    import os
    from PIL import ImageFont
    candidates = [
        os.path.join(os.path.dirname(__file__), 'DejaVuSans.ttf'),  # bundled
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
                        star_radius: int = 25,
                        mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Draw constellation lines onto img (RGB) using a plate solution."""
    h, w = img.shape[:2]
    saved = img.copy() if mask is not None else None
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
    if mask is not None:
        m = mask > 128
        img[m] = saved[m]
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
    v_cel = np.column_stack([np.cos(dec_rad) * np.cos(ra_rad),
                             np.cos(dec_rad) * np.sin(ra_rad),
                             np.sin(dec_rad)])
    px, py, in_front = plate.project_with_mask(v_cel)
    mag_f = mag

    margin = 50
    visible = (in_front &
               (px >= -margin) & (px < w + margin) &
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
                    color: Tuple = (255, 180, 0),
                    mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Draw Bayer designations (with real Greek letters) next to matched star circles."""
    if len(matched_stars) == 0:
        return img

    saved = img.copy() if mask is not None else None
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
    if mask is not None:
        m = mask > 128
        img[m] = saved[m]
    return img


# ── detection overlay ────────────────────────────────────────────────────────

def draw_detections(img: np.ndarray, stars: List[Dict],
                    color: tuple = (0, 255, 0), thickness: int = 5,
                    star_radius: int = 25,
                    max_highlight: int = 50, max_draw: int = 2000) -> np.ndarray:
    """Draw circles around detected stars on img (in-place)."""
    h, w = img.shape[:2]
    subset = stars[:max_highlight]

    items = [(s['x'], s['y'], 1.0) for s in subset]
    _draw_circles_with_alpha(img, items, color, star_radius, thickness)

    if len(stars) < max_highlight:
        
        return img
    
    else:

        subset = stars[max_highlight:max_draw]
        items = [(s['x'], s['y'], 0.25) for s in subset]
        _draw_circles_with_alpha(img, items, color, star_radius, thickness)

        return img


# ── pipeline drawing helpers ──────────────────────────────────────────────────

def _mag_alpha(mag: float, mag_bright: float = 0.5, mag_faint: float = 7.0,
               alpha_bright: float = 1.0, alpha_faint: float = 0.15) -> float:
    """Map catalog magnitude to a [alpha_faint, alpha_bright] opacity factor."""
    t = max(0.0, min(1.0, (mag - mag_bright) / (mag_faint - mag_bright)))
    return alpha_bright - t * (alpha_bright - alpha_faint)


def _draw_circles_with_alpha(img: np.ndarray, items, color, radius, thickness,
                             mask: Optional[np.ndarray] = None):
    """Draw circles with per-circle opacity (alpha blend).

    items: iterable of (cx, cy, alpha) tuples.
    Groups by rounded alpha to minimise addWeighted calls.
    """
    from collections import defaultdict
    saved = img.copy() if mask is not None else None
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

    if mask is not None:
        m = mask > 128
        img[m] = saved[m]


def _draw_refine_labels(img: np.ndarray, matched_stars: list,
                        star_radius: int = 25,
                        color: Tuple = (255, 180, 0),
                        mask: Optional[np.ndarray] = None) -> None:
    """Draw Bayer designations next to matched stars."""
    named = [s for s in matched_stars if s['name']]
    if not named:
        return

    saved = img.copy() if mask is not None else None
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
        r, g, b = color[0], color[1], color[2]
        draw.text((tx, ty), name, font=font, fill=(r, g, b, int(255 * a)))

    img[:] = np.array(pil_img)
    if mask is not None:
        m = mask > 128
        img[m] = saved[m]


def _draw_unknown_labels(img: np.ndarray, unknowns: list,
                         plate: Plate, phot_b: float = 0.0,
                         star_radius: int = 25,
                         color: Tuple = (200, 50, 50),
                         mask: Optional[np.ndarray] = None) -> None:
    """Label unknown detections with nearest catalog distance and mag diff."""
    if not unknowns:
        return

    saved = img.copy() if mask is not None else None
    h, w = img.shape[:2]

    ra_rad, dec_rad, mag_cat, _ = _get_hip_catalog()
    v_cel = np.column_stack([np.cos(dec_rad) * np.cos(ra_rad),
                             np.cos(dec_rad) * np.sin(ra_rad),
                             np.sin(dec_rad)])
    cat_px, cat_py, in_front = plate.project_with_mask(v_cel)
    margin = 100
    vis = (in_front &
           (cat_px >= -margin) & (cat_px < w + margin) &
           (cat_py >= -margin) & (cat_py < h + margin))
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

        draw.text((tx, ty), line1, font=font, fill=color)
        if line2:
            ty2 = ty + (bbox1[3] - bbox1[1]) + 2
            draw.text((tx, ty2), line2, font=font, fill=color)

    img[:] = np.array(pil_img)
    if mask is not None:
        m = mask > 128
        img[m] = saved[m]
