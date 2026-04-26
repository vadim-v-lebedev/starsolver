"""
Drawing functions for star field overlays.

All functions modify the passed-in numpy image (RGB) in-place and return it.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from typing import Dict, List, Optional, Tuple

from plate import Plate
from catalog import (_LINES_RAW, _get_star_names,
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
                        color: Tuple = (255, 180, 0),
                        thickness: int = 4,
                        star_radius: int = 25,
                        mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Draw constellation lines onto img (RGB) using a plate solution."""
    h, w = img.shape[:2]
    saved = img.copy() if mask is not None else None
    try:
        hip_coords = _get_hip_coords()
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
                       mag_limit: float = 7.0,
                       opacity: float = 0.25,
                       color: Tuple = (255, 200, 0),
                       thickness: int = 2) -> np.ndarray:
    """Overlay Hipparcos catalog stars (mag <= mag_limit) on img."""
    h, w = img.shape[:2]
    ra_rad, dec_rad, mag, _, _ = _get_hip_catalog()
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
                    mask: Optional[np.ndarray] = None,
                    font_size: int = 48) -> np.ndarray:
    """Draw Bayer designations (with real Greek letters) next to matched star circles."""
    if len(matched_stars) == 0:
        return img

    saved = img.copy() if mask is not None else None
    matched_stars     = np.asarray(matched_stars,     dtype=np.float64)
    matched_centroids = np.asarray(matched_centroids, dtype=np.float64)
    h, w = img.shape[:2]

    font    = _get_label_font(font_size)
    offset  = star_radius + 6
    pil_img = Image.fromarray(img)
    draw    = ImageDraw.Draw(pil_img)

    for i in range(len(matched_stars)):
        ra_deg, dec_deg = float(matched_stars[i, 0]), float(matched_stars[i, 1])
        hip_id = _hip_id_for_radec(ra_deg, dec_deg)
        name   = _get_star_names().get(hip_id)
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
                        mask: Optional[np.ndarray] = None,
                        font_size: int = 48) -> None:
    """Draw Bayer designations next to matched stars."""
    draw_names = _get_star_names()
    named = [(s, draw_names[s['hip_id']]) for s in matched_stars if s['hip_id'] in draw_names]
    if not named:
        return

    saved = img.copy() if mask is not None else None
    h, w = img.shape[:2]
    font   = _get_label_font(font_size)
    offset = star_radius + 6
    pil_img = Image.fromarray(img)
    draw    = ImageDraw.Draw(pil_img)

    for star, name in named:
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


def _draw_special_labels(img: np.ndarray, specials: list,
                         star_radius: int = 25,
                         color: Tuple = (0, 100, 255),
                         mask: Optional[np.ndarray] = None,
                         font_size: int = 48) -> None:
    """Draw planet name labels next to matched special objects."""
    if not specials:
        return

    saved = img.copy() if mask is not None else None
    h, w  = img.shape[:2]
    font   = _get_label_font(font_size)
    offset = star_radius + 6
    pil_img = Image.fromarray(img)
    draw    = ImageDraw.Draw(pil_img)

    for obj in specials:
        name   = obj['name']
        px, py = int(round(obj['x'])), int(round(obj['y']))
        tx, ty = px + offset, py - 10
        bbox = draw.textbbox((tx, ty), name, font=font)
        tw = bbox[2] - bbox[0]
        if tx + tw > w:
            tx = px - offset - tw
        if ty < 0:
            ty = py + offset
        draw.text((tx, ty), name, font=font, fill=color)

    img[:] = np.array(pil_img)
    if mask is not None:
        m = mask > 128
        img[m] = saved[m]




# ── timestamp overlay ─────────────────────────────────────────────────────────

def draw_timestamp(img: np.ndarray, timestamp: str,
                   color: Tuple = (255, 255, 255),
                   font_size: int = 48) -> np.ndarray:
    """Draw an ISO 8601 timestamp in the bottom-left corner of img."""
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(timestamp)
        if dt.tzinfo is not None:
            label = dt.strftime('%Y-%m-%d  %H:%M:%S  ') + dt.strftime('%z')
            label = label[:-2] + ':' + label[-2:]  # +0300 → +03:00
        else:
            label = dt.strftime('%Y-%m-%d  %H:%M:%S')
    except (ValueError, TypeError):
        label = timestamp

    h, w = img.shape[:2]
    font    = _get_label_font(font_size)
    pil_img = Image.fromarray(img)
    draw    = ImageDraw.Draw(pil_img)

    margin = 20
    bbox   = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = margin
    ty = h - th - margin

    pad     = 8
    bg_box  = [tx - pad, ty - pad, tx + tw + pad, ty + th + pad]
    pil_bg  = Image.fromarray(img.copy())
    ImageDraw.Draw(pil_bg).rectangle(bg_box, fill=(0, 0, 0))
    bg      = np.array(pil_bg).astype(np.float32)
    img[:]  = np.clip(0.55 * bg + 0.45 * img.astype(np.float32), 0, 255).astype(np.uint8)

    pil_img = Image.fromarray(img)
    ImageDraw.Draw(pil_img).text((tx, ty), label, font=font, fill=color)
    img[:] = np.array(pil_img)
    return img
