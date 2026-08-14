"""
Drawing functions for star field overlays.

All functions modify the passed-in numpy image (RGB) in-place and return it.
"""
import io
import os
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from typing import Dict, List, Optional, Tuple

from plate import Plate
from catalog import (_LINES_RAW, _get_star_names,
                     _get_hip_coords, _get_hip_coords_all,
                     _get_hip_catalog, _hip_id_for_radec)


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


# ── constellation art ────────────────────────────────────────────────────────

_ART_CACHE = None


def _load_constellation_art():
    """Load the packed art bundle; defer PNG decoding to first use.

    Returns {abbr: {'png_bytes', 'anchors_img', 'anchors_hip', 'image'}}.
    'image' starts as None and is populated by _get_art_image on first access.

    constellation_art.npz is vendored data, derived once from the western sky
    culture of Stellarium (GPL): its constellationsart.fab table plus the 85
    figure PNGs, converted to 8-bit grayscale.  Layout, for anyone rebuilding it:

        abbr         (N,)      <U3     IAU abbreviation, e.g. 'Ori'
        anchors_img  (N, 3, 2) float32 3 anchor points, art-image pixel coords
        anchors_hip  (N, 3)    int32   HIP id of the star at each anchor
        png_blob     (M,)      uint8   all PNG files concatenated
        png_offsets  (N+1,)    int64   entry i is png_blob[o[i]:o[i+1]]

    The anchor triples are what tie an art image to the sky; everything else in
    the .fab (image dimensions) is recoverable from the PNGs themselves.
    """
    global _ART_CACHE
    if _ART_CACHE is not None:
        return _ART_CACHE
    npz_path = os.path.join(os.path.dirname(__file__), 'constellation_art.npz')
    with np.load(npz_path) as z:
        abbr, anc, hips = z['abbr'], z['anchors_img'], z['anchors_hip']
        blob, offs      = z['png_blob'], z['png_offsets']
        _ART_CACHE = {
            str(abbr[i]): {
                'png_bytes':   blob[offs[i]:offs[i + 1]].tobytes(),
                'anchors_img': anc[i],
                'anchors_hip': hips[i],
                'image':       None,   # decoded on first use
            }
            for i in range(len(abbr))
        }
    return _ART_CACHE


def _get_art_image(entry):
    """Decode PNG on first access; cache the uint8 grayscale array in entry."""
    if entry['image'] is None:
        entry['image'] = np.asarray(
            Image.open(io.BytesIO(entry['png_bytes'])), dtype=np.uint8)
    return entry['image']


def draw_constellation_art(img: np.ndarray, plate,
                           opacity: float = 0.8,
                           color: Tuple = (200, 170, 130),
                           mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Blend Stellarium western-sky-culture art onto img using the plate solution.

    Each output pixel is back-projected to a sky unit vector (undistortion is
    always valid for in-image pixels), then mapped to art-image coordinates via a
    gnomonic tangent-plane transform anchored by the three Hipparcos stars in the
    art bundle.  This avoids applying the distortion polynomial to off-FOV anchor
    stars, which extrapolates unreliably.

    To keep drawing fast the back-projection and blending are done at 1/4
    resolution; the resulting colour delta is bilinearly upsampled before being
    applied to the full-resolution image.  Per-constellation work is also
    skipped via a two-stage cull (coarse centroid distance, then art bbox).
    """
    art_db = _load_constellation_art()

    try:
        # Use full catalog lookup — art anchors are not all in constellation lines.
        hip_coords = _get_hip_coords_all()
    except FileNotFoundError:
        return img

    saved = img.copy() if mask is not None else None
    h, w  = img.shape[:2]
    SCALE = 4
    h_s, w_s = max(1, h // SCALE), max(1, w // SCALE)

    # ── pre-compute celestial unit vectors at ¼ resolution ───────────────────
    # Pixel centres of each SCALE×SCALE block in original-image coordinates.
    f, cx, cy, k1, k2 = plate.f, plate.cx, plate.cy, plate.k1, plate.k2
    px_g = (np.arange(w_s, dtype=np.float32) + 0.5) * SCALE - 0.5   # (w_s,)
    py_g = (np.arange(h_s, dtype=np.float32) + 0.5) * SCALE - 0.5   # (h_s,)
    xd = ((px_g[None, :] - cx) / f).astype(np.float32)   # (1, w_s) → broadcast
    yd = ((py_g[:, None] - cy) / f).astype(np.float32)   # (h_s, 1)
    xd = np.broadcast_to(xd, (h_s, w_s)).copy()
    yd = np.broadcast_to(yd, (h_s, w_s)).copy()
    xn, yn = xd.copy(), yd.copy()
    for _ in range(5):
        r2 = xn * xn + yn * yn
        s  = 1.0 + k1 * r2 + k2 * r2 * r2
        xn = xd / s
        yn = yd / s
    nm    = np.sqrt(1.0 + xn * xn + yn * yn)
    v_cam = np.stack([1.0 / nm, -xn / nm, -yn / nm]).astype(np.float32)  # (3,h_s,w_s)
    R_f   = plate.R.astype(np.float32)
    v_cel = np.einsum('ij,jhw->ihw', R_f.T, v_cam)       # (3, h_s, w_s)

    # Accumulate per-pixel "transparency" t = ∏(1 − α_i).  Since each constellation
    # blends the same constant `color` over the same base image, the final result
    # is order-independent: out = t · base + (1−t) · color.  This lets us do the
    # 3-channel blend just once at the end instead of per-constellation.
    transparency = np.ones((h_s, w_s), dtype=np.float32)

    half_diag_rad = np.radians(plate.fov_deg / 2.0 * np.sqrt(2.0))
    boresight     = plate.R[0].astype(np.float64)
    R64           = plate.R.astype(np.float64)

    for entry in art_db.values():
        anc_img = entry['anchors_img']   # (3, 2) in image-native coords
        anc_hip = entry['anchors_hip']   # (3,) HIP IDs

        # Collect sky unit vectors for the three anchor stars.
        v_list = []
        for k in range(3):
            hip_id = int(anc_hip[k])
            if hip_id not in hip_coords:
                break
            ra_d, dec_d = hip_coords[hip_id]
            ra_r, dec_r = np.radians(ra_d), np.radians(dec_d)
            v_list.append(np.array([
                np.cos(dec_r) * np.cos(ra_r),
                np.cos(dec_r) * np.sin(ra_r),
                np.sin(dec_r),
            ], dtype=np.float64))
        if len(v_list) < 3:
            continue

        v1, v2, v3 = v_list

        # Centroid c and tangent basis at c.
        c = v1 + v2 + v3;  c /= np.linalg.norm(c)
        cos_to_boresight = float(c @ boresight)

        # Coarse cull: if centroid lies more than (half_diag + 3·max_anchor_angle)
        # outside the FOV the constellation cannot intersect the image.
        max_anc_cos = min(float(v1 @ c), float(v2 @ c), float(v3 @ c))
        max_anc_rad = np.arccos(np.clip(max_anc_cos, -1.0, 1.0))
        if cos_to_boresight < np.cos(half_diag_rad + 3.0 * max_anc_rad + np.radians(5.0)):
            continue

        # Orthonormal tangent-plane basis at c.
        arb = np.array([0., 0., 1.]) if abs(c[2]) < 0.9 else np.array([1., 0., 0.])
        e1  = np.cross(c, arb);  e1 /= np.linalg.norm(e1)
        e2  = np.cross(c, e1)

        # Gnomonic project 3 anchor sky vectors → tangent-plane 2-D.
        t_pts = np.array([
            [float(vi @ e1) / float(vi @ c), float(vi @ e2) / float(vi @ c)]
            for vi in [v1, v2, v3]
        ])  # (3, 2)

        # Affine A (2×3): tangent-plane → art-pixel.
        t_h = np.column_stack([t_pts, np.ones(3)])
        try:
            A = np.linalg.solve(t_h, anc_img.astype(np.float64))    # (3, 2)
        except np.linalg.LinAlgError:
            continue

        # Decode PNG lazily — only constellations past the cull pay this cost.
        art_img = _get_art_image(entry)
        ih_a, iw_a = art_img.shape[:2]

        # ── per-constellation bounding box ───────────────────────────────────
        # Inverse-affine the 4 art image corners back to tangent plane, then
        # to unit vectors, then forward-project (ideal pinhole — no distortion)
        # to image pixels.  Slice v_cel to this bbox so per-pixel work scales
        # with the constellation's footprint rather than the whole image.
        # A has shape (3, 2): A.T (2,3) maps [t_x, t_y, 1] → [art_x, art_y].
        A_T = A.T
        try:
            A_2x2_inv = np.linalg.inv(A_T[:, :2])
        except np.linalg.LinAlgError:
            continue
        corners_pix = np.array(
            [[0, 0], [iw_a, 0], [iw_a, ih_a], [0, ih_a]], dtype=np.float64)
        t_corners = (corners_pix - A_T[:, 2]) @ A_2x2_inv.T      # (4, 2)
        norms_c   = np.sqrt(1.0 + (t_corners ** 2).sum(axis=1))  # (4,)
        v_corners = ((c[None, :]
                      + t_corners[:, 0:1] * e1[None, :]
                      + t_corners[:, 1:2] * e2[None, :])
                     / norms_c[:, None])                         # (4, 3)
        v_cam_c   = v_corners @ R64.T                            # (4, 3)
        xc        = v_cam_c[:, 0]
        if (xc <= 0.05).any():
            # at least one corner behind/at camera — fall back to full grid
            y0, y1, x0, x1 = 0, h_s, 0, w_s
        else:
            px = -v_cam_c[:, 1] / xc * plate.f + plate.cx
            py = -v_cam_c[:, 2] / xc * plate.f + plate.cy
            # ¼-resolution pixel indices, with 1-block padding for safety
            x0 = max(0,  int(np.floor(px.min() / SCALE)) - 1)
            x1 = min(w_s, int(np.ceil (px.max() / SCALE)) + 1)
            y0 = max(0,  int(np.floor(py.min() / SCALE)) - 1)
            y1 = min(h_s, int(np.ceil (py.max() / SCALE)) + 1)
            if x0 >= x1 or y0 >= y1:
                continue

        h_b, w_b = y1 - y0, x1 - x0
        v_sub = v_cel[:, y0:y1, x0:x1].reshape(3, -1)            # (3, h_b*w_b)
        M = np.stack([c, e1, e2]).astype(np.float32)             # (3, 3)
        dots = (M @ v_sub).reshape(3, h_b, w_b)
        dot_c, dot_e1, dot_e2 = dots[0], dots[1], dots[2]

        front  = dot_c > 0.01
        safe_c = np.where(front, dot_c, 1.0)
        t_x = dot_e1 / safe_c
        t_y = dot_e2 / safe_c

        A_f = A.astype(np.float32)
        ax = A_f[0, 0]*t_x + A_f[1, 0]*t_y + A_f[2, 0]
        ay = A_f[0, 1]*t_x + A_f[1, 1]*t_y + A_f[2, 1]

        in_art = (front
                  & (ax >= 0.0) & (ax < iw_a - 0.5)
                  & (ay >= 0.0) & (ay < ih_a - 0.5))
        if not bool(in_art.any()):
            continue
        ax_i   = np.clip(np.round(ax).astype(np.int32), 0, iw_a - 1)
        ay_i   = np.clip(np.round(ay).astype(np.int32), 0, ih_a - 1)

        art_vals = art_img[ay_i, ax_i].astype(np.float32)
        alpha    = np.where(in_art, art_vals * (opacity / 255.0), 0.0)

        transparency[y0:y1, x0:x1] *= (1.0 - alpha)

    # ── one combined 3-channel blend, then upsample the colour delta ──────────
    img_small = np.array(
        Image.fromarray(img).resize((w_s, h_s), Image.BOX), dtype=np.float32)
    tint     = np.array(color, dtype=np.float32)
    t_3      = transparency[..., np.newaxis]
    img_s_f  = t_3 * img_small + (1.0 - t_3) * tint            # (h_s, w_s, 3)

    delta_small = img_s_f - img_small                          # (h_s, w_s, 3)
    delta_full  = np.array(
        Image.fromarray(
            np.clip(delta_small + 128.0, 0.0, 255.0).astype(np.uint8)
        ).resize((w, h), Image.BILINEAR),
        dtype=np.float32,
    ) - 128.0                                                  # (h, w, 3)
    img[:] = np.clip(img.astype(np.float32) + delta_full, 0, 255).astype(np.uint8)
    if mask is not None:
        m = mask > 128
        img[m] = saved[m]
    return img


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
