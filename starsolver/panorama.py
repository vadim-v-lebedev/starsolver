"""
Equirectangular panorama builder.

Accumulates multiple plate-solved images onto a shared canvas.
Processed in horizontal strips to keep peak memory low.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from plate import Plate
from draw import load_image


def _sample_bilinear(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Bilinear sample of img (H, W, 3) at float coords xs, ys (flat arrays)."""
    h, w = img.shape[:2]
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1, y1 = x0 + 1, y0 + 1
    fx = (xs - x0)[:, np.newaxis].astype(np.float32)
    fy = (ys - y0)[:, np.newaxis].astype(np.float32)
    x0 = np.clip(x0, 0, w - 1)
    x1 = np.clip(x1, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    y1 = np.clip(y1, 0, h - 1)
    return (img[y0, x0].astype(np.float32) * (1 - fy) * (1 - fx) +
            img[y0, x1].astype(np.float32) * (1 - fy) *      fx  +
            img[y1, x0].astype(np.float32) *      fy  * (1 - fx) +
            img[y1, x1].astype(np.float32) *      fy  *      fx).astype(np.uint8)


def _equirect_coords(ra_deg: float, dec_deg: float, W: int, H: int):
    """Convert (RA °, Dec °) to equirectangular pixel coordinates."""
    x = (ra_deg / 360.0) * W
    y = (0.5 - dec_deg / 180.0) * H
    return x, y


def _draw_equirect_line(draw, x1, y1, x2, y2, W, color, width):
    """Draw a line handling the RA=0/360° wraparound seam."""
    dx = x2 - x1
    if abs(dx) > W / 2:
        # Shortest path crosses the seam — split into two segments
        if dx > 0:
            # x1 near left edge (low RA), x2 near right edge (high RA)
            # path: x1 → left seam (x=0) and right seam (x=W) → x2
            frac = x1 / (x1 + (W - x2))
            y_mid = y1 + frac * (y2 - y1)
            draw.line([(x1, y1), (0, y_mid)],  fill=color, width=width)
            draw.line([(W, y_mid), (x2, y2)],  fill=color, width=width)
        else:
            # x1 near right edge, x2 near left edge
            frac = (W - x1) / ((W - x1) + x2)
            y_mid = y1 + frac * (y2 - y1)
            draw.line([(x1, y1), (W, y_mid)],  fill=color, width=width)
            draw.line([(0, y_mid), (x2, y2)],  fill=color, width=width)
    else:
        draw.line([(x1, y1), (x2, y2)], fill=color, width=width)


def _overlay_composite(canvas: np.ndarray, overlay: Image.Image,
                        blur_radius: float = 1.5) -> None:
    """Blur overlay to soften edges, then alpha-composite onto canvas (in-place).

    Drawing hard lines directly into JPEG causes DCT ringing around each edge.
    Blurring the transparent overlay before compositing gives smooth gradients
    that JPEG can encode cleanly.
    """
    blurred = overlay.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    base    = Image.fromarray(canvas).convert('RGBA')
    merged  = Image.alpha_composite(base, blurred)
    canvas[:] = np.array(merged.convert('RGB'))


def _draw_grid(canvas: np.ndarray,
               color=(30, 50, 60), thickness: int = 2) -> None:
    """Draw RA/Dec coordinate grid onto an equirectangular canvas (in-place)."""
    H, W   = canvas.shape[:2]
    overlay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)
    c = color + (255,)

    # Dec parallels every 30° (-60, -30, 0, 30, 60)
    for dec in range(-60, 61, 30):
        _, y = _equirect_coords(0, dec, W, H)
        draw.line([(0, y), (W - 1, y)], fill=c, width=thickness)

    # RA meridians every 30°
    for ra in range(0, 360, 30):
        x, _ = _equirect_coords(ra, 0, W, H)
        draw.line([(x, 0), (x, H - 1)], fill=c, width=thickness)

    _overlay_composite(canvas, overlay)


def _draw_constellations(canvas: np.ndarray,
                         filter_names=None,
                         color=(170, 120, 0), thickness: int = 3) -> None:
    """Draw constellation lines onto an equirectangular canvas (in-place).

    filter_names: set of full constellation names to draw, or None for all.
    """
    from catalog import _LINES_RAW, _get_hip_coords, CONSTELLATIONS

    hip_coords = _get_hip_coords()
    H, W    = canvas.shape[:2]
    overlay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)
    c = color + (255,)

    for row in _LINES_RAW:
        abbr = row[0]
        name = CONSTELLATIONS.get(abbr, abbr)
        if filter_names is not None and name not in filter_names:
            continue

        hip_list = row[2:]
        for i in range(0, len(hip_list) - 1, 2):
            hip_a, hip_b = hip_list[i], hip_list[i + 1]
            if hip_a not in hip_coords or hip_b not in hip_coords:
                continue
            ra_a, dec_a = hip_coords[hip_a]
            ra_b, dec_b = hip_coords[hip_b]
            x1, y1 = _equirect_coords(ra_a, dec_a, W, H)
            x2, y2 = _equirect_coords(ra_b, dec_b, W, H)
            _draw_equirect_line(draw, x1, y1, x2, y2, W, c, thickness)

    _overlay_composite(canvas, overlay)


class Panorama:
    def __init__(self, width: int = 4096, blend: str = 'overwrite'):
        self.W      = width
        self.H      = width // 2
        self.blend  = blend
        self.canvas = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        self.count  = 0   # images added so far
        if blend == 'average':
            self._accum   = np.zeros((self.H, self.W, 3), dtype=np.float32)
            self._weights = np.zeros((self.H, self.W),    dtype=np.float32)

    def add_image(self, image_path: str, plate: Plate,
                  strip_height: int = 256) -> int:
        """
        Remap image onto the equirectangular canvas (overwriting existing pixels).
        Processed in strips of strip_height rows to limit peak memory.
        Returns number of pixels written.
        """
        img = load_image(image_path)
        ih, iw = img.shape[:2]
        total = 0

        col = np.arange(self.W, dtype=np.float32)

        for y0 in range(0, self.H, strip_height):
            y1  = min(y0 + strip_height, self.H)
            row = np.arange(y0, y1, dtype=np.float32)

            cg, rg = np.meshgrid(col, row)          # (strip, W) each

            ra_rad  = (cg / self.W).ravel() * (2 * np.pi)
            dec_rad = (0.5 - rg / self.H).ravel() * np.pi

            cd = np.cos(dec_rad)
            sd = np.sin(dec_rad)
            cr = np.cos(ra_rad)
            sr = np.sin(ra_rad)
            v_cel = np.column_stack([cd * cr, cd * sr, sd])   # (N, 3)

            px, py, in_front = plate.project_with_mask(v_cel)

            valid = (in_front &
                     (px >= 0.5) & (px < iw - 0.5) &
                     (py >= 0.5) & (py < ih - 0.5))

            idx = np.where(valid)[0]
            if not len(idx):
                continue

            sampled  = _sample_bilinear(img, px[idx], py[idx])
            strip_w  = self.W
            rows_out = idx // strip_w
            cols_out = idx  % strip_w
            if self.blend == 'average':
                self._accum  [y0 + rows_out, cols_out] += sampled.astype(np.float32)
                self._weights[y0 + rows_out, cols_out] += 1.0
            else:
                self.canvas[y0 + rows_out, cols_out] = sampled
            total += len(idx)

        self.count += 1
        return total

    def save(self, output_path: str, quality: int = 95,
             show_grid: bool = False,
             cons_mode: str = 'off',
             observed_names=None) -> None:
        """Save the panorama as a JPEG with Photo Sphere XMP metadata.

        show_grid:      draw RA/Dec coordinate grid
        cons_mode:      'off' | 'all' | 'observed'
        observed_names: iterable of constellation full-names used when
                        cons_mode='observed'
        """
        if self.blend == 'average' and self.count > 0:
            w      = np.maximum(self._weights[:, :, np.newaxis], 1.0)
            filled = self._weights > 0
            canvas = np.where(filled[:, :, np.newaxis],
                              self._accum / w, 0).astype(np.uint8)
        else:
            canvas = self.canvas.copy()

        if show_grid:
            _draw_grid(canvas)

        if cons_mode != 'off':
            names = set(observed_names) if cons_mode == 'observed' else None
            _draw_constellations(canvas, filter_names=names)

        xmp = (
            '<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>'
            '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
            '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            '<rdf:Description rdf:about=""'
            ' xmlns:GPano="http://ns.google.com/photos/1.0/panorama/"'
            f' GPano:ProjectionType="equirectangular"'
            f' GPano:FullPanoWidthPixels="{self.W}"'
            f' GPano:FullPanoHeightPixels="{self.H}"'
            f' GPano:CroppedAreaImageWidthPixels="{self.W}"'
            f' GPano:CroppedAreaImageHeightPixels="{self.H}"'
            ' GPano:CroppedAreaLeftPixels="0"'
            ' GPano:CroppedAreaTopPixels="0"/>'
            '</rdf:RDF></x:xmpmeta>'
            '<?xpacket end="w"?>'
        ).encode('utf-8')
        Image.fromarray(canvas).save(output_path, quality=quality, xmp=xmp)
