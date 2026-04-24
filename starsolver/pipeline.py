"""
Star field processing pipeline.

The Pipeline class holds intermediate state between steps (detected stars,
plate solution) so that detect → solve → refine can be called sequentially
on the same image without re-running earlier stages.

CLI usage:
    python pipeline.py image.jpg [-o output.jpg] [-t threshold]
"""
import os
import sys
import time
import numpy as np
from PIL import Image
from detector import find_stars_multiscale as find_stars
from solver import plate_solve
from plate import Plate
from config import Config
from draw import (load_image, draw_detections, draw_constellations, draw_star_names,
                  _draw_circles_with_alpha, _mag_alpha,
                  _draw_refine_labels, _draw_unknown_labels,
                  _draw_special_labels, draw_timestamp)


class Pipeline:
    def __init__(self, config: Config = None):
        self.config           = config or Config()
        self.stars            = []    # detected stars (list of dicts)
        self.plate            = None  # Plate object (set after successful solve)
        self.timestamp        = None  # ISO 8601 string from EXIF, or None
        self.fov_hint         = None  # None=auto from EXIF; float=explicit (°, long axis); False=disabled
        self.detection_mask   = None  # uint8 numpy array (image coords) from apply_mask, or None

    @staticmethod
    def read_exif(image_path: str) -> dict:
        """Read EXIF metadata from image.

        Returns dict with:
          'timestamp'    — ISO 8601 string or None
          'fov_estimate' — long-axis FOV in degrees derived from
                           FocalLengthIn35mmFilm, or None if tag absent
        """
        result = {'timestamp': None, 'fov_estimate': None}
        try:
            from PIL import Image as _PIL, ImageOps as _Ops
            img  = _PIL.open(image_path)
            exif = img.getexif()
            sub  = exif.get_ifd(0x8769)

            dt = str(sub.get(0x9003) or '')
            if len(dt) >= 19:
                tz  = str(sub.get(0x9011) or '')
                iso = dt[:4] + '-' + dt[5:7] + '-' + dt[8:10] + 'T' + dt[11:]
                result['timestamp'] = iso + tz if tz else iso

            f35 = sub.get(0xA405)
            if f35 and f35 > 0:
                import numpy as np
                # 36 mm is the long side of a 35 mm frame → always gives long-axis FOV
                result['fov_estimate'] = float(
                    np.degrees(2.0 * np.arctan(36.0 / (2.0 * float(f35)))))
        except Exception:
            pass
        return result

    def detect(self, image_path: str, output_path: str,
               ratio_threshold: float = 20.0) -> dict:
        """
        Detect stars in image_path, save annotated image to output_path.
        Returns {'count': int, 'message': str}. Resets solve/refine state.
        """
        self.plate = None

        img  = load_image(image_path)
        gray = load_image(image_path, 'L')
        self.stars = find_stars(gray, ratio_threshold=ratio_threshold)

        if self.detection_mask is not None:
            ih, iw = img.shape[:2]
            self.stars = [
                s for s in self.stars
                if self.detection_mask[
                    min(int(s['y']), ih - 1),
                    min(int(s['x']), iw - 1)
                ] < 128
            ]

        d = self.config.draw
        draw_detections(img, self.stars,
                        color=d.detection_color, thickness=d.detection_thickness,
                        star_radius=d.star_radius, max_draw=d.max_draw)
        Image.fromarray(img).save(output_path, quality=95)

        n = len(self.stars)
        message = f'{n} stars detected · showing 2000' if n > 2000 else f'{n} stars detected'
        return {'count': n, 'message': message}

    def remove_star(self, image_path: str, output_path: str,
                    tap_x: float, tap_y: float,
                    max_dist: float = 200.0) -> dict:
        """
        Remove the detection closest to (tap_x, tap_y) if within max_dist pixels.
        Re-annotates and saves to output_path.
        Returns {'removed': bool, 'message': str, 'count': int}.
        """
        if not self.stars:
            return {'removed': False, 'message': 'No detections to remove.', 'count': 0}

        dists = [((s['x'] - tap_x) ** 2 + (s['y'] - tap_y) ** 2) ** 0.5
                 for s in self.stars]
        idx = int(min(range(len(dists)), key=lambda i: dists[i]))
        nearest_dist = dists[idx]

        if nearest_dist > max_dist:
            return {
                'removed': False,
                'message': 'No detection within range (nearest: %dpx away).' % int(nearest_dist),
                'count': len(self.stars),
            }

        self.stars.pop(idx)
        self.plate = None
        img = load_image(image_path)
        draw_detections(img, self.stars)
        Image.fromarray(img).save(output_path, quality=95)
        return {
            'removed': True,
            'message': 'Removed detection. %d stars remaining.' % len(self.stars),
            'count': len(self.stars),
        }

    def apply_mask(self, image_path: str, output_path: str,
                   mask_bytes: bytes, mask_w: int, mask_h: int) -> dict:
        """
        Remove detections that fall inside the painted mask and redraw.

        mask_bytes: flat uint8 alpha array, row-major, shape (mask_h, mask_w)
        Returns {'count': int, 'message': str}
        """
        if not self.stars:
            return {'count': 0, 'message': 'No detections to mask.'}

        mask = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(mask_h, mask_w)

        img = load_image(image_path)
        ih, iw = img.shape[:2]
        sx = mask_w / iw
        sy = mask_h / ih

        # Scale mask to image coordinates for later use in drawing
        img_mask = np.array(
            Image.fromarray(mask).resize((iw, ih), Image.NEAREST)
        )
        self.detection_mask = img_mask

        before = len(self.stars)
        self.stars = [
            s for s in self.stars
            if mask[min(int(s['y'] * sy), mask_h - 1),
                    min(int(s['x'] * sx), mask_w - 1)] < 128
        ]
        removed = before - len(self.stars)
        self.plate = None

        d = self.config.draw
        draw_detections(img, self.stars,
                        color=d.detection_color, thickness=d.detection_thickness,
                        star_radius=d.star_radius, max_draw=d.max_draw)
        Image.fromarray(img).save(output_path, quality=95)

        n = len(self.stars)
        msg = f'Removed {removed} detection{"s" if removed != 1 else ""}. {n} remaining.'
        return {'count': n, 'message': msg}

    def solve(self, image_path: str, output_path: str,
              timeout_override=None) -> dict:
        """
        Plate-solve the previously detected stars.
        Saves image with constellation lines and matched-star circles to output_path.

        timeout_override: None = use config timeout; 0 = no timeout (unlimited).

        Returns dict with keys:
            status:  'solved' | 'no_solution' | 'too_few_stars'
            RA, Dec, Roll, FOV, RMSE, matches, T_solve  (only when solved)
        """
        if len(self.stars) < 4:
            return {'status': 'too_few_stars'}

        img = load_image(image_path)
        h, w = img.shape[:2]

        if timeout_override == 0:
            timeout = None          # unlimited
        elif timeout_override is not None:
            timeout = timeout_override
        else:
            timeout = self.config.solve.timeout_ms

        fov_estimate = fov_max_error = None
        if self.fov_hint is not False:
            fov_long = None
            if self.fov_hint is not None:
                fov_long = float(self.fov_hint)
            else:
                fov_long = self.read_exif(image_path).get('fov_estimate')
            if fov_long is not None:
                import numpy as _np
                f_est        = max(w, h) / (2.0 * _np.tan(_np.radians(fov_long) / 2.0))
                fov_estimate = float(_np.degrees(2.0 * _np.arctan(w / (2.0 * f_est))))
                fov_max_error = 10.0

        result = plate_solve(self.stars, w, h,
                             fov_estimate=fov_estimate,
                             fov_max_error=fov_max_error,
                             solve_timeout=timeout,
                             return_matches=True)
        if result is None:
            return {'status': 'no_solution'}

        self.plate = Plate.from_dict(result)
        self.plate.timestamp = self.timestamp

        d = self.config.draw
        draw_mask = self.detection_mask if d.mask_constellations else None

        draw_constellations(img, self.plate,
                            color=d.constellation_color,
                            thickness=d.constellation_thickness,
                            star_radius=d.star_radius,
                            mask=draw_mask)
        centroids = result.get('matched_centroids', [])
        if centroids:
            _draw_circles_with_alpha(
                img,
                [(float(yx[1]), float(yx[0]), 1.0) for yx in centroids],
                color=d.match_color, radius=d.star_radius, thickness=d.circle_thickness,
                mask=draw_mask,
            )
        draw_star_names(img,
                        result.get('matched_stars', []),
                        result.get('matched_centroids', []),
                        star_radius=d.star_radius,
                        color=d.match_color,
                        mask=draw_mask,
                        font_size=d.text_size)
        if d.show_timestamp and self.plate.timestamp:
            draw_timestamp(img, self.plate.timestamp, font_size=d.text_size)
        Image.fromarray(img).save(output_path, quality=95)

        return {
            'status':         'solved',
            'RA':             result['RA'],
            'Dec':            result['Dec'],
            'Roll':           result['Roll'],
            'FOV':            result['FOV'],
            'RMSE':           result['RMSE'],
            'matches':        result['matches'],
            'T_solve':        result['T_solve'],
            'constellations': ', '.join(result.get('constellations', [])),
            'dec_min':        result.get('dec_min', 0.0),
            'dec_max':        result.get('dec_max', 0.0),
        }

    def refine(self, image_path: str, output_path: str) -> dict:
        """
        Refine the plate solution with ICP and identify all detected stars.
        Draws constellation lines, gold circles + labels for identified stars,
        and red circles for unidentified detections.

        Returns dict with keys:
            status: 'refined' | 'failed' | 'no_plate'
            RA, Dec, FOV, RMSE, matches, unknowns, T_refine  (when refined)
        """
        if self.plate is None:
            return {'status': 'no_plate'}

        from refine import refine as _refine
        t0 = time.time()
        result = _refine(self.plate, self.stars)
        t_ms = round((time.time() - t0) * 1000.0, 1)

        if result['status'] != 'refined':
            return {'status': 'failed'}

        refined_plate = Plate.from_dict(result)
        refined_plate.timestamp = self.timestamp
        out_img = load_image(image_path)
        d = self.config.draw
        draw_mask = self.detection_mask if d.mask_constellations else None

        draw_constellations(out_img, refined_plate,
                            color=d.constellation_color,
                            thickness=d.constellation_thickness,
                            star_radius=d.star_radius,
                            mask=draw_mask)
        _draw_circles_with_alpha(
            out_img,
            [(int(round(s['x'])), int(round(s['y'])), _mag_alpha(s['mag']))
             for s in result['matched_stars']],
            color=d.match_color, radius=d.star_radius, thickness=d.circle_thickness,
            mask=draw_mask,
        )

        unknowns = result['unknown_detections']

        # ── special object matching (planets + deep-sky) ─────────────────
        special_matches = []
        if d.show_planets:
            arcsec_per_px = refined_plate.fov_deg * 3600.0 / refined_plate.w
            # 15 arcmin covers Jupiter/Saturn great-inequality error (up to ~11')
            special_thr = min(300.0, max(15.0, 900.0 / arcsec_per_px))
            if self.timestamp:
                from planets import match_planets as _match_planets
                special_matches += _match_planets(
                    refined_plate, self.timestamp, unknowns, special_thr)
            from deepsky import match_deepsky as _match_deepsky
            special_matches += _match_deepsky(refined_plate, unknowns, special_thr)

        if special_matches:
            _draw_circles_with_alpha(
                out_img,
                [(int(round(p['x'])), int(round(p['y'])), 1.0) for p in special_matches],
                color=d.special_color, radius=d.star_radius, thickness=d.circle_thickness,
                mask=draw_mask,
            )

        if unknowns:
            _draw_circles_with_alpha(
                out_img,
                [(int(round(u['x'])), int(round(u['y'])), 1.0) for u in unknowns],
                color=d.unknown_color, radius=d.star_radius, thickness=d.circle_thickness,
                mask=draw_mask,
            )

        _draw_refine_labels(out_img, result['matched_stars'], d.star_radius,
                            color=d.match_color, mask=draw_mask,
                            font_size=d.text_size)
        _draw_special_labels(out_img, special_matches, d.star_radius,
                             color=d.special_color, mask=draw_mask,
                             font_size=d.text_size)
        _draw_unknown_labels(out_img, unknowns, refined_plate,
                             d.star_radius,
                             color=d.unknown_color, mask=draw_mask,
                             font_size=d.text_size)
        if d.show_timestamp and refined_plate.timestamp:
            draw_timestamp(out_img, refined_plate.timestamp, font_size=d.text_size)

        Image.fromarray(out_img).save(output_path, quality=95)

        faintest_mag = max((s['mag'] for s in result['matched_stars']), default=0.0)
        return {
            'status':         'refined',
            'RA':             result['RA'],
            'Dec':            result['Dec'],
            'FOV':            result['FOV'],
            'RMSE':           result['RMSE'],
            'matches':        result['matches'],
            'unknowns':       len(unknowns),
            'specials':       ', '.join(s.get('messier', s['name']) for s in special_matches),
            'mag_limit':      round(faintest_mag, 1),
            'T_refine':       t_ms,
            'constellations': ', '.join(result.get('constellations', [])),
            'dec_min':        result.get('dec_min', 0.0),
            'dec_max':        result.get('dec_max', 0.0),
        }


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Plate-solve a night-sky image.')
    ap.add_argument('image')
    ap.add_argument('-o', '--output', help='output path (default: <stem>_solved.<ext>)')
    ap.add_argument('-t', '--threshold', type=float, default=20.0,
                    help='detection variance-ratio threshold (default: 20)')
    ap.add_argument('-f', '--fov', type=float, default=None,
                    help='long-axis FOV hint in degrees (auto-read from EXIF if omitted)')
    args = ap.parse_args()

    stem, ext = os.path.splitext(args.image)
    out = args.output or f'{stem}_solved{ext}'

    pipe = Pipeline()
    if args.fov is not None:
        pipe.fov_hint = args.fov

    r = pipe.detect(args.image, out, args.threshold)
    print(r['message'])
    if r['count'] < 4:
        print('Too few stars detected — lower the threshold with -t.')
        sys.exit(1)

    r = pipe.solve(args.image, out)
    if r['status'] == 'too_few_stars':
        print('Too few stars for plate solving.')
        sys.exit(1)
    if r['status'] == 'no_solution':
        print('No plate solution found.')
        sys.exit(1)
    print(f"Solved:  RA={r['RA']:.2f}°  Dec={r['Dec']:.2f}°  "
          f"FOV={r['FOV']:.1f}°  RMSE={r['RMSE']:.0f}\"  "
          f"{r['matches']} matches  {r['T_solve']/1000:.1f}s")
    if r['constellations']:
        print(f"         {r['constellations']}")

    r = pipe.refine(args.image, out)
    if r['status'] != 'refined':
        print('Refinement failed.')
        sys.exit(1)
    print(f"Refined: RA={r['RA']:.2f}°  Dec={r['Dec']:.2f}°  "
          f"FOV={r['FOV']:.1f}°  RMSE={r['RMSE']:.0f}\"  "
          f"{r['matches']} identified  lim={r['mag_limit']:.1f}m  {r['T_refine']/1000:.1f}s")
    if r['unknowns']:
        print(f"         {r['unknowns']} unidentified detections")
    if r['constellations']:
        print(f"         {r['constellations']}")
    print(f"Saved:   {out}")


if __name__ == '__main__':
    main()
