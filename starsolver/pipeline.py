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
                  _draw_refine_labels, _draw_unknown_labels)


class Pipeline:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.stars = []    # detected stars (list of dicts)
        self.plate = None  # Plate object (set after successful solve)

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
        d = self.config.draw
        draw_detections(img, self.stars,
                        color=d.detection_color, thickness=d.detection_thickness,
                        star_radius=d.star_radius, max_draw=d.max_draw)
        Image.fromarray(img).save(output_path)

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
        Image.fromarray(img).save(output_path)
        return {
            'removed': True,
            'message': 'Removed detection. %d stars remaining.' % len(self.stars),
            'count': len(self.stars),
        }

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

        result = plate_solve(self.stars, w, h,
                             solve_timeout=timeout,
                             return_matches=True)
        if result is None:
            return {'status': 'no_solution'}

        self.plate = Plate.from_dict(result)
        d = self.config.draw

        draw_constellations(img, self.plate,
                            color=d.constellation_color,
                            thickness=d.constellation_thickness,
                            star_radius=d.star_radius)
        centroids = result.get('matched_centroids', [])
        if centroids:
            _draw_circles_with_alpha(
                img,
                [(float(yx[1]), float(yx[0]), 1.0) for yx in centroids],
                color=d.match_color, radius=d.star_radius, thickness=d.circle_thickness,
            )
        draw_star_names(img,
                        result.get('matched_stars', []),
                        result.get('matched_centroids', []),
                        star_radius=d.star_radius,
                        color=d.match_color)
        Image.fromarray(img).save(output_path)

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
        out_img = load_image(image_path)
        d = self.config.draw

        draw_constellations(out_img, refined_plate,
                            color=d.constellation_color,
                            thickness=d.constellation_thickness,
                            star_radius=d.star_radius)
        _draw_circles_with_alpha(
            out_img,
            [(int(round(s['x'])), int(round(s['y'])), _mag_alpha(s['mag']))
             for s in result['matched_stars']],
            color=d.match_color, radius=d.star_radius, thickness=d.circle_thickness,
        )

        unknowns = result['unknown_detections']
        if unknowns:
            _draw_circles_with_alpha(
                out_img,
                [(int(round(u['x'])), int(round(u['y'])), 1.0) for u in unknowns],
                color=d.unknown_color, radius=d.star_radius, thickness=d.circle_thickness,
            )

        _draw_refine_labels(out_img, result['matched_stars'], d.star_radius,
                            color=d.match_color)
        _draw_unknown_labels(out_img, unknowns, refined_plate,
                             result.get('phot_b', 0.0), d.star_radius,
                             color=d.unknown_color)

        Image.fromarray(out_img).save(output_path)

        faintest_mag = max((s['mag'] for s in result['matched_stars']), default=0.0)
        return {
            'status':         'refined',
            'RA':             result['RA'],
            'Dec':            result['Dec'],
            'FOV':            result['FOV'],
            'RMSE':           result['RMSE'],
            'matches':        result['matches'],
            'unknowns':       len(result['unknown_detections']),
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
    args = ap.parse_args()

    stem, ext = os.path.splitext(args.image)
    out = args.output or f'{stem}_solved{ext}'

    pipe = Pipeline()

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
