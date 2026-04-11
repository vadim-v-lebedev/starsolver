"""
Android-facing pipeline. All functions take/return file paths and primitives
so Chaquopy can marshal them without extra wrappers.

Module-level state stores intermediate results between steps.

CLI usage:
    python pipeline.py image.jpg [-o output.jpg] [-t threshold]
"""
import os
import sys
import time
import numpy as np
from PIL import Image, ImageDraw
from detector import find_stars_multiscale as find_stars
from solver import plate_solve
from plate import Plate
from draw import (load_image, draw_detections, draw_constellations, draw_star_names,
                  _draw_circles_with_alpha, _mag_alpha,
                  _draw_refine_labels, _draw_unknown_labels)

# ── module state ─────────────────────────────────────────────────────────────
_stars  = []    # detected stars (list of dicts)
_plate  = None  # Plate object (set after successful solve)


def detect(image_path: str, output_path: str,
           ratio_threshold: float = 20.0) -> dict:
    """
    Detect stars in image_path, save annotated image to output_path.
    Returns {'count': int, 'message': str}. Resets solve/refine state.
    """
    global _stars, _plate
    _plate = None

    img  = load_image(image_path)
    gray = load_image(image_path, 'L')
    _stars = find_stars(gray, ratio_threshold=ratio_threshold)
    draw_detections(img, _stars)
    Image.fromarray(img).save(output_path)

    n = len(_stars)
    message = f'{n} stars detected · showing 2000' if n > 2000 else f'{n} stars detected'
    return {'count': n, 'message': message}


def remove_star(image_path: str, output_path: str,
                tap_x: float, tap_y: float,
                max_dist: float = 200.0) -> dict:
    """
    Remove the detection closest to (tap_x, tap_y) if it is within max_dist pixels.
    Re-annotates and saves to output_path.
    Returns {'removed': bool, 'message': str, 'count': int}.
    """
    global _stars, _plate
    if not _stars:
        return {'removed': False, 'message': 'No detections to remove.', 'count': 0}

    dists = [((s['x'] - tap_x) ** 2 + (s['y'] - tap_y) ** 2) ** 0.5 for s in _stars]
    idx = int(min(range(len(dists)), key=lambda i: dists[i]))
    nearest_dist = dists[idx]

    if nearest_dist > max_dist:
        return {
            'removed': False,
            'message': 'No detection within range (nearest: %dpx away).' % int(nearest_dist),
            'count': len(_stars),
        }

    _stars.pop(idx)
    _plate = None
    img = load_image(image_path)
    draw_detections(img, _stars)
    Image.fromarray(img).save(output_path)
    return {
        'removed': True,
        'message': 'Removed detection. %d stars remaining.' % len(_stars),
        'count': len(_stars),
    }


def solve(image_path: str, output_path: str, use_timeout: bool = True) -> dict:
    """
    Plate-solve the previously detected stars.

    use_timeout=True  — first attempt, 10 s tetra3 timeout.
                        On success, re-solves quickly (constrained FOV) to get
                        matched centroids for the overlay.
    use_timeout=False — second attempt (user tapped again), no timeout.

    Saves image with constellation lines and matched-star circles to output_path.

    Returns dict with keys:
        status:  'solved' | 'no_solution' | 'too_few_stars'
        RA, Dec, Roll, FOV, RMSE, matches, T_solve  (only when solved)
    """
    global _plate

    if len(_stars) < 4:
        return {'status': 'too_few_stars'}

    img = load_image(image_path)
    h, w = img.shape[:2]

    if use_timeout:
        result = plate_solve(_stars, w, h, solve_timeout=10000)
        if result is None:
            return {'status': 'no_solution'}
        result = plate_solve(_stars, w, h,
                             fov_estimate=result['FOV'], fov_max_error=2.0,
                             solve_timeout=None, return_matches=True)
        if result is None:
            result = plate_solve(_stars, w, h, solve_timeout=None)
    else:
        result = plate_solve(_stars, w, h, solve_timeout=None, return_matches=True)

    if result is None:
        return {'status': 'no_solution'}

    _plate = Plate.from_dict(result)

    STAR_RADIUS = 25
    out_img = img  # reuse the already-loaded image
    draw_constellations(out_img, _plate, star_radius=STAR_RADIUS)
    centroids = result.get('matched_centroids', [])
    if centroids:
        pil_out = Image.fromarray(out_img)
        idraw   = ImageDraw.Draw(pil_out)
        r = STAR_RADIUS
        for yx in centroids:
            cx, cy = int(round(yx[1])), int(round(yx[0]))
            idraw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 180, 0), width=2)
        out_img[:] = np.array(pil_out)
    draw_star_names(out_img,
                    result.get('matched_stars', []),
                    result.get('matched_centroids', []),
                    star_radius=STAR_RADIUS)
    Image.fromarray(out_img).save(output_path)

    return {
        'status':          'solved',
        'RA':              result['RA'],
        'Dec':             result['Dec'],
        'Roll':            result['Roll'],
        'FOV':             result['FOV'],
        'RMSE':            result['RMSE'],
        'matches':         result['matches'],
        'T_solve':         result['T_solve'],
        'constellations':  ', '.join(result.get('constellations', [])),
        'dec_min':         result.get('dec_min', 0.0),
        'dec_max':         result.get('dec_max', 0.0),
    }


def refine_solution(image_path: str, output_path: str) -> dict:
    """
    Refine the plate solution with ICP and identify all detected stars.

    Draws constellation lines, gold circles + labels for identified stars,
    and dim red circles for unidentified detections.

    Returns dict with keys:
        status: 'refined' | 'failed' | 'no_plate'
        RA, Dec, FOV, RMSE, matches, unknowns, T_refine  (when refined)
    """
    if _plate is None:
        return {'status': 'no_plate'}

    from refine import refine
    t0 = time.time()
    result = refine(_plate, _stars)
    t_ms = round((time.time() - t0) * 1000.0, 1)

    if result['status'] != 'refined':
        return {'status': 'failed'}

    refined_plate = Plate.from_dict(result)

    STAR_RADIUS = 25
    out_img = load_image(image_path)

    # Constellation lines using refined orientation
    draw_constellations(out_img, refined_plate, star_radius=STAR_RADIUS)

    # Matched stars: orange circles, fading by catalog magnitude
    _draw_circles_with_alpha(
        out_img,
        [( int(round(s['x'])), int(round(s['y'])), _mag_alpha(s['mag']) )
         for s in result['matched_stars']],
        color=(255, 180, 0), radius=STAR_RADIUS, thickness=2,
    )

    # Unknown detections: red circles, full opacity
    unknowns = result['unknown_detections']
    if unknowns:
        pil_out = Image.fromarray(out_img)
        idraw   = ImageDraw.Draw(pil_out)
        r = STAR_RADIUS
        for u in unknowns:
            cx, cy = int(round(u['x'])), int(round(u['y']))
            idraw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(200, 50, 50), width=2)
        out_img[:] = np.array(pil_out)

    # Star name labels (Greek letters via PIL)
    _draw_refine_labels(out_img, result['matched_stars'], STAR_RADIUS)

    # Unknown detection labels: nearest catalog distance + magnitude diff
    _draw_unknown_labels(out_img, unknowns, refined_plate,
                         result.get('phot_b', 0.0), STAR_RADIUS)

    Image.fromarray(out_img).save(output_path)

    named = sum(1 for s in result['matched_stars'] if s['name'])
    faintest_mag = max(s['mag'] for s in result['matched_stars']) \
        if result['matched_stars'] else 0.0
    return {
        'status':          'refined',
        'RA':              result['RA'],
        'Dec':             result['Dec'],
        'FOV':             result['FOV'],
        'RMSE':            result['RMSE'],
        'matches':         result['matches'],
        'named':           named,
        'unknowns':        len(result['unknown_detections']),
        'mag_limit':       round(faintest_mag, 1),
        'T_refine':        t_ms,
        'constellations':  ', '.join(result.get('constellations', [])),
        'dec_min':         result.get('dec_min', 0.0),
        'dec_max':         result.get('dec_max', 0.0),
    }


def main():
    import argparse
    p = argparse.ArgumentParser(description='Plate-solve a night-sky image.')
    p.add_argument('image')
    p.add_argument('-o', '--output', help='output path (default: <stem>_solved.<ext>)')
    p.add_argument('-t', '--threshold', type=float, default=20.0,
                   help='detection variance-ratio threshold (default: 20)')
    args = p.parse_args()

    stem, ext = os.path.splitext(args.image)
    out = args.output or f'{stem}_solved{ext}'

    r = detect(args.image, out, args.threshold)
    print(r['message'])
    if r['count'] < 4:
        print('Too few stars detected — lower the threshold with -t.')
        sys.exit(1)

    r = solve(args.image, out)
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

    r = refine_solution(args.image, out)
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
