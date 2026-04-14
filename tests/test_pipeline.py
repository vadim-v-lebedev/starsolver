from pathlib import Path
from pipeline import Pipeline

IMAGE = Path(__file__).parent / 'cloud.jpg'


# ── state management (no external data needed) ────────────────────────────────

def test_initial_state():
    p = Pipeline()
    assert p.stars == []
    assert p.plate is None


def test_detect_resets_plate(tmp_path):
    p = Pipeline()
    p.plate = object()  # simulate a prior solve
    p.detect(str(IMAGE), str(tmp_path / 'out.jpg'))
    assert p.plate is None


def test_solve_too_few_stars(tmp_path):
    p = Pipeline()
    p.stars = [{'x': 10, 'y': 10}, {'x': 20, 'y': 20}]
    result = p.solve(str(IMAGE), str(tmp_path / 'out.jpg'))
    assert result['status'] == 'too_few_stars'


def test_refine_without_solve(tmp_path):
    p = Pipeline()
    result = p.refine(str(IMAGE), str(tmp_path / 'out.jpg'))
    assert result['status'] == 'no_plate'


def test_remove_star_empty(tmp_path):
    p = Pipeline()
    result = p.remove_star(str(IMAGE), str(tmp_path / 'out.jpg'), 100, 100)
    assert result['removed'] is False
    assert result['count'] == 0


# ── integration tests ─────────────────────────────────────────────────────────

def test_detect(tmp_path):
    p = Pipeline()
    result = p.detect(str(IMAGE), str(tmp_path / 'out.jpg'))
    assert result['count'] > 0
    assert isinstance(result['message'], str)
    assert len(p.stars) == result['count']
    assert (tmp_path / 'out.jpg').exists()


def test_solve(tmp_path):
    p = Pipeline()
    p.detect(str(IMAGE), str(tmp_path / 'detect.jpg'))
    result = p.solve(str(IMAGE), str(tmp_path / 'solve.jpg'))
    assert result['status'] == 'solved'
    assert 0 <= result['RA'] < 360
    assert -90 <= result['Dec'] <= 90
    assert 0 < result['FOV'] < 180
    assert result['matches'] > 0
    assert p.plate is not None
    assert (tmp_path / 'solve.jpg').exists()


def test_refine(tmp_path):
    p = Pipeline()
    p.detect(str(IMAGE), str(tmp_path / 'detect.jpg'))
    p.solve(str(IMAGE), str(tmp_path / 'solve.jpg'))
    result = p.refine(str(IMAGE), str(tmp_path / 'refine.jpg'))
    assert result['status'] == 'refined'
    assert result['matches'] > 0
    assert result['mag_limit'] > 0
    assert (tmp_path / 'refine.jpg').exists()
