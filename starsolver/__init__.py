import sys
import os

# Allow bare imports (from detector import ...) whether the package is imported
# as 'starsolver.pipeline' (installed CLI) or as a top-level module (Pyodide).
_pkg_dir = os.path.dirname(__file__)
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)
