import sys
from pathlib import Path

# Allow bare imports (from pipeline import ...) matching the Pyodide environment
sys.path.insert(0, str(Path(__file__).parent.parent / 'starsolver'))
