"""Run: python -m unittest discover -s tests (or python run.py validate)."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from validate import ScientificTests
