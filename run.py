"""Launcher: keep dependencies in this delivery, without modifying system Python."""
from pathlib import Path
import os,sys,runpy
root=Path(__file__).resolve().parent
env=root.parent/'运行环境'
for p in [env/'packages',env/'mcx_packages',root/'src']:
    if p.exists(): sys.path.insert(0,str(p))
os.environ.setdefault('MPLCONFIGDIR',str(root/'logs/matplotlib_cache'))
os.environ.setdefault('NUMBA_CACHE_DIR',str(root/'logs/numba_cache'))
os.environ.setdefault('OMP_NUM_THREADS','4')
if len(sys.argv)<2: raise SystemExit('Usage: python run.py model|mcx_forward|figures|validate [options]')
module=sys.argv.pop(1)
runpy.run_module(module,run_name='__main__')
