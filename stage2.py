"""Run the second transmission experiment. See docs/stage2.md for the data layout."""
import os
import runpy
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
bundle = os.environ.get("NIR_PACKAGES")
if not bundle and os.name == "nt":
    matches = list(root.parents[1].glob("第一次仿真*/运行环境"))
    bundle = matches[0] if len(matches) == 1 else None
if bundle and sys.version_info[:2] == (3, 12):
    sys.path[:0] = [str(Path(bundle) / d) for d in ("mcx_packages", "packages")]
sys.path.insert(0, str(root / "src"))
task = sys.argv.pop(1) if len(sys.argv) > 1 else ""
if task not in ("scan", "reconstruct", "figures", "check"):
    raise SystemExit("Usage: python stage2.py scan|reconstruct|figures|check [options]")
runpy.run_module("stage2_" + task, run_name="__main__")

