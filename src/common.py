"""Project paths and unit conventions. All transport calculations use mm."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def load_config():
    return json.loads((ROOT / 'configs/phantom.json').read_text(encoding='utf-8'))

def write_json(path, value):
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=convert), encoding='utf-8')

def make_model(c=None):
    c = c or load_config()
    h = c['voxel_mm']
    axes = [(np.arange(round(d / h)) + 0.5) * h for d in c['domain_mm']]
    axes[0] -= c['domain_mm'][0] / 2
    axes[1] -= c['domain_mm'][1] / 2
    X, Y, Z = np.meshgrid(*axes, indexing='ij')
    labels = np.ones(X.shape, dtype=np.uint8)
    for j, inc in enumerate(c['inclusions'], 2):
        centre = inc['centre_mm']
        sphere = sum((q - k) ** 2 for q, k in zip((X,Y,Z), centre)) <= (inc['diameter_mm'] / 2) ** 2
        labels[sphere] = j
    mua = np.array([0, c['background']['mua']] + [v['mua'] for v in c['inclusions']])[labels]
    return labels, mua, axes

def property_table(c, case):
    b = c['background']
    prop = [[0, 0, 1, 1], [b['mua'], b['mus'], b['g'], b['n']]]
    for i, inc in enumerate(c['inclusions']):
        a = inc['mua'] if case == 'two_spheres' or (case == 'high_only' and i == 0) else b['mua']
        prop.append([a, b['mus'], b['g'], b['n']])
    return np.array(prop, dtype=float)
