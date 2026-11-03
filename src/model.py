"""Save known ground truth. These arrays are not reconstructed data."""
import numpy as np
from common import ROOT, load_config, make_model, write_json

def main():
    c = load_config()
    labels, mua, axes = make_model(c)
    np.save(ROOT / 'data/phantom_labels.npy', labels)
    np.save(ROOT / 'data/phantom_mua_mm_inverse.npy', mua)
    np.savez_compressed(ROOT / 'data/phantom_coordinates.npz', x_mm=axes[0], y_mm=axes[1], z_mm=axes[2])
    records = []
    for j, inc in enumerate(c['inclusions'], 2):
        ids = np.where(labels == j)
        records.append(dict(inc, voxel_count=len(ids[0]), voxelised_centre_mm=[float(axes[k][ids[k]].mean()) for k in range(3)], voxelised_volume_mm3=len(ids[0]) * c['voxel_mm'] ** 3))
    write_json(ROOT / 'results/model_summary.json', {'array_order':'x,y,z', 'shape':labels.shape, 'targets':records, 'classification':'known ground truth, not reconstruction'})
    print('Saved ground-truth model:', labels.shape, records, flush=True)

if __name__ == '__main__': main()
