"""Extract the 7-ring x 8-angle detector layout from actual MCX exit maps.

The 3x3 patch is 3x3 mm in this pilot, NOT the paper's 3 camera pixels.
Out-of-field samples retain NaN and a validity mask. No extrapolation to zero.
"""
import csv, json
import numpy as np
from common import ROOT, load_config, write_json

def detector_offsets():
    return np.array([(r*np.cos(a), r*np.sin(a)) for r in np.linspace(15,25,7) for a in np.arange(8)*np.pi/4])

def integrate_patch(image, x_mm, y_mm, c, halfwidth=1):
    h=c['voxel_mm'];n=np.array(image.shape)
    # Centred pixel coordinates; nearest output pixel, explicit and fixed.
    q=np.rint((np.array([x_mm,y_mm])+np.array(c['domain_mm'][:2])/2)/h-.5).astype(int)
    if np.any(q-halfwidth<0) or np.any(q+halfwidth>=n): return np.nan
    return float(image[q[0]-halfwidth:q[0]+halfwidth+1,q[1]-halfwidth:q[1]+halfwidth+1].sum(dtype=np.float64)*h*h)

def main():
    c=load_config();offsets=detector_offsets()
    pos=np.array([(x,y) for y in c['scan_grid_mm'] for x in c['scan_grid_mm']])
    vals=np.full((len(pos),len(offsets),2),np.nan)
    rows=[]
    for i,xy in enumerate(pos):
        for k,case in enumerate(['homogeneous','two_spheres']):
            image=np.load(ROOT/f'data/mcx/scan_{i:02}_{case}.npz')['transmission']
            for j,off in enumerate(offsets):
                vals[i,j,k]=integrate_patch(image,*(xy+off),c)
        for j,off in enumerate(offsets):
            h,t=vals[i,j]
            rows.append([i,*xy,j,*off,h,t,bool(np.isfinite(h) and np.isfinite(t)), (t-h)/h if h>0 and np.isfinite(t) else np.nan])
    np.savez_compressed(ROOT/'data/virtual_detector_samples.npz',intensity=vals,valid=np.isfinite(vals).all(axis=2),source_xy_mm=pos,offset_xy_mm=offsets,grid_images=vals.reshape(3,3,56,2))
    with (ROOT/'results/virtual_detector_samples.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['source_id','source_x_mm','source_y_mm','detector_id','dx_mm','dy_mm','background_energy','target_energy','valid','relative_change']);w.writerows(rows)
    write_json(ROOT/'results/virtual_detector_summary.json',dict(offsets=56,sources=9,pairs=504,valid_pairs=int(np.isfinite(vals).all(axis=2).sum()),physical_patch_mm=[3,3],paper_patch='3x3 camera pixels; physically different from this pilot',reconstruction='not performed: 3x3 source grid validates extraction, not reliable depth sampling'))
    print('Saved',len(rows),'source-detector pairs; valid',np.isfinite(vals).all(axis=2).sum())

if __name__=='__main__':main()
