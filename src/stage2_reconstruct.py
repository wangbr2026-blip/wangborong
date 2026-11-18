"""Generate and evaluate tomosynthesis volumes from paired exit images.

Input MCX images are indexed [x,y]. Scan maps use [source_y,source_x].
Output arrays use [z,y,x]. Positions, aperture sizes and path lengths use mm.
"""
import argparse
import csv
import hashlib
import json
import os
import warnings
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
from common import ROOT, load_config, write_json
from virtual_detectors import detector_offsets, integrate_patch

RAW = ROOT/"data/stage2"
RES = ROOT/"results/stage2"
XY = np.arange(-24.,25.)
Z = np.arange(5.,46.)
OFF = np.vstack([detector_offsets(), [0.,0.]])
MUA, MUSP, N = 0.01, 1., 1.4
D = 1/(3*(MUA+MUSP))
K = np.sqrt(MUA/D)
# Extrapolation factor for n>1, as in the original authors' theoryContini.py.
A = np.polynomial.polynomial.polyval(N,[504.332889,-2641.00214,5923.699064,-7376.355814,
                                         5507.53041,-2463.357945,610.956547,-64.8047])
ZE = 2*A*D

def slab_path(diameter=10.):
    """Zero-offset mean transmitted path in an equivalent slab (author convention)."""
    m=np.arange(-10,11)
    a=diameter*(1-2*m)-4*m*ZE-1/MUSP
    b=a+2*ZE
    def terms(z):
        r=np.abs(z)
        return (np.sign(z)*np.exp(-K*r),z/r**3*(1+K*r)*np.exp(-K*r))
    ga,fa=terms(a)
    gb,fb=terms(b)
    return (ga-gb).sum()/(2*D*(fa-fb).sum())

def slab_green(r,z):
    """CW scalar Green function, isotropic point at 1/musp, extrapolated planes."""
    g=np.zeros(np.broadcast_shapes(np.shape(r),np.shape(z)))
    for m in range(-3,4):
        shift=2*m*(50+2*ZE)
        a=np.hypot(r,z-1/MUSP-shift)
        b=np.hypot(r,z+1/MUSP+2*ZE-shift)
        g+=np.exp(-K*a)/a-np.exp(-K*b)/b
    return g/(4*np.pi*D)

def trajectories():
    radii=np.linspace(15,25,7)
    along=np.linspace(0,1,1001)
    ridge=[]
    for r in radii:
        product=slab_green(r*along[None,:], Z[:,None])*slab_green(r*(1-along[None,:]),50-Z[:,None])
        ridge.append(along[np.argmax(product,axis=1)])
    return np.asarray(ridge)

def sample_pair(maps, xy, c):
    return [[integrate_patch(im,*(xy+off),c) for im in maps] for off in OFF]

def legacy_measurements(legacy):
    c=load_config()
    axis=np.asarray(c["scan_grid_mm"])
    rows=[]
    hashes=[]
    for i in range(9):
        xy=np.array([axis[i%3],axis[i//3]])
        images=[]
        for case in ("homogeneous","two_spheres"):
            file=legacy/"data/mcx"/f"scan_{i:02}_{case}.npz"
            with np.load(file) as f:
                np.testing.assert_equal(f["source_xy_mm"],xy)
                images.append(f["transmission"])
            hashes.append((file.name,hashlib.sha256(file.read_bytes()).hexdigest()))
        rows.append(sample_pair(images,xy,c))
    raw=np.array(rows)
    with np.load(legacy/"data/virtual_detector_samples.npz") as old:
        np.testing.assert_allclose(raw[:,:56,:],old["intensity"],equal_nan=True,atol=0,rtol=0)
    with (RES/"legacy_inputs.csv").open("w",newline="",encoding="utf-8") as fp:
        writer=csv.writer(fp);writer.writerow(["file","sha256"]);writer.writerows(hashes)
    return axis,raw.reshape(3,3,57,2)

def wide_measurements():
    c=json.loads((RAW/"config.json").read_text(encoding="utf-8"))
    axis=np.array(c["scan_grid_mm"])
    measurements=[]
    closure=[]; runtime=[]; seeds=[]
    for i in range(81):
        with np.load(RAW/f"pair_{i:02}_r0.npz") as f:
            xy=f["source_xy_mm"]
            np.testing.assert_equal(xy,[axis[i%9],axis[i//9]])
            measurements.append(sample_pair(f["transmission"],xy,c))
    centre=[]
    files=sorted(RAW.glob("pair_*_r*.npz"))
    hashes=[]
    run_rows=[]
    for p in files:
        with np.load(p) as f:
            closure.extend(f["energy_closure"]);runtime.extend(f["elapsed_s"]);seeds.append(int(f["seed"]))
            for k,case in enumerate(("homogeneous","two_spheres")):
                run_rows.append((p.name,case,*f["source_xy_mm"],int(f["seed"]),c["scan_nphoton"],
                                 f["elapsed_s"][k],f["energy_closure"][k],
                                 f["transmission"][k].sum(dtype=np.float64)))
            if p.name.startswith("pair_40_"):
                centre.append(f["transmission"].sum(axis=(1,2),dtype=np.float64))
        hashes.append((p.name,hashlib.sha256(p.read_bytes()).hexdigest()))
    centre=np.array(centre)
    with (RES/"run_index.csv").open("w",newline="",encoding="utf-8") as out:
        wr=csv.writer(out)
        wr.writerow(["paired_file","case","x_mm","y_mm","seed","photons","elapsed_s","energy_closure","T"])
        wr.writerows(run_rows)
    with (RES/"wide_inputs.csv").open("w",newline="",encoding="utf-8") as out:
        wr=csv.writer(out);wr.writerow(["file","sha256"]);wr.writerows(hashes)
    write_json(RES/"scan_summary.json",dict(runs=len(runtime), photons=len(runtime)*c["scan_nphoton"],
              seconds=sum(runtime), max_closure_error=np.max(np.abs(np.array(closure)-1)),
              centre_transmissions=centre,centre_cv=centre.std(axis=0,ddof=1)/centre.mean(axis=0),
              independent_pair_seeds=len(set(seeds)),path_mm=slab_path()))
    return axis,np.array(measurements).reshape(9,9,57,2)

def sample_plane(im, axis, x, y):
    step=axis[1]-axis[0]
    coords=np.array([(y-axis[0])/step,(x-axis[0])/step])
    # Avoid NaN*0 destroying a valid sample exactly on a grid point.
    weight=map_coordinates(np.isfinite(im).astype(float),coords,order=1,mode="constant",cval=0)
    value=map_coordinates(np.nan_to_num(im),coords,order=1,mode="constant",cval=0)
    return np.where(weight>1-1e-8,value,np.nan)

def reconstruct(raw,axis,mode):
    yy,xx=np.meshgrid(XY,XY,indexing="ij")
    ratio=raw[:,:,:56,1]/raw[:,:,:56,0]
    path=slab_path()
    fractions=trajectories() if mode=="ridge" else np.tile(Z/50,(7,1))
    out=[];count=[];log_mu=[]
    for iz,z in enumerate(Z):
        layers=[]; log_layers=[]
        for j,(dx,dy) in enumerate(OFF[:56]):
            f=fractions[j//8,iz]
            sx,sy=xx-f*dx, yy-f*dy
            layers.append(sample_plane(ratio[:,:,j],axis,sx,sy))
            log_layers.append(sample_plane(-np.log(ratio[:,:,j]),axis,sx,sy))
        n=np.isfinite(layers).sum(axis=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",message="All-NaN slice encountered")
            out.append(np.where(n>=28,np.nanmedian(layers,axis=0),np.nan))
            log_mu.append(np.where(n>=28,MUA+np.nanmedian(log_layers,axis=0)/path,np.nan))
        count.append(n)
    rel=np.asarray(out)
    return dict(x=XY,y=XY,z=Z,relative=rel,mu_linear=MUA+(1-rel)/path,
                mu_log=np.asarray(log_mu),support=np.array(count,dtype=np.uint8),shift_fraction=fractions)

def width(values):
    j=int(np.nanargmax(values));level=values[j]/2
    edges=[]
    for direction in (-1,1):
        k=j
        while 0<=k+direction<len(values) and np.isfinite(values[k+direction]) and values[k+direction]>=level:
            k+=direction
        q=k+direction
        if q<0 or q>=len(values) or not np.isfinite(values[q]):
            return None
        edges.append(k+(q-k)*(level-values[k])/(values[q]-values[k]))
    return float(edges[1]-edges[0])

def evaluate(vol,mask=None):
    contrast=1-vol["relative"]
    if mask is not None: contrast=np.where(mask,contrast,np.nan)
    zz,yy,xx=np.meshgrid(Z,XY,XY,indexing="ij")
    scores=[]
    truth_masks=[]
    for obj in load_config()["inclusions"]:
        x,y,z=obj["centre_mm"];truth_masks.append((xx-x)**2+(yy-y)**2+(zz-z)**2<=25)
    for obj,tru in zip(load_config()["inclusions"],truth_masks):
        signed=contrast*np.sign(obj["mua"]-MUA)
        k,j,i=np.unravel_index(np.nanargmax(signed),signed.shape)
        peak=np.array([XY[i],XY[j],Z[k]])
        error=peak-obj["centre_mm"]
        seg=signed>=.5*signed[k,j,i]
        finite=np.isfinite(signed)
        dice=2*(seg&tru).sum()/(seg.sum()+tru.sum())
        bg=finite & ~(truth_masks[0]|truth_masks[1])
        scores.append(dict(target=obj["name"],peak_mm=peak,peak_error_mm=np.linalg.norm(error),
                   z_error_mm=error[2],dice=dice,truth_coverage=(tru&finite).sum()/tru.sum(),
                   fwhm_xyz_mm=[width(signed[k,j,:]),width(signed[k,:,i]),width(signed[:,j,i])],
                   mu_peak=vol["mu_linear"][k,j,i],mu_error_pct=100*(vol["mu_linear"][k,j,i]/obj["mua"]-1),
                   background_rms=float(np.sqrt(np.mean(signed[bg]**2))),
                   outside_truth_fraction=(seg&bg).sum()/seg.sum()))
    return scores

def save_sampling(name, axis,raw):
    np.savez_compressed(RES/f"samples_{name}.npz",source_axis=axis,offset_xy_mm=OFF,energy=raw,
                        valid=np.isfinite(raw).all(axis=-1))
    with (RES/f"samples_{name}.csv").open("w",newline="",encoding="utf-8") as fp:
        w=csv.writer(fp);w.writerow(["sx_mm","sy_mm","j","dx_mm","dy_mm","I0","I","valid","contrast"])
        for j_y,y in enumerate(axis):
            for i_x,x in enumerate(axis):
                for j,(dx,dy) in enumerate(OFF):
                    b,t=raw[j_y,i_x,j]
                    w.writerow((x,y,j,dx,dy,b,t,np.isfinite(b+t),1-t/b if b>0 else np.nan))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--legacy",type=Path)
    args=ap.parse_args()
    RES.mkdir(parents=True,exist_ok=True)
    old=args.legacy
    if old is None:
        old=Path(os.environ["NIR_LEGACY_ROOT"]) if "NIR_LEGACY_ROOT" in os.environ else next(iter(ROOT.parents[1].glob("第一次仿真*/nir_tomography")),None)
    if old and old.exists():
        axis,raw=legacy_measurements(old)
        save_sampling("legacy3",axis,raw)
    else:
        with np.load(RES/"samples_legacy3.npz") as f: axis,raw=f["source_axis"],f["energy"]
    new,wide=wide_measurements()
    groups=[("legacy3",axis,raw),("wide3",new[2:7:2],wide[2:7:2,2:7:2]),
            ("wide5",new[::2],wide[::2,::2]),("wide9",new,wide)]
    scores={}; summaries={}
    for name,axis,raw in groups:
        save_sampling(name,axis,raw)
        summaries[name]={"valid_offset":int(np.isfinite(raw[:,:,:56]).all(axis=-1).sum()),
                         "valid_zero":int(np.isfinite(raw[:,:,56]).all(axis=-1).sum())}
        volumes={}
        for mode in ("line","ridge"):
            vol=reconstruct(raw,axis,mode);volumes[mode]=vol
            np.savez_compressed(RES/f"{name}_{mode}.npz",**vol)
        common=np.isfinite(volumes["line"]["relative"])&np.isfinite(volumes["ridge"]["relative"])
        for mode,vol in volumes.items():
            scores[name+"_"+mode]=evaluate(vol,common)
    write_json(RES/"metrics.json",scores);write_json(RES/"coverage.json",summaries)
    print(summaries)

if __name__=="__main__":
    main()

