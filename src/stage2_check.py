"""Regression checks on sign, coordinate origin, support, data and exported figures."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from stage2_reconstruct import RES,RAW,XY,Z,OFF,reconstruct,sample_plane,slab_path

def main():
    cov=json.loads((RES/"coverage.json").read_text(encoding="utf-8"))
    assert [cov[k]["valid_offset"] for k in ("legacy3","wide3","wide5","wide9")]==[492,504,1400,4536]
    axis=np.arange(-30.,31.,2.)
    y,x=np.meshgrid(axis,axis,indexing="ij")
    raw=np.ones((len(axis),len(axis),57,2))
    centre=(7.,-9.,15.)
    for j,(dx,dy) in enumerate(OFF):
        distance=(x+dx*centre[2]/50-centre[0])**2+(y+dy*centre[2]/50-centre[1])**2
        raw[:,:,j,1]-=.15*np.exp(-distance/32)
    vol=reconstruct(raw,axis,"line")
    k,j,i=np.unravel_index(np.nanargmin(vol["relative"]),vol["relative"].shape)
    assert np.linalg.norm(np.array([XY[i],XY[j],Z[k]])-centre)<=2
    raw[...,1]=1
    flat=reconstruct(raw,axis,"line")
    np.testing.assert_allclose(flat["mu_linear"][np.isfinite(flat["mu_linear"])],.01)
    assert np.isfinite(slab_path()) and slab_path()>10
    for name in cov:
        for mode in ("line","ridge"):
            with np.load(RES/f"{name}_{mode}.npz") as v:
                assert v["relative"].shape==(41,49,49)
                assert np.array_equal(np.isfinite(v["relative"]),v["support"]>=28)
                assert np.all((v["support"]>=0)&(v["support"]<=56))
    for frame in ("background","target"):
        img=np.asarray(Image.open(RES/f"synthetic_{frame}_Mono12.tiff"))
        assert img.shape==(1026,1282) and img.dtype==np.uint16 and img.max()<=4095
    for filename in ("01_geometry.png","03_offsets_wide9.png","03_offsets_legacy3.png","04_forward.png",
                     "05_relative.png","06_orthogonal.png","07_overlay.png","08_profiles.png","09_camera.png"):
        with Image.open(RES/"figures"/filename) as figure: figure.verify()
    assert (RES/"三维重建.html").stat().st_size>100_000
    print(f"PASS: source coverage, signed single-target geometry {XY[i],XY[j],Z[k]}, null target, path, 8 volumes, frames and figures.")
if __name__=="__main__": main()

