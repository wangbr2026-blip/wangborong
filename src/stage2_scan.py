"""Paired MCX scans on the wider finite slab. Keep one atomic file per pair."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pmcx
from common import ROOT, load_config, make_model, write_json
from mcx_forward import build_cfg

OUT = ROOT / "data" / "stage2"
AXIS = np.linspace(-30, 30, 9)

def simulate(c, xy, index, repeat):
    destination = OUT / f"pair_{index:02}_r{repeat}.npz"
    old_grid = all(v in (-15, 0, 15) for v in xy)
    if old_grid and repeat == 0:
        seed = 20261203 + [-15, 0, 15].index(xy[0]) + 3 * [-15, 0, 15].index(xy[1])
    else:
        seed = 20261118 + 200 + index + repeat * 10000
    configs = [build_cfg(c, case, c["scan_nphoton"], seed, xy)
               for case in ("homogeneous", "two_spheres")]
    serial = [{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in cfg.items() if k!="vol"}
              for cfg in configs]
    serial[0]["labels_sha256_F"] = hashlib.sha256(configs[0]["vol"].tobytes(order="F")).hexdigest()
    signature = json.dumps(serial, sort_keys=True)
    if destination.exists():
        with np.load(destination) as data:
            if data["inputs"].item() != signature:
                raise ValueError(f"Inputs changed for {destination}; select a new output location.")
        return
    maps, energy, elapsed, fluences = [], [], [], []
    for cfg in configs:
        start = time.perf_counter()
        out = pmcx.run(cfg)
        elapsed.append(time.perf_counter()-start)
        boundary = np.asarray(out["dref"]).sum(axis=3)
        maps.append(boundary[1:-1,1:-1,-1].copy())
        energy.append(float(boundary.sum(dtype=np.float64)*c["voxel_mm"]**2
                            + out["stat"]["energyabs"]/out["stat"]["energytot"]))
        if index == 40 and repeat == 0:
            fluences.append(out["flux"][1:-1,1:-1,1:-1,:].sum(axis=3))
    if not np.isfinite(maps).all() or np.max(np.abs(np.array(energy)-1))>0.001:
        raise RuntimeError(f"Nonfinite exit map or energy imbalance at {xy}: {energy}")
    temp = destination.with_suffix(".tmp")
    with temp.open("wb") as fp:
        np.savez_compressed(fp, transmission=maps, energy_closure=energy, elapsed_s=elapsed,
                            source_xy_mm=xy, seed=seed, inputs=signature, fluence=fluences)
    os.replace(temp,destination)
    print(f"Saved {destination.name} {xy}, {sum(elapsed):.1f} s", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat-centre", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    c=load_config()
    c.update(domain_mm=[120,120,50], scan_grid_mm=AXIS.tolist(), name="wide_slab_9x9",
             date="2026-11-18", lateral_domain_provenance="120 mm finite domain; scan extent +/-30 mm")
    write_json(OUT/"config.json",c)
    write_json(OUT/"environment.json",dict(pmcx=pmcx.__version__, gpu=pmcx.gpuinfo(), numpy=np.__version__))
    order=sorted(range(81),key=lambda i:(i!=40, bool((i//9)%2 or (i%9)%2),i))
    for i in order:
        simulate(c,(AXIS[i%9],AXIS[i//9]),i,0)
    for r in range(1,args.repeat_centre+1):
        simulate(c,(0.,0.),40,r)

if __name__=="__main__":
    main()

