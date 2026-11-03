"""Real MCX transport; export outward boundary energy rather than last-layer fluence.

All cases retain identical voxel labels and change only optical absorption.
Matched seeds/threads reduce noise of differences; independent seeds assess variance.
The data are an ideal all-angle planar exit tally, not Basler ADU or lens radiance.
"""
import argparse, csv, time, sys, platform, hashlib,json
from pathlib import Path
import numpy as np
import pmcx
from common import ROOT, load_config, make_model, property_table, write_json

def build_cfg(c, case, nphoton, seed, source_xy=(0,0), tend=None):
    labels, _, _ = make_model(c)
    h = c['voxel_mm']
    return dict(nphoton=int(nphoton), vol=np.asfortranarray(np.pad(labels, 1)),
        unitinmm=h, issrcfrom0=1,
        srcpos=[(source_xy[0]+c['domain_mm'][0]/2)/h+1, (source_xy[1]+c['domain_mm'][1]/2)/h+1, 1.0001],
        srcdir=[0,0,1], srctype='pencil', prop=property_table(c,case),
        tstart=0, tend=tend or c['time_window_s'], tstep=c['time_gate_s'],
        isreflect=1, isspecular=0, isnormalized=1, issaveref=1,
        outputtype='fluence', issavedet=0, minenergy=0,
        seed=int(seed), autopilot=0, nthread=c['nthread'], nblocksize=c['nblocksize'],
        gpuid=1, flog=2)

def run_one(c, case, nphoton, seed, tag, source_xy=(0,0), tend=None, volume=True):
    dest = ROOT / 'data/mcx'; dest.mkdir(exist_ok=True)
    fn = dest / (tag + '.npz')
    meta = dest / (tag + '.json')
    cfg = build_cfg(c, case, nphoton, seed, source_xy, tend)
    serial = {k:v for k,v in cfg.items() if k!='vol'}
    serial['vol_file'] = 'data/phantom_labels.npy with one-voxel zero padding'
    serial['vol_shape'] = cfg['vol'].shape
    serial['volume_sha256']=hashlib.sha256(cfg['vol'].tobytes(order='F')).hexdigest()
    # Cached data are reusable only for exactly the same input and voxel labels.
    normal=lambda x:json.loads(json.dumps(x,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else list(a)))
    if fn.exists() and meta.exists():
        old=json.loads((dest/(tag+'_input.json')).read_text(encoding='utf-8'))
        if old != normal(serial):
            raise RuntimeError('Existing run has different inputs: '+tag+'. Use a new run tag/output directory; existing measurements were retained.')
        print('Reuse verified completed run',tag,flush=True)
        return json.loads(meta.read_text(encoding='utf-8'))
    write_json(dest/(tag+'_input.json'), serial)
    print(f'RUN {tag}: {nphoton:,} photons, source={source_xy}, padded shape={cfg["vol"].shape}, one float32 output volume=1.4 MB', flush=True)
    t = time.perf_counter()
    r = pmcx.run(cfg)
    elapsed = time.perf_counter()-t
    fluence=np.asarray(r['flux'],dtype=np.float32).sum(axis=3)
    dref=np.asarray(r['dref'],dtype=np.float32).sum(axis=3)
    if not np.isfinite(fluence).all() or not np.isfinite(dref).all(): raise ValueError('Non-finite MCX data')
    trans=dref[1:-1,1:-1,-1].copy()
    refl=dref[1:-1,1:-1,0].copy()
    stat=r['stat']
    a=float(stat['energyabs']/stat['energytot'])
    escaped=float(dref.sum(dtype=np.float64)*c['voxel_mm']**2)
    payload=dict(transmission=trans, reflection=refl, source_xy_mm=np.array(source_xy), voxel_mm=c['voxel_mm'])
    if volume: payload.update(fluence=fluence[1:-1,1:-1,1:-1], boundary_all=dref)
    np.savez_compressed(fn, **payload)
    row=dict(tag=tag, case=case, nphoton=int(nphoton), seed=int(seed), source_xy_mm=source_xy,
        elapsed_s=elapsed, stat=stat, transmission_fraction=float(trans.sum(dtype=np.float64)*c['voxel_mm']**2),
        reflection_fraction=float(refl.sum(dtype=np.float64)*c['voxel_mm']**2),
        escaped_fraction=escaped, absorbed_fraction=a,
        energy_closure=escaped+a, t_end_s=cfg['tend'], output_shape=list(trans.shape),
        transmission_per_gate_fraction=np.asarray(r['dref'])[1:-1,1:-1,-1,:].sum(axis=(0,1),dtype=np.float64)*c['voxel_mm']**2,
        unit='exiting energy / launched energy / mm^2; all outgoing angles',
        data_file=str(fn.relative_to(ROOT)), classification='MCX Monte Carlo forward simulation; not reconstruction')
    write_json(meta,row)
    print('DONE', tag, 'seconds',round(elapsed,3),'T=',row['transmission_fraction'],'A+escape=',row['energy_closure'],flush=True)
    return row

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['pilot','single','scan','convergence'],default='pilot');args=ap.parse_args()
    c=load_config();write_json(ROOT/'logs/mcx_environment.json',dict(python=sys.version, platform=platform.platform(),pmcx_version=pmcx.__version__,mcx_version=pmcx.version(),numpy_version=np.__version__,gpuinfo=pmcx.gpuinfo()))
    rows=[]
    if args.stage=='pilot':
        for n in [100000,1000000]:
            for case in ['homogeneous','high_only','two_spheres']:
                rows.append(run_one(c,case,n,c['seed_base'],f'centre_{case}_{n}'))
    if args.stage=='single':
        for case in ['homogeneous','high_only','two_spheres']:
            rows.append(run_one(c,case,10000000,c['seed_base'],f'centre_{case}_10000000'))
    if args.stage=='scan':
        positions=[(x,y) for y in c['scan_grid_mm'] for x in c['scan_grid_mm']]
        with (ROOT/'data/source_positions.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.writer(f);w.writerow(['source_id','x_mm','y_mm','z_mm']);w.writerows((i,x,y,0) for i,(x,y) in enumerate(positions))
        for i,xy in enumerate(positions):
            for case in ['homogeneous','two_spheres']:
                rows.append(run_one(c,case,c['scan_nphoton'],c['seed_base']+i+100,f'scan_{i:02}_{case}',xy,volume=False))
    if args.stage=='convergence':
        for n in [100000,1000000]:
            for j in range(4):
                rows.append(run_one(c,'homogeneous',n,c['seed_base']+500+j,f'variance_N{n}_seed{j}',volume=False))
        rows.append(run_one(c,'homogeneous',1000000,c['seed_base'],'window_20ns_twogates',tend=2e-8,volume=False))
        rows.append(run_one(c,'homogeneous',1000000,c['seed_base'],'repeat_fixed_seed',volume=False))
    write_json(ROOT/f'results/{args.stage}_runs.json',rows)

if __name__=='__main__': main()
