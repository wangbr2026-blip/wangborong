"""Fresh MCX simulation for a live demonstration; existing data are not reused."""
import time
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mcx_forward import build_cfg
from common import ROOT,load_config,write_json
import pmcx

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--no-open',action='store_true');args=parser.parse_args()
    c=load_config();stamp=datetime.now().strftime('%Y%m%d_%H%M%S');out=ROOT/'data/live'/stamp;out.mkdir(parents=True,exist_ok=False)
    n=1000000;seed=int(time.time())%2000000000
    print('现场重新发射光子：每条件 1,000,000；三条件合计 3,000,000。',flush=True)
    print('体素 80×80×50，约数 MB 输出；结果位置：',out,flush=True)
    images=[];rows=[]
    for case in ['homogeneous','high_only','two_spheres']:
        cfg=build_cfg(c,case,n,seed);cfg['flog']=0
        start=time.perf_counter();r=pmcx.run(cfg);a=r['dref'][1:-1,1:-1,-1,:].sum(axis=2)
        images.append(a.reshape(20,4,20,4).mean(axis=(1,3)))
        np.savez_compressed(out/(case+'.npz'),transmission=a,fluence=r['flux'][1:-1,1:-1,1:-1,:].sum(axis=3))
        row=dict(case=case,seed=seed,nphoton=n,elapsed_s=time.perf_counter()-start,transmission_fraction=float(a.sum(dtype=np.float64)),stat=r['stat']);rows.append(row)
        print(case,'实算完成；总透射比例',row['transmission_fraction'],flush=True)
    write_json(out/'run_summary.json',rows)
    plt.rcParams['font.family']=['Microsoft YaHei','DejaVu Sans']
    f,axs=plt.subplots(1,3,figsize=(11,4),layout='constrained');mx=max(a.max() for a in images)
    for ax,a,title in zip(axs,images,['均匀背景','高吸收球','两个球']):
        im=ax.imshow(a.T,origin='lower',extent=[-40,40,-40,40],norm=LogNorm(mx*1e-4,mx),cmap='viridis');ax.set(title=title,xlabel='x / mm',ylabel='y / mm')
    f.colorbar(im,ax=axs,label='出射能量面密度 / mm^-2')
    f.suptitle('现场重新计算的 MCX 出口透射图  '+stamp)
    f.savefig(out/'live_transmission.png',dpi=160,bbox_inches='tight');plt.close(f)
    (out/'打开现场结果.html').write_text('<!doctype html><meta charset="utf-8"><title>现场实算结果</title><h1>刚完成的 MCX 蒙特卡罗仿真</h1><p>'+stamp+'；3 × 100 万光子；随机种子 '+str(seed)+'。这是正向仿真，未做逆重建。</p><img style="max-width:100%" src="live_transmission.png"><p><a href="run_summary.json">运行参数与数值</a></p>',encoding='utf-8')
    print('现场结果已保存：',out/'打开现场结果.html',flush=True)
    import os
    if not args.no_open and hasattr(os,'startfile'):os.startfile(out/'打开现场结果.html')

if __name__=='__main__':main()
