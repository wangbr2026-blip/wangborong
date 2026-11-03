"""Scientific figures generated entirely from saved numerical arrays."""
import json, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.patches import Circle
from common import ROOT,load_config,make_model,write_json

OUT=ROOT/'results'
plt.rcParams.update({'font.family':['Microsoft YaHei','DejaVu Sans'],'font.size':10,'axes.unicode_minus':False,'figure.dpi':130,'savefig.dpi':180})
CASES=['homogeneous','high_only','two_spheres']
NAMES=['均匀背景','只改变 I1 的吸收','同时加入 I1 和 I2']

def save(fig,name):
    fig.savefig(OUT/name,bbox_inches='tight',facecolor='white');plt.close(fig)

def bin_image(a,b=4):
    return a.reshape(a.shape[0]//b,b,a.shape[1]//b,b).mean(axis=(1,3))

def main():
    c=load_config();labels,mua,axes=make_model(c)
    records={case:np.load(ROOT/f'data/mcx/centre_{case}_10000000.npz') for case in CASES}
    runs=[json.loads(p.read_text(encoding='utf-8')) for p in (ROOT/'data/mcx').glob('*.json') if not p.stem.endswith('_input')]
    summary={r['case']:r for r in runs if r['tag'].startswith('centre_') and r['nphoton']==10000000}
    # Actual 3-D ground truth, not an illustration of an assumed recovery.
    f=plt.figure(figsize=(10,6));ax=f.add_subplot(projection='3d')
    for x in [-40,40]:
        for y in [-40,40]:ax.plot([x,x],[y,y],[0,50],c='#81909c',lw=.8)
    for z in [0,50]:
        ax.plot([-40,40,40,-40,-40],[-40,-40,40,40,-40],[z]*5,c='#81909c',lw=.8)
    u,v=np.mgrid[0:2*np.pi:40j,0:np.pi:22j]
    for inc,col in zip(c['inclusions'],['#d65f43','#4586bb']):
        x,y,z=inc['centre_mm'];r=inc['diameter_mm']/2
        ax.plot_surface(x+r*np.cos(u)*np.sin(v),y+r*np.sin(u)*np.sin(v),z+r*np.cos(v),color=col,alpha=.8,linewidth=0)
        ax.text(x+6,y,z,f"{inc['name']}\n{inc['centre_mm']} mm\nμa={inc['mua']*10:g} cm⁻¹",fontsize=9)
    ax.scatter([0],[0],[0],marker='*',s=100,c='#111111',label='入射点源')
    ax.set(xlabel='x / mm',ylabel='y / mm',zlabel='z / mm',xlim=(-40,40),ylim=(-40,40),zlim=(50,0))
    ax.set_box_aspect((80,80,50));ax.view_init(22,-65)
    f.suptitle('已建立的体素模型真值  80 × 80 × 50 mm',fontsize=15)
    f.text(.08,.02,'NumPy 数据；1 mm 体素。横向尺寸为本次自选测试域；此图不是重建结果。',fontsize=10)
    save(f,'01_phantom_3d.png')
    f,axs=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for j,inc in enumerate(c['inclusions']):
        zidx=int(np.argmin(abs(axes[2]-inc['centre_mm'][2])))
        yidx=int(np.argmin(abs(axes[1]-inc['centre_mm'][1])))
        im=axs[0,j].imshow(mua[:,:,zidx].T*10,origin='lower',extent=[-40,40,-40,40],vmin=.05,vmax=.2,cmap='coolwarm',interpolation='nearest')
        axs[0,j].set(title=f"XY 真值  z={axes[2][zidx]:g} mm",xlabel='x / mm',ylabel='y / mm')
        axs[1,j].imshow(mua[:,yidx,:].T*10,origin='lower',extent=[-40,40,0,50],vmin=.05,vmax=.2,cmap='coolwarm',interpolation='nearest',aspect='equal')
        axs[1,j].set(title=f"XZ 真值  y={axes[1][yidx]:g} mm",xlabel='x / mm',ylabel='z / mm',ylim=(50,0))
    f.colorbar(im,ax=axs,label='已知吸收系数 μa / cm⁻¹',shrink=.75)
    f.suptitle('两个目标的不同深度切片  所有子图共用物理色标',fontsize=14)
    save(f,'02_phantom_slices.png')
    f,axs=plt.subplots(1,2,figsize=(10,4.3),layout='constrained')
    for j,inc in enumerate(c['inclusions']):
        xi=int(np.argmin(abs(axes[0]-inc['centre_mm'][0])))
        im=axs[j].imshow(mua[xi,:,:].T*10,origin='lower',extent=[-40,40,0,50],vmin=.05,vmax=.2,cmap='coolwarm',interpolation='nearest',aspect='equal')
        axs[j].set(title=f'YZ 真值  x={axes[0][xi]:g} mm',xlabel='y / mm',ylabel='z / mm',ylim=(50,0))
    f.colorbar(im,ax=axs,label='已知 μa / cm⁻¹')
    f.suptitle('YZ 方向补充切片  按真实目标所在横向位置取面')
    save(f,'02b_phantom_yz.png')
    # Fluence is integrated over 0-10 ns and normalised to launched unit energy.
    vmax=max(float(records[k]['fluence'].max()) for k in CASES)
    norm=LogNorm(vmin=1e-8,vmax=max(1e-4,vmax))
    f,axs=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for row,inc in enumerate(c['inclusions']):
        yi=int(np.argmin(abs(axes[1]-inc['centre_mm'][1])))
        for j,case in enumerate(CASES):
            im=axs[row,j].imshow(records[case]['fluence'][:,yi,:].T,origin='lower',extent=[-40,40,0,50],norm=norm,cmap='magma',aspect='equal')
            axs[row,j].set(title=NAMES[j]+f"\ny={axes[1][yi]:g} mm",xlabel='x / mm',ylabel='z / mm',ylim=(50,0))
            x,y,z=inc['centre_mm'];rr=np.sqrt(25-(axes[1][yi]-y)**2)
            axs[row,j].add_patch(Circle((x,z),rr,fill=False,ec='cyan',lw=1))
    f.colorbar(im,ax=axs,label='体内 fluence / mm⁻²（每单位发射能量）',shrink=.8)
    f.suptitle('MCX 实际光传播结果  每种条件 10⁷ 光子  0—10 ns 积分',fontsize=14)
    save(f,'03_mcx_internal_fluence.png')
    images=[bin_image(records[k]['transmission']) for k in CASES]
    top=max(a.max() for a in images)
    diffs=[images[1]-images[0],images[2]-images[1],images[2]-images[0]]
    lim=max(abs(a).max() for a in diffs)
    f,axs=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    for j in range(3):
        im=axs[0,j].imshow(images[j].T,origin='lower',extent=[-40,40,-40,40],norm=LogNorm(max(top*1e-4,1e-12),top),cmap='viridis')
        axs[0,j].set(title=NAMES[j],xlabel='x / mm',ylabel='y / mm')
        di=axs[1,j].imshow(diffs[j].T,origin='lower',extent=[-40,40,-40,40],vmin=-lim,vmax=lim,cmap='RdBu_r')
        axs[1,j].set(title=['I1 引起的变化：单球 − 背景','I2 的增量变化：双球 − 单球','总变化：双球 − 背景'][j],xlabel='x / mm',ylabel='y / mm')
    f.colorbar(im,ax=axs[0,:],label='出口面能量面密度 / mm⁻²',shrink=.85)
    f.colorbar(di,ax=axs[1,:],label='能量面密度差 / mm⁻²',shrink=.85)
    f.suptitle('出口边界透射对照  同一光源、同一光子数、统一色标',fontsize=14)
    f.text(.08,-.02,'由空气壳层 dref 提取；4×4 mm 面元取平均以降低噪声。全出射角积分，未模拟镜头收光和相机灰度。',fontsize=9)
    save(f,'04_exit_transmission_comparison.png')
    # Noise study: four independent random-seed groups, raw measurements retained.
    noise={};f,axs=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    for n in [100000,1000000]:
        rr=[r for r in runs if r['tag'].startswith(f'variance_N{n}_')]
        values=np.array([r['transmission_fraction'] for r in sorted(rr,key=lambda r:r['seed'])])
        noise[str(n)]={'values':values,'mean':values.mean(),'sample_sd':values.std(ddof=1),'cv':values.std(ddof=1)/values.mean(),'independent_seeds':4}
        axs[0].plot(np.arange(1,5),values*1e5,'o-',label=f'N={n:g}')
    axs[0].set(xlabel='独立随机种子组',ylabel='总透射比例 × 10⁵',title='光子数增加后的重复波动');axs[0].legend()
    axs[1].bar(['10⁵ 光子','10⁶ 光子'],[noise[str(n)]['cv']*100 for n in [100000,1000000]],color=['#c7a36c','#397c92'])
    axs[1].set(ylabel='总透射比例的样本 CV / %',title='每档 4 组随机种子')
    vals=[summary[k]['transmission_fraction'] for k in CASES]
    axs[2].bar(['背景','只含 I1','I1 + I2'],np.array(vals)*1e5,color=['#8a969e','#cb6f52','#598bb0'])
    axs[2].set(ylabel='总透射比例 × 10⁵',title='中心源 10⁷ 光子对照')
    save(f,'05_noise_and_transmission.png')
    # 3 x 3 source positions, actual two-inclusion forward maps.
    f,axs=plt.subplots(3,3,figsize=(10,9),layout='constrained')
    scan=[]
    for i,(x,y) in enumerate((x,y) for y in c['scan_grid_mm'] for x in c['scan_grid_mm']):
        dat=np.load(ROOT/f'data/mcx/scan_{i:02}_two_spheres.npz');a=bin_image(dat['transmission'])
        scan.append(a)
    mx=max(a.max() for a in scan)
    for i,ax in enumerate(axs.ravel()):
        x,y=np.load(ROOT/f'data/mcx/scan_{i:02}_two_spheres.npz')['source_xy_mm']
        im=ax.imshow(scan[i].T,origin='lower',extent=[-40,40,-40,40],norm=LogNorm(mx*1e-4,mx),cmap='viridis')
        ax.scatter([x],[y],s=35,marker='+',color='white')
        ax.set(title=f'S{i}  入射源 ({x:g}, {y:g}) mm',xlabel='出口面 x / mm',ylabel='出口面 y / mm')
    f.colorbar(im,ax=axs,label='出口面能量面密度 / mm⁻²',shrink=.8)
    f.suptitle('3 × 3 光源扫描实算结果  每位置 10⁷ 光子',fontsize=14)
    save(f,'06_nine_source_scan.png')
    vd=np.load(ROOT/'data/virtual_detector_samples.npz');offs=vd['offset_xy_mm']
    f,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    axs[0].scatter(offs[:,0],offs[:,1],c=np.repeat(np.arange(7),8),cmap='viridis',s=25)
    axs[0].scatter([0],[0],c='red',marker='*',s=80)
    axs[0].set(xlabel='相对光源 dx / mm',ylabel='相对光源 dy / mm',title='7 环 × 8 角 = 56 个虚拟探测位置',aspect='equal')
    values=vd['intensity'];relative=np.divide(values[:,:,1]-values[:,:,0],values[:,:,0],where=values[:,:,0]>0,out=np.full(values.shape[:2],np.nan))*100
    lim=max(.1,float(np.nanmax(abs(relative))))
    cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#bcbcbc')
    im=axs[1].imshow(relative,origin='lower',cmap=cmap,aspect='auto',vmin=-lim,vmax=lim)
    axs[1].set(xlabel='虚拟探测器编号',ylabel='扫描光源编号',title='实算信号变化 / %（灰白缺口为无效采样）')
    f.colorbar(im,ax=axs[1],label='(双球 − 背景) / 背景 / %')
    save(f,'07_virtual_detectors.png')
    # Store machine-readable metrics and one-line-per-run table.
    result=dict(date='2026-11-03',runs=len(runs),photons_total=sum(r['nphoton'] for r in runs),
        wall_seconds=sum(r['elapsed_s'] for r in runs),kernel_seconds=sum(r['stat']['runtime']/1000 for r in runs),
        max_energy_closure_error=max(abs(r['energy_closure']-1) for r in runs),single=summary,noise=noise,
        high_vs_background_pct=(vals[1]/vals[0]-1)*100,two_vs_high_pct=(vals[2]/vals[1]-1)*100,
        two_vs_background_pct=(vals[2]/vals[0]-1)*100,
        virtual_detector_pairs=504,valid_virtual_detector_pairs=int(vd['valid'].sum()),
        reconstruction_status='not run; sparse pilot source grid, MCX forward and detector extraction completed')
    write_json(OUT/'summary.json',result)
    with (OUT/'run_summary.csv').open('w',encoding='utf-8-sig',newline='') as fp:
        w=csv.writer(fp);w.writerow(['tag','case','photons','seed','source_x_mm','source_y_mm','wall_seconds','kernel_ms','T_fraction','R_fraction','A_fraction','energy_closure'])
        for r in runs:w.writerow([r['tag'],r['case'],r['nphoton'],r['seed'],*r['source_xy_mm'],r['elapsed_s'],r['stat']['runtime'],r['transmission_fraction'],r['reflection_fraction'],r['absorbed_fraction'],r['energy_closure']])
    print('Saved seven scientific figures and summary.json',flush=True)

if __name__=='__main__':main()
