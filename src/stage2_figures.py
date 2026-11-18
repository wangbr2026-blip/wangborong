"""Scientific figures, sensor-format scenario and offline interactive 3-D view."""
import csv
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.colors import Normalize
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_matrix
import plotly.graph_objects as go
from common import ROOT, write_json, load_config
from stage2_reconstruct import RAW,RES,XY,Z,OFF,MUA

FIG=RES/"figures"
plt.rcParams.update({"font.sans-serif":["Microsoft YaHei","SimHei","DejaVu Sans"],
                     "axes.unicode_minus":False,"axes.spines.top":False,"axes.spines.right":False,
                     "font.size":10,"figure.facecolor":"white","savefig.dpi":165})
OBJ=load_config()["inclusions"]
def save(fig,name):
    fig.savefig(FIG/name,bbox_inches="tight",pad_inches=.15)
    plt.close(fig)

def sampling():
    old=np.load(RES/"samples_legacy3.npz")
    f,axs=plt.subplots(1,3,figsize=(12,3.9),layout="constrained")
    for m in (-15,0,15):
        for n in (-15,0,15):
            axs[0].plot(n,m,"o",color="#196c78")
            axs[0].annotate(f"S{int((m+15)/15)*3+int((n+15)/15)}",(n,m),xytext=(3,4),textcoords="offset points")
    axs[0].set(title="入口面  z = 0 mm",xlim=(-40,40),ylim=(-40,40))
    for j,xy in enumerate(OFF):
        axs[1].add_patch(Rectangle(xy-1.5,3,3,facecolor="#20918c",edgecolor="#124252",alpha=.5))
    axs[1].set(title="出口面  S4 对应的 56 个窗口及零偏移",xlim=(-30,30),ylim=(-30,30))
    grid=old["valid"][:,:,:56].sum(axis=-1)
    im=axs[2].imshow(grid,origin="lower",extent=[-22.5,22.5,-22.5,22.5],vmin=50,vmax=56,cmap="Blues")
    for i,y in enumerate(old["source_axis"]):
        for j,x in enumerate(old["source_axis"]):
            axs[2].text(x,y,str(grid[i,j]),ha="center",va="center",color="white" if grid[i,j]>=54 else "black")
    axs[2].set(title="旧平板每个源位的有效窗口数")
    for a in axs: a.set(xlabel="x / mm",ylabel="y / mm",aspect="equal")
    save(f,"01_geometry.png")
    oldroot=Path(os.environ["NIR_LEGACY_ROOT"]) if "NIR_LEGACY_ROOT" in os.environ else next(iter(ROOT.parents[1].glob("第一次仿真*/nir_tomography")),None)
    if oldroot is not None:
        f,axs=plt.subplots(3,6,figsize=(12,6),layout="constrained")
        maps=[]
        for i in range(9):
            for case in ("homogeneous","two_spheres"):
                with np.load(oldroot/f"data/mcx/scan_{i:02}_{case}.npz") as raw: maps.append(raw["transmission"])
        vmax=max(x.max() for x in maps)
        for i,(ax,im) in enumerate(zip(axs.ravel(),maps)):
            p=ax.imshow(im.T,origin="lower",extent=(-40,40,-40,40),cmap="inferno",vmin=0,vmax=vmax)
            ax.set_title(f"S{i//2} · "+("双球" if i%2 else "均匀"),fontsize=9)
            ax.set_xticks([-30,0,30]);ax.set_yticks([-30,0,30]);ax.tick_params(labelsize=7)
        f.colorbar(p,ax=axs,shrink=.7,label="归一化出射能量 / mm$^{-2}$")
        f.supxlabel("x / mm");f.supylabel("y / mm");save(f,"02_legacy_exit18.png")
    for name in ("legacy3","wide9"):
        dat=np.load(RES/f"samples_{name}.npz")
        ratio=1-dat["energy"][:,:,:,1]/dat["energy"][:,:,:,0]
        f,axs=plt.subplots(7,8,figsize=(12,11),layout="constrained")
        lim=np.nanmax(abs(ratio[:,:,:56]))
        for j,ax in enumerate(axs.flat):
            pc=ax.imshow(ratio[:,:,j]*100,origin="lower",cmap="RdBu_r",vmin=-lim*100,vmax=lim*100,interpolation="nearest")
            ax.set_xticks([]);ax.set_yticks([])
            if j<8: ax.set_title(f"{45*j}°")
            if j%8==0: ax.set_ylabel(f"{np.linalg.norm(OFF[j]):.1f} mm")
        f.colorbar(pc,ax=axs,shrink=.65,label="透射对比 (1 − I/$I_0$) / %")
        f.suptitle(("旧板 3 × 3" if name=="legacy3" else "扩板 9 × 9")+" 光源网格上的 56 张偏移图；白格表示缺测")
        save(f,f"03_offsets_{name}.png")
    dat=np.load(RAW/"pair_40_r0.npz")
    f,axs=plt.subplots(1,3,figsize=(11,3.6),layout="constrained")
    fl=dat["fluence"][1]
    for ax,im,title,extent in zip(axs,[fl[:,60,:].T, dat["transmission"][1].T,np.load(RES/"samples_wide9.npz")["energy"][:,:,56,1]],
              ["体内通量对数剖面  y = 0.5 mm","出口面能量密度  z = 50 mm","零偏移积分信号  9 × 9 源位"],
              [(-60,60,0,50),(-60,60,-60,60),(-33.75,33.75,-33.75,33.75)]):
        arr=np.log10(np.maximum(im,1e-15)) if ax==axs[0] else im
        pc=ax.imshow(arr,origin="lower",extent=extent,cmap="magma",aspect="auto")
        f.colorbar(pc,ax=ax,shrink=.65,format="%.1g");ax.set_title(title,fontsize=10);ax.set_xlabel("x / mm")
    axs[0].set_ylabel("z / mm");axs[1].set_ylabel("y / mm");axs[2].set_ylabel("源 y / mm")
    save(f,"04_forward.png")

def coords_volume():
    zz,yy,xx=np.meshgrid(Z,XY,XY,indexing="ij")
    return xx,yy,zz

def voxel_shell(mask):
    """Triangulate exposed faces of the selected 1-mm reconstruction voxels."""
    padded=np.pad(mask,1)
    vertices=[]; triangles=[]; start=0
    for dim in range(3):
        tangents=[i for i in range(3) if i!=dim]
        for side in (-1,1):
            section=[slice(1,-1)]*3
            section[dim]=slice(None,-2) if side==-1 else slice(2,None)
            centers=np.argwhere(mask & ~padded[tuple(section)]).astype(float)
            if not len(centers): continue
            centers+=np.array([Z[0],XY[0],XY[0]])
            delta=np.zeros((4,3))
            delta[:,dim]=side*.5
            delta[:,tangents]=np.array([[-.5,-.5],[-.5,.5],[.5,.5],[.5,-.5]])
            # Work in [z,y,x], then return [x,y,z] for the renderer.
            vertices.append((centers[:,None,:]+delta[None,:,:]).reshape(-1,3)[:,::-1])
            base=start+4*np.arange(len(centers))[:,None]
            triangles.extend([base+np.array([0,1,2]),base+np.array([0,2,3])])
            start+=4*len(centers)
    return np.concatenate(vertices).T,np.concatenate(triangles).T

def slices(v):
    f,axs=plt.subplots(2,3,figsize=(11,6.9),layout="constrained")
    a=v["mu_linear"];lim=np.nanmax(np.abs(a-MUA))
    for row,obj in enumerate(OBJ):
        x,y,z=obj["centre_mm"]; ix=np.argmin(abs(XY-x));iy=np.argmin(abs(XY-y));iz=np.argmin(abs(Z-z))
        for ax,arr,ext,center,plane in zip(axs[row],
                [a[iz],a[:,iy,:],a[:,:,ix]],[(-24.5,24.5,-24.5,24.5),(-24.5,24.5,4.5,45.5),(-24.5,24.5,4.5,45.5)],
                [(x,y),(x,z),(y,z)],[f"XY   z={Z[iz]:.0f}",f"XZ   y={XY[iy]:.0f}",f"YZ   x={XY[ix]:.0f}"]):
            pc=ax.imshow(arr,origin="lower",extent=ext,cmap="RdBu_r",vmin=MUA-lim,vmax=MUA+lim)
            ax.add_patch(Circle(center,5,fill=False,color="black",ls="--",lw=1.2))
            ax.set_title(f"{obj['name']}    {plane} mm");ax.set_xlabel(plane[0].lower()+" / mm");ax.set_ylabel(plane[1].lower()+" / mm")
    f.colorbar(pc,ax=axs,shrink=.75,label="表观吸收系数 / mm$^{-1}$")
    save(f,"06_orthogonal.png")
    f,axs=plt.subplots(2,3,figsize=(10,6.2),layout="constrained")
    for ax,z in zip(axs.flat,[5,12,20,28,36,44]):
        pc=ax.imshow(v["relative"][int(z-5)],origin="lower",extent=(-24.5,24.5,-24.5,24.5),cmap="RdBu",
                     vmin=np.nanmin(v["relative"]),vmax=np.nanmax(v["relative"]))
        ax.set(title=f"z = {z} mm",xlabel="x / mm",ylabel="y / mm")
    f.colorbar(pc,ax=axs,label="重建相对光强 I/$I_0$");save(f,"05_relative.png")

def display3d(v):
    xx,yy,zz=coords_volume()
    with np.load(RES/"wide9_line.npz") as baseline:
        response=np.where(np.isfinite(baseline["relative"]),1-v["relative"],np.nan)
    f=plt.figure(figsize=(10,4.8),layout="constrained")
    ax=f.add_subplot(121,projection="3d"); bx=f.add_subplot(122,projection="3d")
    html=go.Figure()
    for s,obj,col in zip((1,-1),OBJ,("#bb4038","#216ba5")):
        x,y,z=obj["centre_mm"]
        a,b=np.meshgrid(np.linspace(0,2*np.pi,35),np.linspace(0,np.pi,18))
        sphere=[x+5*np.cos(a)*np.sin(b),y+5*np.sin(a)*np.sin(b),z+5*np.cos(b)]
        ax.plot_surface(*sphere,color=col,alpha=.75)
        bx.plot_wireframe(*sphere,rstride=3,cstride=4,linewidth=.55,color="black")
        html.add_trace(go.Surface(x=sphere[0],y=sphere[1],z=sphere[2],name=obj["name"]+" 真值",showscale=False,
                      colorscale=[[0,"#303030"],[1,"#303030"]],opacity=.28,showlegend=True))
        cut=.5*np.nanmax(s*response)
        selected=(s*response>=cut)
        # Surface of a thresholded grid; points at mask boundaries remain visible.
        from scipy.ndimage import binary_erosion
        shell=selected & ~binary_erosion(selected)
        bx.scatter(xx[shell],yy[shell],zz[shell],s=2,color=col,alpha=.17,depthshade=False)
        vertices,faces=voxel_shell(selected)
        html.add_trace(go.Mesh3d(x=vertices[0],y=vertices[1],z=vertices[2],
                       i=faces[0],j=faces[1],k=faces[2],
                       opacity=.45,name=obj["name"]+" 半峰壳层",showlegend=True,color=col,
                       hovertemplate="x %{x:.1f} mm · y %{y:.1f} mm · z %{z:.1f} mm<extra>重建边界</extra>"))
    for axis,title in [(ax,"输入的双球真值"),(bx,"重建半峰壳层及真值线框")]:
        axis.set(xlim=(-24,24),ylim=(-24,24),zlim=(5,45),xlabel="x / mm",ylabel="y / mm",zlabel="z / mm",title=title)
        axis.set_box_aspect((48,48,40));axis.view_init(23,-63)
    save(f,"07_overlay.png")
    forward=np.load(RAW/"pair_40_r0.npz")["fluence"][1,::4,::4,::2]
    gx,gy,gz=np.meshgrid(np.arange(-59.5,60,4),np.arange(-59.5,60,4),np.arange(.5,50,2),indexing="ij")
    field=np.log10(np.maximum(forward,1e-15))
    html.add_trace(go.Volume(x=gx.ravel(),y=gy.ravel(),z=gz.ravel(),value=field.ravel(),
                   isomin=float(np.percentile(field,45)),isomax=float(np.percentile(field,99.5)),
                   opacity=.1,surface_count=16,colorscale="Inferno",visible=False,
                   colorbar_title="log₁₀ 通量",name="中心光源正向传播"))
    buttons=[dict(label="双球重建与真值",method="update",args=[{"visible":[True,True,True,True,False]}]),
             dict(label="中心照明的 MCX 通量体",method="update",args=[{"visible":[False,False,False,False,True]}])]
    html.update_layout(title="散射平板三维成像 · 双球重建与中心照明",
                       updatemenus=[dict(buttons=buttons,x=0,y=1.09,xanchor="left",yanchor="top")],
                       margin=dict(t=130),
                       scene=dict(aspectmode="data",xaxis_title="x / mm",yaxis_title="y / mm",zaxis_title="z / mm"))
    html.write_html(str(RES/"三维重建.html"),include_plotlyjs=True)

def comparison():
    metrics=json.loads((RES/"metrics.json").read_text(encoding="utf-8"))
    f,axs=plt.subplots(1,3,figsize=(12,3.8),layout="constrained")
    for name,ls in (("line","--"),("ridge","-")):
        v=np.load(RES/f"wide9_{name}.npz")
        for j,obj in enumerate(OBJ):
            x,y,z=obj["centre_mm"]
            i=np.argmin(abs(XY-x));k=np.argmin(abs(XY-y))
            axs[0].plot(Z,(1-v["relative"][:,k,i])*100,ls=ls,color=["#b94137","#206da4"][j],
                        label=f"{obj['name']} "+("几何" if name=="line" else "扩散"))
        axs[1].plot(Z,20*v["shift_fraction"][3],ls=ls,label=("几何" if name=="line" else "扩散"))
    axs[0].set(xlabel="z / mm",ylabel="透射对比 / %",title="过参考球心的轴向剖面");axs[0].legend(fontsize=8)
    axs[1].set(xlabel="z / mm",ylabel="相对光源横移 / mm",title="20 mm 偏移的移位曲线");axs[1].legend()
    for j,obj in enumerate(OBJ):
        axs[2].plot(["旧3×3","新3×3","新5×5","新9×9"],[metrics[n+"_ridge"][j]["peak_error_mm"] for n in
                  ("legacy3","wide3","wide5","wide9")],"o-",label=obj["name"])
    axs[2].set(title="全局极值的定位误差",ylabel="三维距离 / mm");axs[2].legend()
    save(f,"08_profiles.png")

def camera():
    h,w=1026,1282;pitch=120/h
    def overlaps(n):
        left=(np.arange(n)-n/2)*pitch
        voxel=np.arange(-60,60)
        return csr_matrix(np.maximum(0,np.minimum((left+pitch)[:,None],voxel+1)-np.maximum(left[:,None],voxel)))
    wx,wy=overlaps(w),overlaps(h)
    maps=np.load(RAW/"pair_40_r0.npz")["transmission"]
    expectation=[]
    for im in maps:
        sm=gaussian_filter(im.T.astype(float),.65,mode="constant")
        expectation.append((wx@(wy@sm).T).T[::-1])
    scale=1800/np.max(expectation[0])
    rng=np.random.default_rng(20261118)
    f,axs=plt.subplots(1,2,figsize=(8.7,3.7),layout="constrained")
    for name,mean,ax in zip(("background","target"),expectation,axs):
        dn=np.clip(np.rint(64+(rng.poisson(mean*scale)+rng.normal(0,3,mean.shape))/1.5),0,4095).astype(np.uint16)
        Image.fromarray(dn).save(RES/f"synthetic_{name}_Mono12.tiff",compression="tiff_deflate")
        pc=ax.imshow(dn,cmap="gray",vmin=64,vmax=1264);ax.set(xlabel="传感器列 / px",ylabel="传感器行 / px",
            title="均匀背景" if name=="background" else "双球");f.colorbar(pc,ax=ax,shrink=.65,label="模拟 DN")
    f.suptitle("中心照明 · 同一响应尺度下的合成相机帧");save(f,"09_camera.png")
    write_json(RES/"camera.json",dict(model="Basler acA1300-60gm",lens="BL 2.4/16-11",hw=[h,w],
              pixel_um=5.3,object_pixel_mm=pitch,field_xy_mm=[w*pitch,120],magnification=.0053/pitch,
              thin_lens_object_distance_mm=16*(1+pitch/.0053),psf_sigma_object_mm=.65,
              read_noise_e=3,gain_e_per_dn=1.5,offset_dn=64,background_peak_expected_e=1800,
              seed=20261118,format="Mono12 numeric range, stored in uint16 TIFF without bit packing",
              acquisition="synthetic scenario; effective exposure-throughput scale fixed from uniform background; no hardware frames"))
def main():
    FIG.mkdir(parents=True,exist_ok=True)
    sampling()
    with np.load(RES/"wide9_ridge.npz") as v:
        slices(v);display3d(v)
    comparison();camera()
if __name__=="__main__": main()

