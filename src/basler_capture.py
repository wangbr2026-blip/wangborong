"""Real-camera acquisition helper. No images are fabricated when no camera is present.

Example: python run.py basler_capture --list
         python run.py basler_capture --serial 123456 --exposure-us 5000 --frames 20 --label dark
Requires Basler pylon/pypylon; run the Viewer to check the network first.
"""
import argparse,time
from datetime import datetime
import numpy as np
from common import ROOT,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--list',action='store_true');p.add_argument('--serial');p.add_argument('--exposure-us',type=float,default=5000);p.add_argument('--frames',type=int,default=20);p.add_argument('--label',default='sample');a=p.parse_args()
    try: from pypylon import pylon,genicam
    except ImportError: raise SystemExit('请先安装 Basler pylon 和 pypylon。当前未采集任何真实图像。下载入口见软件说明。')
    factory=pylon.TlFactory.GetInstance();devices=factory.EnumerateDevices()
    if a.list:
        for d in devices:print(d.GetModelName(),d.GetSerialNumber(),d.GetFriendlyName())
        if not devices:print('当前未枚举到相机；检查供电、网线、IP 和 pylon Viewer。')
        return
    if not a.serial:raise SystemExit('请用 --serial 明确选择相机序列号；先运行 --list。')
    matches=[d for d in devices if d.GetSerialNumber()==a.serial]
    if not matches:raise SystemExit('找不到指定序列号的相机；未采集图像。')
    cam=pylon.InstantCamera(factory.CreateDevice(matches[0]));cam.Open()
    def setnode(names,value):
        for name in names:
            try:
                node=cam.GetNodeMap().GetNode(name)
                if node is not None and genicam.IsWritable(node):node.SetValue(value);return name
            except (genicam.GenericException,AttributeError):pass
        raise RuntimeError('当前相机无法设置节点 '+str(names))
    try:
        setnode(['ExposureAuto'],'Off');setnode(['GainAuto'],'Off')
        setnode(['GainRaw','Gain'],0)
        fmt=cam.PixelFormat.GetSymbolics()
        selected=next((v for v in ['Mono12','Mono12Packed','Mono8'] if v in fmt),None)
        if selected is None:raise RuntimeError('未找到支持的单色原始格式')
        cam.PixelFormat.SetValue(selected)
        setnode(['ExposureTimeAbs','ExposureTime'],a.exposure_us)
        dest=ROOT/'data/hardware'/datetime.now().strftime('%Y%m%d_%H%M%S')
        dest.mkdir(parents=True,exist_ok=False)
        frames=[];ticks=[];cam.StartGrabbingMax(a.frames)
        while cam.IsGrabbing():
            r=cam.RetrieveResult(5000,pylon.TimeoutHandling_ThrowException)
            try:
                if not r.GrabSucceeded():raise RuntimeError(r.GetErrorDescription())
                frames.append(r.Array.copy());ticks.append(time.time())
            finally:r.Release()
        np.save(dest/(a.label+'_raw.npy'),np.stack(frames))
        write_json(dest/'metadata.json',dict(model=cam.GetDeviceInfo().GetModelName(),serial=a.serial,requested_exposure_us=a.exposure_us,pixel_format=selected,label=a.label,frame_times_unix=ticks,shape=np.stack(frames).shape,classification='real camera frames; uncorrected raw grayscale'))
        print('已保存真实原始帧',dest)
    finally:
        if cam.IsGrabbing():cam.StopGrabbing()
        cam.Close()

if __name__=='__main__':main()
