# 第二次仿真　从出口透射图到三维吸收响应

日期为 2026 年 11 月 18 日。本轮从第一次工程的提交 c5db39d 接续。正式运行入口为 stage2.py。原工程 run.py、原始模型与首轮 18 张扫描图保留原样。

## 如何复算

在本机工程目录中运行 `.\run_stage2.ps1 check` 可直接检查现有成果。将最后一个参数换为 scan、reconstruct 或 figures 可分别执行扫描、重建和制图。入口会定位本机已安装的计算环境；自选解释器时设置 NIR_PYTHON。

使用 Python 3.12、PMCX 0.8.1、NumPy、SciPy、Matplotlib、Pillow、Plotly。扫描需要可运行 PMCX 的 NVIDIA GPU。本机选择第一次仿真配套的“运行环境”目录；其他电脑可在自己的 Python 环境安装依赖，或用 NIR_PACKAGES 指定包含 packages 和 mcx_packages 的离线目录。

```powershell
python stage2.py scan
python stage2.py reconstruct --legacy "第一次工程的 nir_tomography 绝对路径"
python stage2.py figures
python stage2.py check
```

在当前课题目录中，第二步和制图会自动定位同级第一次仿真。旧环境搬迁到其他位置时，可设置 NIR_LEGACY_ROOT 指向第一次工程，或将现有 results/stage2/samples_legacy3.npz 和原始旧图概览一起归档。首次独立安装先运行旧工程的 model、mcx_forward --stage scan 和 virtual_detectors，或取得原始 18 张扫描图及旧采样表。各批光子量较大，scan 会核对输入并复用已完成的扩板配对。结果明细在本地，公开仓库保存代码、实验口径和数字汇总。

## 数组和坐标

- data/stage2/pair_00_r0.npz 至 pair_80_r0.npz 保留 81 个位置的数据，编号为 9×行+列，行沿 y 增大。中心为 40。r1 至 r3 是中心的额外独立重复。transmission 的轴为条件、x、y，条件顺序为均匀、双球。中心额外保存体内通量。每个文件含输入、种子、能量闭合和分条件耗时。
- results/stage2/samples_*.npz 的 energy 轴为源 y、源 x、偏移、条件。0–55 号为七环八角，56 号为零偏移。CSV 的 I0 为均匀背景，I 为双球。虚拟窗口按原程序吸附到最近的面元中心，双精度求和。旧数据中12个越界窗口保留NaN。
- *_line.npz 与 *_ridge.npz 中 relative、mu_linear、mu_log、support 的轴为 z、y、x。坐标 x,y = −24…24 mm，z = 5…45 mm；每步1 mm。support 为实际参与的非零偏移数，低于28的体素不进入评价。零偏移单独供诊断。
- 偏移图按光源坐标编排。某一深度的目标横坐标为源坐标加 fρ，将图像反向采样到目标网格时取源坐标 x−fρ。直线 f=z/L；若改以出口探测器为坐标原点，等价的位移幅度为 (1−f)ρ。扩散模型使用外推边界无限横向平板的连续波源场与互易探测场乘积，求其横向极大值。本轮以点探测中心近似窗口；正向数据仍对完整窗口积分。
- 线性结果使用 μa,0+(1−I/I0)/ℓ；对数结果先对每通道求 −ln(I/I0)，再插值、融合。ℓ 由10 mm已知直径的等效平板平均光程计算。配对背景参考与作者使用的位置平均参考不同。绝对数值受尺寸先验和光程近似影响。
- metrics.json 在两种移位共同有效的体素上按正负全局极值定位。Dice 使用各自符号峰值50%的全部体素；真值按完整球离散，truth_coverage 给出可评价比例。FWHM 沿峰值所在轴线、用相邻取样点线性插值得到；null 表示半高一侧未在有效域内闭合。outside_truth_fraction 为阈值体素中位于两球以外的比例，background_rms 是真值两球外的相对强度对比均方根。
- figures 为静态图件，三维重建.html 包含离线交互所需的全部资源。模拟相机帧为16位TIFF容器内的12位数值，几何与假设在 camera.json。scan_summary.json、run_index.csv、wide_inputs.csv 和 legacy_inputs.csv 分别记录批次、逐次统计和文件校验值。

后续版本按[阶段图件与结果约定](阶段图件与结果约定.md)出图。方法参考 [Carbone 等](https://doi.org/10.1088/2057-1976/abc633)、[Contini 等](https://doi.org/10.1364/AO.36.004587)、[MCX](https://doi.org/10.1364/OE.17.020178) 及 [Basler 型号页](https://docs.baslerweb.com/aca1300-60gm)。

