# FastPrep BFC 1.1.0：可靠组织核心偏置场矫正工具包

采样修订：`reliable-core-v2`；RBF 模型修订：`bounded-sigmoid-v1`。默认 **RBF / CUDA**；Gaussian 为显式 CPU 备选。工具包版本表示接口与封装，不表示已经通过全脑或临床验证。

## 1. 之前亮斑的原因

不是 GM/WM 编号接反，也不是单纯显示调窗问题。已证实的主要问题是 **拟合支持域、标签含义和场应用域混在了一起**。

- **标签 0 混合背景与未建模脑实质。** 原映射只用大脑皮层 GM、大脑 WM/胼胝体和 CSF；丘脑、基底节、小脑、脑干等仍在脑掩膜内，但映射为 0。五个 session 的这部分占脑域约 17–19%。不参与拟合不应等同于背景或不需要校正。没有把所有深部结构强塞进皮层 GM。
- **Gaussian 的“最近填充”实际读取镜像位置。** 原函数先找最近有效点，再读取 `2*nearest-outside` 的反射位置，未保证该位置仍可靠。首例冻结旧热点的 257 个供体中，214 个落在 CSF；错误的负 log 残差进入平滑与累积，最终 `10**(-bias)` 把局部增亮。仅修正索引有改善，但不足以消除全部伪影。
- **仅空间腐蚀仍可能保留错误组织样本。** 第一版修复仍有过亮点，其近邻虽标为 WM，原始强度中位数为 65，而全 WM 核心中位数为 102。因此增加同类 log 强度的鲁棒离群点排除；这不保证分割绝对正确，也可能排除强真实偏置。
- **旧 RBF 把标签 0 当作背景，随后又恢复其原强度。** 拟合时未建模脑区被制造为伪暗背景，平滑软标签权重又泄漏进去；应用时标签 0 不随邻域校正，形成与标签边界一致的亮块和接缝。
- **最初“脑内核权重下溢”的猜测未获本例支持。** 保存的控制点复核显示，首次失败尝试的脑内旧 divisor 仍为正，约 0.735–1.639，稳定权重版本与其接近。失败检查当时错误覆盖了未应用的整幅脑外背景。稳定 softmax 是独立数值改进，不能拿它解释全部旧脑内伪影。

Gaussian 修订后仍有小脑/脑干局部过校正，当前真实首例候选不通过 QC。逐轮外推并累积残差可能影响缺乏可靠样本的区域，但尚未完成该机制的独立消融验证。不要把这一推断写成已彻底定位的新原因。

## 2. 两种方法的共同流程

1. 检查非负有限 3D 强度、整数组织标签、二值脑掩膜、相同物理网格与有效空间信息；不隐式重采样。
2. 三组织编码固定为 `0=未建模/背景`、`1=CSF`、`2=GM`、`3=WM`。GM、WM 分别取距自身边界大于 1 mm、2 mm 的正信号核心，不合并后腐蚀、不闭运算。
3. 各类空间核心内按 log 强度中位数 ±3.5×`1.4826*MAD` 排除拟合离群点；参数可配置，样本不足明确失败，不退回污染样本。原标签和原信号不改写。
4. 软先验限制在所属可靠核心，核心外权重严格为 0；拟合域和最终应用脑域分开。
5. 在包括标签 0 的整个脑掩膜内连续乘/除原始图像；不减最小值、不按标签恢复原值、不做强度范围裁剪。脑外场设为 1，脑外原信号及零信号保持不变。
6. 原场必须整体有限、脑域内严格为正；仅脑外未应用的非正估计可替换为 1，并记录计数。拒绝覆盖现有输出，保存支持域、参数、源码路径、实际设备和状态。

3.5 MAD、核心距离、控制点数等都是拟合参数，不是 QC 放行阈值。

### T2 pial refinement 输入

同会话原始 T2 经已发布的刚性变换一次线性重采样至重建网格，再进行组织引导偏置场校正和 WM=110 全局标定，得到 `T2.norm.mgz`。最终传给 `mris_place_surface --mmvol T2.mgz T2` 的 `T2.mgz` 是该校正结果的逐字节副本，保留重建视野内的完整头部信号，不再按 `brain.finalsurfs` 或脑掩膜置零。完整头部仅表示不剥头皮，不会补回原始采集或重建视野之外的数据。

偏置场仍以可靠脑组织拟合，并按原算法只在脑域应用；脑外信号保留，随后与全图一起进行相同的 WM 全局标定。这不等于已估计并校正头皮、颅骨等脑外组织的偏置。保留 `T2.prenorm.nii.gz`、`T2.norm.mgz`、`T2.mgz`、场和来源记录，旧掩膜版本不得继续作为该修订的缓存结果。

## 3. RBF：默认 GPU 方法

- 使用未修改的上游 `ScatterBiasFieldCorrectorFast`，通过子类限制软标签与核心拟合并替换核权重计算；不是未修改的官方 CLI 配方。
- 用可靠支持内最大值进行比例归一化，保留原脑内信号；拟合类别为 GM/WM，固定原标签。
- 默认 48 个散点控制点、48 个近邻、`rbf_sigma=0.05`、25 次迭代、`soft_sigma=0.5`、`tol=1e-6`、随机种子 0。
- 拟合与导出均使用 `softmax(-distance²/(2*sigma²))`，不在归一化分母添加 epsilon；远场恒定控制值仍保持恒定。
- RBF 控制系数用 `1e-6 + (100 - 1e-6) * sigmoid(logit)` 参数化，初始化仍对应系数 1。优化的是 logit，每次损失计算和全图导出都使用同一个有界系数，不是拟合结束后裁剪失败场。加权均值仅为防止浮点舍入越界限制在同一区间，最终平滑场另检查有限、严格大于 0 且不超过 100；不放宽应用域或原有效性检查。
- `fitted-control-points.npz` 中 `point_values` 保留实际有界系数，另存 `point_logits`；`support.json` 记录模型修订和上下限。该约束针对原始 divisor，不是校正后影像或 WM110 标定后的乘法因子的范围保证。下限 1e-6 只是严格正值的数值保护，不能代替外推及过校正 QC。
- 输入按 ITK z,y,x 轴拟合和导出，再恢复原 NIfTI 轴；保存原物理网格。导出时关闭 autograd。
- `cuda:0` 默认；支持 `cuda`、`cuda:N` 或可见设备序号。CUDA 不可用、索引越界或请求 CPU 时失败，不回退 CPU，也不自动改跑 Gaussian。设备序号服从 `CUDA_VISIBLE_DEVICES`。
- 独立 CLI 的原场为 **divisor**：`corrected = original / field`。仅支持保留拟合组织对比的 `--modality None`，不冒充实现了固定 T1/T2 目标均值。

## 4. Gaussian：显式 CPU 备选

- 在不重叠的可靠 GM/WM 核心内计算条件化 log10 残差；软置信度只用于初始类均值。
- 使用真正最近的有效供体外推残差，保留上游核宽调度、每步 ±0.20 的 log 残差限制和默认 10 次迭代；平滑后累积 log 场。
- 默认 `--modality None` 固定输入类均值；显式 T1/T2 目标模式只对 Gaussian 生效。
- 独立 CLI 的原场是 **multiplier**：`corrected = original * field`。
- NumPy/SciPy 在 CPU 运行。当前真实首例仍有局部过校正，只保留研究对照用途，不能以“程序完成”替代 QC。

## 5. 独立使用

源码位于 `bin/tissue_bfc/`，兼容入口为 `bin/run_tissue_bfc.py`；上游两份源码随包放在 `vendor/tissue_guided_bfc/`，不再要求外部实验路径。工具包不依赖 Nextflow、FreeSurfer 安装或 FastPrep runtime；输入须自行准备为上述组织标签和脑掩膜，不能直接把 aseg 编号当作 0/1/2/3 输入。

依赖配方为 `envs/bfc.yml` 和 `envs/bfc-requirements.txt`；CUDA 版 PyTorch 必须匹配目标 GPU/驱动。配方不是锁文件，也未在全新机器安装验证。实际 CUDA 可用性在运行时严格检查；Gaussian 所需 CPU 依赖另见 `envs/helper-requirements.txt`。

```bash
python bin/run_tissue_bfc.py \
  --input_img input.nii.gz --input_lab tissues.nii.gz --input_msk brain-mask.nii.gz \
  --output_img new-rbf/corrected.nii.gz --output_bias new-rbf/field.nii.gz \
  --device cuda:0

python bin/run_tissue_bfc.py --method gaussian \
  --input_img input.nii.gz --input_lab tissues.nii.gz --input_msk brain-mask.nii.gz \
  --output_img new-gaussian/corrected.nii.gz --output_bias new-gaussian/field.nii.gz
```

也可用 `PYTHONPATH=bin python -m tissue_bfc` 加同样参数。高级对照可显式传 `--source-script`，来源会记录；通常不需要。

每次输出 `corrected.nii.gz`、指定名称的原始场、`fit-mask.nii.gz`、`fit-labels.nii.gz`、`support.json`；RBF 另存 `fitted-control-points.npz`。这些 CLI 结果**尚未做 WM110 全局标定**。失败记录保留；修订后请使用新目录。

## 6. FastPrep 默认接线

- T1 与可选 T2 的 `parameters.method` / `parameters.bias.method` 默认均为 `rbf`；配置显式记录工具包版本与采样修订。全局默认 `--device cuda:0`，SynthSeg 等设备感知步骤也接收此选择。
- `software.local.json` 新增 `python.bfc`，指向具备上述依赖及 CUDA PyTorch 的解释器。控制和 Gaussian 仍用 `python.helper`。`prepare.py` 检查导入依赖；实际执行 RBF 前检查 GPU，允许在无 GPU 的登录节点准备配置。
- `prepare.py --bfc-method gaussian` 同时为 T1/T2 显式选择备选方法；要全流程 CPU 运行，还需在 prepare 时选择 `--mni-backend ants`，在 Nextflow 命令传 `--device cpu --t1_placement_backend cpu --t2_pial_backend cpu --topology_backend cpu`。保留 RBF 或 FireANTs 却请求 CPU 会明确失败。
- 适配器在工作目录保留原场 `field.nii.gz`，并统一生成 `factor.nii.gz`：RBF 取 `1/field`，Gaussian 取 `field`。**发布的 `bias_factor_nii` 和 T2 factor 始终是乘法因子**，不包含后续 WM110 全局尺度。输出先验证 `corrected=original*factor`，再单独做 WM110 标定并记录系数。
- 重新生成 control/runtime，使用新的 work 和结果目录；不能用旧绝对路径 runtime 或缓存假装切换成功。切换默认不等于本轮已经重跑既有数据，也不等于科学批准。

## 7. 验证与限制

此前首例 `sub-0001_ses-01` 的 RBF 修订已在 NVIDIA L20 / CUDA0 运行。两种方法的支持域同为 315638 体素，拟合域外软权重为 0；原始标签、脑域、物理网格与场应用一致性检查通过。冻结旧热点经同一 WM110 标定的中位数：原图 89.10，旧 Gaussian 194.24，新 Gaussian 84.52；旧 RBF 105.38，新 RBF 82.23。原图不是无偏置真值，靠近原图不自动证明正确。

实际共享宽窗复核显示：旧中线亮块和 RBF 标签形状接缝减弱；Gaussian 的小脑/脑干局部过校正仍存在，候选不通过。RBF 非拟合组织的对比变化尚待进一步核验；只有一个真实 T1 session 的修订对照，不能外推为全部 session 或真实 T2 已通过。完整历史诊断和失败尝试保存在原实验目录，本工具包不携带 MRI 数据。

回归测试覆盖取样、供体、场应用、无 CPU 回退、T1/T2 默认接线、RBF divisor 到流程 multiplier 的转换，以及解压后独立运行。GPU 集成测试需显式启用 `FASTPREP_TEST_BFC_CUDA=cuda:0`，实际执行合成 T1/T2，不是 mock GPU，也不等于真实脑影像验证。

2026-11-18 最终运行：完整 FastPrep 测试共 60 项，58 项通过、2 项跳过，用时约 75.9 秒；已显式启用 CUDA0 的 RBF 合成 T1/T2 集成测试。跳过的是未启用环境的原生 FreeSurfer 接口及完整 QC renderer 集成测试。两份上游源码逐字节一致、49 节点流程连接验证、独立解包运行和全量发布文件清单检查通过。

不画估计场来代替真实图像审核，不以 CV 单独决定胜负；比较须使用相同支持域、空间网格、全局标定和显示映射，保留失败产物与未完成检查。

## 8. 来源和打包

Gaussian 上游源码保持不变；RBF 源码在 2026-11-24 增加执行优化：缓存固定组织权重和、类别有效性及点间距正则的对角项，减少每个 Adam 步骤中的 GPU→CPU 标量同步。自适应权重下限为 0.2，因此总权重至少为 1 的类别不必每步再判断是否低于 0.001；极小类别仍执行原判断。损失仅在原有日志点及 epoch 结束时回传，保持原目标函数、求和顺序、模型参数、迭代上限与收敛规则。

可靠核心、连续场应用与有界 RBF 等方法修订仍在适配包中实现。来源、本地改动及上游说明见 `vendor/tissue_guided_bfc/SOURCE.md`、`UPSTREAM_README.md`。实现等价性验证应比较校正图、场、控制参数、支持域和物理网格；耗时改善不构成科学 QC 批准。不得把本工具包名称或版本当作上游作者对修订算法的认可。

在完整 FastPrep 根目录运行 `python bin/package_bfc.py --output dist/fastprep-bfc-1.1.0.tar.gz` 可生成独立小包；拒绝覆盖已有压缩包，不含模型权重、MRI 数据、个人 license、绝对路径 runtime 或内容校验锁。
