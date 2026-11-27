# SUGAR model bundle

本目录包含 SUGAR rigid 和四层 nonlinear 推理所需的最小源码、层级球面资源和双侧权重。

`run_registration.py` 要求 `--template-metadata` JSON 显式绑定 `template=fsaverage6`、半球、40962 顶点数及 sphere/sulc 的精确路径；随后还会比较随包内置 fsaverage6 坐标与 topology。nonlinear 计算结果最终重采样回 rigid 输入的 native source topology。

- `run_registration.py rigid`：单半球全局旋转。
- `run_registration.py nonlinear`：fsaverage3→6 四层非线性配准及原流程负面积修复。
- `source`：模型类、推理函数、球面插值和层级几何资源。
- `weights`：`fsaverage6/<hemi>_Rigid_904_fsaverage6.model` 与
  `<level>/<hemi>_NoRigid_904_<level>.model`，`lh`、`rh` 两侧齐备。

入口按上游命名规则解析默认权重：rigid 取 `weights/fsaverage6/` 下的 `Rigid` 权重，nonlinear 取 `weights/<level>/` 下的 `NoRigid` 权重；两种阶段的名字不能靠子串区分，因为 `rigid` 是 `NoRigid` 的子串，所以校验要求文件名与阶段、半球、层级三者精确对应。`--checkpoint*` 仍可覆盖默认值，缺少文件时在推理前明确失败。`curv` 和 `aparc.annot` 未收录，因为当前 DeepPrep 配置的模型特征始终为 `sulc`。

本项目的 `sugar` 环境配置见 `envs/sugar.yml` 与 `envs/sugar-requirements.txt`。已验证运行组合为 Python 3.9.25、PyTorch 2.8.0+cu128、PyG 2.2.0、PyTorch3D 0.7.9，以及针对 CUDA 12.8 和 H100 `sm_90` 源码编译的 torch-scatter 2.1.2。目标机器需按本机 CUDA/PyTorch 选择兼容构建。CUDA 模式会在加载模型前检查 GPU 编号、torch-scatter CUDA kernel 和 PyTorch3D CUDA KNN 的设备与结果；任一扩展缺少 CUDA 支持都会在推理前直接报错。`--device auto` 优先使用 GPU，`--device cuda` 或 `--device cuda:0` 使用可见 GPU 0，`cuda:N` 或非负整数 `N` 选择对应的可见 GPU。编号遵循外层 `CUDA_VISIBLE_DEVICES`，不重新解释为物理 GPU 编号；显式 GPU 请求不可用或越界时直接报错，不回退 CPU。
