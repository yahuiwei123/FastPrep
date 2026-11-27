# FreeSurfer 8.1 集群运行

支持的表面运行环境为 Linux x86-64、FreeSurfer Ubuntu 20 的 8.1.0 构建 `f30dfed`。在所有节点挂载相同的项目、BIDS 数据、工作目录、软件环境和 TemplateFlow 缓存。先按 README 准备 `software.local.json`，设置 Java、Nextflow 及 GPU 环境。可用 `FASTPREP_SOFTWARE` 指向另一份配置，用 `FASTPREP_NEXTFLOW` 指定可执行文件。

在共享目录生成一次计划。三个节点各自运行一个 rank，每节点分配五张可见 GPU。

```bash
export FASTPREP_BIDS_ROOT=/shared/smri
export TEMPLATEFLOW_HOME=/shared/templateflow
run_dir=/shared/fastprep-runs/adult-01
python3 cluster15/cluster.py plan --root "$FASTPREP_BIDS_ROOT" --batch-size 4 --run-dir "$run_dir"
# 在第一个节点运行。第二、三个节点分别使用 --rank 1 和 --rank 2，并设置相同的环境变量。
bash cluster15/on-node.sh --rank 0 --root "$FASTPREP_BIDS_ROOT" --run-dir "$run_dir"
```

每个分片启动一个 Nextflow 流程，处理该分片的多个 session。新分片采用 6 核模型步骤和 8 核表面步骤。恢复时使用相同的 rank、计划及工作目录，脚本会请求 `-resume`。已有缓存是否命中取决于 Nextflow 的输入、命令、资源配置和工作文件。迁移到新的项目路径后，准备阶段会生成新路径，部分任务可能重新计算。

成人批处理暂时排除门诊中心 `PT002_Nigerian_Clinical`，并排除婴儿数据集 `ds004776`、`ds006169` 和 `PT023_Infant_Development_Brain`。恢复旧计划时保留其余分片的序号和 GPU 分配。分片的 `session-status.json` 记录完成与失败情况。一个 session 缺少必要产物时，其余 session 可以继续。

若希望在 Slurm 中为整个 BIDS 数据集启动一个 Nextflow 流程，在提交节点设置 `FASTPREP_PARTITION`、`FASTPREP_NODES`（三个节点的 Slurm 名称）、`FASTPREP_BIDS_ROOT` 和 `TEMPLATEFLOW_HOME`，运行 `bash cluster15/one-dataset-slurm.sh PT001_ClevelandCCF /shared/fastprep-runs/cleveland`。

直接使用 `batch_fastprep_anat.py` 也默认隔离 session 错误。若跳过批处理脚本、直接执行 `nextflow run`，需要显式传入 `--isolate_session_errors true` 才会隔离错误。

报告见各 session 的 `outputs.json` 和批次的 `qc/index.html`。`quality_warnings` 汇总了产生可用科学产物时记录的质量问题，`qc_stages` 保留实际判定。直接使用 Nextflow 时可显式设置 `--qc_mode strict`。

## 重建 GPU 后端

仓库包含验证过的 8.1 表面驱动和 CUDA 12.8、13.2 两套核函数。实际卡型还需与驱动、PyTorch 和模型环境兼容。重建需要对应的 FreeSurfer 源码及安装目录、CUDA 工具链、Conda GCC 12 主机编译器、patchelf、binutils、cuobjdump、dpkg-deb。Ubuntu 20 编译器和 sysroot 及 ITK 开发包的确切版本列在 `build-tools/compat-build/packages.txt`。把对应的 amd64 deb 放在独立目录。

```bash
export FASTPREP_CUDA_HOST_PREFIX=/opt/conda/envs/gcc12
python3 -B build-tools/build81.py --stage-only --work-dir /scratch/build-fs81 \
    --freesurfer "$FREESURFER_HOME" --source "$FS_SOURCE" --packages "$DEB_DIR" \
    --cuda12 "$CUDA12_CONDA_PREFIX" --nvcc13 /usr/local/cuda-13.2/bin/nvcc
```

`--source` 接受源码根目录或以 `include/`、`utils/` 等目录开头的构建输入归档。`--stage-only` 将产物写进指定的 work 目录；省略该参数会在 ABI 和 GPU 架构校验完成后安装到本仓库的 `vendor/`。检查随仓库提供的库可运行 `python3 -B build-tools/compat-build/verify.py vendor`。已编译架构和实机验证范围见 `build-tools/compat-build/verified.json`。

SUGAR 使用独立 Python 环境。配套的 `build-tools/build-pytorch3d81.py` 面向 CPython 3.9、PyTorch 2.8/cu128 和 PyTorch3D 0.7.9，复用其源码树中匹配的 CPU 对象文件。设置同一个 `FASTPREP_CUDA_HOST_PREFIX`，并传入 `--work-dir`、`--python-prefix` 和 `--source`。需要安装已验证扩展时传 `--install-dir /path/to/pytorch3d`。用配置中的 SUGAR 解释器运行 `build-tools/sugar-knn-check.py` 可检查当前可见 GPU 的实际 CUDA KNN 运算。
