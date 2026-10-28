# Reproducing the paper's measurements

Each experiment has a `collect.sh` that runs its configs and writes raw results to
`results/<experiment>/<device>/`. The two devices are an Apple MacBook Pro with an
M3 Pro (`m3pro`) and an NVIDIA DGX Spark (`gb10`). Run everything from the repository root.

## Setup

On each device, create the environment and install radT (see the main README):

```bash
conda env create -f environments/macos.yaml     # m3pro
conda env create -f environments/nvidia.yaml    # gb10
```

The Self-RAG engine runs and the collocation runs also need vLLM in its own environment:

```bash
conda env create -f environments/vllm_nvidia.yaml                     # gb10
export VLLM_EXECUTABLE=$(conda run -n gulliver_vllm which vllm)
```

On the M3 Pro, install [vllm-metal](https://github.com/vllm-project/vllm-metal) and point
`VLLM_EXECUTABLE` at its `vllm` binary.

Runs are recorded to the MLflow server in `MLFLOW_TRACKING_URI` (experiment `MLFLOW_EXPERIMENT`).
The framework-overhead and modularity runs use a local store under `results/` instead.

On GB10, install DCGM (`dcgmi`) and Nsight Systems (`nsys`), allow GPU profiling without
root, and before collecting:

```bash
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-profiling.conf   # then reboot
sudo sysctl kernel.perf_event_paranoid=-1
sudo chmod 666 /dev/cpu_dma_latency                 # modularity overhead holds the CPU latency QoS
```

On the M3 Pro, install `macmon` (`brew install macmon`).

Models and datasets are downloaded from the Hugging Face Hub and torchvision on first use,
except KiTS19 and the 3D-UNet checkpoint:

```bash
git clone https://github.com/neheller/kits19 tmp/kits19
(cd tmp/kits19 && pip install -r requirements.txt && python -m starter_code.get_imaging)
mkdir -p data/kits19 && mv tmp/kits19/data data/kits19/raw
mkdir -p models/3dunet_kits19 && wget -O models/3dunet_kits19/3dunet_kits19_pytorch.ptc \
  "https://zenodo.org/record/5597155/files/3dunet_kits19_pytorch.ptc?download=1"
```

## Runs

Every `collect.sh` accepts `--dry-run` (print the runs), `--smoke` (one run per cell,
two queries, local store; for collocation only the 4B cells and one MPS cell per larger
model) and `--runs N`. Completed runs are skipped when a script is restarted.

| Paper | Command | Device |
|---|---|---|
| Fig. 3 | `evaluation/framework_overhead/collect.sh <device>` | both |
| Fig. 4 | `evaluation/modularity_overhead/collect.sh <device>` | both |
| Fig. 5 | `evaluation/unet3d/collect.sh <device> --pass perf` | both |
| Tab. 2 | `evaluation/unet3d/collect.sh <device> --pass acc`, and `--pass reference` on gb10 | both |
| Fig. 6, Tab. 3, App. C | `evaluation/self_rag/collect.sh <device> --pass serial` | both |
| §5.2 accuracy, Fig. 11 | `evaluation/self_rag/collect.sh m3pro --pass acc` | m3pro |
| Fig. 11, Fig. 12 | `evaluation/self_rag/collect.sh gb10 --pass hw` | gb10 |
| Fig. 7, Fig. 8, Fig. 13 | `evaluation/self_rag/collect.sh <device> --pass load` | both |
| Tab. 1, Fig. 9 | `evaluation/collocation/collect.sh gb10 --pass alone`, then `--pass colloc` | gb10 |
| Fig. 10 | `evaluation/collocation/collect.sh gb10 --pass capture` | gb10 |

If a collocation or Self-RAG `hw` cell fails with "Already under profiling" in its log, the GPU's
profiling session was left open; reboot and rerun the pass.

## Configs

The configs are generated; `generate_configs.py` or `generate_configs.sh` in each
experiment directory rewrites them. The Self-RAG configs are derived from the base configs
`self_rag/configs/<task>_<arm>_<mlx|cuda>.yml`, and the collocation foreground from
`self_rag/configs/factoid_decomposed_shared_server_pipe1024r32_cuda.yml`.
