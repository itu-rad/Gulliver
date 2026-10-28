# Gulliver

Gulliver composes machine learning workloads into pipelines of stages, runs them end to end
(data loading and preprocessing included) under a configurable load, and ties per-query
traces to hardware metrics, also across collocated workloads. It runs on a single Linux or
macOS machine, from Apple silicon laptops to NVIDIA servers.

- **Stages** wrap a unit of work: a data loader, a preprocessing step, model inference or
  training, a retriever, a router. Each stage runs in its own thread and passes queries to
  the next through queues, so pipelines can fan out, branch and loop.
- **Pipelines** are declared in YAML: the stages, their parameters and how they connect.
  Swapping a model, a backend or a whole stage is a config change.
- **Load generators** drive a pipeline: synchronous (one query at a time) or Poisson arrivals.
- **radT** runs each pipeline in its own process, sets up
  collocation (e.g. NVIDIA MPS), and attaches hardware listeners such as `nvidia-smi`,
  DCGM, Nsight or `macmon`. Results are recorded to MLflow.

## Installation

```bash
git clone <repository-url> gulliver && cd gulliver
conda env create -f environments/nvidia.yaml   # or environments/macos.yaml on Apple silicon
conda activate gulliver_nvidia                 # or gulliver_macos
pip install "radt @ git+<radt-repository>@<commit>#subdirectory=radt"
```

LLM serving through `stages.llm_server` needs vLLM (`environments/vllm_nvidia.yaml`), vllm-metal
or Ollama, installed separately; set `VLLM_EXECUTABLE` if `vllm` is not on `PATH`.

## Usage

```bash
python main.py pipeline_configs/torchvision_inference.yml
```

runs every pipeline of the config through radT, each in its own process, with the listeners
the config names. To run one pipeline in the current process instead:

```bash
python main.py <config> -p 0 [--label NAME] [--max-queries N]
```

Each run writes per-query CSV rows to `results/<label>.csv` (override the directory with
`SUITE_OUTPUT_DIR`) and records parameters, spans and listener metrics to the MLflow server
in `MLFLOW_TRACKING_URI` (experiment `-e`).

A config:

```yaml
name: "Image classification"
listeners: [smi, top]
pipelines:
  - name: "Inference"
    inputs: [0]
    outputs: [2]
    dataset_stage_id: 0
    loadgen:
      component: loadgen.PoissonLoadScheduler
      queue_depth: 10
      max_queries: 100
      timeout: 600
      config:
        rate: 5            # queries per second
    stages:
      - name: Data loader
        id: 0
        outputs: [1]
        component: stages.torchvision_classification.TorchVisionDataLoader
        config: {...}
      - name: Model
        id: 1
        outputs: [2]
        component: stages.torchvision_classification.TorchVisionClassification
        config: {...}
      - name: End
        id: 2
        component: stages.TerminalCapture
```

`pipeline_configs/` has working examples: TorchVision training and inference, LLM generation
and fine-tuning with Hugging Face, torchtune and MLX, RAG serving in-process or
through vLLM/Ollama, multimodal VQA, and MLPerf-style workloads. Several pipelines in one
config run concurrently; `collocation: mps` runs them under NVIDIA MPS.
`python scripts/render_pipeline_graph.py <config>` draws a config's graph.

See [docs/](docs/README.md) for the configuration reference, the component library,
tracing, and how to add stages, schedulers and polling policies.

## Reproducing the paper

[evaluation/](evaluation/README.md) holds the configs and collection scripts for every
measurement in the paper.
