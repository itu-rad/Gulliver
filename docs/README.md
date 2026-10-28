# Gulliver reference

## Configuration

A config (`utils/schemas/`) has a `name`, the radT `listeners` to attach, an optional
`collocation` mechanism (e.g. `mps`), and one or more `pipelines`. A pipeline has:

| Field | Meaning |
|---|---|
| `inputs`, `outputs` | ids of the entry and exit stages |
| `dataset_stage_id` | the stage that reports dataset splits to the load generator |
| `loadgen` | `component`, `max_queries`, `timeout` (s), `queue_depth`, scheduler `config` |
| `stages` | list of stages: `id`, `name`, `component`, `outputs`, `config`, optional `polling_policy`, `disable_logs`, `max_input_queue_depth` |
| `disable_logs` | suppress the pipeline-level per-query trace rows |
| `serialize_queries` | release the next query only when the previous one has left the pipeline |
| `stop_with` | name of another pipeline; stop generating load when it finishes (`--max-queries` does not apply) |
| `env` | environment variables set in the pipeline's process before it starts |

`component` is a Python import path, so custom stages, schedulers and polling policies are
referenced the same way as the built-in ones.

## Components

Load schedulers (`loadgen`):

| Component | Load |
|---|---|
| `OfflineLoadScheduler` | synchronous: the next query when the previous one completes |
| `PoissonLoadScheduler` | Poisson arrivals at `config.rate` queries/s (`config.seed`) |
| `MultiStreamScheduler` | fixed interval `config.interval` s |
| `SaturatingOfflineScheduler` | keeps `queue_depth` queries in flight |

Queue polling policies for stages with several inputs (`utils.queues.polling`):
`SingleQueuePolicy` (default, one input), `FirstSubmittedPolicy` (oldest query first),
`MergePolicy` (combine one item from each input).

Stages (`stages.<package>.<Class>`):

| Package | Stages |
|---|---|
| `torchvision_classification` | `TorchVisionDataLoader`, `TorchVisionPreprocessFromWeights`, `TorchVisionClassification`, `TorchVisionDetection`, `DetectionPostprocess` |
| `llm_huggingface` | `HuggingFaceDataLoader`, `MockDataLoader`, `Inference`, `Finetune` |
| `llm_mlx` | `Inference` |
| `llm_server` | `Inference` (vLLM, vllm-metal or Ollama server, started and stopped by the stage) |
| `llm_torchtune` | `TorchTuneDataLoader`, `Inference`, `Finetune` |
| `self_rag` | `SelfRAGDataLoader`, `ChromaRetriever`, `RetrievalGraderFormatter`, `RelevanceFilterRouter`, `AnswerGeneratorFormatter`, `HallucinationGraderFormatter`, `QuestionRewriterFormatter`, `BinaryRouter`, `MonolithFormatter`, `MonolithRouter` |
| `rag_indexing` | `CorpusChunkLoader`, `EmbedStage`, `ChromaIndexer` |
| `multimodal_vqa` | `VQADataLoader`, `CLIPImagePreprocess`, `CLIPVisionEncoder`, `CLIPVisionEncoderCoreML`, `FAISSImageRetriever`, `VQAPromptFormatter` |
| `unet3d_kits19` | `KiTS19CaseLoader`, `KiTS19Preprocess`, `UNet3DInference`, `KiTS19DiceScore` |
| `evaluation` | `CopyStage`, `DataInjector` (synthetic payloads) |
| `stages` | `Stage` (identity), `TerminalCapture` (writes each query's output to JSONL) |

Stage-specific config keys are documented in each class's docstring.

## Extending

A stage subclasses `stages.Stage` and implements `run(query) -> {output_stage_id: query}`;
`prepare()` runs once before the first query and `pre_run()`/`post_run()` run on the
stage's own thread. Configuration is available as `self.extra_config`.

```python
from stages import Stage

class Uppercase(Stage):
    def run(self, query):
        query.data = [s.upper() for s in query.data]
        return {idx: query for idx in self.output_queues}
```

A stage can call methods of another stage directly with `self.dispatch_call(stage_id, name, ...)`,
e.g. to share one loaded model. Schedulers subclass `loadgen.schedulers.scheduler.LoadScheduler`
and implement `generate(queue, event)`; polling policies subclass
`utils.queues.polling.polling_policy.PollingPolicy`.

## Traces

A run records two kinds of trace, switched independently.

**CSV rows.** Each run writes `results/<label>.csv`, one event per line:

```
<wall time>, <pipeline>, <stage>, <phase>, <start|end>[, <extras>], <perf_counter_ns>
```

`phase` is `prepare` or `run`; LLM stages add `first_token`, `n_prompt_tokens`,
`n_generated_tokens` and reasoning-token rows. `disable_logs` on a stage or pipeline suppresses
its rows; `main.py --pipeline-logs on|off` overrides the pipeline's setting.

**Spans.** The same timings are recorded as spans in the run's MLflow store, in one of three modes:

| Setting | Spans |
|---|---|
| neither variable set | MLflow tracing in the pipeline process |
| `SUITE_PROC_TRACE=1` | radT tracing, exported from a separate radT process |
| `SUITE_DISABLE_TRACING=1` | none |

Spans can be exported to Perfetto with `utils/export_trace.py`.

Environment variables:

| Variable | Effect |
|---|---|
| `SUITE_OUTPUT_DIR` | directory for CSV traces and outputs (default `results/`) |
| `SUITE_PROC_TRACE` | spans through radT (see above) |
| `SUITE_DISABLE_TRACING` | no spans; CSV rows are unaffected |
| `SUITE_DATA_SEED` | shuffle seed of `TorchVisionDataLoader` |
| `PIPELINE_DRAIN_TIMEOUT_S` | how long to wait for in-flight queries at the end (default 3600) |
| `SERVER_STOP_TIMEOUT_S` | how long an inference server gets to exit (default 120) |
| `VLLM_EXECUTABLE` | the `vllm` binary used by `stages.llm_server` |
