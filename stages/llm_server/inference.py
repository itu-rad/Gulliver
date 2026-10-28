import threading
from concurrent.futures import ThreadPoolExecutor

import json
import litellm
import requests
import mlflow
from utils.trace_span import trace_span
from transformers import AutoTokenizer

from stages.stage import (Stage, log_phase, log_first_token,
                          log_generated_tokens, log_prompt_tokens,
                          log_reasoning_tokens)
from utils.chat import THINK_CLOSE, THINK_FORCE_CLOSE, split_thinking
from utils.component import get_component
from utils.schemas import StageModel, PipelineModel, Query
from utils.schemas.server import ServerModel
from utils.server_manager import ServerManager


def _translate_gen_kwargs(gen_kwargs: dict) -> dict:
    """Map HuggingFace-style gen kwargs to the OpenAI/litellm names the server
    endpoints expect. Unknown keys pass through.
    """
    out = dict(gen_kwargs)
    if "max_new_tokens" in out:
        out["max_tokens"] = out.pop("max_new_tokens")
    # HF `do_sample: false` == greedy decoding == temperature 0.
    do_sample = out.pop("do_sample", None)
    if do_sample is False and "temperature" not in out:
        out["temperature"] = 0
    return out


class Inference(Stage):
    """Inference stage backed by a local engine server (vLLM / Ollama).

    The *owner* stage (no ``depends_on_id``) launches the engine as a subprocess
    via :class:`ServerManager` and serves requests over its OpenAI-compatible
    HTTP API using litellm. Dependent stages reuse the same running server.

    Prompts arrive already chat-templated (formatter stages call
    ``get_tokenizer().apply_chat_template``), so the stage uses the
    text-completion endpoint to avoid the engine applying a template again.

    YAML config example::

        config:
          server:
            engine: vllm
            model: Qwen/Qwen3.5-4B
            port: 0
            gpu_memory_utilization: 0.9
          model:
            gen_kwargs:
              max_new_tokens: 256
              do_sample: false
    """

    def __init__(self, stage_config: StageModel, pipeline_config: PipelineModel):
        super().__init__(stage_config, pipeline_config)

        self._max_queries = pipeline_config.loadgen.max_queries

        mcfg = self.extra_config.get("model", {})
        gk = dict(mcfg.get("gen_kwargs", {}))
        gk.pop("think", None)          # thinking is a prompt-side switch here
        self._gen_kwargs = _translate_gen_kwargs(gk)
        # As in llm_mlx / llm_huggingface: `max_new_tokens` is the answer
        # budget, `thinking_budget` the reasoning cap before a forced close.
        # The forced close re-prompts with prompt + reasoning + closing text.
        self._answer_budget = int(self._gen_kwargs.pop("max_tokens", 256))
        self._thinking_budget = int(mcfg.get("thinking_budget", 0))
        self._num_ctx = int(mcfg.get("num_ctx", 8192))

        # Optional pydantic model the output is validated against.
        self._data_model = None
        data_model_path = self.extra_config.get("data_model", None)
        if data_model_path:
            self._data_model = get_component(data_model_path)

        self._depends_on_id = self.extra_config.get("depends_on_id")

        # Requests this stage keeps in flight; >1 lets the engine batch. With
        # a shared server (depends_on_id) the load is the sum over stages.
        self._concurrency = int(self.extra_config.get("concurrency", 1))

        self._server: ServerManager | None = None
        self._litellm_model: str | None = None
        self._litellm_api_base: str | None = None

        if not self._depends_on_id:
            self._server_config = ServerModel(**self.extra_config["server"])
            # Formatter stages call get_tokenizer() during their prepare(), so
            # load it eagerly. `tokenizer_name` overrides the served model id
            # for engines whose model id is not a HF repo (e.g. Ollama tags).
            tokenizer_name = self.extra_config.get(
                "tokenizer_name", self._server_config.model
            )
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            self._server_config = None
            self._tokenizer = None

    def get_tokenizer(self):
        """Getter for the tokenizer, matching the HF/MLX Inference interface."""
        return self._tokenizer

    def get_server_handle(self) -> tuple[str, str]:
        """Return (litellm_model, litellm_api_base) for dependent stages."""
        return self._litellm_model, self._litellm_api_base

    @log_phase
    def prepare(self):
        # Start the server before super().prepare() starts the worker thread,
        # so dependent stages and the first queries find it listening.
        if not self._depends_on_id:
            print("Launching inference server in ", self.name)
            self._server = ServerManager(self._server_config)
            self._server.start()
            self._litellm_model = self._server.litellm_model
            self._litellm_api_base = self._server.litellm_api_base

        super().prepare()

    def pre_run(self):
        if self._depends_on_id:
            self._litellm_model, self._litellm_api_base = self.dispatch_call(
                self._depends_on_id, "get_server_handle"
            )
            self._tokenizer = self.dispatch_call(self._depends_on_id, "get_tokenizer")

    def run_wrapper(self):
        """Concurrent dispatcher (overrides Stage.run_wrapper).

        Keeps up to ``concurrency`` requests in flight so the server can batch
        them. Each query is handled by a pool worker via ``_process_query``,
        keeping the per-query tracing and logging. On the terminator, drain
        in-flight requests, then propagate ``None`` downstream.

        ``concurrency == 1`` is equivalent to the base sequential loop.
        """
        self.pre_run()
        executor = ThreadPoolExecutor(max_workers=self._concurrency)
        pending = []
        try:
            while True:
                with trace_span(
                    name=f"{self.name}.get_input",
                    attributes={"thread_id": threading.get_ident()},
                ):
                    query = self._get_input_from_queues()
                if not query:
                    break
                pending.append(executor.submit(self._process_query, query))
                # Reap finished workers: bounds the list and surfaces a worker
                # exception promptly instead of swallowing it.
                for done in [f for f in pending if f.done()]:
                    done.result()
                pending = [f for f in pending if not f.done()]
            executor.shutdown(wait=True)
            for f in pending:
                f.result()
        except Exception:
            # As in Stage.run_wrapper: propagate the terminator so the
            # pipeline does not hang, then re-raise.
            import traceback
            print(f"[stage FATAL] {self.name}: exception in a server worker; "
                  f"terminating pipeline", flush=True)
            traceback.print_exc()
            executor.shutdown(wait=False, cancel_futures=True)
            self._push_to_all_outputs(None)
            raise
        self._push_to_all_outputs(None)
        self.post_run()

    # ------------------------------------------------------------------
    def _is_ollama(self) -> bool:
        return str(self._litellm_model).startswith("ollama/")

    def _stream(self, prompt: str, max_tokens: int):
        """Yield generated text pieces, one per decode step.

        Ollama: the native generate endpoint with `raw: true`, since the prompt
        is already chat-templated; the OpenAI-compatible endpoint re-applies a
        template and returns reasoning on a separate channel.
        vLLM: its completions endpoint is raw already; litellm streams it.
        """
        if self._is_ollama():
            tag = str(self._litellm_model)[len("ollama/"):]
            body = {"model": tag, "prompt": prompt, "raw": True, "stream": True,
                    "options": {"num_predict": int(max_tokens),
                                "temperature": float(self._gen_kwargs.get("temperature", 0)),
                                "num_ctx": self._num_ctx}}
            with requests.post(f"{self._litellm_api_base}/api/generate", json=body,
                               stream=True, timeout=3600) as r:
                if r.status_code != 200:
                    raise RuntimeError(f"ollama {r.status_code}: {r.text[:300]}")
                n_pieces = 0
                for line in r.iter_lines():
                    if not line:
                        continue
                    d = json.loads(line)
                    if "error" in d:
                        # Surface an engine error inside a 200 stream instead
                        # of returning an empty completion.
                        raise RuntimeError(f"ollama stream error: {d['error']!s:.300}")
                    piece = d.get("response", "")
                    if piece:
                        n_pieces += 1
                        yield piece
                    if d.get("done"):
                        if n_pieces == 0:
                            raise RuntimeError(
                                f"ollama returned an empty completion (done_reason="
                                f"{d.get('done_reason')!r}, prompt_eval_count="
                                f"{d.get('prompt_eval_count')!r})")
                        return
        else:
            for chunk in litellm.text_completion(
                model=self._litellm_model, prompt=prompt,
                api_base=self._litellm_api_base, api_key="EMPTY",
                stream=True, max_tokens=int(max_tokens), **self._gen_kwargs,
            ):
                piece = chunk.choices[0].text or ""
                if piece:
                    yield piece

    def _generate_one(self, prompt: str, first_token_cb):
        """Returns (reasoning, answer, n_reasoning, n_answer, closed, forced).
        One streamed piece == one decode step (server-side tokenisation)."""
        budget, answer_budget = self._thinking_budget, self._answer_budget
        if budget <= 0:
            text, n = "", 0
            for piece in self._stream(prompt, answer_budget):
                if first_token_cb: first_token_cb(); first_token_cb = None
                text += piece; n += 1
            reasoning, ans, closed = split_thinking(text)
            if not closed:
                return "", text.strip(), 0, n, False, False
            n_r = len(self._tokenizer.encode(reasoning, add_special_tokens=False)) + 1 \
                if self._tokenizer else 0
            return reasoning, ans, n_r, max(n - n_r, 0), True, False

        text, n, closed_at = "", 0, None
        for piece in self._stream(prompt, budget + answer_budget):
            if first_token_cb: first_token_cb(); first_token_cb = None
            text += piece; n += 1
            if closed_at is None and THINK_CLOSE in text:
                closed_at = n
            if closed_at is None and n >= budget:
                break                       # budget exhausted, block still open
            if closed_at is not None and n - closed_at >= answer_budget:
                break
        if closed_at is not None:
            reasoning, ans, _ = split_thinking(text)
            return reasoning, ans, closed_at, n - closed_at, True, False
        # Forced close: re-prompt with the reasoning so far plus the closing
        # text. After a cancelled stream, llama.cpp's slot cache can be stale
        # and the model ends at once; retry once, then fail.
        ans, n_a = "", 0
        for attempt in range(2):
            ans, n_a = "", 0
            try:
                for piece in self._stream(prompt + text + THINK_FORCE_CLOSE, answer_budget):
                    ans += piece; n_a += 1
            except RuntimeError as e:
                if attempt == 0 and "empty completion" in str(e):
                    print(f"{self.name}: empty forced-close continuation, retrying once")
                    continue
                raise
            if ans.strip():
                break
        reasoning = split_thinking(text)[0] + THINK_FORCE_CLOSE.split(THINK_CLOSE)[0]
        return reasoning, ans.strip(), n, n_a, True, True

    def run(self, query: Query) -> dict[int, Query]:
        batch = query.data
        if isinstance(batch, str):
            batch = [batch]

        if not self.disable_logs:
            _enc = getattr(self.get_tokenizer(), "encode", None)
            if callable(_enc):
                log_prompt_tokens(self, sum(len(_enc(p)) for p in batch))

        log_first_token(self, "start")
        model_out, reasonings = [], []
        tot_r = tot_a = 0
        closed_all, forced_any = True, False
        for i, prompt in enumerate(batch):
            cb = (lambda: log_first_token(self, "end")) if i == 0 else None
            reasoning, answer, n_r, n_a, closed, forced = self._generate_one(prompt, cb)
            model_out.append(answer); reasonings.append(reasoning)
            tot_r += n_r; tot_a += n_a
            closed_all &= closed; forced_any |= forced
        log_generated_tokens(self, tot_r + tot_a)
        log_reasoning_tokens(self, tot_r, tot_a, closed_all, forced_any)

        if self._data_model:
            model_out = [self._data_model.model_validate_json(x) for x in model_out]

        ctx = query.context if query.context is not None else {}
        ctx["reasoning"] = reasonings[0] if len(reasonings) == 1 else reasonings
        ctx["reasoning_tokens"] = ctx.get("reasoning_tokens", 0) + tot_r
        ctx["llm_calls"] = ctx.get("llm_calls", 0) + len(batch)
        query.context = ctx
        query.data = model_out

        return {idx: query for idx in self.output_queues}

    def post_run(self):
        # post_run() may be skipped on timeout or crash; main.py also kills
        # all servers on exit.
        if self._server is not None:
            self._server.stop()
