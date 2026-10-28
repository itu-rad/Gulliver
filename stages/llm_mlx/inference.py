from threading import Lock
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import make_prompt_cache
from transformers import AutoTokenizer

from stages.stage import (Stage, log_phase, log_first_token,
                          log_generated_tokens, log_prompt_tokens,
                          log_reasoning_tokens, log_peak_memory)
from utils.chat import THINK_CLOSE, THINK_FORCE_CLOSE, split_thinking
from utils.component import get_component
from utils.schemas import StageModel, PipelineModel, Query


class Inference(Stage):
    """MLX inference stage.

    YAML config example:
        config:
          model:
            name: mlx-community/Qwen3.5-4B-bf16
            gen_kwargs:
              max_tokens: 256          # ANSWER budget (tokens after </think>)
            thinking_budget: 1024      # reasoning tokens before a forced close;
                                       # 0 disables the budget logic entirely
          depends_on_id: 3             # share another stage's model + lock

    With a thinking budget the stage generates until the model closes its
    reasoning block, then up to `max_tokens` more for the answer. If the
    block is still open at the budget, the stage appends a closing text and
    continues from the SAME prompt cache (no re-prefill) for the answer.
    Downstream stages only ever see the answer; the reasoning goes to
    `query.context["reasoning"]` and the per-call counts to the trace.
    """

    def __init__(self, stage_config: StageModel, pipeline_config: PipelineModel):
        super().__init__(stage_config, pipeline_config)

        self._max_queries = pipeline_config.loadgen.max_queries

        self._model_path = self.extra_config["model"]["name"]
        self._tokenizer_config = self.extra_config["model"].get("tokenizer_config", {})
        self._gen_kwargs = dict(self.extra_config["model"].get("gen_kwargs", {}))
        self._thinking_budget = int(self.extra_config["model"].get("thinking_budget", 0))
        self._answer_budget = int(self._gen_kwargs.pop("max_tokens", 256))

        # Optional pydantic model the output is validated against.
        self._data_model = None
        data_model_path = self.extra_config.get("data_model", None)
        if data_model_path:
            self._data_model = get_component(data_model_path)

        self._model = None

        # Load the tokenizer eagerly so get_tokenizer() is available to other
        # stages' prepare(); the model is loaded in prepare().
        self._depends_on_id = self.extra_config.get("depends_on_id")
        if not self._depends_on_id:
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        else:
            self._tokenizer = None

        self._mutex = None
        if not self._depends_on_id:
            self._mutex = Lock()

    def get_tokenizer(self):
        """Getter for the tokenizer, matching the HuggingFace Inference interface."""
        return self._tokenizer

    def get_model_lock(self):
        return self._model, self._mutex

    def _setup_model(self):
        print(f"Loading MLX model from {self._model_path}")
        self._model, self._tokenizer = load(
            self._model_path, tokenizer_config=self._tokenizer_config
        )

    @log_phase
    def prepare(self):
        # Load before super().prepare() starts the worker thread; other
        # stages may dispatch get_tokenizer() during their prepare().
        if not self._depends_on_id:
            print("Setting up model in ", self.name)
            self._setup_model()

        super().prepare()

    def pre_run(self):
        if self._depends_on_id:
            self._model, self._mutex = self.dispatch_call(
                self._depends_on_id, "get_model_lock"
            )
            self._tokenizer = self.dispatch_call(self._depends_on_id, "get_tokenizer")

    def post_run(self):
        # Process-wide peak of MLX's allocator (system memory figures miss
        # mmapped weights).
        try:
            log_peak_memory(self, mx.get_peak_memory(), "mlx.get_peak_memory")
        except Exception as e:  # telemetry must never fail the run
            print(f"{self.name}: peak memory unavailable: {e}")

    # ------------------------------------------------------------------
    def _generate_one(self, prompt: str, first_token_cb):
        """Generate for one prompt. Returns (reasoning, answer, n_reasoning,
        n_answer, closed, forced)."""
        close_id = self._tokenizer.convert_tokens_to_ids(THINK_CLOSE)
        budget = self._thinking_budget
        if budget <= 0:
            # No budget: one bounded generation, reasoning (if any) split off
            # afterwards.
            text, n = "", 0
            if isinstance(prompt, str):
                prompt = self._tokenizer.encode(prompt)
            for r in stream_generate(self._model, self._tokenizer, prompt=prompt,
                                     max_tokens=self._answer_budget, **self._gen_kwargs):
                if first_token_cb: first_token_cb(); first_token_cb = None
                text += r.text; n = r.generation_tokens
            reasoning, answer, closed = split_thinking(text)
            if not closed:
                # no reasoning block at all (thinking off): everything is answer
                reasoning, answer = "", text.strip()
                n_r, n_a = 0, n
            else:
                n_r = len(self._tokenizer.encode(reasoning, add_special_tokens=False)) + 1
                n_a = max(n - n_r, 0)
            return reasoning, answer, n_r, n_a, closed, False

        cache = make_prompt_cache(self._model)
        # Tokenise outside the first-token window, as the other inference
        # stages do, so the window covers only prefill.
        if isinstance(prompt, str):
            prompt = self._tokenizer.encode(prompt)
        text, n, closed_at = "", 0, None
        for r in stream_generate(self._model, self._tokenizer, prompt=prompt,
                                 prompt_cache=cache,
                                 max_tokens=budget + self._answer_budget,
                                 **self._gen_kwargs):
            if first_token_cb: first_token_cb(); first_token_cb = None
            text += r.text; n = r.generation_tokens
            if closed_at is None and r.token == close_id:
                closed_at = n
            if closed_at is None and n >= budget:
                break                      # budget exhausted, block still open
            if closed_at is not None and n - closed_at >= self._answer_budget:
                break                      # answer budget exhausted
            if r.finish_reason == "stop":
                break
        if closed_at is not None:
            reasoning, answer, _ = split_thinking(text)
            return reasoning, answer, closed_at, n - closed_at, True, False

        # Forced close: continue from the cache with the closing text.
        # generate_step has already fed every yielded token to the model, so
        # the cache is exactly prompt + generated and the closing text is the
        # only new prefill.
        force_ids = self._tokenizer.encode(THINK_FORCE_CLOSE, add_special_tokens=False)
        answer, n_a = "", 0
        for r in stream_generate(self._model, self._tokenizer, prompt=force_ids,
                                 prompt_cache=cache, max_tokens=self._answer_budget,
                                 **self._gen_kwargs):
            answer += r.text; n_a = r.generation_tokens
            if r.finish_reason == "stop":
                break
        reasoning = split_thinking(text)[0] + THINK_FORCE_CLOSE.split(THINK_CLOSE)[0]
        return reasoning, answer.strip(), n, n_a, True, True

    def run(self, query: Query) -> dict[int, Query]:
        batch = query.data
        if isinstance(batch, str):
            batch = [batch]

        if self._mutex:
            self._mutex.acquire()
        try:
            encode = getattr(self._tokenizer, "encode", None)
            if not self.disable_logs and callable(encode):
                log_prompt_tokens(self, sum(len(encode(pr)) for pr in batch))
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
        finally:
            if self._mutex:
                self._mutex.release()

        if self._data_model:
            model_out = [self._data_model.model_validate_json(x) for x in model_out]

        # Per-query bookkeeping persisted by the end stage.
        ctx = query.context if query.context is not None else {}
        ctx["reasoning"] = reasonings[0] if len(reasonings) == 1 else reasonings
        ctx["reasoning_tokens"] = ctx.get("reasoning_tokens", 0) + tot_r
        ctx["llm_calls"] = ctx.get("llm_calls", 0) + len(batch)
        query.context = ctx
        query.data = model_out

        return {idx: query for idx in self.output_queues}
