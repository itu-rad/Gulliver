from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)
import torch
import transformers

import outlines


from threading import Lock

from stages.stage import (Stage, log_phase, log_first_token,
                          log_generated_tokens, log_prompt_tokens,
                          log_reasoning_tokens, log_peak_memory)
from utils.chat import THINK_CLOSE, THINK_FORCE_CLOSE, split_thinking
from utils.component import get_component
from utils.schemas import StageModel, PipelineModel, Query


def _sync_device(device) -> None:
    """Block until queued work on `device` has completed. No-op on CPU."""
    kind = getattr(device, "type", str(device))
    if kind == "cuda":
        torch.cuda.synchronize(device)
    elif kind == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


class _FirstTokenCriteria(StoppingCriteria):
    """Never stops generation; records the first-token instant.

    transformers evaluates stopping criteria once per generated token, right
    after it is appended, so the first invocation is the first-token instant.
    Later invocations return a cached all-False tensor.
    """

    def __init__(self, on_first_token):
        self._on_first_token = on_first_token
        self._not_done = None

    def __call__(self, input_ids, scores, **kwargs):
        if self._not_done is None:
            # The prefill is an asynchronous device launch and the criteria run
            # before any synchronisation; synchronise so the timestamp marks
            # the end of prefill rather than its enqueue.
            _sync_device(input_ids.device)
            self._on_first_token()
            self._not_done = torch.zeros(
                input_ids.shape[0], dtype=torch.bool, device=input_ids.device
            )
        return self._not_done


class _ThinkingBudgetCriteria(StoppingCriteria):
    """Stop when the reasoning budget runs out with the block still open, or
    when the answer budget runs out after the block was closed.

    Tracks the step at which the close token appeared (per row; batch 1 in
    practice). `closed_at` is None until then. Same budget semantics as the
    MLX stage.
    """

    def __init__(self, close_id: int, prompt_len: int, budget: int, answer_budget: int):
        self._close_id = close_id
        self._prompt_len = prompt_len
        self._budget = budget
        self._answer = answer_budget
        self.closed_at = None

    def __call__(self, input_ids, scores, **kwargs):
        n = input_ids.shape[1] - self._prompt_len
        if self.closed_at is None and bool((input_ids[:, -1] == self._close_id).any()):
            self.closed_at = n
        if self.closed_at is None:
            stop = n >= self._budget
        else:
            stop = n - self.closed_at >= self._answer
        return torch.full((input_ids.shape[0],), stop, dtype=torch.bool,
                          device=input_ids.device)


class Inference(Stage):
    """HuggingFace transformers inference stage.

    YAML config example:
        config:
          device: cuda
          dtype: torch.bfloat16
          model:
            name: Qwen/Qwen3.5-4B
            quantize: false
            gen_kwargs:
              max_new_tokens: 256      # ANSWER budget (tokens after </think>)
              do_sample: false
            thinking_budget: 1024      # reasoning tokens before a forced close;
                                       # 0 disables the budget logic entirely

    Same budget semantics as stages.llm_mlx.Inference: generate until the
    model closes its reasoning block, then up to `max_new_tokens` more; if the
    block is still open at the budget, append the closing text and generate
    the answer from a re-prefill. Downstream stages only see the answer.
    """

    def __init__(self, stage_config: StageModel, pipeline_config: PipelineModel):
        super().__init__(stage_config, pipeline_config)

        self._device = self._parse_device(self.extra_config.get("device", None))
        print(f"Device: {self._device}")
        self._dtype = get_component(self.extra_config.get("dtype", "torch.float32"))

        self._max_queries = pipeline_config.loadgen.max_queries

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.extra_config["model"]["name"]
        )

        # Optional pydantic model for structured generation.
        self._data_model = None
        data_model_path = self.extra_config.get("data_model", None)
        if data_model_path:
            self._data_model = get_component(data_model_path)

        self._outlines_generator = None

        self._gen_kwargs = dict(self.extra_config["model"].get("gen_kwargs", {}))
        self._thinking_budget = int(self.extra_config["model"].get("thinking_budget", 0))
        self._answer_budget = int(self._gen_kwargs.pop("max_new_tokens", 256))

        self._model = None

        self._depends_on_id = self.extra_config.get("depends_on_id")
        self._mutex = None
        if not self._depends_on_id:
            self._mutex = Lock()

    def _parse_device(self, device: str | None) -> torch.device:
        if device:
            return torch.device(device)
        else:
            if torch.backends.mps.is_available():
                return torch.device("mps")
            elif torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")

    def get_device(self) -> torch.device:
        return self._device

    def get_tokenizer(self) -> transformers.PreTrainedTokenizer:
        return self._tokenizer

    def get_accelerator(self) -> None:
        return None

    def get_model_lock(self):
        return self._model, self._mutex

    def _setup_model(self):
        if self.extra_config["model"].get("quantize"):
            self._model = AutoModelForCausalLM.from_pretrained(
                self.extra_config["model"]["name"],
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=(
                        torch.bfloat16
                        if self._device == torch.device("cuda")
                        and torch.cuda.is_bf16_supported()
                        else torch.float16
                    ),
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                ),
                device_map="auto",
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.extra_config["model"]["name"],
                torch_dtype=self._dtype if self._dtype is not torch.float32 else "auto",
                device_map="auto",
            )

    def _setup_outlines(self) -> None:
        if self._data_model and not self._outlines_generator:
            self._outlines_model = outlines.models.Transformers(
                self._model, self._tokenizer
            )
            self._outlines_generator = outlines.Generator(
                self._outlines_model, self._data_model
            )

    @log_phase
    def prepare(self):
        super().prepare()

        if not self._depends_on_id:
            print("Setting up model in ", self.name)
            self._setup_model()
            self._setup_outlines()

    def pre_run(self):
        if self._depends_on_id:
            self._model, self._mutex = self.dispatch_call(
                self._depends_on_id, "get_model_lock"
            )
            self._setup_outlines()

    def post_run(self):
        try:
            kind = getattr(self._device, "type", str(self._device))
            if kind == "cuda":
                log_peak_memory(self, torch.cuda.max_memory_allocated(),
                                "torch.cuda.max_memory_allocated")
            elif kind == "mps" and hasattr(torch, "mps"):
                log_peak_memory(self, torch.mps.driver_allocated_memory(),
                                "torch.mps.driver_allocated_memory")
        except Exception as e:  # telemetry must never fail the run
            print(f"{self.name}: peak memory unavailable: {e}")

    # ------------------------------------------------------------------
    def _generate(self, model_inputs, first_token: bool):
        """One generate() call over the (batched) inputs with the thinking
        budget applied. Returns (answers, reasonings, n_reasoning, n_answer,
        closed, forced)."""
        prompt_len = model_inputs.input_ids.shape[1]
        crit = []
        if first_token and not self.disable_logs:
            crit.append(_FirstTokenCriteria(lambda: log_first_token(self, "end")))
        budget = self._thinking_budget
        close_id = self._tokenizer.convert_tokens_to_ids(THINK_CLOSE)
        gen_kwargs = dict(self._gen_kwargs)
        if budget > 0:
            think_crit = _ThinkingBudgetCriteria(close_id, prompt_len, budget,
                                                 self._answer_budget)
            crit.append(think_crit)
            max_new = budget + self._answer_budget
        else:
            think_crit = None
            max_new = self._answer_budget
        out = self._model.generate(
            **model_inputs, **gen_kwargs, max_new_tokens=max_new,
            stopping_criteria=StoppingCriteriaList(crit) if crit else None,
            return_dict_in_generate=True,
        )
        seqs = out.sequences
        gen_ids = seqs[:, prompt_len:]
        n_gen = int(gen_ids.shape[1])
        forced = False
        if think_crit is not None and think_crit.closed_at is None and n_gen >= budget:
            # Budget exhausted with the block open: force it closed and
            # generate the answer from a full re-prefill. Resuming from
            # `out.past_key_values` corrupts the context with hybrid
            # (linear-attention + attention) caches such as Qwen3.5's.
            forced = True
            force_ids = torch.tensor(
                [self._tokenizer.encode(THINK_FORCE_CLOSE, add_special_tokens=False)]
                * seqs.shape[0], device=seqs.device)
            ext = torch.cat([seqs, force_ids], dim=1)
            attn = torch.ones_like(ext)
            out2 = self._model.generate(
                input_ids=ext, attention_mask=attn, **gen_kwargs,
                max_new_tokens=self._answer_budget, return_dict_in_generate=True)
            ans_ids = out2.sequences[:, ext.shape[1]:]
            answers = self._tokenizer.batch_decode(ans_ids, skip_special_tokens=True)
            reasonings = [split_thinking(t)[0] + THINK_FORCE_CLOSE.split(THINK_CLOSE)[0]
                          for t in self._tokenizer.batch_decode(gen_ids, skip_special_tokens=True)]
            return ([a.strip() for a in answers], reasonings, n_gen,
                    int(ans_ids.shape[1]), True, forced)

        texts = self._tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        answers, reasonings = [], []
        closed_all = True
        for t in texts:
            reasoning, answer, closed = split_thinking(t)
            if not closed:            # thinking off, or no block at all
                reasoning, answer = "", t.strip()
            answers.append(answer); reasonings.append(reasoning)
            closed_all &= closed
        if think_crit is not None and think_crit.closed_at is not None:
            n_r, n_a = think_crit.closed_at, n_gen - think_crit.closed_at
        elif closed_all and texts:
            n_r = len(self._tokenizer.encode(reasonings[0], add_special_tokens=False)) + 1
            n_a = max(n_gen - n_r, 0)
        else:
            n_r, n_a = 0, n_gen
        return answers, reasonings, n_r, n_a, closed_all, False

    def run(self, query: Query) -> dict[int, Query]:
        batch = query.data
        if isinstance(batch, str):
            batch = [batch]

        if self._outlines_generator:
            if self._mutex:
                self._mutex.acquire()
            try:
                model_out = [
                    self._outlines_generator(prompt, **self._gen_kwargs)
                    for prompt in batch
                ]
            finally:
                if self._mutex:
                    self._mutex.release()
            reasonings, tot_r, tot_a = [""] * len(batch), 0, 0
        else:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            model_inputs = self._tokenizer(batch, return_tensors="pt", padding=True).to(
                self._model.device
            )
            if self._mutex:
                self._mutex.acquire()
            try:
                if not self.disable_logs:
                    log_prompt_tokens(self, int(model_inputs.input_ids.numel()))
                log_first_token(self, "start")
                model_out, reasonings, tot_r, tot_a, closed, forced = self._generate(
                    model_inputs, first_token=True)
                log_generated_tokens(self, tot_r + tot_a)
                log_reasoning_tokens(self, tot_r, tot_a, closed, forced)
            finally:
                if self._mutex:
                    self._mutex.release()

        if self._data_model:
            model_out = [self._data_model.model_validate_json(x) for x in model_out]

        ctx = query.context if query.context is not None else {}
        ctx["reasoning"] = reasonings[0] if len(reasonings) == 1 else reasonings
        ctx["reasoning_tokens"] = ctx.get("reasoning_tokens", 0) + tot_r
        ctx["llm_calls"] = ctx.get("llm_calls", 0) + len(batch)
        query.context = ctx
        query.data = model_out

        return {idx: query for idx in self.output_queues}
