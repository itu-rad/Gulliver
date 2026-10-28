from torchtune.training.checkpointing import FullModelHFCheckpointer
from torchtune.training import MODEL_KEY, set_default_dtype
from torchtune.generation import generate
import torch

from stages.stage import Stage, log_phase
from utils.component import get_component
from utils.schemas import StageModel, PipelineModel, Query


class Inference(Stage):

    def __init__(self, stage_config: StageModel, pipeline_config: PipelineModel):
        super().__init__(stage_config, pipeline_config)

        self._checkpoint_dict = self._load_checkpoint(self.extra_config["checkpointer"])

        self._device = self._parse_device(self.extra_config["device"])
        self._dtype = get_component(self.extra_config.get("dtype", "torch.float32"))

        self._max_queries = pipeline_config.loadgen.max_queries

        self._temperature = self.extra_config.get("temperature", 0.6)
        self._top_k = self.extra_config.get("top_k", 300)
        self._max_new_tokens = self.extra_config.get("max_new_tokens", 300)

    def _parse_device(self, device: str | None) -> torch.device:
        """
        Parse the device string and return the appropriate torch.device.
        If no device is specified, return the appropriate torch.device based on the available devices.
        Args:
            device (str | None): The device string, or None if no device is specified.
        Returns:
            torch.device: The parsed device.
        """

        if device:
            return torch.device(device)
        else:
            if torch.backends.mps.is_available():
                return torch.device("mps")
            elif torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")

    def _load_checkpoint(self, checkpoint_conf):
        """Load a checkpoint from the checkpoint config.

        Args:
            checkpoint_conf (dict): The checkpoint configuration.

        Returns:
            dict: The loaded checkpoint.
        """
        checkpointer = FullModelHFCheckpointer(
            checkpoint_dir=checkpoint_conf["checkpoint_dir"],
            checkpoint_files=checkpoint_conf["checkpoint_files"],
            model_type=checkpoint_conf["model_type"],
            output_dir=checkpoint_conf["output_dir"],
        )

        checkpoint_dict = checkpointer.load_checkpoint()
        return checkpoint_dict

    def _setup_model(self):
        with set_default_dtype(self._dtype), self._device:
            self._model = get_component(self.extra_config["model"]["component"])()

        base_model_state_dict = self._checkpoint_dict[MODEL_KEY]
        self._model.load_state_dict(base_model_state_dict, strict=False)

        batch_size = self.dispatch_call(
            self.extra_config.get("batch_size_stage_id", 1), "get_batch_size"
        )

        if self.extra_config["model"].get("KV_cache", False):
            with self._device:
                self._model.setup_caches(batch_size=batch_size, dtype=self._dtype)

        self._model.eval()

    def _update_stop_tokens_tracker(
        self,
        tokens: torch.Tensor,
        stop_tokens: torch.Tensor,
        stop_token_reached: torch.Tensor,
    ) -> torch.Tensor:
        """Updates which sequences have reached a stop token."""
        # tokens: [bsz, 1]
        # stop_tokens: [num_stop_tokens]
        # stop_token_reached: [bsz]
        stop_token_reached_curr = torch.isin(tokens, stop_tokens).flatten()
        stop_token_reached |= stop_token_reached_curr
        return stop_token_reached

    @log_phase
    def prepare(self):
        super().prepare()

        self._setup_model()
        self._tokenizer = self.dispatch_call(
            self.extra_config["tokenizer_stage_id"], "get_tokenizer"
        )

    def run(self, query: Query) -> dict[int, Query]:
        with torch.no_grad():
            batch = query.data

            prompt = batch["tokens"]

            prompt = prompt.to(self._device)

            generated_tokens, _ = generate(
                model=self._model,
                prompt=prompt,
                max_generated_tokens=self._max_new_tokens,
                pad_id=self._tokenizer.pad_id,
                temperature=self._temperature,
                top_k=self._top_k,
                stop_tokens=self._tokenizer.stop_tokens,
            )

            query.data = self._tokenizer.decode(generated_tokens[0].tolist())
            print(query.data)
            outputs = {idx: query for idx in self.output_queues}
            return outputs
