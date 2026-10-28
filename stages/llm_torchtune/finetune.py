from torchtune.training.checkpointing import FullModelHFCheckpointer
from torchtune.modules.peft import (
    get_adapter_params,
    set_trainable_params,
)
from torchtune.training import (
    set_default_dtype,
    set_activation_checkpointing,
    MODEL_KEY,
)
from torchtune.modules import TransformerSelfAttentionLayer
import torch

from stages.stage import Stage, log_phase
from utils.component import get_component
from utils.schemas import Query, StageModel, PipelineModel


class Finetune(Stage):

    def __init__(self, stage_config: StageModel, pipeline_config: PipelineModel):
        super().__init__(stage_config, pipeline_config)

        self._loss_fn = get_component(self.extra_config["loss"]["component"])()

        self._checkpoint_dict = self._load_checkpoint(self.extra_config["checkpointer"])

        self._device = self._parse_device(self.extra_config["device"])
        self._dtype = get_component(self.extra_config["dtype"])

        self._max_queries = pipeline_config.loadgen.max_queries

        self._gradient_accumulation_steps = self.extra_config[
            "gradient_accumulation_steps"
        ]
        self._running_loss = 0.0
        self._current_step = 0

    def get_loss_fn(self):
        """
        Returns the loss function used for training.
        Returns:
            Callable: The loss function.
        """

        return self._loss_fn

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
        """
        Sets up the model for fine-tuning by performing the following steps:

        1. Configures the device and precision settings.
        2. Initializes the model using the configuration specified in `self.extra_config`.
        3. Sets adapter parameters as trainable.
        4. Enables activation checkpointing for memory efficiency.
        5. Loads the base model checkpoint into the model.
        """
        with set_default_dtype(self._dtype), self._device:
            model_cls = get_component(self.extra_config["model"]["component"])
            self._model = model_cls(
                lora_attn_modules=self.extra_config["model"]["lora_attn_modules"],
                apply_lora_to_mlp=self.extra_config["model"]["apply_lora_to_mlp"],
                apply_lora_to_output=self.extra_config["model"]["apply_lora_to_output"],
                lora_rank=self.extra_config["model"]["lora_rank"],
                lora_alpha=self.extra_config["model"]["lora_alpha"],
            )

        adapter_params = get_adapter_params(self._model)
        set_trainable_params(self._model, adapter_params)

        set_activation_checkpointing(
            self._model, auto_wrap_policy={TransformerSelfAttentionLayer}
        )

        base_model_state_dict = self._checkpoint_dict[MODEL_KEY]
        self._model.load_state_dict(base_model_state_dict, strict=False)

    def _setup_optimizer(self):
        config = self.extra_config["optimizer"]
        optimizer_cls = get_component(config.pop("component"))
        self._optimizer = optimizer_cls(params=self._model.parameters(), **config)

    def _setup_lr_scheduler(self):
        config = self.extra_config["lr_scheduler"]
        lr_scheduler_cls = get_component(config.pop("component"))
        self._lr_scheduler = lr_scheduler_cls(
            optimizer=self._optimizer,
            num_training_steps=self._max_queries,
            **config,
        )

    @log_phase
    def prepare(self):
        super().prepare()

        self._setup_model()
        self._setup_optimizer()
        self._setup_lr_scheduler()

    def run(self, query: Query) -> dict[int, Query]:
        batch = query.data

        tokens, labels = batch["tokens"], batch["labels"]
        # Get the attention mask and position ids from the dataset if they
        # exist. Currently, only sample packing in PackedDataset returns these
        mask = batch.get("mask", None)  # shape [b, s, s]
        input_pos = batch.get("input_pos", None)  # shape [b, s]

        tokens = tokens.to(self._device)
        labels = labels.to(self._device)
        mask = mask.to(self._device) if mask is not None else None
        input_pos = input_pos.to(self._device) if input_pos is not None else None

        logits = self._model(tokens, mask=mask, input_pos=input_pos)
        # Shift so that tokens < n predict n
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()
        logits = logits.transpose(1, 2)
        loss = self._loss_fn(logits, labels)
        loss = loss / self._gradient_accumulation_steps
        self._running_loss += loss
        loss.backward()

        if (self._current_step + 1) % self._gradient_accumulation_steps == 0:
            self._optimizer.step()
            self._optimizer.zero_grad(set_to_none=True)
            self._lr_scheduler.step()

            print(
                f"{self._current_step + 1}/{self._max_queries} | Running loss: {self._running_loss.item()}"
            )
            # Reset running stats for the next step
            self._running_loss = 0

        self._current_step += 1

        query.data = loss.item()
        outputs = {idx: query for idx in self.output_queues}
        return outputs
