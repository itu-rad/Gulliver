from ast import parse
from os import replace
from pandas import lreshape
import torch
from torch.nn import Linear, CrossEntropyLoss
from torch.optim import SGD
from torchvision.models import get_model

from stages.stage import Stage, log_phase, log_phase_single
from utils.schemas import Query, StageModel, PipelineModel
from utils.component import get_component


class TorchVisionDetection(Stage):
    """This stage loads a TorchVision model and performs transfer learning
    and/or inference using the pretrained model."""

    def __init__(self, stage_config: StageModel, pipeline_config: PipelineModel):
        """Initialize the stage by parsing the stage configuration.

        Args:
            stage_config (dict): Configuration, which includes the name of the model,
            possible checkpoint and change to the head of the classifier.

        Raises:
            Exception: Missing model name.
            Exception: Model checkpoint path is required for inference.
        """
        super().__init__(stage_config, pipeline_config)

        self._model = None
        self._optimizer = None
        self._criterion = None

        self._device = self._parse_device(self.extra_config["device"])
        self._num_classes = self.extra_config["model"].get("num_classes", 1000)

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

    @log_phase
    def prepare(self):
        """Build the model according to the config and load the weights"""
        super().prepare()

        model_cfg = self.extra_config["model"]

        model_cls = model_cfg.pop("component")

        self._model = get_component(model_cls)(**model_cfg)

        model_checkpoint_path = self.extra_config["model"].get(
            "model_checkpoint_path", None
        )
        if model_checkpoint_path:
            self._model.load_state_dict(torch.load(model_checkpoint_path))

        self._model = self._model.to(self._device)

        self._amp = self.extra_config["model"].get("amp", False)

        self._scaler = torch.amp.GradScaler(enabled=self._amp)

        optimizer_cfg = self.extra_config.get("optimizer", None)

        if optimizer_cfg:
            optimizer_cls = optimizer_cfg.pop("component")
            self._optimizer = get_component(optimizer_cls)(
                [x for x in self._model.parameters() if x.requires_grad],
                **optimizer_cfg
            )

    def run(self, query: Query) -> dict[int, Query]:
        batch = query.data
        [inputs, targets] = batch
        inputs = [x.to(self._device) for x in inputs]
        targets = [{k: v.to(self._device) for k, v in t.items()} for t in targets]

        if query.split == "val":
            self._model.eval()
        else:
            self._model.train()

        with torch.set_grad_enabled(query.split == "train"):
            with torch.amp.autocast(enabled=self._amp):
                if query.split == "val":
                    outputs = self._model(inputs)
                    query.data = {
                        "outputs": outputs,
                        "labels": targets,
                    }
                else:
                    outputs = self._model(inputs, targets)
                    losses = sum(loss for loss in outputs.values())
                    loss_value = losses.item()
                    self._scaler.scale(losses).backward()
                    self._scaler.step(self._optimizer)
                    self._scaler.update()
                    self._optimizer.zero_grad()
                    query.data = loss_value

        output = {idx: query for idx in self.output_queues}
        return output
