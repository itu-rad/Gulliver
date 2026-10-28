import torch
from torch.nn import Linear, CrossEntropyLoss
from torch.optim import SGD
from torchvision.models import get_model

from stages.stage import Stage, log_phase, log_phase_single
from utils.schemas import Query, StageModel, PipelineModel
from utils.component import get_component


class TorchVisionClassification(Stage):
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

    def _replace_last_module(self, module):
        """Replace the last module/layer inside the classifier head of the model.

        Args:
            module (tuple[str, torch.Module]): The last (name, module) tuple in the neural network
        """
        last_module_name = module[0].split(".")

        if len(last_module_name) == 1:
            self._model[last_module_name[0]] = Linear(
                module[1].in_features, self._num_classes
            )
        else:
            self._model.__getattr__(".".join(last_module_name[:-1]))[
                int(last_module_name[-1])
            ] = Linear(module[1].in_features, self._num_classes)

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

        model_cls = self.extra_config["model"]["component"]
        weights_cls = self.extra_config["model"].get("weights", None)

        self._model = get_component(model_cls)(weights=weights_cls)

        replace_classifier = self.extra_config["model"].get("replace_classifier", False)
        *_, last_module = self._model.named_modules()
        if replace_classifier:
            self._replace_last_module(last_module)

        model_checkpoint_path = self.extra_config["model"].get(
            "model_checkpoint_path", None
        )
        if model_checkpoint_path:
            self._model.load_state_dict(torch.load(model_checkpoint_path))

        # Transfer learning: freeze everything, then unfreeze the last module.
        for param in self._model.parameters():
            param.requires_grad = False

        optimizer_config = self.extra_config.get("optimizer", None)

        # TODO: Parametrize the number of layers to be trainable
        params_to_update = []
        if optimizer_config:
            for name, param in self._model.named_parameters():
                if last_module[0] in name:
                    param.requires_grad = True
                    params_to_update.append(param)

        self._model = self._model.to(self._device)

        if optimizer_config:
            optimizer_cls = get_component(optimizer_config.pop("component"))
            self._optimizer = optimizer_cls(params_to_update, **optimizer_config)

        criterion_config = self.extra_config.get("criterion", None)
        if criterion_config:
            self._criterion = get_component(criterion_config["component"])()

        trainable_params = sum(
            p.numel() for p in self._model.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in self._model.parameters())
        print(f"Stage {self.name}: Number of trainable parameters: {trainable_params}")
        print(f"Stage {self.name}: Number of total parameters: {total_params}")

    def run(self, query: Query) -> dict[int, Query]:
        batch = query.data
        [inputs, labels] = batch
        inputs = inputs.to(self._device)
        labels = labels.type(torch.LongTensor)
        labels = labels.to(self._device)

        if query.split == "val":
            self._model.eval()
        else:
            self._model.train()

        with torch.set_grad_enabled(query.split == "train"):
            # Inference configs may omit `optimizer`.
            if query.split == "train" and self._optimizer is not None:
                self._optimizer.zero_grad()

            outputs = self._model(inputs)

            if query.split == "train":
                loss = self._criterion(outputs, labels)
                loss.backward()
                self._optimizer.step()

        # Synchronize so the stage's timing includes the device work.
        if self._device.type == "mps":
            torch.mps.synchronize()
        elif self._device.type == "cuda":
            torch.cuda.synchronize()

        output = {idx: query for idx in self.output_queues}
        return output
