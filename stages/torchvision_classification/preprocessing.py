import torch
from torchvision.models import get_weight

from stages.stage import Stage, log_phase
from utils.schemas import Query, StageModel, PipelineModel


class TorchVisionPreprocessFromWeights(Stage):
    """TorchVision stage for preprocessing. The preprocessing steps stem from the
    preprocessing pipeline associated with the pretrained weights of a TorchVision model.
    """

    def __init__(self, stage_config: StageModel, pipeline_config: PipelineModel):
        """Initialize the stage by parsing the stage configuration.

        Args:
            stage_config (_type_): Stage configuration, such as the TorchVision weight names
            to extract preprocessing steps from.
        """
        super().__init__(stage_config, pipeline_config)

        self._preprocess = None

    @log_phase
    def prepare(self):
        """Load in the weights and the corresponding preprocessing pipeline"""
        super().prepare()

        weight_name = self.extra_config.get("weights", None)

        if weight_name is None:
            raise ValueError("Weight name is required.")

        weights = get_weight(weight_name)
        self._preprocess = weights.transforms()

    def run(self, query: Query) -> dict[int, Query]:
        batch = query.data

        preprocessed_inputs = []
        labels = []
        for [sample, label] in batch:
            preprocessed_sample = self._preprocess(sample)
            preprocessed_inputs.append(preprocessed_sample)
            labels.append(label)

        preprocessed_inputs = torch.stack(preprocessed_inputs)
        labels = torch.Tensor(labels)

        query.data = [preprocessed_inputs, labels]
        output = {idx: query for idx in self.output_queues}
        return output
