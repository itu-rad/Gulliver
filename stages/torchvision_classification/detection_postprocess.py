import torch
from torchvision.models import get_weight

from stages.stage import Stage, log_phase
from utils.schemas import Query, StageModel, PipelineModel


class DetectionPostprocess(Stage):
    """Scores TorchVision detection outputs against the expected classes."""

    def __init__(self, stage_config: StageModel, pipeline_config: PipelineModel):
        """Initialize the stage by parsing the stage configuration.

        Args:
            stage_config (_type_): Stage configuration, such as the TorchVision weight names
            to extract preprocessing steps from.
        """
        super().__init__(stage_config, pipeline_config)

    @log_phase
    def prepare(self):
        """Load in the weights and the corresponding preprocessing pipeline"""
        super().prepare()

    def run(self, query: Query) -> dict[int, Query]:
        if query.split == "train":
            return {idx: query for idx in self.output_queues}

        outputs = query.data["outputs"]
        labels = query.data["labels"]

        bboxes_ = [e["boxes"].cpu() for e in outputs]
        labels_ = [e["labels"].cpu() for e in outputs]
        scores_ = [e["scores"].cpu() for e in outputs]
        results = [bboxes_, labels_, scores_]

        good = 0
        total = 0

        processed_results = []
        bs = len(results[0])
        for idx in range(0, bs):
            processed_results.append([])
            detection_boxes = results[0][idx]
            detection_classes = results[1][idx]
            expected_classes = labels[idx]["category_id"]
            scores = results[2][idx]
            for detection, _ in enumerate(scores):
                if scores[detection] < 0.0:
                    break
                detection_class = int(detection_classes[detection])
                if detection_class in expected_classes:
                    good += 1
                box = detection_boxes[detection]
                # The model returns boxes as xmin, ymin, xmax, ymax.
                processed_results[idx].append(
                    [
                        box[1].item(),
                        box[0].item(),
                        box[3].item(),
                        box[2].item(),
                        scores[detection].item(),
                        float(detection_class),
                    ]
                )
                total += 1
        query.data = good / total
        print(f"Detection Postprocess: {good} / {total} = {query.data*100:.2f}%")
        output = {idx: query for idx in self.output_queues}
        return output
