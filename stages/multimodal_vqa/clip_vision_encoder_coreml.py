import numpy as np
import torch
import coremltools as ct
from transformers import CLIPModel, CLIPProcessor

from stages.stage import Stage, log_phase
from stages.multimodal_vqa.clip_vision_encoder import _get_clip_text_features
from utils.schemas import StageModel, PipelineModel, Query


class CLIPVisionEncoderCoreML(Stage):
    """Encodes images using a pre-exported Core ML CLIP vision encoder.

    Core ML decides the compute unit (CPU, GPU or Neural Engine).

    YAML config example:
        config:
          model:
            coreml_path: tmp/clip_vit_b32_vision.mlpackage
            hf_name: openai/clip-vit-base-patch32
          device: ane
    """

    def __init__(self, stage_config: StageModel,
                 pipeline_config: PipelineModel):
        """Initialize CoreML CLIP vision encoder.

        Args:
            stage_config: Stage configuration from YAML.
            pipeline_config: Pipeline configuration from YAML.
        """
        super().__init__(stage_config, pipeline_config)

        self._coreml_path = self.extra_config["model"]["coreml_path"]
        self._hf_name = self.extra_config["model"]["hf_name"]

        # Load processor eagerly for get_clip_processor() availability.
        self._processor = CLIPProcessor.from_pretrained(self._hf_name)
        self._coreml_model = None
        # For encode_texts() during FAISSRetriever.prepare()
        self._text_model = None

    def get_clip_processor(self):
        """Return the CLIP processor for external use.

        Returns:
            CLIPProcessor: The loaded CLIP processor.
        """
        return self._processor

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into CLIP text embeddings.

        Loads a CLIPModel onto MPS for fast one-shot indexing, then
        releases it. The CoreML image encoder (ANE) remains primary;
        the text path is only exercised during FAISSRetriever.prepare().

        Args:
            texts: List of text strings to encode.

        Returns:
            np.ndarray: Text embeddings array of shape (N, embed_dim).
        """
        text_device = "mps" if torch.backends.mps.is_available() else "cpu"
        text_model = CLIPModel.from_pretrained(self._hf_name)
        text_model = text_model.to(text_device)
        text_model.eval()

        batch_size = 64
        all_embeddings = []
        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                inputs = self._processor(
                    text=batch, return_tensors="pt",
                    padding=True, truncation=True
                )
                inputs = {k: v.to(text_device) for k, v in inputs.items()}
                with torch.no_grad():
                    text_features = _get_clip_text_features(text_model, inputs)
                all_embeddings.append(text_features.cpu().numpy())
        finally:
            del text_model
            if text_device == "mps":
                torch.mps.empty_cache()
        return np.concatenate(all_embeddings, axis=0)

    @log_phase
    def prepare(self):
        """Load CoreML model before starting the worker thread."""
        print(
            f"CLIPVisionEncoderCoreML: loading CoreML model "
            f"from {self._coreml_path}"
        )
        self._coreml_model = ct.models.MLModel(
            self._coreml_path,
            compute_units=ct.ComputeUnit.ALL
        )

        super().prepare()

    def run(self, query: Query) -> dict[int, Query]:
        """Encode a PIL image into a CLIP embedding via CoreML.

        Args:
            query: Query with data=(PIL_image, question_str).

        Returns:
            dict[int, Query]: Query with data=image_embedding (numpy).
        """
        pil_image, question = query.data

        # Fast image processors only emit torch tensors (return_tensors="np"
        # raises), so request "pt" and convert to numpy for Core ML.
        inputs = self._processor(
            images=pil_image, return_tensors="pt"
        )
        pixel_values = inputs["pixel_values"].numpy().astype(np.float32)

        prediction = self._coreml_model.predict(
            {"pixel_values": pixel_values}
        )

        # CoreML output key depends on export; use first available.
        output_key = list(prediction.keys())[0]
        embedding = prediction[output_key]
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)

        # Ensure shape is (1, embed_dim)
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        query.data = embedding

        output = {idx: query for idx in self.output_queues}
        return output
