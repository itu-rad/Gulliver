import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

from stages.stage import Stage, log_phase
from utils.schemas import StageModel, PipelineModel, Query


def _extract_projected(out, projection):
    """Pull a projected (embed_dim) tensor out of a CLIP get_*_features result.

    transformers 4.x returns the projected Tensor directly.
    transformers 5.x returns a BaseModelOutputWithPooling: pooler_output
    may be already-projected (`embed_dim`) or pre-projection (`hidden_size`).
    """
    if isinstance(out, torch.Tensor):
        tensor = out
    else:
        pooled = getattr(out, "pooler_output", None)
        if pooled is None:
            pooled = out[1] if isinstance(out, tuple) and len(out) > 1 else out.last_hidden_state[:, 0]
        tensor = pooled
    # If the tensor dim matches the projection's INPUT dim, project; else
    # it's already the output of the projection (embed_dim) — return as-is.
    expected_in = projection.in_features
    if tensor.shape[-1] == expected_in and tensor.shape[-1] != projection.out_features:
        return projection(tensor)
    return tensor


def _get_clip_text_features(model, inputs):
    """Robust CLIP text feature extraction across transformers 4.x and 5.x."""
    return _extract_projected(model.get_text_features(**inputs), model.text_projection)


def _get_clip_image_features(model, inputs):
    """Robust CLIP image feature extraction across transformers 4.x and 5.x."""
    return _extract_projected(model.get_image_features(**inputs), model.visual_projection)


class CLIPVisionEncoder(Stage):
    """Encodes images into CLIP embeddings using transformers.

    YAML config example:
        config:
          model:
            name: openai/clip-vit-base-patch32
          device: mps
    """

    def __init__(self, stage_config: StageModel,
                 pipeline_config: PipelineModel):
        """Initialize CLIP vision encoder.

        Args:
            stage_config: Stage configuration from YAML.
            pipeline_config: Pipeline configuration from YAML.
        """
        super().__init__(stage_config, pipeline_config)

        self._model_name = self.extra_config["model"]["name"]
        self._device = self.extra_config.get("device", "mps")

        # Load processor eagerly so get_clip_processor() and
        # encode_texts() are available during other stages' prepare().
        self._processor = CLIPProcessor.from_pretrained(
            self._model_name
        )
        self._model = None

    def get_clip_processor(self):
        """Return the CLIP processor for external use.

        Returns:
            CLIPProcessor: The loaded CLIP processor.
        """
        return self._processor

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into CLIP text embeddings.

        Called by FAISSRetriever during prepare() to build the index.
        Not called during run(), so no mutex needed.

        Args:
            texts: List of text strings to encode.

        Returns:
            np.ndarray: Text embeddings array of shape (N, embed_dim).
        """
        batch_size = 64
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self._processor(
                text=batch, return_tensors="pt",
                padding=True, truncation=True
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                text_features = _get_clip_text_features(self._model, inputs)
            all_embeddings.append(text_features.cpu().numpy())
            # Periodically release cached MPS memory; a long indexing loop
            # otherwise accumulates intermediates until it runs out of memory.
            if self._device == "mps" and (i // batch_size) % 16 == 15:
                torch.mps.empty_cache()
        return np.concatenate(all_embeddings, axis=0)

    @log_phase
    def prepare(self):
        """Load the CLIP model before starting the worker thread."""
        print(f"CLIPVisionEncoder: loading {self._model_name}")
        self._model = CLIPModel.from_pretrained(self._model_name)
        self._model = self._model.to(self._device)
        self._model.eval()

        super().prepare()

    def run(self, query: Query) -> dict[int, Query]:
        """Encode a PIL image into a CLIP embedding.

        Args:
            query: Query with data=(PIL_image or pixel_values tensor, question_str).

        Returns:
            dict[int, Query]: Query with data=image_embedding (numpy).
        """
        image, question = query.data

        if isinstance(image, torch.Tensor):
            # already preprocessed by CLIPImagePreprocess (pixel_values on the CPU)
            inputs = {"pixel_values": image.to(self._device)}
        else:
            inputs = self._processor(
                images=image, return_tensors="pt"
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = _get_clip_image_features(self._model, inputs)

        query.data = image_features.cpu().numpy()

        output = {idx: query for idx in self.output_queues}
        return output


class CLIPImagePreprocess(Stage):
    """CPU-side CLIP image preprocessing (resize, crop, normalize) as its own
    stage, so the following encoder stage covers only the device work.

    YAML config example:
        config:
          model:
            name: openai/clip-vit-large-patch14

    Input: query.data=(PIL_image, question_str). Output: query.data=(pixel_values tensor on the CPU, question_str).
    """

    def __init__(self, stage_config: StageModel, pipeline_config: PipelineModel):
        super().__init__(stage_config, pipeline_config)
        self._processor = CLIPProcessor.from_pretrained(self.extra_config["model"]["name"])

    def run(self, query: Query) -> dict[int, Query]:
        pil_image, question = query.data
        pixel_values = self._processor(images=pil_image, return_tensors="pt")["pixel_values"]
        query.data = (pixel_values, question)
        return {idx: query for idx in self.output_queues}
