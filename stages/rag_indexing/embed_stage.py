import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from stages.stage import Stage, log_phase
from utils.schemas import Query


class EmbedStage(Stage):
    """Embeds a batch of passages with a transformers sentence encoder
    (mean pooling + L2 normalisation).

    YAML config example:
        component: stages.rag_indexing.EmbedStage
        config:
          model:
            name: sentence-transformers/all-MiniLM-L6-v2
          device: mps        # mps | cuda | cpu
          max_length: 256
    """

    def __init__(self, stage_config, pipeline_config):
        super().__init__(stage_config, pipeline_config)
        self._model_name = self.extra_config["model"]["name"]
        self._device = self.extra_config.get("device", "cpu")
        self._max_length = int(self.extra_config.get("max_length", 256))
        self._tokenizer = None
        self._model = None

    @log_phase
    def prepare(self):
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModel.from_pretrained(self._model_name)
        self._model = self._model.to(self._device).eval()
        super().prepare()

    def run(self, query: Query) -> dict[int, Query]:
        texts: list[str] = query.data or []
        if not texts:
            query.data = ([], np.zeros((0, 0), dtype=np.float32))
            return {idx: query for idx in self.output_queues}
        inputs = self._tokenizer(texts, padding=True, truncation=True,
                                 max_length=self._max_length,
                                 return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            summed = (out.last_hidden_state * mask).sum(dim=1)
            emb = summed / mask.sum(dim=1).clamp(min=1e-9)
            emb = torch.nn.functional.normalize(emb, dim=1)
        query.data = (texts, emb.cpu().numpy().astype(np.float32))
        return {idx: query for idx in self.output_queues}
