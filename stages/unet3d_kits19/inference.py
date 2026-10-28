import numpy as np
import torch

from stages.stage import Stage, log_phase
from utils.schemas import Query

from . import kits19_lib as kl


class UNet3DInference(Stage):
    """Gaussian-weighted sliding-window inference for the MLPerf 3D-UNet
    (nnU-Net) KiTS19 model.

    Port of MLPerf's base_SUT.infer_single_query: tile the volume into ROIs
    with overlap, accumulate Gaussian-weighted logits, normalise by the
    coverage map and take the argmax. Config: ``model_path`` (TorchScript)
    and optional ``device`` (cuda, mps or cpu; auto-detected if unset).

    Input : query.data = {case, image, label, aux, n_subvolumes}
    Output: query.data = {case, segmentation[1,1,D,H,W] uint8, label, aux}
    """

    def __init__(self, stage_config, pipeline_config):
        super().__init__(stage_config, pipeline_config)
        self._model_path = self.extra_config["model_path"]
        self._device_str = self.extra_config.get("device")
        self._roi = kl.ROI_SHAPE
        self._overlap = kl.SLIDE_OVERLAP_FACTOR
        self._device = None
        self._model = None

    def _parse_device(self):
        if self._device_str:
            return torch.device(self._device_str)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @log_phase
    def prepare(self):
        self._device = self._parse_device()
        print(f"UNet3DInference: loading TorchScript model {self._model_path} "
              f"on {self._device}")
        self._model = torch.jit.load(self._model_path, map_location=self._device)
        self._model.eval()
        super().prepare()

    @torch.no_grad()
    def _infer_volume(self, image: np.ndarray) -> np.ndarray:
        # image: [1, D, H, W] -> add batch dim -> [1, 1, D, H, W]
        image = image[np.newaxis, ...]
        result, norm_map, norm_patch = kl.prepare_arrays(image, self._roi)

        t_image = torch.from_numpy(image).float().to(self._device)
        t_result = torch.from_numpy(result).float().to(self._device)
        t_norm_map = torch.from_numpy(norm_map).float().to(self._device)
        t_norm_patch = torch.from_numpy(norm_patch).float().to(self._device)

        for i, j, k in kl.get_slice_for_sliding_window(image, self._roi, self._overlap):
            di, dj, dk = self._roi
            sl = (..., slice(i, i + di), slice(j, j + dj), slice(k, k + dk))
            out = self._model(t_image[sl])
            t_result[sl] += out * t_norm_patch
            t_norm_map[sl] += t_norm_patch

        result = t_result.cpu().numpy().astype(np.float64)
        norm_map = t_norm_map.cpu().numpy().astype(np.float64)
        return kl.finalize(result, norm_map)

    def run(self, query: Query) -> dict[int, Query]:
        payload = query.data
        segmentation = self._infer_volume(payload["image"])
        query.data = {
            "case": payload["case"],
            "segmentation": segmentation,
            "label": payload.get("label"),
            "aux": payload.get("aux"),
        }
        return {idx: query for idx in self.output_queues}
