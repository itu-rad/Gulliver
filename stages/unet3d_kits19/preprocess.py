from stages.stage import Stage, log_phase, marker_span
from utils.schemas import Query

from . import kits19_lib as kl


class KiTS19Preprocess(Stage):
    """Resample, clip/normalize and pad a KiTS19 case. MLPerf runs this offline;
    here it is a timed stage.

    Input  : query.data = {case, image_path, label_path} (from KiTS19CaseLoader)
    Output : query.data = {case, image[1,D,H,W] float32, label|None, aux,
                           n_subvolumes}
    """

    def __init__(self, stage_config, pipeline_config):
        super().__init__(stage_config, pipeline_config)
        self._keep_label = self.extra_config.get("keep_label", True)

    @log_phase
    def prepare(self):
        super().prepare()

    def run(self, query: Query) -> dict[int, Query]:
        payload = query.data
        label_path = payload.get("label_path") if self._keep_label else None
        image, label, aux = kl.preprocess_volume(
            payload["image_path"], label_path)

        n_sub = kl.count_subvolumes(image[None, ...] if image.ndim == 4 else image)
        query.data = {
            "case": payload["case"],
            "image": image,
            "label": label,
            "aux": aux,
            "n_subvolumes": n_sub,
        }
        # Record the case size on this query's trace. marker_span ignores
        # `disable_logs`, so the size is kept even with CSV logging off.
        marker_span(self, "case_size", {
            "case": payload["case"],
            "n_subvolumes": n_sub,
            # Post-resample shape, i.e. what inference tiles over.
            "image_shape": "x".join(str(d) for d in image.shape[-3:]),
        })
        return {idx: query for idx in self.output_queues}
