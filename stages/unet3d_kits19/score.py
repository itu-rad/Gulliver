import csv
import os
import threading

import numpy as np

from stages.stage import Stage, log_phase
from utils.schemas import Query


# Smoothing terms from MLPerf's accuracy_kits.py, kept at the reference values.
SMOOTH_NR = 1e-6
SMOOTH_DR = 1e-6
N_CLASSES = 3            # background, kidney, tumor


def dice_per_class(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-class DICE for kidney and tumor, MLPerf's formula.

    Port of `get_dice_score` in the reference harness: 2|P n T| / (|P| + |T|)
    per non-background class. Both arrays must be the preprocessed, padded
    volumes (the label goes through the same chain as the image).
    """
    pred = np.asarray(prediction).reshape(-1)
    targ = np.asarray(target).reshape(-1)
    if pred.shape != targ.shape:
        raise ValueError(f"prediction {pred.shape} != label {targ.shape}; the "
                         f"two did not go through the same preprocessing")
    out = np.empty(N_CLASSES - 1, dtype=np.float64)
    for c in range(1, N_CLASSES):
        p = pred == c
        t = targ == c
        inter = float(np.count_nonzero(p & t))
        out[c - 1] = ((2.0 * inter + SMOOTH_NR)
                      / (float(np.count_nonzero(p)) + float(np.count_nonzero(t))
                         + SMOOTH_DR))
    return out


class KiTS19DiceScore(Stage):
    """End stage of an accuracy pipeline: score each case and write one CSV row.

    Meant for accuracy runs only; scoring is too expensive to sit inside a
    timed pipeline. Requires `keep_label: true` on the preprocess stage.

    Input : query.data = {case, segmentation, label, aux}
    Config: output_path -- CSV to write (default: dice.csv in the run's output directory)
    """

    def __init__(self, stage_config, pipeline_config):
        super().__init__(stage_config, pipeline_config)
        self._path = self.extra_config.get(
            "output_path",
            os.path.join(os.environ.get("SUITE_OUTPUT_DIR", "results"), "dice.csv"))
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        self._lock = threading.Lock()
        # Truncate so a new run does not append to stale data.
        with open(self._path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["case", "dice_kidney", "dice_tumor", "dice_mean"])

    @log_phase
    def prepare(self):
        super().prepare()

    def run(self, query: Query) -> dict[int, Query]:
        payload = query.data
        label = payload.get("label")
        if label is None:
            raise ValueError(
                f"{self.name}: case {payload.get('case')} arrived without a "
                f"label. The accuracy pipeline must set `keep_label: true` on "
                f"the preprocess stage; without it there is nothing to score "
                f"against and a silently empty DICE column would look like a "
                f"result.")
        d = dice_per_class(payload["segmentation"], label)
        with self._lock, open(self._path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([payload["case"], f"{d[0]:.6f}",
                                    f"{d[1]:.6f}", f"{float(d.mean()):.6f}"])
        return {idx: query for idx in self.output_queues}
