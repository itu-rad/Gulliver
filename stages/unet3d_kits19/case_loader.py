import json
from pathlib import Path

from stages.stage import Stage, log_phase
from utils.schemas import Query


class KiTS19CaseLoader(Stage):
    """Source stage for the MLPerf 3D-UNet / KiTS19 medical-segmentation workload.

    Emits one case per query (case id and file paths only); loading and
    preprocessing happen in the downstream ``KiTS19Preprocess`` stage.

    YAML config::

        config:
          raw_data_dir: data/kits19/raw          # contains case_00000/... dirs
          cases_json: null                       # optional explicit case list
          max_cases: null                        # cap for smoke runs
    """

    def __init__(self, stage_config, pipeline_config):
        super().__init__(stage_config, pipeline_config)
        self._raw_data_dir = Path(self.extra_config["raw_data_dir"])
        self._cases_json = self.extra_config.get("cases_json")
        self._max_cases = self.extra_config.get("max_cases")
        self._cases: list[str] = []
        self._data_index = 0

    def get_dataset_splits(self) -> dict[str, int]:
        return {"inference": len(self._cases)}

    @log_phase
    def prepare(self):
        super().prepare()
        if self._cases_json:
            cases = json.loads(Path(self._cases_json).read_text())
        else:
            cases = sorted(p.name for p in self._raw_data_dir.iterdir()
                           if p.is_dir() and p.name.startswith("case"))
        if self._max_cases:
            cases = cases[: int(self._max_cases)]
        self._cases = cases
        print(f"KiTS19CaseLoader: {len(self._cases)} cases from {self._raw_data_dir}")

    def run(self, query: Query) -> dict[int, Query]:
        if query.batch == 0:
            self._data_index = 0
        case = self._cases[self._data_index % len(self._cases)]
        self._data_index += 1

        case_dir = self._raw_data_dir / case
        image_path = case_dir / "imaging.nii.gz"
        label_path = case_dir / "segmentation.nii.gz"
        query.data = {
            "case": case,
            "image_path": str(image_path),
            "label_path": str(label_path) if label_path.exists() else None,
        }
        query.context = {"case": case}
        return {idx: query for idx in self.output_queues}
