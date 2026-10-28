# 3D-UNet / KiTS19: model and data

## 1. Python dependency

Preprocessing loads NIfTI volumes with `nibabel` (imported lazily):

    pip install nibabel

## 2. Model (TorchScript)

The inference stage loads a TorchScript module with `torch.jit.load`. Use the
MLPerf "PyTorch model" from Zenodo record 5597155, which is already TorchScript
(the "PyTorch/checkpoint" variant is a raw state_dict and will not load):

    mkdir -p models/3dunet_kits19
    wget -O models/3dunet_kits19/3dunet_kits19_pytorch.ptc \
      "https://zenodo.org/record/5597155/files/3dunet_kits19_pytorch.ptc?download=1"

The model takes `[1,1,128,128,128]` float input and returns
`[1,3,128,128,128]` logits (background, kidney, tumor).

## 3. KiTS19 data

    git clone https://github.com/neheller/kits19
    cd kits19 && python -m starter_code.get_imaging

The MLPerf inference cases are listed in the reference
`meta/inference_cases.json`. Point the case loader's `raw_data_dir` at the
directory holding the `case_XXXXX/` directories (each with `imaging.nii.gz`
and `segmentation.nii.gz`). Preprocessing is done by the `KiTS19Preprocess`
stage at run time, so no offline preprocessing step is needed.

## 4. Running

    python main.py <config>.yml

Set `device` on the inference stage (`cuda`, `mps` or `cpu`). For accuracy
runs, end the pipeline with `KiTS19DiceScore` and set `keep_label: true` on
the preprocess stage.
