"""Port of the MLPerf-Inference 3D-UNet / KiTS19 reference math.

Mirrors `vision/medical_imaging/3d-unet-kits19` in mlcommons/inference
(global_vars.py, preprocess.py, inference_utils.py, base_SUT.py). Depends only
on numpy and scipy; nibabel is needed only to load volumes.
"""

from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import zoom

# --- global_vars.py (verbatim) ---------------------------------------------
MEAN_VAL = 101.0
STDDEV_VAL = 76.9
MIN_CLIP_VAL = -79.0
MAX_CLIP_VAL = 304.0
PADDING_VAL = -2.2
TARGET_SPACING = [1.6, 1.2, 1.2]
ROI_SHAPE = [128, 128, 128]
SLIDE_OVERLAP_FACTOR = 0.5


# --- preprocess.py: resample / normalize / pad -----------------------------

def load_and_resample(image_path, label_path, target_spacing=TARGET_SPACING):
    """nibabel-load a CT image (+ optional segmentation) and resample to the
    common voxel spacing. Returns (image[1,D,H,W] float32, label or None, aux).

    order=1 (trilinear) for the image, order=0 (nearest) for the integer label,
    exactly as the reference. nibabel is imported lazily so importing this module
    never requires it.
    """
    import nibabel  # lazy: only preprocessing needs it

    image_nii = nibabel.load(str(image_path))
    image_spacings = image_nii.header["pixdim"][1:4].tolist()
    original_affine = image_nii.affine
    image = image_nii.get_fdata().astype(np.float32)

    label = None
    if label_path is not None:
        label = nibabel.load(str(label_path)).get_fdata().astype(np.uint8)

    zoom_factor = (np.array(image_spacings) / np.array(target_spacing)).tolist()

    if image_spacings != list(target_spacing):
        image = zoom(image, zoom_factor, order=1, mode="constant",
                     cval=image.min(), grid_mode=False)
        if label is not None:
            label = zoom(label, zoom_factor, order=0, mode="constant",
                         cval=label.min(), grid_mode=False)

    aux = {"original_affine": original_affine, "zoom_factor": zoom_factor}
    image = np.expand_dims(image, 0)
    if label is not None:
        label = np.expand_dims(label, 0)
    return image, label, aux


def normalize_intensity(image, min_val=MIN_CLIP_VAL, max_val=MAX_CLIP_VAL,
                        mean=MEAN_VAL, std=STDDEV_VAL):
    """Clip to [min,max] then standardize by the KiTS19 corpus mean/std."""
    image = np.clip(image, min_val, max_val)
    return (image - mean) / std


def pad_to_min_shape(image, roi_shape=ROI_SHAPE):
    """Edge-pad so every spatial dim is >= the ROI (128)."""
    current = image.shape[1:]
    bounds = [max(0, roi_shape[i] - current[i]) for i in range(3)]
    paddings = [(0, 0)] + [(bounds[i] // 2, bounds[i] - bounds[i] // 2)
                           for i in range(3)]
    return np.pad(image, paddings, mode="edge")


def _constant_pad_volume(volume, roi_shape, strides, padding_val, dim=3):
    bounds = [(strides[i] - volume.shape[1:][i] % strides[i]) % strides[i]
              for i in range(dim)]
    bounds = [bounds[i] if (volume.shape[1:][i] + bounds[i]) >= roi_shape[i]
              else bounds[i] + strides[i] for i in range(dim)]
    paddings = [(0, 0)] + [(bounds[i] // 2, bounds[i] - bounds[i] // 2)
                           for i in range(dim)]
    return np.pad(volume, paddings, mode="constant", constant_values=padding_val)


def adjust_shape_for_sliding_window(image, roi_shape=ROI_SHAPE,
                                    overlap=SLIDE_OVERLAP_FACTOR,
                                    padding_val=PADDING_VAL):
    """Crop the small remainder then constant-pad so the volume tiles cleanly
    under the sliding window (edges become stride-divisible)."""
    image_shape = list(image.shape[1:])
    strides = [int(roi_shape[i] * (1 - overlap)) for i in range(3)]
    bounds = [image_shape[i] % strides[i] for i in range(3)]
    bounds = [bounds[i] if bounds[i] < strides[i] // 2 else 0 for i in range(3)]
    image = image[...,
                  bounds[0] // 2: image_shape[0] - (bounds[0] - bounds[0] // 2),
                  bounds[1] // 2: image_shape[1] - (bounds[1] - bounds[1] // 2),
                  bounds[2] // 2: image_shape[2] - (bounds[2] - bounds[2] // 2)]
    return _constant_pad_volume(image, roi_shape, strides, padding_val)


def preprocess_volume(image_path, label_path=None):
    """Full reference preprocessing chain for one case → (image, label, aux).

    image is [1, D, H, W] float32, ready for sliding-window inference.
    """
    image, label, aux = load_and_resample(image_path, label_path)
    image = normalize_intensity(image.copy())
    image = pad_to_min_shape(image)
    if label is not None:
        label = pad_to_min_shape(label)
    image = adjust_shape_for_sliding_window(image)
    if label is not None:
        label = adjust_shape_for_sliding_window(label, padding_val=0)
    return image, label, aux


# --- inference_utils.py / base_SUT.py: sliding window ----------------------

def gaussian_kernel(n, std):
    """Cube-root-normalized separable 3-D Gaussian patch weight (reference)."""
    g1d = signal.windows.gaussian(n, std)
    g2d = np.outer(g1d, g1d)
    g3d = np.outer(g2d, g1d).reshape(n, n, n)
    g3d = np.cbrt(g3d)
    g3d /= g3d.max()
    return g3d


def prepare_arrays(image, roi_shape=ROI_SHAPE):
    """Empty accumulators for weighted sliding-window aggregation."""
    image_shape = list(image.shape[2:])
    result = np.zeros(shape=(1, 3, *image_shape), dtype=image.dtype)
    norm_map = np.zeros_like(result)
    norm_patch = gaussian_kernel(roi_shape[0], 0.125 * roi_shape[0]).astype(
        norm_map.dtype)
    return result, norm_map, norm_patch


def get_slice_for_sliding_window(image, roi_shape=ROI_SHAPE,
                                 overlap=SLIDE_OVERLAP_FACTOR):
    """Yield (i,j,k) top-left corners tiling the volume with the given overlap."""
    image_shape = list(image.shape[2:])
    strides = [int(roi_shape[i] * (1 - overlap)) for i in range(3)]
    size = [(image_shape[i] - roi_shape[i]) // strides[i] + 1 for i in range(3)]
    for i in range(0, strides[0] * size[0], strides[0]):
        for j in range(0, strides[1] * size[1], strides[1]):
            for k in range(0, strides[2] * size[2], strides[2]):
                yield i, j, k


def finalize(result, norm_map):
    """Divide accumulated logits by the coverage map, then argmax → uint8 mask."""
    result = result / norm_map
    result = np.argmax(result, axis=1).astype(np.uint8)
    return np.expand_dims(result, axis=0)


def count_subvolumes(image, roi_shape=ROI_SHAPE, overlap=SLIDE_OVERLAP_FACTOR):
    """Number of forward passes for a preprocessed volume."""
    return sum(1 for _ in get_slice_for_sliding_window(image, roi_shape, overlap))
