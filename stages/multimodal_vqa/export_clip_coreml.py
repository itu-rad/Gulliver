"""Offline script to export the CLIP vision tower to CoreML.

Usage:
    python stages/multimodal_vqa/export_clip_coreml.py \
        --model openai/clip-vit-base-patch32 \
        --output tmp/clip_vit_b32_vision.mlpackage
"""

import argparse
import os

import coremltools as ct
import torch
from transformers import CLIPModel


class _CLIPImageFeaturesWrapper(torch.nn.Module):
    """Wraps CLIPModel to return projected image features as a tensor.

    Replicates get_image_features() so the CoreML output matches the
    512-dim projected embeddings used by encode_texts(), ensuring FAISS
    index dimensions match at query time.

    Core ML cannot convert dict-returning ops, so intermediate outputs are
    reduced to plain tensors before tracing.
    """

    def __init__(self, clip_model):
        super().__init__()
        self.vision_model = clip_model.vision_model
        self.visual_projection = clip_model.visual_projection

    def forward(self, pixel_values):
        vision_outputs = self.vision_model(pixel_values)
        pooled_output = vision_outputs[1]  # pooler_output
        return self.visual_projection(pooled_output)


def export_clip_vision_coreml(model_name: str, output_path: str):
    """Export the CLIP vision encoder + projection to CoreML format.

    The exported model replicates CLIPModel.get_image_features(), so
    output embeddings have the same dimension as CLIP text embeddings
    (512 for ViT-B/32) and can be used directly with a FAISS index
    built from text embeddings.

    Args:
        model_name: HuggingFace model identifier.
        output_path: Path to save the .mlpackage.
    """
    print(f"Loading CLIPModel from {model_name}")
    clip_model = CLIPModel.from_pretrained(model_name)
    clip_model.eval()
    model = _CLIPImageFeaturesWrapper(clip_model)

    # CLIP ViT-B/32 expects (1, 3, 224, 224)
    dummy_input = torch.randn(1, 3, 224, 224)

    print("Tracing model with torch.jit.trace")
    traced_model = torch.jit.trace(model, dummy_input, strict=False)

    print("Converting to CoreML")
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(
                name="pixel_values",
                shape=(1, 3, 224, 224)
            )
        ],
        compute_precision=ct.precision.FLOAT16,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print(f"Saving CoreML model to {output_path}")
    mlmodel.save(output_path)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export CLIP vision tower to CoreML"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="HuggingFace model name (e.g. openai/clip-vit-base-patch32)"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output path for .mlpackage"
    )
    args = parser.parse_args()

    export_clip_vision_coreml(args.model, args.output)
