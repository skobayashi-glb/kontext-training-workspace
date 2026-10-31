from __future__ import annotations

import base64
from io import BytesIO

import torch
from PIL import Image

from .augmentation import load_rgba_as_rgb
from .schemas import EvaluationPreview, TrainingPair


class EvaluationPreviewBuilder:
    def __init__(self, resolution: int, inference_steps: int = 16) -> None:
        self.resolution = resolution
        self.inference_steps = inference_steps

    @torch.no_grad()
    def build(
        self,
        pipe,
        pair: TrainingPair,
        epoch: int,
        step: int,
        generator: torch.Generator,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
    ) -> EvaluationPreview:
        input_rgb = load_rgba_as_rgb(pair.input_image_path)
        try:
            prediction = pipe(
                image=input_rgb,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                height=self.resolution,
                width=self.resolution,
                num_inference_steps=self.inference_steps,
                generator=generator,
                output_type="pil",
            ).images[0]
            input_preview = self._to_data_url(input_rgb)
        finally:
            input_rgb.close()

        target_rgb = load_rgba_as_rgb(pair.target_image_path)
        try:
            target_preview = self._to_data_url(target_rgb)
        finally:
            target_rgb.close()

        return EvaluationPreview(
            epoch=epoch,
            step=step,
            prompt=pair.instruction,
            input_image=input_preview,
            target_image=target_preview,
            prediction_image=self._to_data_url(prediction.convert("RGB")),
        )

    def _to_data_url(self, image: Image.Image) -> str:
        preview = image.copy()
        preview.thumbnail((512, 512), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        preview.save(buffer, format="PNG", optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
