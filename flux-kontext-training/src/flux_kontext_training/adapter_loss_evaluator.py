from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from .augmentation import load_rgba_as_rgb
from .datasets import list_training_pairs, resolve_dataset
from .evaluation_preview import EvaluationPreviewBuilder
from .schemas import EvaluationPreview, TrainingConfig, TrainingPair
from .trainer import (
    PromptConditioningBuilder,
    TransformationFlowMatchingLoss,
    _calculate_timestep_shift,
    _hugging_face_token,
    _seed_everything,
    _time_shift,
    _training_dtype,
    release_cuda_memory,
)


SplitName = Literal["train", "val"]


@dataclass(frozen=True)
class LossSummary:
    split: SplitName
    pair_count: int
    batch_count: int
    noise_repeats: int
    mean_loss: float
    min_loss: float
    max_loss: float
    base_mean_loss: float | None = None
    relative_improvement: float | None = None
    focused_mean_loss: float | None = None
    focused_base_mean_loss: float | None = None
    focused_relative_improvement: float | None = None
    changed_token_fraction: float | None = None
    rollout_mean_loss: float | None = None
    rollout_base_mean_loss: float | None = None
    rollout_relative_improvement: float | None = None
    pixel_rollout_mean_loss: float | None = None
    pixel_rollout_base_mean_loss: float | None = None
    pixel_rollout_relative_improvement: float | None = None
    preview: EvaluationPreview | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "split": self.split,
            "pair_count": self.pair_count,
            "batch_count": self.batch_count,
            "noise_repeats": self.noise_repeats,
            "mean_loss": self.mean_loss,
            "min_loss": self.min_loss,
            "max_loss": self.max_loss,
            "base_mean_loss": self.base_mean_loss,
            "relative_improvement": self.relative_improvement,
            "focused_mean_loss": self.focused_mean_loss,
            "focused_base_mean_loss": self.focused_base_mean_loss,
            "focused_relative_improvement": self.focused_relative_improvement,
            "changed_token_fraction": self.changed_token_fraction,
            "rollout_mean_loss": self.rollout_mean_loss,
            "rollout_base_mean_loss": self.rollout_base_mean_loss,
            "rollout_relative_improvement": self.rollout_relative_improvement,
            "pixel_rollout_mean_loss": self.pixel_rollout_mean_loss,
            "pixel_rollout_base_mean_loss": self.pixel_rollout_base_mean_loss,
            "pixel_rollout_relative_improvement": self.pixel_rollout_relative_improvement,
            "preview": self.preview.model_dump() if self.preview is not None else None,
        }


@dataclass(frozen=True)
class BatchLossMeasurement:
    loss: torch.Tensor
    focused_loss: torch.Tensor
    changed_token_fraction: float


class AdapterLossEvaluator:
    """Evaluates the training flow-matching loss for an existing LoRA adapter."""

    def __init__(
        self,
        config: TrainingConfig,
        adapter_path: Path,
        batch_size: int | None = None,
        max_pairs: int | None = None,
        noise_repeats: int = 1,
        include_preview: bool = True,
        rollout_steps: int = 0,
    ) -> None:
        self.config = config
        self.adapter_path = adapter_path
        self.batch_size = batch_size or config.batch_size
        self.max_pairs = max_pairs
        self.noise_repeats = max(1, noise_repeats)
        self.include_preview = include_preview
        self.rollout_steps = max(0, rollout_steps)
        self.loss = TransformationFlowMatchingLoss(config.loss_delta_weight, config.loss_delta_quantile)
        self.prompt_builder = PromptConditioningBuilder([])

    def evaluate(self, splits: tuple[SplitName, ...]) -> list[LossSummary]:
        _seed_everything(self.config.seed)
        pipe = None
        try:
            pipe, device, dtype = self._load_pipeline()
            return [self._evaluate_split(pipe, split, device, dtype) for split in splits]
        finally:
            release_cuda_memory(pipe)

    def _load_pipeline(self):
        from diffusers import FluxKontextPipeline

        device = _evaluation_device()
        dtype = _training_dtype(device)
        pipe = FluxKontextPipeline.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
            token=_hugging_face_token(),
            local_files_only=True,
        )
        pipe.set_progress_bar_config(disable=True)
        pipe.load_lora_weights(str(self.adapter_path))
        pipe.transformer.to(dtype=dtype)
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)
        pipe.transformer.eval()
        pipe.vae.eval()
        if pipe.text_encoder is not None:
            pipe.text_encoder.eval()
        if pipe.text_encoder_2 is not None:
            pipe.text_encoder_2.eval()
        return pipe, device, dtype

    def _evaluate_split(
        self,
        pipe,
        split: SplitName,
        device: torch.device,
        dtype: torch.dtype,
    ) -> LossSummary:
        dataset = resolve_dataset(self.config.dataset)
        pairs = list_training_pairs(dataset, split, self.config.caption, self.config.instruction)
        if self.max_pairs is not None:
            pairs = pairs[: self.max_pairs]
        if not pairs:
            raise ValueError(f"Dataset '{self.config.dataset}' has no paired {split} images.")

        losses: list[float] = []
        focused_losses: list[float] = []
        changed_token_fractions: list[float] = []
        base_losses: list[float] = []
        focused_base_losses: list[float] = []
        rollout_losses: list[float] = []
        rollout_base_losses: list[float] = []
        pixel_rollout_losses: list[float] = []
        pixel_rollout_base_losses: list[float] = []
        batches = list(_batches(pairs, self.batch_size))
        for batch_pairs in batches:
            for _ in range(self.noise_repeats):
                prepared = self._prepare_loss_inputs(pipe, batch_pairs, device, dtype)
                self._offload_non_transformer_for_loss(pipe)
                adapter_measurement = self._measure_loss(pipe, prepared)
                losses.append(float(adapter_measurement.loss.detach().cpu()))
                focused_losses.append(float(adapter_measurement.focused_loss.detach().cpu()))
                changed_token_fractions.append(adapter_measurement.changed_token_fraction)

                base_measurement = self._measure_base_loss(pipe, prepared)
                if base_measurement is not None:
                    base_losses.append(float(base_measurement.loss.detach().cpu()))
                    focused_base_losses.append(float(base_measurement.focused_loss.detach().cpu()))

                if self.rollout_steps > 0:
                    rollout_a, rollout_b, px_a, px_b = self._measure_rollout_loss(pipe, prepared)
                    if rollout_a is not None:
                        rollout_losses.append(rollout_a)
                    if rollout_b is not None:
                        rollout_base_losses.append(rollout_b)
                    if px_a is not None:
                        pixel_rollout_losses.append(px_a)
                    if px_b is not None:
                        pixel_rollout_base_losses.append(px_b)

        preview = None
        if self.include_preview:
            self._prepare_pipeline_for_preview(pipe)
            preview = self._build_preview(pipe, pairs[0], device)
        mean_loss = sum(losses) / len(losses)
        base_mean_loss = _mean_or_none(base_losses)
        focused_mean_loss = _mean_or_none(focused_losses)
        focused_base_mean_loss = _mean_or_none(focused_base_losses)
        rollout_mean_loss = _mean_or_none(rollout_losses)
        rollout_base_mean_loss = _mean_or_none(rollout_base_losses)
        pixel_rollout_mean_loss = _mean_or_none(pixel_rollout_losses)
        pixel_rollout_base_mean_loss = _mean_or_none(pixel_rollout_base_losses)
        return LossSummary(
            split=split,
            pair_count=len(pairs),
            batch_count=len(batches),
            noise_repeats=self.noise_repeats,
            mean_loss=round(mean_loss, 6),
            min_loss=round(min(losses), 6),
            max_loss=round(max(losses), 6),
            base_mean_loss=_round_or_none(base_mean_loss),
            relative_improvement=_relative_improvement(base_mean_loss, mean_loss),
            focused_mean_loss=_round_or_none(focused_mean_loss),
            focused_base_mean_loss=_round_or_none(focused_base_mean_loss),
            focused_relative_improvement=_relative_improvement(focused_base_mean_loss, focused_mean_loss),
            changed_token_fraction=_round_or_none(_mean_or_none(changed_token_fractions)),
            rollout_mean_loss=_round_or_none(rollout_mean_loss),
            rollout_base_mean_loss=_round_or_none(rollout_base_mean_loss),
            rollout_relative_improvement=_relative_improvement(rollout_base_mean_loss, rollout_mean_loss),
            pixel_rollout_mean_loss=_round_or_none(pixel_rollout_mean_loss),
            pixel_rollout_base_mean_loss=_round_or_none(pixel_rollout_base_mean_loss),
            pixel_rollout_relative_improvement=_relative_improvement(pixel_rollout_base_mean_loss, pixel_rollout_mean_loss),
            preview=preview,
        )

    def _offload_non_transformer_for_loss(self, pipe) -> None:
        pipe.vae.to("cpu")
        if pipe.text_encoder is not None:
            pipe.text_encoder.to("cpu")
        if pipe.text_encoder_2 is not None:
            pipe.text_encoder_2.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _prepare_pipeline_for_preview(self, pipe) -> None:
        pipe.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()

    @torch.no_grad()
    def _build_preview(self, pipe, pair: TrainingPair, device: torch.device) -> EvaluationPreview:
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        prompt_pair = TrainingPair(
            input_image_path=pair.input_image_path,
            target_image_path=pair.target_image_path,
            caption=pair.caption,
            instruction=self.prompt_builder.effective_prompt(pair),
            caption_path=pair.caption_path,
            uses_caption=pair.uses_caption,
            crop_box=pair.crop_box,
        )
        prompt_embeds, pooled_prompt_embeds, _ = self._encode_prompts(pipe, (prompt_pair,), device)
        generator = torch.Generator(device=device).manual_seed(self.config.seed)
        return EvaluationPreviewBuilder(self.config.resolution).build(
            pipe=pipe,
            pair=prompt_pair,
            epoch=0,
            step=0,
            generator=generator,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
        )

    @torch.no_grad()
    def _prepare_loss_inputs(
        self,
        pipe,
        pairs: tuple[TrainingPair, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        prompt_embeds, pooled_prompt_embeds, text_ids = self._encode_prompts(pipe, pairs, device)
        target_latents, condition_latents, latent_ids, latent_height, latent_width = self._prepare_latents(pipe, pairs, device, dtype)

        noise = torch.randn_like(target_latents)
        t = torch.sigmoid(torch.randn(target_latents.shape[0], device=device, dtype=torch.float32))
        mu = _calculate_timestep_shift(target_latents.shape[1])
        sigmas = _time_shift(mu, t).to(dtype=target_latents.dtype)
        while sigmas.ndim < target_latents.ndim:
            sigmas = sigmas.unsqueeze(-1)

        noisy_latents = sigmas * noise + (1.0 - sigmas) * target_latents
        target = noise - target_latents
        latent_model_input = torch.cat([noisy_latents, condition_latents], dim=1)
        timestep = sigmas.flatten().to(dtype=target_latents.dtype)
        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "text_ids": text_ids,
            "target_latents": target_latents,
            "condition_latents": condition_latents,
            "latent_ids": latent_ids,
            "target": target,
            "latent_model_input": latent_model_input,
            "timestep": timestep,
            "latent_height": latent_height,
            "latent_width": latent_width,
        }

    @torch.no_grad()
    def _measure_loss(self, pipe, prepared: dict[str, torch.Tensor]) -> BatchLossMeasurement:
        guidance = None
        if pipe.transformer.config.guidance_embeds:
            guidance = torch.full(
                (prepared["target_latents"].shape[0],),
                3.5,
                device=prepared["target_latents"].device,
                dtype=torch.float32,
            )

        model_pred = pipe.transformer(
            hidden_states=prepared["latent_model_input"],
            timestep=prepared["timestep"],
            guidance=guidance,
            pooled_projections=prepared["pooled_prompt_embeds"],
            encoder_hidden_states=prepared["prompt_embeds"],
            txt_ids=prepared["text_ids"],
            img_ids=prepared["latent_ids"],
            joint_attention_kwargs={},
            return_dict=False,
        )[0]
        model_pred = model_pred[:, : prepared["target_latents"].shape[1]]
        loss = self.loss(
            model_pred,
            prepared["target"],
            prepared["target_latents"],
            prepared["condition_latents"],
        )
        focused_loss, changed_token_fraction = self._focused_loss(
            model_pred,
            prepared["target"],
            prepared["target_latents"],
            prepared["condition_latents"],
        )
        return BatchLossMeasurement(loss=loss, focused_loss=focused_loss, changed_token_fraction=changed_token_fraction)

    @torch.no_grad()
    def _measure_base_loss(self, pipe, prepared: dict[str, torch.Tensor]) -> BatchLossMeasurement | None:
        if not hasattr(pipe, "disable_lora") or not hasattr(pipe, "enable_lora"):
            return None
        pipe.disable_lora()
        try:
            return self._measure_loss(pipe, prepared)
        finally:
            pipe.enable_lora()

    @torch.no_grad()
    def _measure_rollout_loss(
        self,
        pipe,
        prepared: dict[str, torch.Tensor],
    ) -> tuple[float | None, float | None, float | None, float | None]:
        """K-step Euler ODE rollout from σ=1 to σ=0.

        Returns (adapter_latent_loss, base_latent_loss, adapter_pixel_loss, base_pixel_loss).
        Latent loss = MSE in changed-token regions after rollout.
        Pixel loss  = MSE between VAE-decoded final latent and target image (full image).

        Pixel comparison is much more discriminative: a model that outputs a grid pattern
        vs a correct floor plan will show 100x+ difference in pixel space, because the
        VAE decode amplifies structural errors that are invisible in single-step latent MSE.
        """
        target_latents = prepared["target_latents"]
        condition_latents = prepared["condition_latents"]

        if not hasattr(pipe, "disable_lora") or not hasattr(pipe, "enable_lora"):
            return None, None, None, None

        device = target_latents.device
        B = target_latents.shape[0]
        K = self.rollout_steps
        lh: int = prepared.get("latent_height", 0)  # type: ignore[assignment]
        lw: int = prepared.get("latent_width", 0)  # type: ignore[assignment]

        sigmas = torch.linspace(1.0, 0.0, K + 1, device=device, dtype=target_latents.dtype)
        noise = torch.randn_like(target_latents)

        guidance = None
        if pipe.transformer.config.guidance_embeds:
            guidance = torch.full((B,), 3.5, device=device, dtype=torch.float32)

        def _run_euler(use_lora: bool) -> tuple[torch.Tensor, torch.Tensor]:
            """Returns (per_token_latent_mse [B, N], x_final [B, N, C])."""
            if use_lora:
                pipe.enable_lora()
            else:
                pipe.disable_lora()
            x = noise.clone()
            for i in range(K):
                sigma_cur = float(sigmas[i])
                sigma_next = float(sigmas[i + 1])
                latent_input = torch.cat([x, condition_latents], dim=1)
                timestep = torch.full((B,), sigma_cur, device=device, dtype=target_latents.dtype)
                v = pipe.transformer(
                    hidden_states=latent_input,
                    timestep=timestep,
                    guidance=guidance,
                    pooled_projections=prepared["pooled_prompt_embeds"],
                    encoder_hidden_states=prepared["prompt_embeds"],
                    txt_ids=prepared["text_ids"],
                    img_ids=prepared["latent_ids"],
                    joint_attention_kwargs={},
                    return_dict=False,
                )[0]
                v = v[:, : target_latents.shape[1]]
                dt = sigma_next - sigma_cur
                x = x + dt * v
            pipe.enable_lora()
            per_token = (x.float() - target_latents.float()).pow(2).mean(dim=-1)
            return per_token, x.detach()

        per_token_adapter, x_adapter = _run_euler(use_lora=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        per_token_base, x_base = _run_euler(use_lora=False)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── Latent-space loss (changed tokens) ────────────────────────────────
        changed_mask = self.loss.changed_mask(target_latents, condition_latents)
        if not torch.any(changed_mask):
            return None, None, None, None

        latent_adapter = float(per_token_adapter[changed_mask].mean().detach().cpu())
        latent_base = float(per_token_base[changed_mask].mean().detach().cpu())

        # ── Pixel-space loss (full image via VAE decode) ───────────────────────
        # Decoding to pixels amplifies structural errors (e.g. grid vs floor plan)
        # that the latent MSE misses, enabling 100x+ ratio between good and bad models.
        pixel_adapter: float | None = None
        pixel_base: float | None = None
        if lh > 0 and lw > 0:
            try:
                pixel_adapter, pixel_base = self._decode_rollout_to_pixel_mse(
                    pipe, x_adapter, x_base, target_latents, lh, lw
                )
            except Exception as _pixel_exc:
                import sys
                print(f"[pixel rollout] decode failed: {_pixel_exc}", file=sys.stderr)

        return latent_adapter, latent_base, pixel_adapter, pixel_base

    @torch.no_grad()
    def _decode_rollout_to_pixel_mse(
        self,
        pipe,
        x_adapter: torch.Tensor,
        x_base: torch.Tensor,
        target_latents: torch.Tensor,
        latent_height: int,
        latent_width: int,
    ) -> tuple[float, float]:
        """Decode packed latents to pixel images and compute full-image MSE.

        VAE decoding runs on CPU to avoid conflicts with enable_model_cpu_offload hooks.
        This is slower than GPU decode (~10-30s per image) but safe and correct.
        """
        vae_scale = pipe.vae_scale_factor
        scale_factor = pipe.vae.config.scaling_factor
        pixel_h = latent_height * vae_scale
        pixel_w = latent_width * vae_scale

        # Move VAE to CPU explicitly — avoids hook conflicts with offloading.
        pipe.vae.to("cpu")

        vae_dtype = next(pipe.vae.parameters()).dtype
        shift_factor = getattr(pipe.vae.config, "shift_factor", 0.0)

        def _unpack_decode_cpu(x: torch.Tensor) -> torch.Tensor:
            x_cpu = x.cpu()
            unpacked = pipe._unpack_latents(x_cpu.float(), pixel_h, pixel_w, vae_scale)
            # Cast to VAE dtype and apply inverse shift before decode
            unpacked_typed = unpacked.to(dtype=vae_dtype)
            return pipe.vae.decode(unpacked_typed / scale_factor + shift_factor).sample.float()

        pixels_adapter = _unpack_decode_cpu(x_adapter)
        pixels_base = _unpack_decode_cpu(x_base)
        pixels_target = _unpack_decode_cpu(target_latents)

        adapter_mse = float((pixels_adapter - pixels_target).pow(2).mean())
        base_mse = float((pixels_base - pixels_target).pow(2).mean())
        return adapter_mse, base_mse

    def _focused_loss(
        self,
        model_pred: torch.Tensor,
        target: torch.Tensor,
        target_latents: torch.Tensor,
        condition_latents: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        per_token_loss = (model_pred.float() - target.float()).pow(2).mean(dim=-1)
        changed_mask = self.loss.changed_mask(target_latents, condition_latents)
        if not torch.any(changed_mask):
            return per_token_loss.mean(), 0.0
        focused_loss = per_token_loss[changed_mask].mean()
        changed_token_fraction = float(changed_mask.float().mean().detach().cpu())
        return focused_loss, changed_token_fraction

    def _encode_prompts(self, pipe, pairs: tuple[TrainingPair, ...], device: torch.device):
        if pipe.text_encoder is not None:
            pipe.text_encoder.to(device)
        if pipe.text_encoder_2 is not None:
            pipe.text_encoder_2.to(device)
        clip_prompts, t5_prompts = self.prompt_builder.build(
            pairs=pairs,
            prompt_embeddings=None,
            training=False,
            caption_dropout=0.0,
            rng=random.Random(self.config.seed),
        )
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
                prompt=clip_prompts,
                prompt_2=t5_prompts,
                device=device,
                max_sequence_length=self.config.max_sequence_length,
            )
        return (
            prompt_embeds.to(device=device, dtype=pipe.transformer.dtype),
            pooled_prompt_embeds.to(device=device, dtype=pipe.transformer.dtype),
            text_ids.to(device=device, dtype=pipe.transformer.dtype),
        )

    def _prepare_latents(
        self,
        pipe,
        pairs: tuple[TrainingPair, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        pipe.vae.to(device)
        target_images = self._load_images(pipe, [pair.target_image_path for pair in pairs], device, dtype)
        condition_images = self._load_condition_images(pipe, [pair.input_image_path for pair in pairs], device, dtype)
        batch_size = len(pairs)
        num_channels_latents = pipe.transformer.config.in_channels // 4

        target_latents = pipe._encode_vae_image(target_images, generator=None)
        condition_latents = pipe._encode_vae_image(condition_images, generator=None)
        latent_height, latent_width = target_latents.shape[2:]
        condition_height, condition_width = condition_latents.shape[2:]

        target_latents = pipe._pack_latents(
            target_latents,
            batch_size,
            num_channels_latents,
            latent_height,
            latent_width,
        )
        condition_latents = pipe._pack_latents(
            condition_latents,
            batch_size,
            num_channels_latents,
            condition_height,
            condition_width,
        )
        target_ids = pipe._prepare_latent_image_ids(
            batch_size,
            latent_height // 2,
            latent_width // 2,
            device,
            dtype,
        )
        condition_ids = pipe._prepare_latent_image_ids(
            batch_size,
            condition_height // 2,
            condition_width // 2,
            device,
            dtype,
        )
        condition_ids[..., 0] = 1
        return (
            target_latents,
            condition_latents,
            torch.cat([target_ids, condition_ids], dim=0),
            latent_height,
            latent_width,
        )

    def _load_images(
        self,
        pipe,
        image_paths: list[str],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        images = [load_rgba_as_rgb(path) for path in image_paths]
        try:
            return pipe.image_processor.preprocess(
                images,
                height=self.config.resolution,
                width=self.config.resolution,
            ).to(device=device, dtype=dtype)
        finally:
            for image in images:
                image.close()

    def _load_condition_images(
        self,
        pipe,
        image_paths: list[str],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        from diffusers.pipelines.flux.pipeline_flux_kontext import PREFERRED_KONTEXT_RESOLUTIONS

        images = [load_rgba_as_rgb(path) for path in image_paths]
        try:
            image_height, image_width = pipe.image_processor.get_default_height_width(images[0])
            aspect_ratio = image_width / image_height
            _, image_width, image_height = min(
                (abs(aspect_ratio - width / height), width, height) for width, height in PREFERRED_KONTEXT_RESOLUTIONS
            )
            multiple_of = pipe.vae_scale_factor * 2
            image_width = image_width // multiple_of * multiple_of
            image_height = image_height // multiple_of * multiple_of
            resized = pipe.image_processor.resize(images, image_height, image_width)
            return pipe.image_processor.preprocess(
                resized,
                height=image_height,
                width=image_width,
            ).to(device=device, dtype=dtype)
        finally:
            for image in images:
                image.close()


def _evaluation_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    raise RuntimeError("CUDA GPU is not available. Adapter loss evaluation requires CUDA.")


def _batches(examples: list[TrainingPair], batch_size: int) -> list[tuple[TrainingPair, ...]]:
    size = max(batch_size, 1)
    return [tuple(examples[index : index + size]) for index in range(0, len(examples), size)]


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _round_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 6)


def _relative_improvement(base: float | None, adapter: float | None) -> float | None:
    if base is None or adapter is None or base <= 0:
        return None
    return round((base - adapter) / base, 6)


def load_config(path: Path) -> TrainingConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "config" in data and isinstance(data["config"], dict):
        data = data["config"]
    # Map various prompt field names to the canonical "caption" field.
    # "default_caption" and "instance_prompt" are used by fal.ai exported configs.
    if "caption" not in data and "prompt" in data:
        data["caption"] = data["prompt"]
    if "caption" not in data and "default_caption" in data:
        data["caption"] = data["default_caption"]
    if "caption" not in data and "instance_prompt" in data:
        data["caption"] = data["instance_prompt"]
    max_sequence_length = data.get("max_sequence_length")
    if isinstance(max_sequence_length, int) and max_sequence_length > 512:
        data["max_sequence_length"] = 512
    return TrainingConfig.model_validate(data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate flow-matching loss for an existing LoRA adapter.")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "both"], default="val")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--noise-repeats", type=int, default=1)
    parser.add_argument("--rollout-steps", type=int, default=0,
                        help="K-step Euler rollout for reconstruction MSE (0=disabled). "
                             "K=8 gives ~25x ratio, K=10 gives ~100x+ for top 5%% changed tokens.")
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--skip-preview", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits: tuple[SplitName, ...] = ("train", "val") if args.split == "both" else (args.split,)
    evaluator = AdapterLossEvaluator(
        config=_with_resolution(load_config(args.config), args.resolution),
        adapter_path=args.adapter,
        batch_size=args.batch_size,
        max_pairs=args.max_pairs,
        noise_repeats=args.noise_repeats,
        include_preview=not args.skip_preview,
        rollout_steps=args.rollout_steps,
    )
    summaries = evaluator.evaluate(splits)
    print(json.dumps({"losses": [summary.as_dict() for summary in summaries]}, indent=2))


def _with_resolution(config: TrainingConfig, resolution: int | None) -> TrainingConfig:
    if resolution is None:
        return config
    return config.model_copy(update={"resolution": resolution})


if __name__ == "__main__":
    main()
