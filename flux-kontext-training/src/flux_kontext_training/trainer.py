from __future__ import annotations

import base64
import contextlib
import inspect
import json
import math
import os
import random
import gc
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub.errors import GatedRepoError
from PIL import Image

from .augmentation import FloorPlanAugmenter, load_rgba_as_rgb
from .checkpoints import TrainingCheckpointRegistry
from .datasets import list_training_pairs, resolve_dataset
from .evaluation_preview import EvaluationPreviewBuilder
from .schemas import (
    CheckpointComponent,
    EvaluationPreview,
    MetricEvent,
    PreviewHistoryEntry,
    PreviewSource,
    TrainingConfig,
    TrainingPair,
)


MetricCallback = Callable[[MetricEvent], None]
PreviewCallback = Callable[[EvaluationPreview], None]
CancelCallback = Callable[[], bool]
LearningRateCallback = Callable[[], float | None]


class PromptConditioningBuilder:
    """Builds the text conditioning used by Flux Kontext training."""

    def __init__(self, prompt_token_strings: list[str]) -> None:
        self.prompt_token_strings = prompt_token_strings

    def build(
        self,
        pairs: tuple[TrainingPair, ...],
        prompt_embeddings: torch.Tensor | torch.nn.Parameter | None,
        training: bool,
        caption_dropout: float,
        rng: random.Random,
    ) -> tuple[list[str], list[str]]:
        clip_prompts = [pair.caption for pair in pairs]
        t5_prompts = [self.effective_prompt(pair) for pair in pairs]

        if training and caption_dropout > 0.0:
            for index in range(len(clip_prompts)):
                if rng.random() < caption_dropout:
                    clip_prompts[index] = ""
                    t5_prompts[index] = ""

        if prompt_embeddings is None:
            return clip_prompts, t5_prompts

        return clip_prompts, [self._with_prompt_tokens(prompt) for prompt in t5_prompts]

    def effective_prompt(self, pair: TrainingPair) -> str:
        instruction = pair.instruction.strip()
        if instruction:
            return instruction
        return pair.caption.strip()

    def _with_prompt_tokens(self, prompt: str) -> str:
        return " ".join([*self.prompt_token_strings, prompt]).strip()


class TransformationFlowMatchingLoss:
    """Standard Flux flow-matching velocity loss.

    The default is the same objective used by common Flux/SD3 LoRA training:
    mean squared error between the transformer velocity prediction and
    ``noise - target_latents``. Optional delta weighting can be enabled for
    experiments, but it is disabled by default because it changes the loss
    scale and made adapter quality comparisons misleading.
    """

    def __init__(self, delta_weight: float, delta_quantile: float) -> None:
        self.delta_weight = delta_weight
        self.delta_quantile = delta_quantile

    def __call__(
        self,
        model_pred: torch.Tensor,
        target: torch.Tensor,
        target_latents: torch.Tensor,
        condition_latents: torch.Tensor,
    ) -> torch.Tensor:
        per_token_loss = (model_pred.float() - target.float()).pow(2).mean(dim=-1)
        if self.delta_weight <= 0.0:
            return per_token_loss.mean()

        weights = self._delta_weights(target_latents, condition_latents)
        return (per_token_loss * weights).sum() / weights.sum().clamp_min(1e-6)

    def _delta_weights(self, target_latents: torch.Tensor, condition_latents: torch.Tensor) -> torch.Tensor:
        delta = self._delta_map(target_latents, condition_latents)
        flat_delta = delta.flatten(start_dim=1)
        low = torch.quantile(flat_delta, 0.5, dim=1, keepdim=True)
        high = torch.quantile(flat_delta, self.delta_quantile, dim=1, keepdim=True)
        scale = (high - low).clamp_min(1e-6)
        saliency = ((flat_delta - low) / scale).clamp(0.0, 1.0).view_as(delta)
        return 1.0 + self.delta_weight * saliency

    def changed_mask(self, target_latents: torch.Tensor, condition_latents: torch.Tensor) -> torch.Tensor:
        delta = self._delta_map(target_latents, condition_latents)
        threshold = torch.quantile(delta.flatten(start_dim=1), self.delta_quantile, dim=1, keepdim=True)
        return delta >= threshold

    def _delta_map(self, target_latents: torch.Tensor, condition_latents: torch.Tensor) -> torch.Tensor:
        condition_latents = self._align_condition_latents(target_latents, condition_latents)
        delta = (target_latents.float() - condition_latents.float()).abs().mean(dim=-1)
        return delta

    def _align_condition_latents(self, target_latents: torch.Tensor, condition_latents: torch.Tensor) -> torch.Tensor:
        if target_latents.shape == condition_latents.shape:
            return condition_latents
        if (
            target_latents.shape[0] != condition_latents.shape[0]
            or target_latents.shape[2] != condition_latents.shape[2]
        ):
            return target_latents
        return F.interpolate(
            condition_latents.float().transpose(1, 2),
            size=target_latents.shape[1],
            mode="linear",
            align_corners=False,
        ).transpose(1, 2).to(device=target_latents.device, dtype=target_latents.dtype)


@dataclass(frozen=True)
class ReconstructionLossWeights:
    latent: float
    pixel: float


@dataclass(frozen=True)
class TrainingLossResult:
    total: torch.Tensor
    flow: torch.Tensor
    reconstruction_latent: torch.Tensor | None
    reconstruction_pixel: torch.Tensor | None
    reconstruction_weights: ReconstructionLossWeights


@dataclass(frozen=True)
class FullParameterMemoryEstimate:
    trainable_parameters: int
    required_gb: float
    available_gb: float


class FullParameterMemoryEstimator:
    """Conservative VRAM estimate for full transformer fine-tuning."""

    def __init__(self, optimizer_name: str, batch_size: int, resolution: int) -> None:
        self.optimizer_name = optimizer_name
        self.batch_size = batch_size
        self.resolution = resolution

    def estimate(self, transformer: torch.nn.Module, device: torch.device, dtype: torch.dtype) -> FullParameterMemoryEstimate:
        trainable_parameters = sum(parameter.numel() for parameter in transformer.parameters())
        parameter_bytes = trainable_parameters * _dtype_size(dtype)
        gradient_bytes = parameter_bytes
        optimizer_bytes = trainable_parameters * self._optimizer_bytes_per_parameter()
        activation_reserve_bytes = self._activation_reserve_bytes()
        runtime_margin_bytes = 4 * 1024**3
        required_bytes = parameter_bytes + gradient_bytes + optimizer_bytes + activation_reserve_bytes + runtime_margin_bytes
        available_bytes = _cuda_total_memory(device)
        return FullParameterMemoryEstimate(
            trainable_parameters=trainable_parameters,
            required_gb=required_bytes / 1024**3,
            available_gb=available_bytes / 1024**3,
        )

    def _optimizer_bytes_per_parameter(self) -> float:
        return _optimizer_bytes_per_parameter(self.optimizer_name)

    def _activation_reserve_bytes(self) -> int:
        megapixels = (self.resolution * self.resolution * self.batch_size) / 1_000_000
        reserve_gb = max(8.0, 5.0 * megapixels)
        return int(reserve_gb * 1024**3)


@dataclass(frozen=True)
class LoRARankPlan:
    rank: int
    base_parameters: int
    lora_parameters: int
    ratio_percent: float


class LoRAParameterRatioRankResolver:
    """Chooses one shared LoRA rank from a target adapter/base parameter ratio."""

    def __init__(self, ratio_percent: float) -> None:
        self.ratio = ratio_percent / 100.0

    def resolve(self, modules: list[torch.nn.Linear]) -> LoRARankPlan:
        if not modules:
            raise RuntimeError("No LoRA target Linear modules were found.")
        base_parameters = sum(module.in_features * module.out_features for module in modules)
        rank_denominator = sum(module.in_features + module.out_features for module in modules)
        if rank_denominator <= 0:
            raise RuntimeError("LoRA target modules have no trainable dimensions.")
        unconstrained_rank = round((base_parameters * self.ratio) / rank_denominator)
        dimension_limit = min(min(module.in_features, module.out_features) for module in modules)
        rank = min(max(1, unconstrained_rank), dimension_limit)
        lora_parameters = rank * rank_denominator
        ratio_percent = (lora_parameters / base_parameters) * 100.0 if base_parameters else 0.0
        return LoRARankPlan(
            rank=rank,
            base_parameters=base_parameters,
            lora_parameters=lora_parameters,
            ratio_percent=ratio_percent,
        )


class ReconstructionLossWeightScheduler:
    """Schedules reconstruction loss weights across training progress."""

    def __init__(self, config: TrainingConfig) -> None:
        self.latent_max = config.reconstruction_latent_loss_weight
        self.pixel_max = config.reconstruction_pixel_loss_weight
        self.schedule = config.reconstruction_loss_schedule
        self.start_fraction = config.reconstruction_loss_start_fraction
        self.warmup_fraction = config.reconstruction_loss_warmup_fraction
        self.wave_amplitude = config.reconstruction_loss_wave_amplitude
        self.wave_cycles = config.reconstruction_loss_wave_cycles

    def weights(self, progress: float) -> ReconstructionLossWeights:
        multiplier = self._multiplier(progress)
        return ReconstructionLossWeights(
            latent=self.latent_max * multiplier,
            pixel=self.pixel_max * multiplier,
        )

    def _multiplier(self, progress: float) -> float:
        progress = min(max(progress, 0.0), 1.0)
        if self.schedule in {"constant", "wave"}:
            envelope = 1.0
        else:
            envelope = self.start_fraction + (1.0 - self.start_fraction) * self._ramp(progress)

        if self.schedule in {"wave", "cosine_ramp_wave"} and self.wave_cycles > 0.0 and self.wave_amplitude > 0.0:
            wave = 1.0 + self.wave_amplitude * math.sin(2.0 * math.pi * self.wave_cycles * progress)
            envelope *= max(0.0, wave)
        return envelope

    def _ramp(self, progress: float) -> float:
        if self.warmup_fraction <= 0.0:
            return 1.0
        phase = min(progress / self.warmup_fraction, 1.0)
        if self.schedule == "linear_ramp":
            return phase
        return 0.5 * (1.0 - math.cos(math.pi * phase))


class OneStepReconstructionLoss:
    """Reconstructs x0 from one velocity prediction and compares it to the target."""

    def __init__(self, pixel_crop_fraction: float) -> None:
        self.pixel_crop_fraction = min(max(pixel_crop_fraction, 1e-6), 1.0)

    def __call__(
        self,
        pipe,
        model_pred: torch.Tensor,
        noisy_latents: torch.Tensor,
        sigmas: torch.Tensor,
        target_latents: torch.Tensor,
        target_images: torch.Tensor,
        latent_height: int,
        latent_width: int,
        weights: ReconstructionLossWeights,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if weights.latent <= 0.0 and weights.pixel <= 0.0:
            return None, None

        pred_target_latents = noisy_latents - sigmas * model_pred
        latent_loss = None
        pixel_loss = None
        if weights.latent > 0.0:
            latent_loss = (pred_target_latents.float() - target_latents.float()).pow(2).mean()
        if weights.pixel > 0.0:
            pred_pixels, target_pixels = self._decode_latent_crop(
                pipe,
                pred_target_latents,
                target_images,
                latent_height,
                latent_width,
            )
            pixel_loss = (pred_pixels.float() - target_pixels.float()).pow(2).mean()
        return latent_loss, pixel_loss

    def _decode_latent_crop(
        self,
        pipe,
        packed_latents: torch.Tensor,
        target_images: torch.Tensor,
        latent_height: int,
        latent_width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vae_scale = pipe.vae_scale_factor
        pixel_height = latent_height * vae_scale
        pixel_width = latent_width * vae_scale
        unpacked = pipe._unpack_latents(packed_latents.float(), pixel_height, pixel_width, vae_scale)
        latent_crop, target_crop = self._crop_for_pixel_loss(unpacked, target_images)
        vae_dtype = next(pipe.vae.parameters()).dtype
        scale_factor = pipe.vae.config.scaling_factor
        shift_factor = getattr(pipe.vae.config, "shift_factor", 0.0) or 0.0
        decoded = pipe.vae.decode(latent_crop.to(dtype=vae_dtype) / scale_factor + shift_factor).sample
        if decoded.shape[-2:] != target_crop.shape[-2:]:
            target_crop = F.interpolate(target_crop, size=decoded.shape[-2:], mode="bilinear", align_corners=False)
        return decoded, target_crop

    def _crop_for_pixel_loss(
        self,
        unpacked_latents: torch.Tensor,
        target_images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pixel_crop_fraction >= 0.999:
            return unpacked_latents, target_images

        _, _, latent_h, latent_w = unpacked_latents.shape
        _, _, pixel_h, pixel_w = target_images.shape
        crop_h = max(1, int(round(latent_h * self.pixel_crop_fraction)))
        crop_w = max(1, int(round(latent_w * self.pixel_crop_fraction)))
        max_y = max(latent_h - crop_h, 0)
        max_x = max(latent_w - crop_w, 0)
        y0 = int(torch.randint(max_y + 1, (1,), device=unpacked_latents.device).item()) if max_y else 0
        x0 = int(torch.randint(max_x + 1, (1,), device=unpacked_latents.device).item()) if max_x else 0
        latent_crop = unpacked_latents[:, :, y0 : y0 + crop_h, x0 : x0 + crop_w]

        y_scale = pixel_h / latent_h
        x_scale = pixel_w / latent_w
        py0 = int(round(y0 * y_scale))
        px0 = int(round(x0 * x_scale))
        py1 = min(pixel_h, int(round((y0 + crop_h) * y_scale)))
        px1 = min(pixel_w, int(round((x0 + crop_w) * x_scale)))
        target_crop = target_images[:, :, py0:py1, px0:px1]
        return latent_crop, target_crop


class FluxKontextTrainer:
    """Supervised Flux Kontext trainer for paired start/end images."""

    def __init__(
        self,
        config: TrainingConfig,
        on_metric: MetricCallback,
        on_preview: PreviewCallback | None,
        get_learning_rate: LearningRateCallback | None,
        is_cancelled: CancelCallback,
        output_dir: Path | None = None,
    ) -> None:
        if config.method not in {"lora", "prompt_tuning", "full"}:
            raise ValueError(f"Unsupported training method: {config.method}.")
        self.config = config
        self.on_metric = on_metric
        self.on_preview = on_preview
        self.get_learning_rate = get_learning_rate
        self.is_cancelled = is_cancelled
        self.output_dir = output_dir
        self._soft_prompt: torch.nn.Parameter | None = None
        self._frozen_soft_prompt: torch.Tensor | None = None
        self._prompt_embedding_delta: torch.nn.Parameter | None = None
        self._frozen_prompt_embedding_delta: torch.Tensor | None = None
        self._prompt_token_strings: list[str] = []
        self._trainable_params: list[torch.nn.Parameter] = []
        self._loaded_components: list[CheckpointComponent] = []
        self._loaded_lora = False
        self._best_val_loss: float | None = None
        self._checkpoint_registry = TrainingCheckpointRegistry()
        self._augmenter: FloorPlanAugmenter | None = None
        self._aug_rng: random.Random = random.Random(config.seed)
        self._image_cache: dict[str, np.ndarray] = {}
        self._image_sizes: dict[str, tuple[int, int]] = {}
        self._loss = TransformationFlowMatchingLoss(config.loss_delta_weight, config.loss_delta_quantile)
        self._reconstruction_loss = OneStepReconstructionLoss(config.reconstruction_pixel_loss_crop_fraction)
        self._reconstruction_scheduler = ReconstructionLossWeightScheduler(config)
        # キャッシュされたプロンプト埋め込み: key=(clip_prompt, t5_prompt), value=(embed, pooled, text_ids) on CPU
        self._prompt_embed_cache: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None

    def run(self) -> None:
        _seed_everything(self.config.seed)
        _validate_instruction_token_length(self.config)
        dataset = resolve_dataset(self.config.dataset)
        train_pairs = list_training_pairs(dataset, "train", self.config.caption, self.config.instruction)
        val_pairs = list_training_pairs(dataset, "val", self.config.caption, self.config.instruction)
        self._preload_images([*train_pairs, *val_pairs])
        augment = self.config.augment
        rotation_mode = self.config.rotation_mode
        tile_size = 2048 if self.config.use_tiling else None
        needs_augmenter = augment or rotation_mode != "none" or tile_size is not None
        self._augmenter = (
            FloorPlanAugmenter(flip=augment, rotation_mode=rotation_mode, tile_size=tile_size)
            if needs_augmenter
            else None
        )
        self._aug_rng = random.Random(self.config.seed)
        if not train_pairs:
            raise ValueError(f"Dataset '{self.config.dataset}' must contain paired *_start/*_end train images.")
        if not val_pairs:
            raise ValueError(f"Dataset '{self.config.dataset}' must contain paired *_start/*_end val images.")
        if self.config.train_text_encoder_2 and self.config.method != "lora":
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
            if vram_gb < 79:
                raise RuntimeError(
                    f"train T5 encoder (text_encoder_2) には約 80GB の VRAM が必要です "
                    f"(検出: {vram_gb:.0f}GB)。"
                    " T5 は 4.7B パラメーターのため、モデル重み (~24GB) + T5 重み (~9.4GB) + "
                    "T5 勾配 (~9.4GB) + T5 オプティマイザーステート (~9.4GB) で合計 ~52GB 以上必要です。"
                    " train T5 encoder を無効にするか、80GB 以上の GPU を使用してください。"
                )

        pipe = None
        optimizer = None
        try:
            pipe, device, dtype = self._load_pipeline()
            self._validate_pair_prompt_token_lengths(pipe, [*train_pairs, *val_pairs])
            self._load_source_checkpoint(pipe, device, dtype)
            self._build_prompt_embed_cache(pipe, train_pairs, val_pairs, device)
            self._publish_preview(pipe, val_pairs[0], epoch=0, step=0, device=device, source="val")
            self._publish_preview(pipe, train_pairs[0], epoch=0, step=0, device=device, source="train")
            optimizer = self._configure_trainable_parameters(pipe)
            global_step = 0
            optimizer_steps = 0
            batches_per_epoch = self._count_training_batches(train_pairs)
            total_training_steps = max(1, self.config.epochs * batches_per_epoch)
            # lr_scheduler.step() is only invoked at gradient-accumulation boundaries (see below),
            # so the scheduler must be paced by optimizer steps, not by the micro-batch counter
            # that total_training_steps/global_step track for loss_progress.
            optimizer_steps_per_epoch = math.ceil(batches_per_epoch / max(self.config.gradient_accumulation_steps, 1))
            total_scheduler_steps = max(1, self.config.epochs * optimizer_steps_per_epoch)
            lr_scheduler = _build_lr_scheduler(
                optimizer,
                self.config.lr_scheduler,
                total_scheduler_steps,
                self.config.lr_warmup_fraction,
                self.config.lr_num_cycles,
                self.config.lr_cosine_decay_fraction,
                self.config.lr_cosine_mid_ratio,
                self.config.lr_cosine_final_ratio,
            )

            for epoch in range(1, self.config.epochs + 1):
                if self.is_cancelled():
                    return
                if self.config.use_buckets:
                    train_batches = self._bucketed_batches(train_pairs)
                else:
                    random.shuffle(train_pairs)
                    res = self.config.resolution
                    train_batches = [(b, res, res) for b in _batches(train_pairs, self.config.batch_size)]
                optimizer.zero_grad(set_to_none=True)

                for batch_index, (batch_pairs, bw, bh) in enumerate(train_batches, start=1):
                    if self.is_cancelled():
                        return
                    manual_learning_rate = self._sync_learning_rate(optimizer)
                    global_step += 1
                    loss_progress = min((global_step - 1) / max(total_training_steps - 1, 1), 1.0)
                    train_loss = self._training_step(
                        pipe,
                        batch_pairs,
                        device,
                        dtype,
                        training=True,
                        width=bw,
                        height=bh,
                        loss_progress=loss_progress,
                    )
                    scaled_loss = train_loss.total / self.config.gradient_accumulation_steps
                    scaled_loss.backward()
                    gradient_norm = None
                    if batch_index % self.config.gradient_accumulation_steps == 0 or batch_index == len(train_batches):
                        norm = torch.nn.utils.clip_grad_norm_(self._trainable_parameters(), 1.0)
                        gradient_norm = float(norm.detach().cpu())
                        optimizer.step()
                        optimizer_steps += 1
                        if lr_scheduler is not None and not manual_learning_rate:
                            lr_scheduler.step()
                        optimizer.zero_grad(set_to_none=True)

                    val_loss = None
                    if batch_index == len(train_batches):
                        val_loss = self._validate_epoch(pipe, val_pairs, device, dtype, loss_progress)
                        self._save_best_trainable_parameters(pipe, val_loss)
                        self._publish_preview(pipe, val_pairs[0], epoch, global_step, device, source="val")
                        self._publish_preview(pipe, train_pairs[0], epoch, global_step, device, source="train")
                    self.on_metric(
                        MetricEvent(
                            step=global_step,
                            epoch=epoch,
                            train_loss=round(float(train_loss.total.detach().cpu()), 6),
                            val_loss=val_loss,
                            learning_rate=optimizer.param_groups[0]["lr"],
                            gradient_norm=gradient_norm,
                            optimizer_step=optimizer_steps,
                            train_flow_loss=round(float(train_loss.flow.detach().cpu()), 6),
                            train_reconstruction_latent_loss=_round_loss_component(train_loss.reconstruction_latent),
                            train_reconstruction_pixel_loss=_round_loss_component(train_loss.reconstruction_pixel),
                            reconstruction_latent_weight=round(train_loss.reconstruction_weights.latent, 6),
                            reconstruction_pixel_weight=round(train_loss.reconstruction_weights.pixel, 6),
                        )
                    )

        finally:
            if optimizer is not None:
                try:
                    optimizer.zero_grad(set_to_none=True)
                except Exception:
                    pass
            self._release_trainable_tensors()
            release_cuda_memory(pipe)

    def _load_pipeline(self):
        try:
            from diffusers import FluxKontextPipeline
        except ImportError as exc:
            raise RuntimeError("Flux Kontext training requires diffusers with FluxKontextPipeline support.") from exc

        device = _training_device()
        dtype = _training_dtype(device)
        try:
            pipe = FluxKontextPipeline.from_pretrained(
                self.config.model_name,
                torch_dtype=dtype,
                token=_hugging_face_token(),
            )
        except Exception as exc:
            if _is_hugging_face_auth_error(exc):
                raise RuntimeError(_hugging_face_auth_message(self.config.model_name)) from exc
            raise
        pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        pipe.vae.requires_grad_(False)
        pipe.vae.eval()
        if pipe.text_encoder is not None:
            pipe.text_encoder.requires_grad_(False)
            if self.config.train_text_encoder:
                pipe.text_encoder.train()
            else:
                pipe.text_encoder.eval()
        if pipe.text_encoder_2 is not None:
            pipe.text_encoder_2.requires_grad_(False)
            if self.config.train_text_encoder_2:
                pipe.text_encoder_2.train()
                if hasattr(pipe.text_encoder_2, "gradient_checkpointing_enable"):
                    pipe.text_encoder_2.gradient_checkpointing_enable()
            else:
                pipe.text_encoder_2.eval()
        pipe.transformer.requires_grad_(False)
        pipe.transformer.train()
        if hasattr(pipe.transformer, "enable_gradient_checkpointing"):
            pipe.transformer.enable_gradient_checkpointing()
        return pipe, device, dtype

    def _configure_trainable_parameters(self, pipe) -> torch.optim.Optimizer:
        if self.config.method == "lora":
            self._attach_lora(pipe)
        elif self.config.method == "full":
            self._configure_full_parameter_training(pipe)
        elif self._soft_prompt is None:
            self._attach_soft_prompt(pipe)
        else:
            self._soft_prompt.requires_grad_(True)

        if self._soft_prompt is not None:
            self._configure_prompt_embedding_delta(pipe)
            self._trainable_params = [self._soft_prompt]
            parameter_groups: list[dict[str, object]] = [{"params": [self._soft_prompt], "lr": self.config.learning_rate}]
            if self._prompt_embedding_delta is not None:
                self._trainable_params.append(self._prompt_embedding_delta)
                parameter_groups.append(
                    {
                        "params": [self._prompt_embedding_delta],
                        "lr": self.config.prompt_embedding_learning_rate,
                    }
                )
        else:
            transformer_params = [parameter for parameter in pipe.transformer.parameters() if parameter.requires_grad]
            self._trainable_params = list(transformer_params)
            parameter_groups = [{"params": transformer_params, "lr": self.config.learning_rate}]
        te_lr = self.config.text_encoder_learning_rate or self.config.learning_rate
        if self.config.train_text_encoder and pipe.text_encoder is not None:
            te_params = self._text_encoder_trainable_parameters(pipe.text_encoder)
            self._trainable_params.extend(te_params)
            parameter_groups.append({"params": te_params, "lr": te_lr})
        if self.config.train_text_encoder_2 and pipe.text_encoder_2 is not None:
            te2_params = self._text_encoder_trainable_parameters(pipe.text_encoder_2)
            self._trainable_params.extend(te2_params)
            parameter_groups.append({"params": te2_params, "lr": te_lr})
        if not self._trainable_params:
            raise RuntimeError("No trainable parameters were created.")
        return _build_optimizer(parameter_groups, self.config.optimizer)

    def _sync_learning_rate(self, optimizer: torch.optim.Optimizer) -> bool:
        if self.get_learning_rate is None:
            return False
        learning_rate = self.get_learning_rate()
        if learning_rate is None or learning_rate <= 0:
            return False
        optimizer.param_groups[0]["lr"] = learning_rate
        self.config.learning_rate = learning_rate
        return True

    def _load_source_checkpoint(self, pipe, device: torch.device, dtype: torch.dtype) -> None:
        if not self.config.source_checkpoint_id:
            return
        manifest = self._checkpoint_registry.resolve_manifest(self.config.source_checkpoint_id)
        if manifest.model_name != self.config.model_name:
            raise RuntimeError(
                "Selected checkpoint was trained from a different base model. "
                f"Checkpoint model: '{manifest.model_name}', current model: '{self.config.model_name}'."
            )

        for component in manifest.components:
            if component.method == "lora":
                self._load_lora_component(pipe, Path(component.path))
            elif component.method == "prompt_tuning":
                self._load_prompt_tuning_component(pipe, Path(component.path), device, dtype)
            elif component.method == "full":
                raise RuntimeError("Full-parameter checkpoints cannot be used as adapter training sources.")
            self._loaded_components.append(component)

    def _load_lora_component(self, pipe, path: Path) -> None:
        if self._loaded_lora:
            raise RuntimeError("Only one LoRA component can be loaded as a training source.")
        if not path.exists():
            raise RuntimeError(f"LoRA checkpoint path does not exist: {path}")
        if not hasattr(pipe, "load_lora_weights"):
            raise RuntimeError("The installed diffusers pipeline does not support loading LoRA weights.")
        pipe.load_lora_weights(str(path))
        pipe.transformer.requires_grad_(False)
        if pipe.text_encoder is not None:
            self._load_text_encoder_component(pipe.text_encoder, path, "text_encoder")
        if pipe.text_encoder_2 is not None:
            self._load_text_encoder_component(pipe.text_encoder_2, path, "text_encoder_2")
        self._loaded_lora = True

    def _load_text_encoder_component(self, text_encoder, path: Path, component_name: str) -> None:
        lora_path = path / f"{component_name}_lora.pt"
        if lora_path.exists():
            if not _has_peft_adapter(text_encoder):
                self._attach_text_encoder_lora_from_metadata(text_encoder, path, component_name)
            from peft import set_peft_model_state_dict

            adapters = list(getattr(text_encoder, 'peft_config', {}))
            if len(adapters) != 1:
                raise RuntimeError(f'Expected one encoder adapter for {component_name}, found {adapters}.')
            # Diffusers can name the loaded adapter default_0; PEFT otherwise
            # assumes default and raises KeyError for checkpoints with CLIP LoRA.
            set_peft_model_state_dict(text_encoder, _torch_load(lora_path), adapter_name=adapters[0])
            text_encoder.requires_grad_(False)
            return

        legacy_path = path / f"{component_name}.pt"
        if legacy_path.exists():
            text_encoder.load_state_dict(_torch_load(legacy_path))

    def _attach_text_encoder_lora_from_metadata(self, text_encoder, path: Path, component_name: str) -> None:
        metadata_path = path / "encoder_lora_config.json"
        if not metadata_path.is_file():
            raise RuntimeError(f"Missing encoder LoRA metadata for {component_name}: {metadata_path}")
        with metadata_path.open(encoding="utf-8") as file:
            metadata = json.load(file)
        component = metadata.get(component_name)
        if not isinstance(component, dict):
            raise RuntimeError(f"Encoder LoRA metadata does not contain {component_name}.")
        rank = int(component.get("rank", 0))
        target_modules = component.get("target_modules")
        if rank <= 0 or not isinstance(target_modules, list) or not all(isinstance(item, str) for item in target_modules):
            raise RuntimeError(f"Invalid encoder LoRA metadata for {component_name}.")

        from peft import LoraConfig

        text_encoder.add_adapter(
            LoraConfig(
                r=rank,
                lora_alpha=rank,
                init_lora_weights=True,
                target_modules=target_modules,
            )
        )

    def _load_prompt_tuning_component(
        self,
        pipe,
        path: Path,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if self._soft_prompt is not None or self._frozen_soft_prompt is not None:
            raise RuntimeError("Only one Prompt-tuning component can be loaded as a training source.")
        if not path.is_file():
            raise RuntimeError(f"Prompt-tuning checkpoint path does not exist: {path}")
        state = _torch_load(path)
        embeddings = state.get("prompt_tuning_embeddings")
        token_strings = state.get("token_strings")
        if not isinstance(embeddings, torch.Tensor):
            raise RuntimeError("Prompt-tuning checkpoint does not contain prompt_tuning_embeddings.")
        if not isinstance(token_strings, list) or not all(isinstance(token, str) for token in token_strings):
            raise RuntimeError("Prompt-tuning checkpoint does not contain token_strings.")
        if self.config.method == "prompt_tuning" and embeddings.shape[0] != self.config.prompt_tuning_tokens:
            raise RuntimeError(
                "Prompt-tuning token count does not match the selected checkpoint. "
                f"Checkpoint tokens: {embeddings.shape[0]}, current setting: {self.config.prompt_tuning_tokens}."
            )

        self._prepare_prompt_tokens(pipe, token_strings)
        prompt_embeddings = embeddings.to(device=device, dtype=dtype)
        if self.config.method == "prompt_tuning":
            self._soft_prompt = torch.nn.Parameter(prompt_embeddings.clone())
        else:
            self._frozen_soft_prompt = prompt_embeddings.detach()

        delta = state.get("prompt_embedding_delta")
        if delta is None:
            return
        if not isinstance(delta, torch.Tensor):
            raise RuntimeError("Prompt-tuning checkpoint prompt_embedding_delta is not a tensor.")
        prompt_delta = delta.to(device=device, dtype=dtype)
        if self.config.method == "prompt_tuning" and self.config.prompt_tuning_scope == "soft_prompt_and_prompt":
            self._prompt_embedding_delta = torch.nn.Parameter(prompt_delta.clone())
        else:
            self._frozen_prompt_embedding_delta = prompt_delta.detach()

    def _attach_lora(self, pipe) -> None:
        from peft import LoraConfig

        if self._loaded_lora:
            self._unfreeze_lora_parameters(pipe)
            if self.config.train_text_encoder and pipe.text_encoder is not None:
                if _has_peft_adapter(pipe.text_encoder):
                    self._unfreeze_lora_parameters(pipe.text_encoder)
                else:
                    self._attach_text_encoder_lora(pipe.text_encoder, "CLIP text encoder")
                pipe.text_encoder.train()
            if self.config.train_text_encoder_2 and pipe.text_encoder_2 is not None:
                if _has_peft_adapter(pipe.text_encoder_2):
                    self._unfreeze_lora_parameters(pipe.text_encoder_2)
                else:
                    self._attach_text_encoder_lora(pipe.text_encoder_2, "T5 text encoder")
                pipe.text_encoder_2.train()
            pipe.transformer.train()
            return

        target_modules = _fal_kontext_lora_target_modules(pipe)
        target_linear_modules = _linear_modules_by_name(pipe.transformer, target_modules)
        rank_plan = LoRAParameterRatioRankResolver(
            ratio_percent=self.config.transformer_lora_parameter_ratio_percent,
        ).resolve(target_linear_modules)
        if self.config.lora_rank_override is not None:
            rank = min(self.config.lora_rank_override, min(min(m.in_features, m.out_features) for m in target_linear_modules))
            parameters = rank * sum(m.in_features + m.out_features for m in target_linear_modules)
            rank_plan = LoRARankPlan(rank=rank, base_parameters=rank_plan.base_parameters,
                                     lora_parameters=parameters, ratio_percent=100 * parameters / rank_plan.base_parameters)
        print(f'LoRA rank={rank_plan.rank}, trainable_parameters={rank_plan.lora_parameters}, ratio={rank_plan.ratio_percent:.3f}%', flush=True)
        self._validate_lora_memory(rank_plan, pipe)
        lora_config = LoraConfig(
            r=rank_plan.rank,
            lora_alpha=rank_plan.rank,
            init_lora_weights=True,
            target_modules=target_modules,
        )
        pipe.transformer.add_adapter(lora_config)
        self._unfreeze_lora_parameters(pipe)
        pipe.transformer.train()
        if self.config.train_text_encoder and pipe.text_encoder is not None:
            self._attach_text_encoder_lora(pipe.text_encoder, "CLIP text encoder")
        if self.config.train_text_encoder_2 and pipe.text_encoder_2 is not None:
            self._attach_text_encoder_lora(pipe.text_encoder_2, "T5 text encoder")

    def _validate_lora_memory(self, rank_plan: LoRARankPlan, pipe) -> None:
        device = next(pipe.transformer.parameters()).device
        if device.type != "cuda":
            return
        dtype = next(pipe.transformer.parameters()).dtype
        parameter_bytes = rank_plan.lora_parameters * _dtype_size(dtype)
        gradient_bytes = parameter_bytes
        optimizer_bytes = int(rank_plan.lora_parameters * _optimizer_bytes_per_parameter(self.config.optimizer))
        required_bytes = parameter_bytes + gradient_bytes + optimizer_bytes
        free_bytes = _cuda_free_memory(device)
        if free_bytes <= 0:
            return
        if required_bytes > free_bytes * 0.85:
            raise RuntimeError(
                "LoRA adapter is estimated to exceed available VRAM. "
                f"target ratio: {self.config.transformer_lora_parameter_ratio_percent:g}%, "
                f"resolved rank: {rank_plan.rank}, "
                f"adapter/base ratio: {rank_plan.ratio_percent:.3g}%, "
                f"adapter params: {rank_plan.lora_parameters / 1_000_000:.1f}M, "
                f"required additional memory: {required_bytes / 1024**3:.1f}GB, "
                f"free: {free_bytes / 1024**3:.1f}GB. "
                "Choose a smaller LoRA ratio or reduce other trainable components."
            )

    def _attach_text_encoder_lora(self, text_encoder, label: str) -> None:
        from peft import LoraConfig

        if _has_peft_adapter(text_encoder):
            self._unfreeze_lora_parameters(text_encoder)
            text_encoder.train()
            return

        target_modules = _all_linear_module_names(text_encoder)
        rank_plan = LoRAParameterRatioRankResolver(
            ratio_percent=self.config.text_encoder_lora_parameter_ratio_percent,
        ).resolve(_linear_modules_by_name(text_encoder, target_modules))
        lora_config = LoraConfig(
            r=rank_plan.rank,
            lora_alpha=rank_plan.rank,
            init_lora_weights=True,
            target_modules=target_modules,
        )
        try:
            text_encoder.add_adapter(lora_config)
        except Exception as exc:
            raise RuntimeError(f"{label} does not support LoRA adapters in this environment.") from exc
        self._unfreeze_lora_parameters(text_encoder)
        text_encoder.train()

    def _text_encoder_trainable_parameters(self, text_encoder) -> list[torch.nn.Parameter]:
        if self.config.method == "lora":
            return [parameter for name, parameter in text_encoder.named_parameters() if "lora" in name.lower()]
        for parameter in text_encoder.parameters():
            parameter.requires_grad_(True)
        return list(text_encoder.parameters())

    def _attach_soft_prompt(self, pipe) -> None:
        if pipe.tokenizer_2 is None or pipe.text_encoder_2 is None:
            raise RuntimeError("Prompt-tuning requires the Flux Kontext T5 tokenizer and text encoder.")
        self._prompt_token_strings = [f"<kontext_pt_{index}>" for index in range(self.config.prompt_tuning_tokens)]
        self._prepare_prompt_tokens(pipe, self._prompt_token_strings)

        hidden_size = pipe.transformer.config.joint_attention_dim
        dtype = next(pipe.transformer.parameters()).dtype
        device = next(pipe.transformer.parameters()).device
        soft_prompt = torch.empty(self.config.prompt_tuning_tokens, hidden_size, device=device, dtype=dtype)
        torch.nn.init.normal_(soft_prompt, mean=0.0, std=0.02)
        self._soft_prompt = torch.nn.Parameter(soft_prompt)

    def _configure_prompt_embedding_delta(self, pipe) -> None:
        if self.config.prompt_tuning_scope != "soft_prompt_and_prompt":
            if self._prompt_embedding_delta is not None:
                self._frozen_prompt_embedding_delta = self._prompt_embedding_delta.detach()
                self._prompt_embedding_delta = None
            return
        if self._prompt_embedding_delta is not None:
            self._prompt_embedding_delta.requires_grad_(True)
            return
        if self._frozen_prompt_embedding_delta is not None:
            self._prompt_embedding_delta = torch.nn.Parameter(self._frozen_prompt_embedding_delta.clone())
            self._frozen_prompt_embedding_delta = None
            return
        hidden_size = pipe.transformer.config.joint_attention_dim
        dtype = next(pipe.transformer.parameters()).dtype
        device = next(pipe.transformer.parameters()).device
        max_prompt_tokens = max(self.config.max_sequence_length - self.config.prompt_tuning_tokens, 1)
        self._prompt_embedding_delta = torch.nn.Parameter(
            torch.zeros(max_prompt_tokens, hidden_size, device=device, dtype=dtype)
        )

    def _prepare_prompt_tokens(self, pipe, token_strings: list[str]) -> None:
        if pipe.tokenizer_2 is None or pipe.text_encoder_2 is None:
            raise RuntimeError("Prompt-tuning requires the Flux Kontext T5 tokenizer and text encoder.")
        self._prompt_token_strings = token_strings
        added = pipe.tokenizer_2.add_tokens(token_strings)
        if added:
            pipe.text_encoder_2.resize_token_embeddings(len(pipe.tokenizer_2))
        pipe.text_encoder_2.requires_grad_(False)

    def _unfreeze_lora_parameters(self, module_or_pipe) -> None:
        module = getattr(module_or_pipe, "transformer", module_or_pipe)
        for name, parameter in module.named_parameters():
            if "lora" in name.lower():
                parameter.requires_grad_(True)

    def _configure_full_parameter_training(self, pipe) -> None:
        if self._loaded_components:
            raise RuntimeError("Full-parameter training cannot start from an adapter or prompt-tuning checkpoint.")
        if self.config.train_text_encoder or self.config.train_text_encoder_2:
            raise RuntimeError("Full-parameter training does not support training text encoders in this 48GB-oriented setup.")
        estimator = FullParameterMemoryEstimator(
            optimizer_name=self.config.optimizer,
            batch_size=self.config.batch_size,
            resolution=self.config.resolution,
        )
        device = next(pipe.transformer.parameters()).device
        dtype = next(pipe.transformer.parameters()).dtype
        estimate = estimator.estimate(pipe.transformer, device, dtype)
        allow_oom_risk = os.getenv("FLUX_KONTEXT_ALLOW_FULL_TRAINING_OOM_RISK") == "1"
        if not allow_oom_risk and estimate.required_gb > estimate.available_gb * 0.92:
            raise RuntimeError(
                "Full-parameter training is estimated to exceed available VRAM. "
                f"trainable parameters: {estimate.trainable_parameters / 1_000_000_000:.2f}B, "
                f"required: {estimate.required_gb:.1f}GB, detected: {estimate.available_gb:.1f}GB. "
                "On a 48GB GPU this is expected to OOM because bf16 weights + gradients already consume "
                "almost the whole device before optimizer state and activations. "
                "Use LoRA, reduce the trainable scope, or set FLUX_KONTEXT_ALLOW_FULL_TRAINING_OOM_RISK=1 "
                "only if you intentionally want to bypass this guard."
            )
        pipe.transformer.requires_grad_(True)
        pipe.transformer.train()

    def _trainable_parameters(self) -> list[torch.nn.Parameter]:
        return self._trainable_params

    def _validate_pair_prompt_token_lengths(self, pipe, pairs: list[TrainingPair]) -> None:
        if pipe.tokenizer_2 is None:
            return
        limit = self.config.max_sequence_length
        if self.config.method == "prompt_tuning":
            limit = max(1, limit - self.config.prompt_tuning_tokens)
        prompt_builder = PromptConditioningBuilder(self._prompt_token_strings)
        too_long: list[tuple[str, int]] = []
        for pair in pairs:
            prompt = prompt_builder.effective_prompt(pair)
            if not prompt:
                continue
            token_ids = pipe.tokenizer_2(
                prompt,
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=False,
                verbose=False,
            )["input_ids"]
            if len(token_ids) > limit:
                too_long.append((Path(pair.input_image_path).name, len(token_ids)))
        if not too_long:
            return
        example_name, token_count = too_long[0]
        pt_note = (
            f" prompt tuning で {self.config.prompt_tuning_tokens} トークンが使用されるため、実効上限は {limit} です。"
            if self.config.method == "prompt_tuning"
            else ""
        )
        raise ValueError(
            f"T5 prompt が長すぎます。例: {example_name} は {token_count} トークンで、"
            f"Flux Kontext の上限 {self.config.max_sequence_length} を超えています。{pt_note}"
            " instruction/caption を短くしてください。"
        )

    def _build_prompt_embed_cache(
        self,
        pipe,
        train_pairs: list[TrainingPair],
        val_pairs: list[TrainingPair],
        device: torch.device,
    ) -> None:
        """全ペアのプロンプトを事前エンコードしてCPUキャッシュに格納する。
        キャッシュ構築後はテキストエンコーダーをパイプラインから除去してVRAMを完全解放する。
        テキストエンコーダーを学習対象にしている場合はスキップする。"""
        if self.config.train_text_encoder or self.config.train_text_encoder_2:
            return

        prompt_builder = PromptConditioningBuilder(self._prompt_token_strings)
        prompt_embeddings = self._active_prompt_embeddings()

        unique_prompts: set[tuple[str, str]] = set()
        for pair in [*train_pairs, *val_pairs]:
            clip_prompts, t5_prompts = prompt_builder.build(
                pairs=(pair,),
                prompt_embeddings=prompt_embeddings,
                training=False,
                caption_dropout=0.0,
                rng=self._aug_rng,
            )
            unique_prompts.add((clip_prompts[0], t5_prompts[0]))

        if self.config.caption_dropout > 0:
            unique_prompts.add(("", ""))

        # プレビュー用プロンプト (inference_instruction が設定されている場合はそちらも)
        if val_pairs:
            inference_instruction = self.config.inference_instruction.strip()
            preview_base = val_pairs[0]
            preview_pair_obj = (
                TrainingPair(
                    input_image_path=preview_base.input_image_path,
                    target_image_path=preview_base.target_image_path,
                    caption=preview_base.caption,
                    instruction=inference_instruction,
                    uses_caption=preview_base.uses_caption,
                )
                if inference_instruction
                else preview_base
            )
            clip_prompts, t5_prompts = prompt_builder.build(
                pairs=(preview_pair_obj,),
                prompt_embeddings=prompt_embeddings,
                training=False,
                caption_dropout=0.0,
                rng=self._aug_rng,
            )
            unique_prompts.add((clip_prompts[0], t5_prompts[0]))

        dtype = next(pipe.transformer.parameters()).dtype
        self._prompt_embed_cache = {}
        with torch.no_grad():
            for clip_prompt, t5_prompt in unique_prompts:
                prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
                    prompt=[clip_prompt],
                    prompt_2=[t5_prompt],
                    device=device,
                    max_sequence_length=self.config.max_sequence_length,
                )
                # バッチ次元を除去してCPUに保存 (4MB/プロンプト程度)
                self._prompt_embed_cache[(clip_prompt, t5_prompt)] = (
                    prompt_embeds.squeeze(0).cpu(),
                    pooled_prompt_embeds.squeeze(0).cpu(),
                    text_ids.cpu(),
                )

        # テキストエンコーダーをパイプラインから除去して VRAM を完全解放する。
        # pipe.text_encoder = None にすることで pipe.device の set 反復が transformer を見つけ
        # CUDA デバイスを確実に返すようになる (推論時の _execution_device バグ対策)。
        pipe.text_encoder = None
        pipe.text_encoder_2 = None
        gc.collect()
        torch.cuda.empty_cache()

    def _offload_text_encoders(self, pipe) -> None:
        """Encoding 後にテキストエンコーダーを CPU へ退避してVRAMを解放する。
        キャッシュが構築済みの場合はテキストエンコーダーはすでに除去済みのため何もしない。
        テキストエンコーダーを学習対象にしている場合は backward で勾配が必要なため退避しない。"""
        if self._prompt_embed_cache is not None:
            return
        if self.config.train_text_encoder or self.config.train_text_encoder_2:
            return
        if pipe.text_encoder is not None:
            pipe.text_encoder.to("cpu")
        if pipe.text_encoder_2 is not None:
            pipe.text_encoder_2.to("cpu")
        torch.cuda.empty_cache()

    def _restore_text_encoders(self, pipe, device: torch.device, dtype: torch.dtype) -> None:
        """Encoding の前にテキストエンコーダーを GPU へ戻す。
        キャッシュが構築済みの場合は不要なため何もしない。"""
        if self._prompt_embed_cache is not None:
            return
        if self.config.train_text_encoder or self.config.train_text_encoder_2:
            return
        if pipe.text_encoder is not None and next(pipe.text_encoder.parameters()).device.type == "cpu":
            pipe.text_encoder.to(device=device, dtype=dtype)
        if pipe.text_encoder_2 is not None and next(pipe.text_encoder_2.parameters()).device.type == "cpu":
            pipe.text_encoder_2.to(device=device, dtype=dtype)

    def _training_step(
        self,
        pipe,
        pairs: tuple[TrainingPair, ...],
        device: torch.device,
        dtype: torch.dtype,
        training: bool = True,
        width: int | None = None,
        height: int | None = None,
        loss_progress: float = 1.0,
    ) -> TrainingLossResult:
        self._restore_text_encoders(pipe, device, dtype)
        prompt_embeds, pooled_prompt_embeds, text_ids = self._encode_prompts(pipe, pairs, device, training=training)
        self._offload_text_encoders(pipe)
        target_latents, condition_latents, latent_ids, target_images, latent_height, latent_width = self._prepare_latents(
            pipe,
            pairs,
            device,
            dtype,
            training=training,
            width=width,
            height=height,
        )

        noise = torch.randn_like(target_latents)
        # Logit-normal base sampling + resolution-dependent timestep shift.
        # The shift matches inference (FluxKontextPipeline uses calculate_shift / time_shift),
        # concentrating training around the sigma values actually visited at inference time.
        u = torch.randn(target_latents.shape[0], device=device, dtype=torch.float32)
        t = torch.sigmoid(u)
        image_seq_len = target_latents.shape[1]
        mu = _calculate_timestep_shift(image_seq_len)
        sigmas = _time_shift(mu, t).to(dtype=target_latents.dtype)
        while sigmas.ndim < target_latents.ndim:
            sigmas = sigmas.unsqueeze(-1)

        noisy_latents = sigmas * noise + (1.0 - sigmas) * target_latents
        target = noise - target_latents
        latent_model_input = torch.cat([noisy_latents, condition_latents], dim=1)
        timestep = sigmas.flatten().to(dtype=target_latents.dtype)

        guidance = None
        if pipe.transformer.config.guidance_embeds:
            guidance = torch.full((target_latents.shape[0],), 3.5, device=device, dtype=torch.float32)

        model_pred = pipe.transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_ids,
            joint_attention_kwargs={},
            return_dict=False,
        )[0]
        model_pred = model_pred[:, : target_latents.shape[1]]
        flow_loss = self._loss(model_pred, target, target_latents, condition_latents)
        reconstruction_weights = self._reconstruction_scheduler.weights(loss_progress)
        reconstruction_latent_loss, reconstruction_pixel_loss = self._reconstruction_loss(
            pipe=pipe,
            model_pred=model_pred,
            noisy_latents=noisy_latents,
            sigmas=sigmas,
            target_latents=target_latents,
            target_images=target_images,
            latent_height=latent_height,
            latent_width=latent_width,
            weights=reconstruction_weights,
        )
        total_loss = flow_loss
        if reconstruction_latent_loss is not None:
            total_loss = total_loss + reconstruction_weights.latent * reconstruction_latent_loss
        if reconstruction_pixel_loss is not None:
            total_loss = total_loss + reconstruction_weights.pixel * reconstruction_pixel_loss
        return TrainingLossResult(
            total=total_loss,
            flow=flow_loss,
            reconstruction_latent=reconstruction_latent_loss,
            reconstruction_pixel=reconstruction_pixel_loss,
            reconstruction_weights=reconstruction_weights,
        )

    @torch.no_grad()
    def _validate_epoch(
        self,
        pipe,
        pairs: list[TrainingPair],
        device: torch.device,
        dtype: torch.dtype,
        loss_progress: float,
    ) -> float | None:
        total, count = 0.0, 0
        modules = [m for m in [pipe.transformer, pipe.text_encoder, pipe.text_encoder_2] if m is not None]
        modes = [m.training for m in modules]
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
        try:
            for module in modules:
                module.eval()
            # Repeat exactly the same validation noise and VAE sampling at
            # every epoch, without perturbing subsequent training randomness.
            with torch.random.fork_rng(devices=devices):
                torch.random.default_generator.manual_seed(self.config.validation_seed)
                for index in devices:
                    torch.cuda.default_generators[index].manual_seed(self.config.validation_seed)
                for _ in range(self.config.validation_noise_repeats):
                    for batch_pairs in _batches(pairs, self.config.batch_size):
                        if self.is_cancelled():
                            return None
                        result = self._training_step(pipe, batch_pairs, device, dtype, training=False, loss_progress=1.0)
                        # A fixed flow objective is comparable across epochs;
                        # scheduled reconstruction weights must not choose the winner.
                        total += float(result.flow.detach().cpu()) * len(batch_pairs)
                        count += len(batch_pairs)
            return round(total / count, 6) if count else None
        finally:
            for module, mode in zip(modules, modes):
                module.train(mode)

    def _publish_preview(
        self,
        pipe,
        pair: TrainingPair,
        epoch: int,
        step: int,
        device: torch.device,
        source: PreviewSource = "val",
    ) -> None:
        if self.on_preview is None or self.is_cancelled():
            return
        inference_instruction = self.config.inference_instruction.strip()
        preview_pair = (
            TrainingPair(
                input_image_path=pair.input_image_path,
                target_image_path=pair.target_image_path,
                caption=pair.caption,
                instruction=inference_instruction,
                uses_caption=pair.uses_caption,
            )
            if inference_instruction
            else pair
        )
        prompt_builder = PromptConditioningBuilder(self._prompt_token_strings)
        preview_pair = TrainingPair(
            input_image_path=preview_pair.input_image_path,
            target_image_path=preview_pair.target_image_path,
            caption=preview_pair.caption,
            instruction=prompt_builder.effective_prompt(preview_pair),
            caption_path=preview_pair.caption_path,
            uses_caption=preview_pair.uses_caption,
            crop_box=preview_pair.crop_box,
        )
        dtype = next(pipe.transformer.parameters()).dtype
        self._restore_text_encoders(pipe, device, dtype)
        prompt_embeds, pooled_prompt_embeds, _ = self._encode_prompts(pipe, (preview_pair,), device)
        prompt_embeds = prompt_embeds.detach()
        pooled_prompt_embeds = pooled_prompt_embeds.detach()
        # テキストエンコーダーは pipe() 完了後にオフロードする。
        # pipe._execution_device が pipe.device に依存しており、T5 を先に CPU へ移動すると
        # pipe.device が cpu を返し、VAE への入力が CPU テンソルになってクラッシュするため。
        generator = torch.Generator(device=device).manual_seed(self.config.seed)
        was_training = pipe.transformer.training
        pipe.transformer.eval()
        try:
            preview = EvaluationPreviewBuilder(self.config.resolution).build(
                pipe=pipe,
                pair=preview_pair,
                epoch=epoch,
                step=step,
                generator=generator,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
            )
            preview = preview.model_copy(update={"source": source})
        finally:
            if was_training:
                pipe.transformer.train()
            self._offload_text_encoders(pipe)
        self.on_preview(preview)
        self._save_preview_history(preview)

    def _save_preview_history(self, preview: EvaluationPreview) -> None:
        if self.output_dir is None:
            return
        preview_dir = self.output_dir / "previews" / f"{preview.source}_step_{preview.step:06d}"
        preview_dir.mkdir(parents=True, exist_ok=True)
        (preview_dir / "input.png").write_bytes(_decode_data_url(preview.input_image))
        (preview_dir / "prediction.png").write_bytes(_decode_data_url(preview.prediction_image))
        (preview_dir / "target.png").write_bytes(_decode_data_url(preview.target_image))
        entry = PreviewHistoryEntry(epoch=preview.epoch, step=preview.step, prompt=preview.prompt, source=preview.source)
        with (self.output_dir / "previews.jsonl").open("a", encoding="utf-8") as file:
            file.write(entry.model_dump_json() + "\n")

    def _encode_prompts(self, pipe, pairs: tuple[TrainingPair, ...], device: torch.device, training: bool = False):
        prompt_embeddings = self._active_prompt_embeddings()
        prompt_delta = self._active_prompt_embedding_delta()
        prompt_builder = PromptConditioningBuilder(self._prompt_token_strings)
        clip_prompts, t5_prompts = prompt_builder.build(
            pairs=pairs,
            prompt_embeddings=prompt_embeddings,
            training=training,
            caption_dropout=self.config.caption_dropout,
            rng=self._aug_rng,
        )

        if self._prompt_embed_cache is not None:
            dtype = next(pipe.transformer.parameters()).dtype
            embeds_list: list[torch.Tensor] = []
            pooled_list: list[torch.Tensor] = []
            text_ids: torch.Tensor | None = None
            for clip, t5 in zip(clip_prompts, t5_prompts):
                cached = self._prompt_embed_cache.get((clip, t5))
                if cached is None:
                    raise KeyError(
                        f"プロンプトがキャッシュに見つかりません: clip={clip!r}, t5={t5!r}\n"
                        "学習開始後に inference_instruction が変更された可能性があります。"
                    )
                embed, pooled, t_ids = cached
                embeds_list.append(embed)
                pooled_list.append(pooled)
                if text_ids is None:
                    text_ids = t_ids
            prompt_embeds = torch.stack(embeds_list, dim=0).to(device=device, dtype=dtype)
            pooled_prompt_embeds = torch.stack(pooled_list, dim=0).to(device=device, dtype=dtype)
            text_ids = text_ids.to(device=device, dtype=dtype)
        else:
            training_encoders = training and (self.config.train_text_encoder or self.config.train_text_encoder_2)
            encode_ctx = contextlib.nullcontext() if training_encoders else torch.no_grad()
            with encode_ctx:
                prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
                    prompt=clip_prompts,
                    prompt_2=t5_prompts,
                    device=device,
                    max_sequence_length=self.config.max_sequence_length,
                )

        if prompt_embeddings is not None:
            token_count = min(prompt_embeddings.shape[0], prompt_embeds.shape[1])
            soft_prompt = prompt_embeddings[:token_count].to(device=device, dtype=prompt_embeds.dtype)
            soft_prompt = soft_prompt.unsqueeze(0).expand(prompt_embeds.shape[0], -1, -1)
            prompt_embeds = torch.cat([soft_prompt, prompt_embeds[:, token_count:]], dim=1)
            if prompt_delta is not None and token_count < prompt_embeds.shape[1]:
                delta_count = min(prompt_delta.shape[0], prompt_embeds.shape[1] - token_count)
                delta = prompt_delta[:delta_count].to(device=device, dtype=prompt_embeds.dtype)
                delta = delta.unsqueeze(0).expand(prompt_embeds.shape[0], -1, -1)
                prompt_embeds = prompt_embeds.clone()
                prompt_embeds[:, token_count : token_count + delta_count] += delta
        return prompt_embeds, pooled_prompt_embeds, text_ids

    def _active_prompt_embeddings(self) -> torch.Tensor | torch.nn.Parameter | None:
        return self._soft_prompt if self._soft_prompt is not None else self._frozen_soft_prompt

    def _active_prompt_embedding_delta(self) -> torch.Tensor | torch.nn.Parameter | None:
        if self._prompt_embedding_delta is not None:
            return self._prompt_embedding_delta
        return self._frozen_prompt_embedding_delta

    @torch.no_grad()
    def _prepare_latents(
        self,
        pipe,
        pairs: tuple[TrainingPair, ...],
        device: torch.device,
        dtype: torch.dtype,
        training: bool,
        width: int | None = None,
        height: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        w = width or self.config.resolution
        h = height or self.config.resolution
        if training and self._augmenter is not None:
            condition_pil: list[Image.Image] = []
            target_pil: list[Image.Image] = []
            for pair in pairs:
                cond, tgt = self._augmenter.transform_pair(
                    self._open_cached(pair.input_image_path),
                    self._open_cached(pair.target_image_path),
                    pair.crop_box,
                    self._aug_rng,
                )
                condition_pil.append(cond)
                target_pil.append(tgt)
            target_images = self._preprocess_pil(target_pil, pipe, device, dtype, width=w, height=h)
            condition_images = self._preprocess_kontext_condition_pil(condition_pil, pipe, device, dtype)
        else:
            target_images = self._load_images(
                [(pair.target_image_path, pair.crop_box) for pair in pairs], pipe, device, dtype, width=w, height=h
            )
            condition_images = self._load_condition_images(
                [(pair.input_image_path, pair.crop_box) for pair in pairs], pipe, device, dtype
            )
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
        latent_ids = torch.cat([target_ids, condition_ids], dim=0)
        return target_latents, condition_latents, latent_ids, target_images, latent_height, latent_width

    def _load_images(
        self,
        path_box_pairs: list[tuple[str, tuple[int, int, int, int] | None]],
        pipe,
        device: torch.device,
        dtype: torch.dtype,
        width: int | None = None,
        height: int | None = None,
    ) -> torch.Tensor:
        w = width or self.config.resolution
        h = height or self.config.resolution
        images = []
        for path, box in path_box_pairs:
            img = self._open_cached(path)
            if box is not None:
                img = img.crop(box)
            images.append(img)
        try:
            return pipe.image_processor.preprocess(
                images,
                height=h,
                width=w,
            ).to(device=device, dtype=dtype)
        finally:
            for image in images:
                image.close()

    def _preprocess_pil(
        self,
        images: list[Image.Image],
        pipe,
        device: torch.device,
        dtype: torch.dtype,
        width: int | None = None,
        height: int | None = None,
    ) -> torch.Tensor:
        w = width or self.config.resolution
        h = height or self.config.resolution
        try:
            return pipe.image_processor.preprocess(
                images,
                height=h,
                width=w,
            ).to(device=device, dtype=dtype)
        finally:
            for image in images:
                image.close()

    def _load_condition_images(
        self,
        path_box_pairs: list[tuple[str, tuple[int, int, int, int] | None]],
        pipe,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        images = []
        for path, box in path_box_pairs:
            img = self._open_cached(path)
            if box is not None:
                img = img.crop(box)
            images.append(img)
        return self._preprocess_kontext_condition_pil(images, pipe, device, dtype)

    def _preprocess_kontext_condition_pil(
        self,
        images: list[Image.Image],
        pipe,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        from diffusers.pipelines.flux.pipeline_flux_kontext import PREFERRED_KONTEXT_RESOLUTIONS

        if not images:
            raise ValueError("No condition images were provided.")
        image_height, image_width = pipe.image_processor.get_default_height_width(images[0])
        aspect_ratio = image_width / image_height
        _, image_width, image_height = min(
            (abs(aspect_ratio - width / height), width, height) for width, height in PREFERRED_KONTEXT_RESOLUTIONS
        )
        multiple_of = pipe.vae_scale_factor * 2
        image_width = image_width // multiple_of * multiple_of
        image_height = image_height // multiple_of * multiple_of
        try:
            resized = pipe.image_processor.resize(images, image_height, image_width)
            return pipe.image_processor.preprocess(
                resized,
                height=image_height,
                width=image_width,
            ).to(device=device, dtype=dtype)
        finally:
            for image in images:
                image.close()

    def _save_trainable_parameters(self, pipe) -> None:
        if self.output_dir is None:
            return
        weights_dir = self.output_dir / "weights"
        if weights_dir.exists():
            shutil.rmtree(weights_dir)
        weights_dir.mkdir(parents=True, exist_ok=True)
        if self._soft_prompt is not None:
            prompt_path = weights_dir / "prompt_tuning.pt"
            torch.save(
                {
                    "method": "prompt_tuning",
                    "model_name": self.config.model_name,
                    "prompt_tuning_tokens": self.config.prompt_tuning_tokens,
                    "prompt_tuning_scope": self.config.prompt_tuning_scope,
                    "prompt_embedding_learning_rate": self.config.prompt_embedding_learning_rate,
                    "token_strings": self._prompt_token_strings,
                    "prompt_tuning_embeddings": self._soft_prompt.detach().cpu(),
                    "prompt_embedding_delta": (
                        self._active_prompt_embedding_delta().detach().cpu()
                        if self._active_prompt_embedding_delta() is not None
                        else None
                    ),
                },
                prompt_path,
            )
            self._write_checkpoint_manifest(weights_dir, CheckpointComponent(method="prompt_tuning", path=str(prompt_path)))
            return

        if self.config.method == "full":
            transformer_dir = weights_dir / "transformer"
            pipe.transformer.save_pretrained(transformer_dir, safe_serialization=True)
            self._write_checkpoint_manifest(weights_dir, CheckpointComponent(method="full", path=str(transformer_dir)))
            return

        from peft import get_peft_model_state_dict

        lora_state_dict = get_peft_model_state_dict(pipe.transformer)
        text_encoder_lora = None
        text_encoder_2_lora = None
        if self.config.train_text_encoder and pipe.text_encoder is not None:
            text_encoder_lora = get_peft_model_state_dict(pipe.text_encoder)
        if self.config.train_text_encoder_2 and pipe.text_encoder_2 is not None:
            text_encoder_2_lora = get_peft_model_state_dict(pipe.text_encoder_2)
        self._save_pipeline_lora_weights(pipe, weights_dir, lora_state_dict, text_encoder_lora, text_encoder_2_lora)
        if text_encoder_lora is not None:
            torch.save(text_encoder_lora, weights_dir / "text_encoder_lora.pt")
        if text_encoder_2_lora is not None:
            torch.save(text_encoder_2_lora, weights_dir / "text_encoder_2_lora.pt")
        self._save_encoder_lora_metadata(pipe, weights_dir)
        self._write_checkpoint_manifest(weights_dir, CheckpointComponent(method="lora", path=str(weights_dir)))

    def _save_pipeline_lora_weights(
        self,
        pipe,
        weights_dir: Path,
        transformer_lora: dict[str, torch.Tensor],
        text_encoder_lora: dict[str, torch.Tensor] | None,
        text_encoder_2_lora: dict[str, torch.Tensor] | None,
    ) -> None:
        kwargs: dict[str, object] = {"transformer_lora_layers": transformer_lora}
        signature = inspect.signature(pipe.save_lora_weights)
        if text_encoder_lora is not None and "text_encoder_lora_layers" in signature.parameters:
            kwargs["text_encoder_lora_layers"] = text_encoder_lora
        if text_encoder_2_lora is not None and "text_encoder_2_lora_layers" in signature.parameters:
            kwargs["text_encoder_2_lora_layers"] = text_encoder_2_lora
        pipe.save_lora_weights(weights_dir, **kwargs)

    def _save_encoder_lora_metadata(self, pipe, weights_dir: Path) -> None:
        metadata: dict[str, object] = {}
        if self.config.train_text_encoder and pipe.text_encoder is not None and _has_peft_adapter(pipe.text_encoder):
            metadata["text_encoder"] = _active_lora_metadata(pipe.text_encoder)
        if self.config.train_text_encoder_2 and pipe.text_encoder_2 is not None and _has_peft_adapter(pipe.text_encoder_2):
            metadata["text_encoder_2"] = _active_lora_metadata(pipe.text_encoder_2)
        if not metadata:
            return
        with (weights_dir / "encoder_lora_config.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)

    def _save_best_trainable_parameters(self, pipe, val_loss: float | None) -> None:
        if val_loss is None:
            return
        if self._best_val_loss is not None and val_loss >= self._best_val_loss:
            return
        self._best_val_loss = val_loss
        self._save_trainable_parameters(pipe)

    def _write_checkpoint_manifest(self, weights_dir: Path, current_component: CheckpointComponent) -> None:
        inherited = [component for component in self._loaded_components if component.method != current_component.method]
        self._checkpoint_registry.write_manifest(
            weights_dir,
            model_name=self.config.model_name,
            components=[*inherited, current_component],
        )

    def _preload_images(self, pairs: list[TrainingPair]) -> None:
        paths = {p.input_image_path for p in pairs} | {p.target_image_path for p in pairs}
        for path in sorted(paths):
            if path in self._image_cache:
                continue
            img = load_rgba_as_rgb(path)
            self._image_sizes[path] = img.size
            self._image_cache[path] = np.asarray(img).copy()
            img.close()

    def _open_cached(self, path: str) -> Image.Image:
        arr = self._image_cache.get(path)
        if arr is not None:
            return Image.fromarray(arr)
        return load_rgba_as_rgb(path)

    def _release_trainable_tensors(self) -> None:
        self._soft_prompt = None
        self._frozen_soft_prompt = None
        self._prompt_embedding_delta = None
        self._frozen_prompt_embedding_delta = None
        self._trainable_params = []
        self._image_cache.clear()
        self._image_sizes.clear()
        self._prompt_embed_cache = None


    def _pair_bucket(self, pair: TrainingPair) -> tuple[int, int]:
        from .augmentation import RESOLUTION_BUCKETS, nearest_bucket

        size = self._image_sizes.get(pair.target_image_path)
        if size is None:
            return (self.config.resolution, self.config.resolution)
        bucket = nearest_bucket(size, RESOLUTION_BUCKETS)
        base_pixels = self.config.resolution ** 2
        bw, bh = bucket
        scale = (base_pixels / (bw * bh)) ** 0.5
        sw = max(64, round(bw * scale / 64) * 64)
        sh = max(64, round(bh * scale / 64) * 64)
        return (sw, sh)

    def _count_training_batches(self, pairs: list[TrainingPair]) -> int:
        if not self.config.use_buckets:
            return math.ceil(len(pairs) / max(self.config.batch_size, 1))
        bucket_counts: dict[tuple[int, int], int] = {}
        for pair in pairs:
            bucket = self._pair_bucket(pair)
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        return sum(math.ceil(count / max(self.config.batch_size, 1)) for count in bucket_counts.values())

    def _bucketed_batches(self, pairs: list[TrainingPair]) -> list[tuple[tuple[TrainingPair, ...], int, int]]:
        bucket_groups: dict[tuple[int, int], list[TrainingPair]] = {}
        for pair in pairs:
            bucket = self._pair_bucket(pair)
            if bucket not in bucket_groups:
                bucket_groups[bucket] = []
            bucket_groups[bucket].append(pair)
        batches: list[tuple[tuple[TrainingPair, ...], int, int]] = []
        for bucket, group in bucket_groups.items():
            random.shuffle(group)
            for batch in _batches(group, self.config.batch_size):
                batches.append((batch, bucket[0], bucket[1]))
        random.shuffle(batches)
        return batches


FluxKontextLoraTrainer = FluxKontextTrainer


def _fal_kontext_lora_target_modules(pipe) -> list[str]:
    """Return explicit diffusers modules corresponding to fal.ai Flux Kontext LoRA.

    fal.ai's adapter uses fused names such as ``img_attn.qkv`` and
    ``single.linear1``. Diffusers exposes those as separate Linear modules:
    q/k/v projections plus the single-block MLP projection. We target the
    functional equivalent explicitly by full module name so PEFT does not pick
    up unrelated modules through broad suffix matching.
    """
    double_block_tails = {
        "norm1.linear",
        "norm1_context.linear",
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out.0",
        "attn.add_q_proj",
        "attn.add_k_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "ff.net.0.proj",
        "ff.net.2",
        "ff_context.net.0.proj",
        "ff_context.net.2",
    }
    single_block_tails = {
        "norm.linear",
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "proj_mlp",
        "proj_out",
    }
    targets: list[str] = []
    for name, module in pipe.transformer.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        parts = name.split(".")
        if name.startswith("transformer_blocks.") and len(parts) > 2:
            if ".".join(parts[2:]) in double_block_tails:
                targets.append(name)
        elif name.startswith("single_transformer_blocks.") and len(parts) > 2:
            if ".".join(parts[2:]) in single_block_tails:
                targets.append(name)
        elif name == "proj_out":
            targets.append(name)
    return targets


def _linear_modules_by_name(root: torch.nn.Module, names: list[str]) -> list[torch.nn.Linear]:
    modules = dict(root.named_modules())
    linear_modules: list[torch.nn.Linear] = []
    for name in names:
        module = modules.get(name)
        if not isinstance(module, torch.nn.Linear):
            raise RuntimeError(f"LoRA target module is not a Linear layer: {name}")
        linear_modules.append(module)
    return linear_modules


def _all_linear_module_names(root: torch.nn.Module) -> list[str]:
    return [name for name, module in root.named_modules() if name and isinstance(module, torch.nn.Linear)]


def _has_peft_adapter(module: torch.nn.Module) -> bool:
    peft_config = getattr(module, "peft_config", None)
    return bool(peft_config)


def _active_lora_metadata(module: torch.nn.Module) -> dict[str, object]:
    peft_config = getattr(module, "peft_config", None)
    if not peft_config:
        raise RuntimeError("No PEFT LoRA adapter is attached.")
    adapter_name = next(iter(peft_config))
    config = peft_config[adapter_name]
    target_modules = getattr(config, "target_modules", [])
    if isinstance(target_modules, set):
        target_modules = sorted(target_modules)
    else:
        target_modules = list(target_modules)
    return {
        "rank": int(getattr(config, "r")),
        "target_modules": target_modules,
    }


def _batches(examples: list[TrainingPair], batch_size: int) -> list[tuple[TrainingPair, ...]]:
    size = max(batch_size, 1)
    return [tuple(examples[index : index + size]) for index in range(0, len(examples), size)]


def _round_loss_component(loss: torch.Tensor | None) -> float | None:
    if loss is None:
        return None
    return round(float(loss.detach().cpu()), 6)


def _decode_data_url(data_url: str) -> bytes:
    _, _, encoded = data_url.partition(",")
    return base64.b64decode(encoded)


def _training_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    raise RuntimeError("CUDA GPU is not available. Flux Kontext training requires a CUDA-enabled PyTorch environment.")


def _training_dtype(device: torch.device) -> torch.dtype:
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _dtype_size(dtype: torch.dtype) -> int:
    if dtype in {torch.float16, torch.bfloat16}:
        return 2
    if dtype in {torch.float32, torch.int32}:
        return 4
    if dtype in {torch.float64, torch.int64}:
        return 8
    return 2


def _cuda_total_memory(device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    return torch.cuda.get_device_properties(device).total_memory


def _cuda_free_memory(device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    try:
        free_memory, _ = torch.cuda.mem_get_info(device)
    except Exception:
        return 0
    return free_memory


def _optimizer_bytes_per_parameter(optimizer_name: str) -> float:
    if optimizer_name == "adamw8bit":
        return 1.5
    return 8.0


def release_cuda_memory(pipe=None) -> None:
    if pipe is not None:
        try:
            pipe.to("cpu")
        except Exception:
            pass
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass


def _hugging_face_token() -> str | bool:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or True


def _torch_load(path: Path) -> dict[str, object]:
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        loaded = torch.load(path, map_location="cpu")
    if not isinstance(loaded, dict):
        raise RuntimeError(f"Checkpoint '{path}' is not a dictionary.")
    return loaded


def _is_hugging_face_auth_error(exc: Exception) -> bool:
    if isinstance(exc, GatedRepoError):
        return True
    message = str(exc).lower()
    return (
        "gated repo" in message
        or "gated repositories" in message
        or "401 client error" in message
        or "403 forbidden" in message
        or "restricted" in message
    )


def _hugging_face_auth_message(model_name: str) -> str:
    return (
        f"Cannot access Hugging Face model '{model_name}'. "
        "The model is gated, so approve access on the model page and authenticate with "
        "`.venv312/bin/python -c \"from huggingface_hub.cli.hf import main; main()\" auth login`, "
        "or set the `HF_TOKEN` environment variable before starting the API. "
        "If you use a fine-grained token, enable read access and public gated repository access for this model."
    )


def _validate_instruction_token_length(config: TrainingConfig) -> None:
    effective_prompt = config.instruction.strip() or config.caption.strip()
    if not effective_prompt:
        return
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, subfolder="tokenizer_2", token=_hugging_face_token()
        )
    except Exception:
        return
    token_ids = tokenizer(
        effective_prompt,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=False,
        verbose=False,
    )["input_ids"]
    count = len(token_ids)
    limit = config.max_sequence_length
    if config.method == "prompt_tuning":
        limit = max(1, limit - config.prompt_tuning_tokens)
    if count > limit:
        pt_note = (
            f" (prompt tuning で {config.prompt_tuning_tokens} トークンが使用されるため実効上限は {limit})"
            if config.method == "prompt_tuning"
            else ""
        )
        raise ValueError(
            f"T5 prompt が {count} トークンで上限 {config.max_sequence_length} を超えています{pt_note}。"
            " instruction/caption を短くしてください。Flux Kontext の max sequence length は 512 です。"
        )


def _calculate_timestep_shift(image_seq_len: int) -> float:
    """Compute mu for resolution-dependent sigma shifting (matches FluxKontextPipeline)."""
    base_seq_len = 256
    max_seq_len = 4096
    base_shift = 0.5
    max_shift = 1.15
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return m * image_seq_len + b


def _time_shift(mu: float, t: torch.Tensor) -> torch.Tensor:
    """Apply exponential time shift: f(t) = exp(mu) / (exp(mu) + 1/t - 1)."""
    exp_mu = math.exp(mu)
    return exp_mu / (exp_mu + (1.0 / t - 1.0))


def _build_optimizer(parameter_groups: list[dict[str, object]], optimizer_name: str) -> torch.optim.Optimizer:
    if optimizer_name == "adamw8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise RuntimeError(
                "8-bit AdamW requires bitsandbytes. Install it with: pip install bitsandbytes"
            ) from exc
        return bnb.optim.AdamW8bit(parameter_groups)
    return torch.optim.AdamW(parameter_groups)


def _cosine_segment(progress: float, start_ratio: float, end_ratio: float) -> float:
    progress = min(max(progress, 0.0), 1.0)
    return end_ratio + (start_ratio - end_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    total_steps: int,
    warmup_fraction: float,
    num_cycles: int,
    cosine_decay_fraction: float = 1.0,
    cosine_mid_ratio: float = 0.1,
    cosine_final_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Builds a schedule expressed as a multiplier of the configured learning rate.

    For cosine schedules with cosine_decay_fraction < 1.0, the decay runs in two segments:
    phase 1 drops from 1.0 to cosine_mid_ratio over the first cosine_decay_fraction of the
    post-warmup steps, then phase 2 continues from cosine_mid_ratio down to cosine_final_ratio
    over the remaining steps. Using a single LambdaLR (rather than chaining two
    CosineAnnealingLR instances) avoids a discontinuity: chained schedulers each capture the
    optimizer's original base LR at construction time, so the second leg would otherwise jump
    back up to the start LR instead of continuing from the phase-1 end value.
    """
    if warmup_fraction > 0.0:
        warmup_steps = max(1, int(total_steps * warmup_fraction))
    else:
        warmup_steps = 0
    decay_steps = max(1, total_steps - warmup_steps)

    if scheduler_name == "constant":
        if warmup_steps == 0:
            return None

        def constant_lambda(step: int) -> float:
            if step < warmup_steps:
                return max(1e-6, step / warmup_steps)
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=constant_lambda)

    if scheduler_name == "linear":

        def linear_lambda(step: int) -> float:
            if warmup_steps and step < warmup_steps:
                return max(1e-6, step / warmup_steps)
            progress = min((step - warmup_steps) / decay_steps, 1.0)
            return max(0.0, 1.0 - progress)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=linear_lambda)

    if scheduler_name in {"cosine", "cosine_with_restarts"}:
        phase1_steps = max(1, int(decay_steps * cosine_decay_fraction))
        phase2_steps = max(1, decay_steps - phase1_steps)

        def cosine_lambda(step: int) -> float:
            if warmup_steps and step < warmup_steps:
                return max(1e-6, step / warmup_steps)
            elapsed = step - warmup_steps
            if cosine_decay_fraction >= 1.0:
                if elapsed >= decay_steps:
                    return cosine_final_ratio
                progress = max(0.0, elapsed / decay_steps)
                if scheduler_name == 'cosine_with_restarts':
                    progress = (progress * max(1, num_cycles)) % 1.0
                return _cosine_segment(progress, 1.0, cosine_final_ratio)
            if elapsed < phase1_steps:
                return _cosine_segment(elapsed / phase1_steps, 1.0, cosine_mid_ratio)
            progress = (elapsed - phase1_steps) / phase2_steps
            return _cosine_segment(progress, cosine_mid_ratio, cosine_final_ratio)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_lambda)

    return None


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
