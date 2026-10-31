from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DatasetSummary(BaseModel):
    name: str
    path: str
    train_count: int
    val_count: int
    train_pair_count: int = 0
    val_pair_count: int = 0
    train_caption_count: int = 0
    val_caption_count: int = 0
    train_missing_caption_count: int = 0
    val_missing_caption_count: int = 0
    has_train: bool
    has_val: bool


class TrainingExample(BaseModel):
    image_path: str
    caption: str
    caption_path: str | None = None
    uses_caption: bool


class TrainingPair(BaseModel):
    input_image_path: str
    target_image_path: str
    caption: str
    instruction: str
    caption_path: str | None = None
    uses_caption: bool
    crop_box: tuple[int, int, int, int] | None = None


TrainingMethod = Literal["lora", "prompt_tuning", "full"]
PromptTuningScope = Literal["soft_prompt", "soft_prompt_and_prompt"]
RotationMode = Literal["none", "cardinal", "full"]
AdapterLossSplit = Literal["train", "val", "both"]
ReconstructionLossSchedule = Literal["constant", "linear_ramp", "cosine_ramp", "wave", "cosine_ramp_wave"]
Optimizer = Literal["adamw", "adamw8bit"]
LRScheduler = Literal["constant", "linear", "cosine", "cosine_with_restarts"]


class TrainingConfig(BaseModel):
    dataset: str
    method: TrainingMethod = "lora"
    source_checkpoint_id: str | None = None
    model_name: str = "black-forest-labs/FLUX.1-Kontext-dev"
    caption: str = ""
    instruction: str = ""
    inference_instruction: str = ""
    epochs: int = Field(default=10, ge=1, le=10_000)
    learning_rate: float = Field(default=1e-4, gt=0)
    rank: int = Field(default=16, ge=1, le=256)
    lora_rank_override: int | None = Field(default=None, ge=1, le=256)
    transformer_lora_parameter_ratio_percent: float = Field(default=1.0, ge=0.1, le=50.0)
    text_encoder_lora_parameter_ratio_percent: float = Field(default=1.0, ge=0.1, le=50.0)
    prompt_tuning_tokens: int = Field(default=1, ge=1, le=128)
    prompt_tuning_scope: PromptTuningScope = "soft_prompt"
    prompt_embedding_learning_rate: float = Field(default=1e-5, gt=0)
    batch_size: int = Field(default=2, ge=1, le=128)
    resolution: int = Field(default=1024, ge=256, le=2048)
    gradient_accumulation_steps: int = Field(default=2, ge=1, le=128)
    max_sequence_length: int = Field(default=512, ge=16, le=512)
    seed: int = Field(default=42, ge=0)
    optimizer: Optimizer = "adamw8bit"
    train_text_encoder: bool = False
    train_text_encoder_2: bool = False
    text_encoder_learning_rate: float | None = Field(default=None, gt=0)
    lr_scheduler: LRScheduler = "cosine"
    lr_warmup_fraction: float = Field(default=0.05, ge=0.0, le=1.0)
    lr_num_cycles: int = Field(default=1, ge=1)
    lr_cosine_decay_fraction: float = Field(default=1.0, ge=0.05, le=1.0)
    lr_cosine_mid_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    lr_cosine_final_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    augment: bool = True
    rotation_mode: RotationMode = "cardinal"
    use_tiling: bool = False
    caption_dropout: float = Field(default=0.1, ge=0.0, le=1.0)
    use_buckets: bool = True
    loss_delta_weight: float = Field(default=0.0, ge=0.0, le=100.0)
    loss_delta_quantile: float = Field(default=0.95, ge=0.5, le=1.0)
    reconstruction_latent_loss_weight: float = Field(default=0.0, ge=0.0, le=100.0)
    reconstruction_pixel_loss_weight: float = Field(default=0.0, ge=0.0, le=100.0)
    reconstruction_pixel_loss_crop_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    reconstruction_loss_schedule: ReconstructionLossSchedule = "cosine_ramp"
    reconstruction_loss_start_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    reconstruction_loss_warmup_fraction: float = Field(default=1.0, ge=0.0, le=1.0)
    reconstruction_loss_wave_amplitude: float = Field(default=0.2, ge=0.0, le=1.0)
    reconstruction_loss_wave_cycles: float = Field(default=3.0, ge=0.0, le=100.0)
    validation_seed: int = Field(default=1042, ge=0)
    validation_noise_repeats: int = Field(default=1, ge=1, le=8)
    visual_evaluation_enabled: bool = False
    evaluation_interval: int = Field(default=2, ge=1)
    evaluation_samples: int = Field(default=4, ge=1, le=10)
    evaluation_inference_steps: int = Field(default=28, ge=4, le=50)
    evaluation_guidance_scale: float = Field(default=3.5, ge=1.0, le=10.0)
    early_stopping_patience: int = Field(default=5, ge=1)
    early_stopping_min_epochs: int = Field(default=8, ge=1)
    early_stopping_min_delta: float = Field(default=0.0001, ge=0)
    plateau_lr_patience: int = Field(default=2, ge=1)
    plateau_lr_factor: float = Field(default=0.5, gt=0, lt=1)
    minimum_learning_rate: float = Field(default=1e-6, gt=0)


class LearningRateUpdate(BaseModel):
    learning_rate: float = Field(gt=0)


class MetricEvent(BaseModel):
    step: int
    train_loss: float
    val_loss: float | None = None
    epoch: int | None = None
    learning_rate: float | None = None
    gradient_norm: float | None = None
    optimizer_step: int | None = None
    train_flow_loss: float | None = None
    train_reconstruction_latent_loss: float | None = None
    train_reconstruction_pixel_loss: float | None = None
    reconstruction_latent_weight: float | None = None
    reconstruction_pixel_weight: float | None = None


PreviewSource = Literal["train", "val"]


class EvaluationPreview(BaseModel):
    epoch: int
    step: int
    prompt: str
    input_image: str
    target_image: str
    prediction_image: str
    source: PreviewSource = "val"


class PreviewHistoryEntry(BaseModel):
    epoch: int
    step: int
    prompt: str
    source: PreviewSource = "val"


class CheckpointComponent(BaseModel):
    method: TrainingMethod
    path: str


class CheckpointManifest(BaseModel):
    version: int = 1
    model_name: str
    components: list[CheckpointComponent]


class TrainingCheckpointSummary(BaseModel):
    id: str
    label: str
    run_id: str
    created_at: str
    dataset: str
    model_name: str
    method: TrainingMethod
    components: list[TrainingMethod]
    path: str


class TrainingJob(BaseModel):
    id: str
    status: JobStatus
    config: TrainingConfig
    created_at: str
    updated_at: str
    run_dir: str
    error: str | None = None
    latest_metric: MetricEvent | None = None


class AdapterSummary(BaseModel):
    name: str
    path: str
    size_bytes: int


class AdapterLossRequest(BaseModel):
    adapter_path: str
    config: TrainingConfig
    split: AdapterLossSplit = "val"
    batch_size: int | None = Field(default=None, ge=1, le=128)
    max_pairs: int | None = Field(default=None, ge=1)
    noise_repeats: int = Field(default=1, ge=1, le=32)
    rollout_steps: int = Field(default=0, ge=0, le=64)


class AdapterLossResult(BaseModel):
    split: str
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


class AdapterLossJob(BaseModel):
    id: str
    status: JobStatus
    request: AdapterLossRequest
    created_at: str
    updated_at: str
    run_dir: str
    results: list[AdapterLossResult] = []
    error: str | None = None
