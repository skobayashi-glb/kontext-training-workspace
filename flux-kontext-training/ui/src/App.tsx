import { Activity, Database, Gauge, History, Image as ImageIcon, PauseCircle, Play, RefreshCcw, Settings, Trash2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

type DatasetSummary = {
  name: string;
  path: string;
  train_count: number;
  val_count: number;
  train_pair_count: number;
  val_pair_count: number;
  train_caption_count: number;
  val_caption_count: number;
  train_missing_caption_count: number;
  val_missing_caption_count: number;
  has_train: boolean;
  has_val: boolean;
};

type TrainingMode = {
  id: TrainingMethod;
  label: string;
  enabled: boolean;
};

type TrainingCheckpoint = {
  id: string;
  label: string;
  run_id: string;
  created_at: string;
  dataset: string;
  model_name: string;
  method: TrainingMethod;
  components: TrainingMethod[];
  path: string;
};

type AdapterSummary = {
  name: string;
  path: string;
  size_bytes: number;
};

type AdapterLossResult = {
  split: string;
  pair_count: number;
  batch_count: number;
  noise_repeats: number;
  mean_loss: number;
  min_loss: number;
  max_loss: number;
  base_mean_loss: number | null;
  relative_improvement: number | null;
  focused_mean_loss: number | null;
  focused_base_mean_loss: number | null;
  focused_relative_improvement: number | null;
  changed_token_fraction: number | null;
  preview: EvaluationPreview | null;
};

type AdapterLossJob = {
  id: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  request: {
    adapter_path: string;
    config: TrainingConfig;
    split: AdapterLossSplit;
    batch_size: number | null;
    max_pairs: number | null;
    noise_repeats: number;
  };
  created_at: string;
  updated_at: string;
  run_dir: string;
  results: AdapterLossResult[];
  error: string | null;
};

type MetricEvent = {
  step: number;
  train_loss: number;
  val_loss: number | null;
  epoch: number | null;
  learning_rate: number | null;
  train_flow_loss: number | null;
  train_reconstruction_latent_loss: number | null;
  train_reconstruction_pixel_loss: number | null;
  reconstruction_latent_weight: number | null;
  reconstruction_pixel_weight: number | null;
};

type PreviewSource = "train" | "val";

type EvaluationPreview = {
  epoch: number;
  step: number;
  prompt: string;
  input_image: string;
  target_image: string;
  prediction_image: string;
  source: PreviewSource;
};

type PreviewHistoryEntry = {
  epoch: number;
  step: number;
  prompt: string;
  source: PreviewSource;
};

type TrainingJob = {
  id: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  config: TrainingConfig;
  created_at: string;
  updated_at: string;
  run_dir: string;
  error: string | null;
  latest_metric: MetricEvent | null;
};

type TrainingConfig = {
  dataset: string;
  method: TrainingMethod;
  source_checkpoint_id: string | null;
  model_name: string;
  caption: string;
  instruction: string;
  inference_instruction: string;
  epochs: number;
  learning_rate: number;
  rank: number;
  transformer_lora_parameter_ratio_percent: number;
  text_encoder_lora_parameter_ratio_percent: number;
  prompt_tuning_tokens: number;
  prompt_tuning_scope: PromptTuningScope;
  prompt_embedding_learning_rate: number;
  batch_size: number;
  resolution: number;
  gradient_accumulation_steps: number;
  max_sequence_length: number;
  seed: number;
  optimizer: Optimizer;
  train_text_encoder: boolean;
  train_text_encoder_2: boolean;
  text_encoder_learning_rate: number | null;
  lr_scheduler: LRScheduler;
  lr_warmup_fraction: number;
  lr_num_cycles: number;
  lr_cosine_decay_fraction: number;
  lr_cosine_mid_ratio: number;
  lr_cosine_final_ratio: number;
  augment: boolean;
  rotation_mode: RotationMode;
  use_tiling: boolean;
  caption_dropout: number;
  use_buckets: boolean;
  loss_delta_weight: number;
  loss_delta_quantile: number;
  reconstruction_latent_loss_weight: number;
  reconstruction_pixel_loss_weight: number;
  reconstruction_pixel_loss_crop_fraction: number;
  reconstruction_loss_schedule: ReconstructionLossSchedule;
  reconstruction_loss_start_fraction: number;
  reconstruction_loss_warmup_fraction: number;
  reconstruction_loss_wave_amplitude: number;
  reconstruction_loss_wave_cycles: number;
};

type PromptPreset = {
  id: string;
  label: string;
  task: string;
};

type TrainingMethod = "lora" | "prompt_tuning" | "full";
type PromptTuningScope = "soft_prompt" | "soft_prompt_and_prompt";
type RotationMode = "none" | "cardinal" | "full";
type AdapterLossSplit = "train" | "val" | "both";
type ReconstructionLossSchedule = "constant" | "linear_ramp" | "cosine_ramp" | "wave" | "cosine_ramp_wave";
type Optimizer = "adamw" | "adamw8bit";
type LRScheduler = "constant" | "linear" | "cosine" | "cosine_with_restarts";

const defaultConfig: TrainingConfig = {
  dataset: "",
  method: "lora",
  source_checkpoint_id: null,
  model_name: "black-forest-labs/FLUX.1-Kontext-dev",
  caption: "a photo of TOK floor plan",
  instruction: "",
  inference_instruction: "",
  epochs: 10,
  learning_rate: 0.0001,
  rank: 16,
  transformer_lora_parameter_ratio_percent: 10,
  text_encoder_lora_parameter_ratio_percent: 1,
  prompt_tuning_tokens: 1,
  prompt_tuning_scope: "soft_prompt",
  prompt_embedding_learning_rate: 0.00001,
  batch_size: 2,
  resolution: 1024,
  gradient_accumulation_steps: 2,
  max_sequence_length: 512,
  seed: 42,
  optimizer: "adamw8bit" as Optimizer,
  train_text_encoder: false,
  train_text_encoder_2: false,
  text_encoder_learning_rate: null,
  lr_scheduler: "cosine" as LRScheduler,
  lr_warmup_fraction: 0.02,
  lr_num_cycles: 1,
  lr_cosine_decay_fraction: 0.5,
  lr_cosine_mid_ratio: 0.1,
  lr_cosine_final_ratio: 0.05,
  augment: true,
  rotation_mode: "cardinal" as RotationMode,
  use_tiling: false,
  caption_dropout: 0.1,
  use_buckets: true,
  loss_delta_weight: 0,
  loss_delta_quantile: 0.95,
  reconstruction_latent_loss_weight: 0,
  reconstruction_pixel_loss_weight: 0,
  reconstruction_pixel_loss_crop_fraction: 0.5,
  reconstruction_loss_schedule: "cosine_ramp",
  reconstruction_loss_start_fraction: 0,
  reconstruction_loss_warmup_fraction: 1,
  reconstruction_loss_wave_amplitude: 0.2,
  reconstruction_loss_wave_cycles: 3
};

const defaultModes: TrainingMode[] = [
  { id: "lora", label: "LoRA", enabled: true },
  { id: "prompt_tuning", label: "Prompt-tuning", enabled: true },
  { id: "full", label: "Full", enabled: true }
];
const terminalStatuses = new Set<TrainingJob["status"]>(["completed", "failed", "cancelled"]);

export function App() {
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [modes, setModes] = useState<TrainingMode[]>([]);
  const [presets, setPresets] = useState<PromptPreset[]>([]);
  const [checkpoints, setCheckpoints] = useState<TrainingCheckpoint[]>([]);
  const [adapters, setAdapters] = useState<AdapterSummary[]>([]);
  const [adapterLossJobs, setAdapterLossJobs] = useState<AdapterLossJob[]>([]);
  const [adapterLossJob, setAdapterLossJob] = useState<AdapterLossJob | null>(null);
  const [selectedAdapterPath, setSelectedAdapterPath] = useState("");
  const [adapterLossConfig, setAdapterLossConfig] = useState<TrainingConfig>(defaultConfig);
  const [adapterLossSplit, setAdapterLossSplit] = useState<AdapterLossSplit>("val");
  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [config, setConfig] = useState<TrainingConfig>(defaultConfig);
  const [job, setJob] = useState<TrainingJob | null>(null);
  const [metrics, setMetrics] = useState<MetricEvent[]>([]);
  const [preview, setPreview] = useState<EvaluationPreview | null>(null);
  const [trainPreview, setTrainPreview] = useState<EvaluationPreview | null>(null);
  const [message, setMessage] = useState("");
  const selectedDataset = datasets.find((dataset) => dataset.name === config.dataset);
  const selectedAdapterDataset = datasets.find((dataset) => dataset.name === adapterLossConfig.dataset);
  const compatibleCheckpoints =
    config.method === "full"
      ? []
      : checkpoints.filter((checkpoint) => checkpoint.model_name === config.model_name && !checkpoint.components.includes("full"));
  const activeConfig = job?.config ?? config;
  const activeDataset = datasets.find((dataset) => dataset.name === activeConfig.dataset);
  const isRunning = job?.status === "queued" || job?.status === "running";
  const isAdapterLossRunning = adapterLossJob?.status === "queued" || adapterLossJob?.status === "running";

  useEffect(() => {
    void refreshWorkspace();
  }, []);

  useEffect(() => {
    if (!selectedAdapterPath && adapters[0]) {
      setSelectedAdapterPath(adapters[0].path);
    }
  }, [adapters, selectedAdapterPath]);

  useEffect(() => {
    if (!adapterLossConfig.dataset && datasets[0]) {
      setAdapterLossConfig((current) => ({ ...current, dataset: datasets[0].name }));
    }
  }, [adapterLossConfig.dataset, datasets]);

  useEffect(() => {
    if (!job) {
      return;
    }

    let closed = false;
    void loadMetrics(job.id).then((loadedMetrics) => {
      if (!closed) {
        setMetrics(loadedMetrics);
      }
    }).catch((error: unknown) => {
      if (!closed) setMessage(errorMessage(error, "metrics を読み込めませんでした。"));
    });
    void loadPreview(job.id, "val").then((loadedPreview) => {
      if (!closed) {
        setPreview(loadedPreview);
      }
    }).catch((error: unknown) => {
      if (!closed) setMessage(errorMessage(error, "preview を読み込めませんでした。"));
    });
    void loadPreview(job.id, "train").then((loadedPreview) => {
      if (!closed) {
        setTrainPreview(loadedPreview);
      }
    }).catch((error: unknown) => {
      if (!closed) setMessage(errorMessage(error, "preview を読み込めませんでした。"));
    });

    if (terminalStatuses.has(job.status)) {
      return () => {
        closed = true;
      };
    }

    const source = new EventSource(`/api/jobs/${job.id}/events`);
    source.addEventListener("metric", (event) => {
      try {
        const metric = JSON.parse((event as MessageEvent).data) as MetricEvent;
        setMetrics((current) => mergeMetric(current, metric));
      } catch (error) {
        setMessage(errorMessage(error, "metric event を処理できませんでした。"));
      }
    });
    source.addEventListener("status", (event) => {
      try {
        const nextJob = normalizeJob(JSON.parse((event as MessageEvent).data) as TrainingJob);
        setJob(nextJob);
        setJobs((current) => upsertJob(current, nextJob));
        if (terminalStatuses.has(nextJob.status)) {
          void loadCheckpoints().catch((error: unknown) => {
            setMessage(errorMessage(error, "checkpoints を読み込めませんでした。"));
          });
        }
      } catch (error) {
        setMessage(errorMessage(error, "status event を処理できませんでした。"));
      }
    });
    source.addEventListener("preview", (event) => {
      try {
        const nextPreview = JSON.parse((event as MessageEvent).data) as EvaluationPreview;
        if (nextPreview.source === "train") {
          setTrainPreview(nextPreview);
        } else {
          setPreview(nextPreview);
        }
      } catch (error) {
        setMessage(errorMessage(error, "preview event を処理できませんでした。"));
      }
    });
    source.onerror = () => {
      setMessage("ジョブイベント接続が切断されました。バックエンドログも確認してください。");
      source.close();
    };
    return () => {
      closed = true;
      source.close();
    };
  }, [job?.id, job?.status]);

  useEffect(() => {
    if (!adapterLossJob || terminalStatuses.has(adapterLossJob.status)) {
      return;
    }
    const id = window.setInterval(() => {
      void loadAdapterLossJob(adapterLossJob.id);
    }, 3000);
    return () => window.clearInterval(id);
  }, [adapterLossJob?.id, adapterLossJob?.status]);

  const latest = metrics.at(-1);
  const totalSteps = getTotalSteps(activeConfig, activeDataset);
  const completedPercent = latest ? Math.min(100, Math.round((latest.step / Math.max(totalSteps, 1)) * 100)) : 0;
  const timeInfo = useTimeInfo(job, metrics, totalSteps);
  const adapterLossDisabledReason = getAdapterLossDisabledReason(
    selectedAdapterPath,
    adapterLossConfig,
    isAdapterLossRunning
  );

  async function refreshWorkspace() {
    setMessage("");
    const [loadedDatasets, loadedJobs, , , , loadedAdapters, loadedAdapterLossJobs] = await Promise.all([
      loadDatasets(),
      loadJobs(),
      loadModes(),
      loadCheckpoints(),
      loadPresets(),
      loadAdapters(),
      loadAdapterLossJobs()
    ]);
    if (!job && loadedJobs[0]) {
      setJob(normalizeJob(loadedJobs[0]));
    }
    if (loadedDatasets[0]) {
      setConfig((current) => ({ ...current, dataset: current.dataset || loadedDatasets[0].name }));
    }
    if (loadedDatasets[0]) {
      setAdapterLossConfig((current) => ({ ...current, dataset: current.dataset || loadedDatasets[0].name }));
    }
    if (!selectedAdapterPath && loadedAdapters[0]) {
      setSelectedAdapterPath(loadedAdapters[0].path);
    }
    if (!adapterLossJob && loadedAdapterLossJobs[0]) {
      setAdapterLossJob(loadedAdapterLossJobs[0]);
    }
  }

  async function loadDatasets() {
    const response = await fetch("/api/datasets");
    const data = (await response.json()) as { datasets: DatasetSummary[] };
    setDatasets(data.datasets);
    setConfig((current) => ({
      ...current,
      dataset: current.dataset || data.datasets[0]?.name || ""
    }));
    return data.datasets;
  }

  async function loadPresets() {
    const response = await fetch("/api/prompts");
    const data = (await response.json()) as { presets: PromptPreset[] };
    setPresets(data.presets);
    return data.presets;
  }

  async function loadModes() {
    const response = await fetch("/api/training/modes");
    const data = (await response.json()) as { modes: TrainingMode[] };
    setModes(data.modes);
    return data.modes;
  }

  async function loadCheckpoints() {
    const response = await fetch("/api/checkpoints");
    const data = (await response.json()) as { checkpoints: TrainingCheckpoint[] };
    setCheckpoints(data.checkpoints);
    return data.checkpoints;
  }

  async function loadAdapters() {
    const response = await fetch("/api/adapters");
    const data = (await response.json()) as { adapters: AdapterSummary[] };
    setAdapters(data.adapters);
    return data.adapters;
  }

  async function loadAdapterLossJobs() {
    const response = await fetch("/api/adapter-loss/jobs");
    const data = (await response.json()) as { jobs: AdapterLossJob[] };
    setAdapterLossJobs(data.jobs);
    return data.jobs;
  }

  async function loadAdapterLossJob(jobId: string) {
    const response = await fetch(`/api/adapter-loss/jobs/${jobId}`);
    const nextJob = (await response.json()) as AdapterLossJob;
    setAdapterLossJob(nextJob);
    setAdapterLossJobs((current) => upsertAdapterLossJob(current, nextJob));
    return nextJob;
  }

  async function loadJobs() {
    const response = await fetch("/api/jobs");
    const data = (await response.json()) as { jobs: TrainingJob[] };
    const jobs = data.jobs.map(normalizeJob);
    setJobs(jobs);
    return jobs;
  }

  async function loadMetrics(jobId: string) {
    const response = await fetch(`/api/jobs/${jobId}/metrics`);
    const data = (await response.json()) as { metrics: MetricEvent[] };
    return data.metrics;
  }

  async function loadPreview(jobId: string, source: PreviewSource) {
    const response = await fetch(`/api/jobs/${jobId}/preview?source=${source}`);
    const data = (await response.json()) as { preview: EvaluationPreview | null };
    return data.preview;
  }

  async function startTraining() {
    setMessage("");
    setMetrics([]);
    setPreview(null);
    setTrainPreview(null);
    try {
      const response = await fetch("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(normalizeConfig(config))
      });
      if (!response.ok) {
        const detail = await responseDetail(response);
        setMessage(detail || "学習を開始できませんでした。");
        return;
      }
      const createdJob = normalizeJob((await response.json()) as TrainingJob);
      setJob(createdJob);
      setJobs((current) => upsertJob(current, createdJob));
    } catch (error) {
      setMessage(errorMessage(error, "学習開始中にUIエラーが発生しました。"));
    }
  }

  async function startAdapterLossEvaluation() {
    if (!selectedAdapterPath || isAdapterLossRunning) {
      return;
    }
    setMessage("");
    const response = await fetch("/api/adapter-loss/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        adapter_path: selectedAdapterPath,
        config: adapterLossConfig,
        split: adapterLossSplit,
        batch_size: adapterLossConfig.batch_size,
        max_pairs: 1,
        noise_repeats: 1
      })
    });
    if (!response.ok) {
      const error = await response.json();
      setMessage(error.detail ?? "adapter loss 評価を開始できませんでした。");
      return;
    }
    const createdJob = (await response.json()) as AdapterLossJob;
    setAdapterLossJob(createdJob);
    setAdapterLossJobs((current) => upsertAdapterLossJob(current, createdJob));
  }

  async function cancelTraining() {
    if (!job) {
      return;
    }
    await fetch(`/api/jobs/${job.id}/cancel`, { method: "POST" });
    const refreshedJobs = await loadJobs();
    const refreshedJob = refreshedJobs.find((item) => item.id === job.id);
    if (refreshedJob) {
      setJob(refreshedJob);
    }
  }

  async function selectJob(nextJob: TrainingJob) {
    setMessage("");
    const normalizedJob = normalizeJob(nextJob);
    setJob(normalizedJob);
    setConfig(normalizedJob.config);
    const [nextMetrics, nextPreview, nextTrainPreview] = await Promise.all([
      loadMetrics(normalizedJob.id),
      loadPreview(normalizedJob.id, "val"),
      loadPreview(normalizedJob.id, "train"),
    ]);
    setMetrics(nextMetrics);
    setPreview(nextPreview);
    setTrainPreview(nextTrainPreview);
  }

  async function deleteJob(targetJob: TrainingJob) {
    if (!terminalStatuses.has(targetJob.status)) {
      setMessage("実行中のジョブは停止してから削除してください。");
      return;
    }
    if (!window.confirm(`Run ${shortId(targetJob.id)} を削除しますか？`)) {
      return;
    }
    setMessage("");
    const response = await fetch(`/api/jobs/${targetJob.id}`, { method: "DELETE" });
    if (!response.ok) {
      const error = await response.json();
      setMessage(error.detail ?? "ジョブを削除できませんでした。");
      return;
    }
    const refreshedJobs = await loadJobs();
    await loadCheckpoints();
    if (job?.id !== targetJob.id) {
      return;
    }
    const nextJob = refreshedJobs[0] ?? null;
    setJob(nextJob ? normalizeJob(nextJob) : null);
    if (nextJob) {
      const normalizedJob = normalizeJob(nextJob);
      setConfig(normalizedJob.config);
      const [nextMetrics, nextPreview, nextTrainPreview] = await Promise.all([
        loadMetrics(normalizedJob.id),
        loadPreview(normalizedJob.id, "val"),
        loadPreview(normalizedJob.id, "train"),
      ]);
      setMetrics(nextMetrics);
      setPreview(nextPreview);
      setTrainPreview(nextTrainPreview);
    } else {
      setMetrics([]);
      setPreview(null);
      setTrainPreview(null);
    }
  }

  function changeInitialLearningRate(learning_rate: number) {
    if (!Number.isFinite(learning_rate) || learning_rate <= 0) {
      return;
    }
    setConfig((current) => ({ ...current, learning_rate }));
  }

  async function changeRuntimeLearningRate(learning_rate: number) {
    if (!Number.isFinite(learning_rate) || learning_rate <= 0 || !isRunning || !job) {
      return;
    }
    const response = await fetch(`/api/jobs/${job.id}/learning-rate`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ learning_rate })
    });
    if (!response.ok) {
      const error = await response.json();
      setMessage(error.detail ?? "learning rate を変更できませんでした。");
      return;
    }
    const updatedJob = normalizeJob((await response.json()) as TrainingJob);
    setJob(updatedJob);
    setJobs((current) => upsertJob(current, updatedJob));
  }

  return (
    <main className="shell">
      <section className="topbar">
        <div>
          <p className="eyebrow">Local Trainer</p>
          <h1>Flux Kontext Training</h1>
        </div>
        <button className="iconButton" type="button" onClick={refreshWorkspace} title="再読み込み">
          <RefreshCcw size={18} />
        </button>
      </section>

      <section className="workspace">
        <aside className="panel controls">
          <div className="panelTitle">
            <Database size={18} />
            <span>Dataset</span>
          </div>
          <label>
            <span>ディレクトリ</span>
            <select
              value={config.dataset}
              onChange={(event) => setConfig({ ...config, dataset: event.target.value })}
              disabled={isRunning}
            >
              {datasets.length === 0 && <option value="">datasets がありません</option>}
              {datasets.map((dataset) => (
                <option value={dataset.name} key={dataset.name}>
                  {dataset.name}
                </option>
              ))}
            </select>
          </label>
          {selectedDataset && (
            <dl className="datasetStats">
              <div>
                <dt>train</dt>
                <dd>{selectedDataset.train_count}</dd>
              </div>
              <div>
                <dt>val</dt>
                <dd>{selectedDataset.val_count}</dd>
              </div>
              <div>
                <dt>train captions</dt>
                <dd>{selectedDataset.train_caption_count}</dd>
              </div>
              <div>
                <dt>missing captions</dt>
                <dd>{selectedDataset.train_missing_caption_count + selectedDataset.val_missing_caption_count}</dd>
              </div>
            </dl>
          )}

          <div className="panelTitle">
            <Settings size={18} />
            <span>Model</span>
          </div>
          <label>
            <span>model</span>
            <input
              type="text"
              value={config.model_name}
              disabled={isRunning}
              onChange={(event) => setConfig({ ...config, model_name: event.target.value })}
            />
          </label>
          <label>
            <span>source checkpoint</span>
            <select
              value={config.source_checkpoint_id ?? ""}
              disabled={isRunning || config.method === "full"}
              onChange={(event) => setConfig({ ...config, source_checkpoint_id: event.target.value || null })}
            >
              <option value="">base model</option>
              {compatibleCheckpoints.map((checkpoint) => (
                <option value={checkpoint.id} key={checkpoint.id}>
                  {checkpoint.label}
                </option>
              ))}
            </select>
          </label>
          <NumberField label="seed" value={config.seed} min={0} disabled={isRunning} onChange={(seed) => setConfig({ ...config, seed })} />
          <label>
            <span>caption (CLIP)</span>
            <textarea
              value={config.caption}
              disabled={isRunning}
              rows={3}
              placeholder="a photo of TOK"
              onChange={(event) => setConfig({ ...config, caption: event.target.value })}
            />
          </label>
          {presets.length > 0 && (
            <label>
              <span>instruction preset</span>
              <select
                disabled={isRunning}
                value=""
                onChange={(event) => {
                  const preset = presets.find((p) => p.id === event.target.value);
                  if (preset) setConfig({ ...config, instruction: preset.task });
                }}
              >
                <option value="">-- select preset --</option>
                {presets.map((preset) => (
                  <option value={preset.id} key={preset.id}>
                    {preset.label}
                  </option>
                ))}
              </select>
            </label>
          )}
          <label>
            <span>instruction (T5)</span>
            <textarea
              value={config.instruction}
              disabled={isRunning}
              rows={3}
              placeholder="Simplify this floor plan"
              onChange={(event) => setConfig({ ...config, instruction: event.target.value })}
            />
          </label>
          <label>
            <span>inference instruction</span>
            <textarea
              value={config.inference_instruction}
              disabled={isRunning}
              rows={3}
              placeholder="Leave empty to use training instruction"
              onChange={(event) => setConfig({ ...config, inference_instruction: event.target.value })}
            />
          </label>

          <div className="panelTitle">
            <Activity size={18} />
            <span>Training</span>
          </div>
          <div className="segmented">
            {(modes.length ? modes : defaultModes).map((mode) => (
              <button
                type="button"
                className={config.method === mode.id ? "active" : ""}
                disabled={!mode.enabled || isRunning}
                key={mode.id}
                onClick={() => setConfig({ ...config, method: mode.id, source_checkpoint_id: mode.id === "full" ? null : config.source_checkpoint_id })}
              >
                {mode.label}
              </button>
            ))}
          </div>
          <NumberField label="epochs" value={config.epochs} min={1} disabled={isRunning} onChange={(epochs) => setConfig({ ...config, epochs })} />
          {config.method === "lora" && (
            <>
              <label>
                <span>transformer LoRA ratio</span>
              </label>
              <div className="segmented">
                {[0.1, 1, 5, 10, 50].map((ratio) => (
                  <button
                    key={ratio}
                    type="button"
                    className={config.transformer_lora_parameter_ratio_percent === ratio ? "active" : ""}
                    disabled={isRunning}
                    onClick={() => setConfig({ ...config, transformer_lora_parameter_ratio_percent: ratio })}
                  >
                    {ratio}%
                  </button>
                ))}
              </div>
              {(config.train_text_encoder || config.train_text_encoder_2) && (
                <>
                  <label>
                    <span>encoder LoRA ratio</span>
                  </label>
                  <div className="segmented">
                    {[0.1, 1, 5, 10, 50].map((ratio) => (
                      <button
                        key={ratio}
                        type="button"
                        className={config.text_encoder_lora_parameter_ratio_percent === ratio ? "active" : ""}
                        disabled={isRunning}
                        onClick={() => setConfig({ ...config, text_encoder_lora_parameter_ratio_percent: ratio })}
                      >
                        {ratio}%
                      </button>
                    ))}
                  </div>
                </>
              )}
            </>
          )}
          {config.method === "prompt_tuning" && (
            <>
              <div className="segmented">
                <button
                  type="button"
                  className={config.prompt_tuning_scope === "soft_prompt" ? "active" : ""}
                  disabled={isRunning}
                  onClick={() => setConfig({ ...config, prompt_tuning_scope: "soft_prompt" })}
                >
                  Soft tokens
                </button>
                <button
                  type="button"
                  className={config.prompt_tuning_scope === "soft_prompt_and_prompt" ? "active" : ""}
                  disabled={isRunning}
                  onClick={() => setConfig({ ...config, prompt_tuning_scope: "soft_prompt_and_prompt" })}
                >
                  Tokens + prompt
                </button>
              </div>
              <NumberField
                label="prompt tokens"
                value={config.prompt_tuning_tokens}
                min={1}
                disabled={isRunning}
                onChange={(prompt_tuning_tokens) => setConfig({ ...config, prompt_tuning_tokens })}
              />
              {config.prompt_tuning_scope === "soft_prompt_and_prompt" && (
                <label>
                  <span>prompt part lr</span>
                  <input
                    type="number"
                    value={config.prompt_embedding_learning_rate}
                    step="0.000001"
                    min="0"
                    disabled={isRunning}
                    onChange={(event) =>
                      setConfig({ ...config, prompt_embedding_learning_rate: Number(event.target.value) })
                    }
                  />
                </label>
              )}
            </>
          )}
          <NumberField label="batch" value={config.batch_size} min={1} disabled={isRunning} onChange={(batch_size) => setConfig({ ...config, batch_size })} />
          <NumberField
            label="resolution"
            value={config.resolution}
            min={256}
            disabled={isRunning}
            onChange={(resolution) => setConfig({ ...config, resolution })}
          />
          <NumberField
            label="grad accum"
            value={config.gradient_accumulation_steps}
            min={1}
            disabled={isRunning}
            onChange={(gradient_accumulation_steps) => setConfig({ ...config, gradient_accumulation_steps })}
          />
          <NumberField
            label="max seq len"
            value={config.max_sequence_length}
            min={16}
            disabled={isRunning}
            onChange={(max_sequence_length) => setConfig({ ...config, max_sequence_length })}
          />
          <InitialLearningRateField
            value={config.learning_rate}
            disabled={isRunning}
            onChange={changeInitialLearningRate}
          />
          <label>
            <span>optimizer</span>
          </label>
          <div className="segmented">
            {(["adamw", "adamw8bit"] as Optimizer[]).map((opt) => (
              <button
                key={opt}
                type="button"
                className={config.optimizer === opt ? "active" : ""}
                disabled={isRunning}
                onClick={() => setConfig({ ...config, optimizer: opt })}
              >
                {opt === "adamw" ? "AdamW" : "AdamW 8bit"}
              </button>
            ))}
          </div>
          <label>
            <span>LR scheduler</span>
          </label>
          <div className="segmented">
            {(["constant", "linear", "cosine", "cosine_with_restarts"] as LRScheduler[]).map((sched) => (
              <button
                key={sched}
                type="button"
                className={config.lr_scheduler === sched ? "active" : ""}
                disabled={isRunning}
                onClick={() => setConfig({ ...config, lr_scheduler: sched })}
              >
                {sched === "constant" ? "定数" : sched === "linear" ? "線形" : sched === "cosine" ? "cosine" : "cosine×N"}
              </button>
            ))}
          </div>
          <NumberField
            label="warmup fraction"
            value={config.lr_warmup_fraction}
            min={0}
            max={1}
            step={0.01}
            disabled={isRunning}
            onChange={(lr_warmup_fraction) => setConfig({ ...config, lr_warmup_fraction })}
          />
          {config.lr_scheduler === "cosine_with_restarts" && (
            <NumberField
              label="cycles"
              value={config.lr_num_cycles}
              min={1}
              disabled={isRunning}
              onChange={(lr_num_cycles) => setConfig({ ...config, lr_num_cycles })}
            />
          )}
          {(config.lr_scheduler === "cosine" || config.lr_scheduler === "cosine_with_restarts") && (
            <>
              <NumberField
                label="decay fraction"
                value={config.lr_cosine_decay_fraction}
                min={0.05}
                max={1}
                step={0.05}
                disabled={isRunning}
                onChange={(lr_cosine_decay_fraction) => setConfig({ ...config, lr_cosine_decay_fraction })}
              />
              <NumberField
                label="mid ratio"
                value={config.lr_cosine_mid_ratio}
                min={0}
                max={1}
                step={0.01}
                disabled={isRunning}
                onChange={(lr_cosine_mid_ratio) => setConfig({ ...config, lr_cosine_mid_ratio })}
              />
              <NumberField
                label="final ratio"
                value={config.lr_cosine_final_ratio}
                min={0}
                max={1}
                step={0.01}
                disabled={isRunning}
                onChange={(lr_cosine_final_ratio) => setConfig({ ...config, lr_cosine_final_ratio })}
              />
              {config.lr_cosine_decay_fraction >= 1 && (
                <p className="hint">decay fraction を 1 未満にすると mid/final ratio が適用されます。</p>
              )}
            </>
          )}
          <label>
            <span>encoder LoRA</span>
          </label>
          <div className="segmented">
            <button
              type="button"
              className={!config.train_text_encoder && !config.train_text_encoder_2 ? "active" : ""}
              disabled={isRunning}
              onClick={() => setConfig({ ...config, train_text_encoder: false, train_text_encoder_2: false })}
            >
              Frozen
            </button>
            <button
              type="button"
              className={config.train_text_encoder && !config.train_text_encoder_2 ? "active" : ""}
              disabled={isRunning}
              onClick={() => setConfig({ ...config, train_text_encoder: true, train_text_encoder_2: false })}
            >
              CLIP LoRA
            </button>
            <button
              type="button"
              className={config.train_text_encoder && config.train_text_encoder_2 ? "active" : ""}
              disabled={isRunning}
              onClick={() => setConfig({ ...config, train_text_encoder: true, train_text_encoder_2: true })}
            >
              CLIP + T5 LoRA
            </button>
          </div>
          {(config.train_text_encoder || config.train_text_encoder_2) && (
            <label>
              <span>text encoder lr</span>
              <input
                type="number"
                value={config.text_encoder_learning_rate ?? ""}
                placeholder="(learning rate と同じ)"
                step="0.000001"
                min="0"
                disabled={isRunning}
                onChange={(event) =>
                  setConfig({
                    ...config,
                    text_encoder_learning_rate: event.target.value === "" ? null : Number(event.target.value),
                  })
                }
              />
            </label>
          )}
          <label>
            <span>flip augmentation</span>
            <input
              type="checkbox"
              checked={config.augment}
              disabled={isRunning}
              onChange={(event) => setConfig({ ...config, augment: event.target.checked })}
            />
          </label>
          <label>
            <span>rotation</span>
          </label>
          <div className="segmented">
            {(["none", "cardinal", "full"] as RotationMode[]).map((mode) => (
              <button
                key={mode}
                type="button"
                className={config.rotation_mode === mode ? "active" : ""}
                disabled={isRunning}
                onClick={() => setConfig({ ...config, rotation_mode: mode })}
              >
                {mode === "none" ? "なし" : mode === "cardinal" ? "直角" : "360°"}
              </button>
            ))}
          </div>
          <label>
            <span>tiling</span>
            <input
              type="checkbox"
              checked={config.use_tiling}
              disabled={isRunning}
              onChange={(event) => setConfig({ ...config, use_tiling: event.target.checked })}
            />
          </label>
          <label>
            <span>resolution bucketing</span>
            <input
              type="checkbox"
              checked={config.use_buckets}
              disabled={isRunning}
              onChange={(event) => setConfig({ ...config, use_buckets: event.target.checked })}
            />
          </label>
          <label>
            <span>caption dropout</span>
            <input
              type="number"
              value={config.caption_dropout}
              min={0}
              max={1}
              step={0.05}
              disabled={isRunning}
              onChange={(event) => setConfig({ ...config, caption_dropout: Number(event.target.value) })}
            />
          </label>

          <div className="panelTitle">
            <Gauge size={18} />
            <span>Flow Loss</span>
          </div>
          <NumberField
            label="delta weight"
            value={config.loss_delta_weight}
            min={0}
            max={100}
            step={1}
            disabled={isRunning}
            onChange={(loss_delta_weight) => setConfig({ ...config, loss_delta_weight })}
          />
          <NumberField
            label="delta quantile"
            value={config.loss_delta_quantile}
            min={0.5}
            max={1}
            step={0.01}
            disabled={isRunning}
            onChange={(loss_delta_quantile) => setConfig({ ...config, loss_delta_quantile })}
          />

          <div className="panelTitle">
            <Gauge size={18} />
            <span>Reconstruction Loss</span>
          </div>
          <NumberField
            label="latent weight"
            value={config.reconstruction_latent_loss_weight}
            min={0}
            step={0.01}
            disabled={isRunning}
            onChange={(reconstruction_latent_loss_weight) =>
              setConfig({ ...config, reconstruction_latent_loss_weight })
            }
          />
          <NumberField
            label="pixel weight"
            value={config.reconstruction_pixel_loss_weight}
            min={0}
            step={0.01}
            disabled={isRunning}
            onChange={(reconstruction_pixel_loss_weight) =>
              setConfig({ ...config, reconstruction_pixel_loss_weight })
            }
          />
          <NumberField
            label="pixel crop fraction"
            value={config.reconstruction_pixel_loss_crop_fraction}
            min={0.1}
            max={1}
            step={0.05}
            disabled={isRunning}
            onChange={(reconstruction_pixel_loss_crop_fraction) =>
              setConfig({ ...config, reconstruction_pixel_loss_crop_fraction })
            }
          />
          <label>
            <span>schedule</span>
          </label>
          <div className="segmented scheduleOptions">
            {(["constant", "linear_ramp", "cosine_ramp", "wave", "cosine_ramp_wave"] as ReconstructionLossSchedule[]).map((schedule) => (
              <button
                key={schedule}
                type="button"
                className={config.reconstruction_loss_schedule === schedule ? "active" : ""}
                disabled={isRunning}
                onClick={() => setConfig({ ...config, reconstruction_loss_schedule: schedule })}
              >
                {scheduleLabel(schedule)}
              </button>
            ))}
          </div>
          <NumberField
            label="start fraction"
            value={config.reconstruction_loss_start_fraction}
            min={0}
            max={1}
            step={0.05}
            disabled={isRunning}
            onChange={(reconstruction_loss_start_fraction) =>
              setConfig({ ...config, reconstruction_loss_start_fraction })
            }
          />
          <NumberField
            label="warmup fraction"
            value={config.reconstruction_loss_warmup_fraction}
            min={0}
            max={1}
            step={0.05}
            disabled={isRunning}
            onChange={(reconstruction_loss_warmup_fraction) =>
              setConfig({ ...config, reconstruction_loss_warmup_fraction })
            }
          />
          {config.reconstruction_loss_schedule.includes("wave") && (
            <>
              <NumberField
                label="wave amplitude"
                value={config.reconstruction_loss_wave_amplitude}
                min={0}
                max={1}
                step={0.05}
                disabled={isRunning}
                onChange={(reconstruction_loss_wave_amplitude) =>
                  setConfig({ ...config, reconstruction_loss_wave_amplitude })
                }
              />
              <NumberField
                label="wave cycles"
                value={config.reconstruction_loss_wave_cycles}
                min={0}
                step={0.5}
                disabled={isRunning}
                onChange={(reconstruction_loss_wave_cycles) =>
                  setConfig({ ...config, reconstruction_loss_wave_cycles })
                }
              />
            </>
          )}
          <div className="actions">
            <button className="primary" type="button" onClick={startTraining} disabled={!config.dataset || isRunning}>
              <Play size={18} />
              Start
            </button>
            <button type="button" onClick={cancelTraining} disabled={!isRunning}>
              <PauseCircle size={18} />
              Stop
            </button>
          </div>
          {message && <p className="error">{message}</p>}
          {job?.error && <p className="error">job error: {job.error}</p>}

          <div className="panelTitle">
            <History size={18} />
            <span>Runs</span>
          </div>
          <div className="jobList">
            {jobs.length === 0 && <p className="empty">履歴はまだありません。</p>}
            {jobs.map((item) => (
              <div className={item.id === job?.id ? "jobRow activeJob" : "jobRow"} key={item.id}>
                <button
                  type="button"
                  className="jobItem"
                  onClick={() => void selectJob(item)}
                >
                  <span>
                    <strong>{item.config.dataset || "dataset"}</strong>
                    <small>{methodLabel(item.config.method)} / {shortId(item.id)} / {item.latest_metric?.step ?? 0} steps</small>
                  </span>
                  <i className={`statusPill ${item.status}`}>{item.status}</i>
                </button>
                <button
                  className="deleteJobButton"
                  type="button"
                  title="削除"
                  disabled={!terminalStatuses.has(item.status)}
                  onClick={() => void deleteJob(item)}
                >
                  <Trash2 size={16} />
                </button>
              </div>
            ))}
          </div>
        </aside>

        <section className="panel monitor">
          <div className="statusLine">
            <div>
              <p className="eyebrow">Job</p>
              <h2>{job?.status ?? "idle"}</h2>
            </div>
            <div className="progress">
              <span>{completedPercent}%</span>
              <div><i style={{ width: `${completedPercent}%` }} /></div>
            </div>
          </div>
          <TimeInfo info={timeInfo} />
          <MetricCards latest={latest} />
          <LossChart metrics={metrics} />
          <div className="runtimeControls">
            <LearningRateControl
              value={job?.config.learning_rate ?? config.learning_rate}
              disabled={!isRunning || !job}
              onChange={(learningRate) => void changeRuntimeLearningRate(learningRate)}
            />
          </div>
          <PreviewPanel preview={preview} label="Preview (val)" />
          <PreviewPanel preview={trainPreview} label="Preview (train)" />
          {job && <PreviewHistoryPanel jobId={job.id} source="val" latestStep={job.latest_metric?.step ?? null} />}
          {job && <PreviewHistoryPanel jobId={job.id} source="train" latestStep={job.latest_metric?.step ?? null} />}
          <div className="jobMeta">
            <span>run: {job?.id ?? "-"}</span>
            <span>output: {job?.run_dir ?? "-"}</span>
            <span>model: {job?.config.model_name ?? config.model_name}</span>
            <span>method: {methodLabel(job?.config.method ?? config.method)}</span>
            <span>learning rate: {formatLearningRate(job?.config.learning_rate ?? config.learning_rate)}</span>
          </div>
          {job?.error && <p className="error">{job.error}</p>}
        </section>
      </section>

      <section className="sectionHeader">
        <div>
          <p className="eyebrow">Adapter Evaluation</p>
          <h2>Adapter Loss</h2>
        </div>
      </section>

      <section className="adapterWorkspace">
        <aside className="panel controls">
          <div className="panelTitle">
            <Gauge size={18} />
            <span>Evaluation Setup</span>
          </div>
          <label>
            <span>adapter</span>
            <select
              value={selectedAdapterPath}
              disabled={isAdapterLossRunning}
              onChange={(event) => setSelectedAdapterPath(event.target.value)}
            >
              {adapters.length === 0 && <option value="">models に adapter がありません</option>}
              {adapters.map((adapter) => (
                <option value={adapter.path} key={adapter.path}>
                  {adapter.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>dataset</span>
            <select
              value={adapterLossConfig.dataset}
              disabled={isAdapterLossRunning}
              onChange={(event) => setAdapterLossConfig({ ...adapterLossConfig, dataset: event.target.value })}
            >
              {datasets.length === 0 && <option value="">datasets がありません</option>}
              {datasets.map((dataset) => (
                <option value={dataset.name} key={dataset.name}>
                  {dataset.name}
                </option>
              ))}
            </select>
          </label>
          {selectedAdapterDataset && (
            <dl className="datasetStats">
              <div>
                <dt>train pairs</dt>
                <dd>{selectedAdapterDataset.train_pair_count}</dd>
              </div>
              <div>
                <dt>val pairs</dt>
                <dd>{selectedAdapterDataset.val_pair_count}</dd>
              </div>
            </dl>
          )}
          <label>
            <span>model</span>
            <input
              type="text"
              value={adapterLossConfig.model_name}
              disabled={isAdapterLossRunning}
              onChange={(event) => setAdapterLossConfig({ ...adapterLossConfig, model_name: event.target.value })}
            />
          </label>
          <NumberField
            label="seed"
            value={adapterLossConfig.seed}
            min={0}
            disabled={isAdapterLossRunning}
            onChange={(seed) => setAdapterLossConfig({ ...adapterLossConfig, seed })}
          />
          <label>
            <span>caption (CLIP)</span>
            <textarea
              value={adapterLossConfig.caption}
              disabled={isAdapterLossRunning}
              rows={3}
              onChange={(event) => setAdapterLossConfig({ ...adapterLossConfig, caption: event.target.value })}
            />
          </label>
          <label>
            <span>prompt (T5)</span>
            <textarea
              value={adapterLossConfig.instruction}
              disabled={isAdapterLossRunning}
              rows={4}
              onChange={(event) => setAdapterLossConfig({ ...adapterLossConfig, instruction: event.target.value })}
            />
          </label>
          <label>
            <span>split</span>
          </label>
          <div className="segmented">
            {(["val", "train"] as AdapterLossSplit[]).map((split) => (
              <button
                key={split}
                type="button"
                className={adapterLossSplit === split ? "active" : ""}
                disabled={isAdapterLossRunning}
                onClick={() => setAdapterLossSplit(split)}
              >
                {split}
              </button>
            ))}
          </div>
          <NumberField
            label="batch"
            value={adapterLossConfig.batch_size}
            min={1}
            disabled={isAdapterLossRunning}
            onChange={(batch_size) => setAdapterLossConfig({ ...adapterLossConfig, batch_size })}
          />
          <NumberField
            label="resolution"
            value={adapterLossConfig.resolution}
            min={256}
            disabled={isAdapterLossRunning}
            onChange={(resolution) => setAdapterLossConfig({ ...adapterLossConfig, resolution })}
          />
          <NumberField
            label="max seq len"
            value={adapterLossConfig.max_sequence_length}
            min={16}
            disabled={isAdapterLossRunning}
            onChange={(max_sequence_length) => setAdapterLossConfig({ ...adapterLossConfig, max_sequence_length })}
          />
          <button
            className="primary"
            type="button"
            onClick={() => void startAdapterLossEvaluation()}
            disabled={adapterLossDisabledReason !== null}
            title={adapterLossDisabledReason ?? "Evaluate adapter loss"}
          >
            <Gauge size={18} />
            Evaluate
          </button>
          {adapterLossDisabledReason && <p className="hint">{adapterLossDisabledReason}</p>}
        </aside>

        <section className="panel monitor">
          <AdapterLossPanel job={adapterLossJob} jobs={adapterLossJobs} onSelect={setAdapterLossJob} />
        </section>
      </section>
    </main>
  );
}

function NumberField(props: {
  label: string;
  value: number;
  min: number;
  max?: number;
  step?: number;
  disabled: boolean;
  onChange: (value: number) => void;
}) {
  return (
    <label>
      <span>{props.label}</span>
      <input
        type="number"
        value={props.value}
        min={props.min}
        max={props.max}
        step={props.step}
        disabled={props.disabled}
        onChange={(event) => props.onChange(Number(event.target.value))}
      />
    </label>
  );
}

function InitialLearningRateField(props: { value: number; disabled: boolean; onChange: (value: number) => void }) {
  return (
    <label>
      <span>initial learning rate</span>
      <input
        type="number"
        value={props.value}
        step="0.000001"
        min="0.0000001"
        disabled={props.disabled}
        onChange={(event) => props.onChange(Number(event.target.value))}
      />
    </label>
  );
}

function LearningRateControl(props: { value: number; disabled: boolean; onChange: (value: number) => void }) {
  const externalLog = Math.log10(Math.max(props.value, 1e-7));
  const [localLog, setLocalLog] = useState(externalLog);
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    if (!dragging) {
      setLocalLog(externalLog);
    }
  }, [externalLog, dragging]);

  const commit = (log: number) => {
    setDragging(false);
    props.onChange(Number((10 ** log).toPrecision(6)));
  };

  const ticks = [-7, -6, -5, -4, -3];
  const displayLr = 10 ** localLog;
  return (
    <div className="learningRateControl">
      <div className="learningRateHeader">
        <span>runtime learning rate</span>
        <strong>{formatLearningRate(displayLr)}</strong>
      </div>
      <input
        type="range"
        min="-7"
        max="-3"
        step="0.01"
        value={localLog}
        disabled={props.disabled}
        onChange={(event) => { setDragging(true); setLocalLog(Number(event.target.value)); }}
        onMouseUp={(event) => commit(Number((event.target as HTMLInputElement).value))}
        onTouchEnd={(event) => commit(Number((event.target as HTMLInputElement).value))}
        onKeyUp={(event) => commit(Number((event.target as HTMLInputElement).value))}
      />
      <div className="learningRateScale">
        {ticks.map((tick) => (
          <span key={tick}>{formatLearningRate(10 ** tick)}</span>
        ))}
      </div>
    </div>
  );
}

function MetricCards({ latest }: { latest: MetricEvent | undefined }) {
  return (
    <div className="metrics">
      <div>
        <span>step</span>
        <strong>{latest?.step ?? 0}</strong>
      </div>
      <div>
        <span>epoch</span>
        <strong>{latest?.epoch ?? 0}</strong>
      </div>
      <div>
        <span>learning rate</span>
        <strong>{latest?.learning_rate != null ? latest.learning_rate.toExponential(2) : "-"}</strong>
      </div>
      <div>
        <span>train loss</span>
        <strong>{formatOptionalFixed(latest?.train_loss, 5)}</strong>
      </div>
      <div>
        <span>val loss</span>
        <strong>{formatOptionalFixed(latest?.val_loss, 5)}</strong>
      </div>
      <div>
        <span>flow loss</span>
        <strong>{formatOptionalFixed(latest?.train_flow_loss, 5)}</strong>
      </div>
      <div>
        <span>latent loss</span>
        <strong>{formatOptionalFixed(latest?.train_reconstruction_latent_loss, 5)}</strong>
      </div>
      <div>
        <span>latent weighted</span>
        <strong>{formatOptionalFixed(weightedLatentLoss(latest), 5)}</strong>
      </div>
      <div>
        <span>pixel loss</span>
        <strong>{formatOptionalFixed(latest?.train_reconstruction_pixel_loss, 5)}</strong>
      </div>
      <div>
        <span>pixel weighted</span>
        <strong>{formatOptionalFixed(weightedPixelLoss(latest), 5)}</strong>
      </div>
      <div>
        <span>latent weight</span>
        <strong>{formatOptionalFixed(latest?.reconstruction_latent_weight, 3)}</strong>
      </div>
      <div>
        <span>pixel weight</span>
        <strong>{formatOptionalFixed(latest?.reconstruction_pixel_weight, 3)}</strong>
      </div>
    </div>
  );
}

function LossChart({ metrics }: { metrics: MetricEvent[] }) {
  const chart = useMemo(() => buildPaths(metrics), [metrics]);
  return (
    <div className="chartWrap">
      <svg viewBox="0 0 820 390" role="img" aria-label="loss chart">
        <rect x="0" y="0" width="820" height="390" rx="6" />
        <path className="grid" d="M82 60H790M82 140H790M82 220H790M82 300H790" />
        <path className="axisLine" d="M82 50V310H790" />
        {chart.yTicks.map((tick, index) => (
          <g key={`y-${index}`}>
            <line className="tickLine" x1="78" y1={tick.y} x2="82" y2={tick.y} />
            <text className="axisTick yTick" x="72" y={tick.labelY}>{formatLossTick(tick.value)}</text>
          </g>
        ))}
        {chart.xTicks.map((tick) => (
          <g key={tick.step}>
            <line className="tickLine" x1={tick.x} y1="310" x2={tick.x} y2="315" />
            <text className="axisTick xTick" x={tick.x} y="335">{tick.step}</text>
          </g>
        ))}
        <text className="axisTitle yTitle" x="10" y="184">loss</text>
        <text className="axisTitle xTitle" x="420" y="368">step</text>
        {chart.series.map((series) =>
          series.path ? <path className={`lossLine ${series.className}`} d={series.path} key={series.key} /> : null
        )}
      </svg>
      <div className="legend">
        {chart.series.map((series) => (
          <span key={series.key}><i className={`${series.className}Key`} />{series.label}</span>
        ))}
      </div>
    </div>
  );
}

function AdapterLossPanel(props: {
  job: AdapterLossJob | null;
  jobs: AdapterLossJob[];
  onSelect: (job: AdapterLossJob) => void;
}) {
  return (
    <section className="adapterLossBlock">
      <div className="previewHeader">
        <span>
          <Gauge size={18} />
          Adapter Loss
        </span>
        <small>{props.job ? props.job.status : "not evaluated"}</small>
      </div>
      {props.job ? (
        <>
          <div className="adapterLossResults">
            {props.job.results.length === 0 && (
              <div>
                <span>status</span>
                <strong>{props.job.status}</strong>
              </div>
            )}
            {props.job.results.map((result) => (
              <div key={result.split}>
                <span>{result.split}</span>
                <strong>{result.mean_loss.toFixed(6)}</strong>
                <small>
                  adapter loss
                  {result.base_mean_loss != null ? ` / base ${result.base_mean_loss.toFixed(6)}` : ""}
                  {result.relative_improvement != null ? ` / ${(result.relative_improvement * 100).toFixed(1)}% better` : ""}
                  {result.focused_mean_loss != null ? (
                    <>
                      <br />
                      focused {result.focused_mean_loss.toFixed(6)}
                      {result.focused_base_mean_loss != null ? ` / base ${result.focused_base_mean_loss.toFixed(6)}` : ""}
                      {result.focused_relative_improvement != null
                        ? ` / ${(result.focused_relative_improvement * 100).toFixed(1)}% better`
                        : ""}
                    </>
                  ) : null}
                  {result.changed_token_fraction != null ? (
                    <>
                      <br />
                      changed tokens {(result.changed_token_fraction * 100).toFixed(1)}%
                    </>
                  ) : null}
                </small>
              </div>
            ))}
          </div>
          <p className="previewPrompt">
            {shortAdapterName(props.job.request.adapter_path)} / {props.job.request.split}
            <br />
            output: {props.job.run_dir}
          </p>
          <p className="previewPrompt">
            caption: {props.job.request.config.caption || "-"}
            <br />
            prompt: {props.job.request.config.instruction || "-"}
          </p>
          {props.job.results.map((result) =>
            result.preview ? (
              <div className="previewGrid adapterPreviewGrid" key={`${result.split}-preview`}>
                <PreviewImage title="input" image={result.preview.input_image} />
                <PreviewImage title="output" image={result.preview.prediction_image} />
                <PreviewImage title="label" image={result.preview.target_image} />
              </div>
            ) : null
          )}
          {props.job.error && <p className="error">{props.job.error}</p>}
        </>
      ) : (
        <div className="previewEmpty">adapter loss 評価を開始すると結果が表示されます。</div>
      )}
      {props.jobs.length > 1 && (
        <div className="adapterLossHistory">
          {props.jobs.slice(0, 5).map((job) => (
            <button key={job.id} type="button" onClick={() => props.onSelect(job)}>
              {shortId(job.id)} / {job.status}
            </button>
          ))}
        </div>
      )}
    </section>
  );
}

function PreviewPanel({ preview, label }: { preview: EvaluationPreview | null; label: string }) {
  return (
    <section className="previewBlock">
      <div className="previewHeader">
        <span>
          <ImageIcon size={18} />
          {label}
        </span>
        <small>
          {preview
            ? preview.epoch === 0 && preview.step === 0
              ? "Pre-training"
              : `epoch ${preview.epoch} / step ${preview.step}`
            : "waiting for validation"}
        </small>
      </div>
      {preview ? (
        <>
          <div className="previewGrid">
            <PreviewImage title="input" image={preview.input_image} />
            <PreviewImage title="output" image={preview.prediction_image} />
            <PreviewImage title="label" image={preview.target_image} />
          </div>
          <p className="previewPrompt">{preview.prompt}</p>
        </>
      ) : (
        <div className="previewEmpty">評価が完了すると preview が表示されます。</div>
      )}
    </section>
  );
}

function PreviewImage({ title, image }: { title: string; image: string }) {
  return (
    <figure className="previewItem">
      <img src={image} alt={title} />
      <figcaption>{title}</figcaption>
    </figure>
  );
}

function PreviewHistoryPanel({
  jobId,
  source,
  latestStep,
}: {
  jobId: string;
  source: PreviewSource;
  latestStep: number | null;
}) {
  const [history, setHistory] = useState<PreviewHistoryEntry[]>([]);
  const [selectedStep, setSelectedStep] = useState<number | null>(null);

  useEffect(() => {
    let closed = false;
    fetch(`/api/jobs/${jobId}/previews?source=${source}`)
      .then((response) => response.json())
      .then((data: { previews: PreviewHistoryEntry[] }) => {
        if (closed) return;
        setHistory(data.previews);
        setSelectedStep((current) =>
          current != null && data.previews.some((entry) => entry.step === current)
            ? current
            : data.previews[data.previews.length - 1]?.step ?? null
        );
      })
      .catch(() => {});
    return () => {
      closed = true;
    };
  }, [jobId, source, latestStep]);

  if (history.length === 0) {
    return null;
  }

  const selected = history.find((entry) => entry.step === selectedStep) ?? history[history.length - 1];

  return (
    <section className="previewBlock">
      <div className="previewHeader">
        <span>
          <ImageIcon size={18} />
          Preview history ({source})
        </span>
        <select value={selected.step} onChange={(event) => setSelectedStep(Number(event.target.value))}>
          {history.map((entry) => (
            <option key={entry.step} value={entry.step}>
              {entry.epoch === 0 && entry.step === 0 ? "Pre-training" : `epoch ${entry.epoch} / step ${entry.step}`}
            </option>
          ))}
        </select>
      </div>
      <div className="previewGrid">
        <PreviewImage title="input" image={`/api/jobs/${jobId}/previews/${source}/${selected.step}/input`} />
        <PreviewImage title="output" image={`/api/jobs/${jobId}/previews/${source}/${selected.step}/prediction`} />
        <PreviewImage title="label" image={`/api/jobs/${jobId}/previews/${source}/${selected.step}/target`} />
      </div>
      <p className="previewPrompt">{selected.prompt}</p>
    </section>
  );
}

function buildPaths(metrics: MetricEvent[]) {
  const emptySeries = lossSeriesDefinitions.map((definition) => ({ ...definition, path: "" }));
  if (metrics.length === 0) {
    return { series: emptySeries, yTicks: [], xTicks: [] };
  }
  const values = lossSeriesDefinitions.flatMap((definition) =>
    metrics.flatMap((metric) => {
      const value = definition.value(metric);
      return isFiniteNumber(value) ? [value] : [];
    })
  );
  if (values.length === 0) {
    return { series: emptySeries, yTicks: [], xTicks: [] };
  }
  const max = Math.max(...values);
  const min = Math.min(...values);
  const x = (index: number) => 82 + (index / Math.max(metrics.length - 1, 1)) * 708;
  const y = (value: number) => 310 - ((value - min) / Math.max(max - min, 0.0001)) * 250;
  const toPath = (points: Array<[number, number]>) =>
    points.map(([px, py], index) => `${index === 0 ? "M" : "L"}${px.toFixed(2)} ${py.toFixed(2)}`).join(" ");
  const yTicks = [0, 0.25, 0.5, 0.75, 1].map((ratio) => {
    const value = min + (max - min) * ratio;
    const tickY = y(value);
    const labelY = Math.min(300, Math.max(56, tickY + 4));
    return { value, y: tickY, labelY };
  }).reverse();
  const xTickIndexes = uniqueIndexes([
    0,
    Math.floor((metrics.length - 1) / 2),
    metrics.length - 1
  ]);
  return {
    series: lossSeriesDefinitions.map((definition) => ({
      ...definition,
      path: toPath(
        metrics.flatMap((metric, index) => {
          const value = definition.value(metric);
          return isFiniteNumber(value) ? [[x(index), y(value)] as [number, number]] : [];
        })
      ),
    })),
    yTicks,
    xTicks: xTickIndexes.map((index) => ({ step: metrics[index].step, x: x(index) }))
  };
}

const lossSeriesDefinitions = [
  {
    key: "total",
    label: "total loss",
    className: "totalLine",
    value: (metric: MetricEvent) => metric.train_loss,
  },
  {
    key: "flow",
    label: "flow loss",
    className: "flowLine",
    value: (metric: MetricEvent) => metric.train_flow_loss,
  },
  {
    key: "latent",
    label: "latent loss",
    className: "latentLine",
    value: (metric: MetricEvent) => metric.train_reconstruction_latent_loss,
  },
  {
    key: "pixel",
    label: "pixel loss",
    className: "pixelLine",
    value: (metric: MetricEvent) => metric.train_reconstruction_pixel_loss,
  },
  {
    key: "val",
    label: "val loss",
    className: "valLine",
    value: (metric: MetricEvent) => metric.val_loss,
  },
];

function isFiniteNumber(value: number | null | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function weightedLatentLoss(metric: MetricEvent | undefined) {
  if (!isFiniteNumber(metric?.train_reconstruction_latent_loss) || !isFiniteNumber(metric?.reconstruction_latent_weight)) {
    return null;
  }
  return metric.train_reconstruction_latent_loss * metric.reconstruction_latent_weight;
}

function weightedPixelLoss(metric: MetricEvent | undefined) {
  if (!isFiniteNumber(metric?.train_reconstruction_pixel_loss) || !isFiniteNumber(metric?.reconstruction_pixel_weight)) {
    return null;
  }
  return metric.train_reconstruction_pixel_loss * metric.reconstruction_pixel_weight;
}

function uniqueIndexes(indexes: number[]) {
  return [...new Set(indexes.filter((index) => index >= 0))];
}

function formatLossTick(value: number) {
  if (Math.abs(value) >= 100) {
    return value.toFixed(0);
  }
  if (Math.abs(value) >= 1) {
    return value.toFixed(2);
  }
  return value.toFixed(4);
}

function mergeMetric(metrics: MetricEvent[], metric: MetricEvent) {
  const merged = metrics.some((item) => item.step === metric.step)
    ? metrics.map((item) => (item.step === metric.step ? metric : item))
    : [...metrics, metric];
  return merged.sort((left, right) => left.step - right.step);
}

function upsertJob(jobs: TrainingJob[], job: TrainingJob) {
  const normalizedJob = normalizeJob(job);
  const withoutJob = jobs.filter((item) => item.id !== normalizedJob.id);
  return [normalizedJob, ...withoutJob].sort((left, right) => right.created_at.localeCompare(left.created_at));
}

function upsertAdapterLossJob(jobs: AdapterLossJob[], job: AdapterLossJob) {
  const withoutJob = jobs.filter((item) => item.id !== job.id);
  return [job, ...withoutJob].sort((left, right) => right.created_at.localeCompare(left.created_at));
}

function getAdapterLossDisabledReason(
  adapterPath: string,
  config: TrainingConfig,
  running: boolean
) {
  if (running) {
    return "adapter loss 評価を実行中です。";
  }
  if (!adapterPath) {
    return "adapter を選択してください。";
  }
  if (!config.dataset) {
    return "dataset を選択してください。";
  }
  if (!config.model_name.trim()) {
    return "model を入力してください。";
  }
  return null;
}

function shortId(id: string) {
  return id.slice(0, 8);
}

function shortAdapterName(path: string) {
  return path.split("/").at(-1) ?? path;
}

function methodLabel(method: TrainingMethod) {
  if (method === "prompt_tuning") return "Prompt-tuning";
  if (method === "full") return "Full";
  return "LoRA";
}

function scheduleLabel(schedule: ReconstructionLossSchedule) {
  switch (schedule) {
    case "constant":
      return "一定";
    case "linear_ramp":
      return "linear";
    case "cosine_ramp":
      return "cosine";
    case "wave":
      return "wave";
    case "cosine_ramp_wave":
      return "cos wave";
  }
}

function formatLearningRate(value: number) {
  return value.toExponential(2);
}

function formatOptionalFixed(value: number | null | undefined, digits: number) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "-";
  }
  return value.toFixed(digits);
}

async function responseDetail(response: Response) {
  const fallback = `${response.status} ${response.statusText}`.trim();
  try {
    const data = await response.json();
    if (typeof data?.detail === "string") {
      return data.detail;
    }
    return JSON.stringify(data);
  } catch {
    try {
      return (await response.text()) || fallback;
    } catch {
      return fallback;
    }
  }
}

function errorMessage(error: unknown, fallback: string) {
  if (error instanceof Error) {
    return `${fallback} ${error.message}`;
  }
  if (typeof error === "string") {
    return `${fallback} ${error}`;
  }
  return fallback;
}

function normalizeJob(job: TrainingJob): TrainingJob {
  return { ...job, config: normalizeConfig(job.config) };
}

function normalizeConfig(config: TrainingConfig) {
  const legacyConfig = config as TrainingConfig & { max_steps?: number; prompt?: string; inference_prompt?: string };
  return {
    ...defaultConfig,
    ...config,
    epochs: config.epochs ?? legacyConfig.max_steps ?? defaultConfig.epochs,
    caption: config.caption ?? legacyConfig.prompt ?? "",
    instruction: config.instruction ?? "",
    inference_instruction: config.inference_instruction ?? legacyConfig.inference_prompt ?? "",
    transformer_lora_parameter_ratio_percent:
      config.transformer_lora_parameter_ratio_percent ?? defaultConfig.transformer_lora_parameter_ratio_percent,
    text_encoder_lora_parameter_ratio_percent:
      config.text_encoder_lora_parameter_ratio_percent ?? defaultConfig.text_encoder_lora_parameter_ratio_percent,
    reconstruction_pixel_loss_crop_fraction:
      config.reconstruction_pixel_loss_crop_fraction > 0
        ? config.reconstruction_pixel_loss_crop_fraction
        : defaultConfig.reconstruction_pixel_loss_crop_fraction,
    rotation_mode: config.rotation_mode ?? defaultConfig.rotation_mode,
  };
}

function getTotalSteps(config: TrainingConfig, dataset: DatasetSummary | undefined) {
  const trainCount = dataset?.train_pair_count ?? dataset?.train_count ?? 0;
  const batchesPerEpoch = Math.max(1, Math.ceil(trainCount / Math.max(config.batch_size, 1)));
  return config.epochs * batchesPerEpoch;
}

type TimeInfoData = {
  elapsed: string;
  eta: string | null;
};

function useTimeInfo(job: TrainingJob | null, metrics: MetricEvent[], totalSteps: number): TimeInfoData | null {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

  if (!job || !["running", "queued"].includes(job.status)) return null;

  const startMs = new Date(job.created_at).getTime();
  const elapsedMs = now - startMs;
  const elapsed = formatDuration(elapsedMs);

  const first = metrics[0];
  const latest = metrics.at(-1);
  let eta: string | null = null;
  if (first && latest && latest.step > 0 && totalSteps > 0) {
    const firstMs = new Date(job.created_at).getTime();
    const msPerStep = elapsedMs / latest.step;
    const remainingSteps = Math.max(0, totalSteps - latest.step);
    const remainingMs = remainingSteps * msPerStep;
    eta = formatDuration(remainingMs);
  }

  return { elapsed, eta };
}

function formatDuration(ms: number): string {
  const totalSec = Math.floor(ms / 1000);
  const h = Math.floor(totalSec / 3600);
  const m = Math.floor((totalSec % 3600) / 60);
  const s = totalSec % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

function TimeInfo({ info }: { info: TimeInfoData | null }) {
  if (!info) return null;
  return (
    <div className="timeInfo">
      <span>経過: <strong>{info.elapsed}</strong></span>
      {info.eta !== null && <span>残り: <strong>{info.eta}</strong></span>}
    </div>
  );
}
