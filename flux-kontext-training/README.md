# Flux Kontext Training

ローカルで Flux Kontext の LoRA / Prompt-tuning 学習ジョブを起動し、TypeScript UI から loss を監視するためのアプリです。

## 構成

- `src/flux_kontext_training`: FastAPI バックエンド
- `ui`: React / TypeScript / Vite UI
- `datasets/<name>/train`: 学習用データ。`*_start.png` と `*_end.png` のペアを配置します。
- `datasets/<name>/val`: 検証用データ。`*_start.png` と `*_end.png` のペアを配置します。
- `runs/<job_id>`: 学習設定と metrics の出力先

画像ごとの prompt は start 画像と同じベース名の `.txt` caption を優先します。たとえば
`datasets/sample/train/0001_start.png` に対して `datasets/sample/train/0001_start.txt`
があれば、その内容を prompt として使います。caption が無いペアは UI の `common prompt`
を使います。

## 環境設定

このリポジトリには、ローカル生成物や学習データ、学習済み重みは含めません。clone 後に必要なものを各自の環境へ配置して復元します。

Git に含めないもの:

- `datasets/`: 学習・検証画像
- `runs/`: 学習ログ、metrics、出力 checkpoint
- `models/`: 外部 adapter や fal.ai の設定 JSON
- `data/`: ローカル作業データ
- `.env`, `.env.*`: token や環境変数
- `*.safetensors`, 画像ファイル、UI/Python の生成物

### 1. リポジトリを clone

```bash
git clone <repository-url>
cd flux-kontext-training
```

### 2. Python 環境を作成

Python 3.10 以上を使用します。CUDA で学習する場合は、環境に合った PyTorch が入るようにしてください。

```bash
python3 -m venv .venv312
. .venv312/bin/activate
pip install -U pip
pip install -e .
```

GPU 環境で `torch` が CUDA を認識しない場合は、PyTorch の CUDA 対応 wheel を入れ直してください。

確認:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
PY
```

### 3. Hugging Face 認証

`black-forest-labs/FLUX.1-Kontext-dev` は gated model です。Hugging Face のモデルページで利用申請を承認したあと、同じユーザーで認証します。

```bash
python -c "from huggingface_hub.cli.hf import main; main()" auth login
```

または、API を起動するターミナルで環境変数に設定します。

```bash
export HF_TOKEN="hf_..."
```

`HUGGING_FACE_HUB_TOKEN` でも指定できます。

```bash
export HUGGING_FACE_HUB_TOKEN="hf_..."
```

バックエンドは起動時に `HF_TOKEN` を優先し、無ければ `HUGGING_FACE_HUB_TOKEN` を使います。

```python
os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
```

`.env` に保存する場合も、`.env` は Git に含めないでください。

### 4. 推論時プロンプトを指定

Hugging Face の認証トークンとは別に、Kontext の推論に使う text prompt は UI の `inference instruction` で指定します。

- `caption (CLIP)`: CLIP 側に渡す短い caption
- `instruction (T5)`: 学習時に使う T5 instruction
- `inference instruction`: preview / 推論時だけ上書きして使う T5 instruction

`inference instruction` が空の場合は、学習時の `instruction` が使われます。

内部では、文字列の prompt を `pipe.encode_prompt()` で `prompt_embeds` / `pooled_prompt_embeds` に変換し、`FluxKontextPipeline` の推論に渡します。

### 5. UI 依存関係を入れる

```bash
cd ui
npm install
cd ..
```

### 6. データセットを配置

`datasets/<dataset-name>/train` と `datasets/<dataset-name>/val` に、start/end の画像ペアを配置します。

```text
datasets/<dataset-name>/train/0001_start.png
datasets/<dataset-name>/train/0001_end.png
datasets/<dataset-name>/val/0001_start.png
datasets/<dataset-name>/val/0001_end.png
```

caption を画像ごとに指定する場合は、start 画像と同じベース名の `.txt` を置きます。

```text
datasets/<dataset-name>/train/0001_start.txt
```

### 7. 任意の外部 adapter / checkpoint を復元

fal.ai などで作成した adapter を評価・継続学習に使う場合は、Git ではなくローカルの `models/` などに配置します。

```text
models/<adapter-name>.safetensors
models/<config-name>.json
```

`models/` は Git 管理外です。公開したくない学習データ URL や重みを誤って push しないようにしてください。

## 起動

環境設定が済んだら、バックエンドと UI を別ターミナルで起動します。

バックエンド:

```bash
. .venv312/bin/activate
python main.py
```

UI:

```bash
cd ui
npm run dev
```

ブラウザで `http://127.0.0.1:5173` を開きます。

## 学習

LoRA、Prompt-tuning、Full を選択できます。学習は `FluxKontextPipeline` を読み込み、start 画像を条件画像、end 画像を教師画像として VAE latent 上の flow-matching loss を最適化します。1 epoch は train split の全ペアを1回ずつ学習した単位で、検証は各 epoch の最後に val split 全ペアで実行します。LoRA は対象 Linear 層の総パラメータ数に対する adapter パラメータ比率から rank を自動計算します。Full は transformer 全体を学習する実験用モードで、起動時に推定VRAMを確認し、危険な場合は学習開始前に停止します。

LoRA で `CLIP LoRA` / `CLIP + T5 LoRA` を有効にした場合、text encoder 本体は凍結し、encoder 内の Linear 層に LoRA adapter を追加して学習します。T5 encoder 本体を full-parameter で学習する場合と比べ、48GB VRAM でも試しやすい設定です。

学習済み重みは `runs/<job_id>/weights` に保存されます。LoRA は diffusers の LoRA 形式、Prompt-tuning は `prompt_tuning.pt`、Full は transformer の `save_pretrained` 形式として保存します。
