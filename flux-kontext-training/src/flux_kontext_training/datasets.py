from pathlib import Path

from .paths import DATASETS_DIR
from .schemas import DatasetSummary, TrainingExample, TrainingPair


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def _image_paths(directory: Path) -> list[Path]:
    if not directory.exists() or not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _count_images(directory: Path) -> int:
    return len(_image_paths(directory))


def _pair_key(image_path: Path) -> str | None:
    stem = image_path.stem
    if stem.endswith("_start"):
        return stem[: -len("_start")]
    if stem.endswith("_end"):
        return stem[: -len("_end")]
    return None


def _paired_image_paths(directory: Path) -> list[tuple[Path, Path]]:
    starts: dict[str, Path] = {}
    ends: dict[str, Path] = {}
    for image_path in _image_paths(directory):
        key = _pair_key(image_path)
        if key is None:
            continue
        if image_path.stem.endswith("_start"):
            starts[key] = image_path
        elif image_path.stem.endswith("_end"):
            ends[key] = image_path
    return [(starts[key], ends[key]) for key in sorted(starts.keys() & ends.keys())]


def _count_pairs(directory: Path) -> int:
    return len(_paired_image_paths(directory))


def _caption_path(image_path: Path) -> Path:
    return image_path.with_suffix(".txt")


def _read_caption(image_path: Path) -> tuple[str, Path] | None:
    caption_paths = [_caption_path(image_path)]
    pair_key = _pair_key(image_path)
    if pair_key is not None:
        caption_paths.append(image_path.with_name(f"{pair_key}.txt"))

    for caption_path in caption_paths:
        if not caption_path.is_file():
            continue
        caption = caption_path.read_text(encoding="utf-8").strip()
        if caption:
            return caption, caption_path
    return None


def _count_captions(directory: Path) -> int:
    return sum(1 for image_path in _image_paths(directory) if _read_caption(image_path) is not None)


def list_datasets() -> list[DatasetSummary]:
    DATASETS_DIR.mkdir(exist_ok=True)
    datasets: list[DatasetSummary] = []
    for path in sorted(DATASETS_DIR.iterdir()):
        if not path.is_dir():
            continue
        train_dir = path / "train"
        val_dir = path / "val"
        train_count = _count_images(train_dir)
        val_count = _count_images(val_dir)
        train_pair_count = _count_pairs(train_dir)
        val_pair_count = _count_pairs(val_dir)
        train_caption_count = _count_captions(train_dir)
        val_caption_count = _count_captions(val_dir)
        datasets.append(
            DatasetSummary(
                name=path.name,
                path=str(path),
                train_count=train_count,
                val_count=val_count,
                train_pair_count=train_pair_count,
                val_pair_count=val_pair_count,
                train_caption_count=train_caption_count,
                val_caption_count=val_caption_count,
                train_missing_caption_count=train_count - train_caption_count,
                val_missing_caption_count=val_count - val_caption_count,
                has_train=train_dir.is_dir(),
                has_val=val_dir.is_dir(),
            )
        )
    return datasets


def resolve_dataset(name: str) -> Path:
    dataset = (DATASETS_DIR / name).resolve()
    datasets_root = DATASETS_DIR.resolve()
    if datasets_root not in dataset.parents:
        raise ValueError("Invalid dataset path.")
    if not dataset.is_dir():
        raise ValueError(f"Dataset '{name}' does not exist.")
    if not (dataset / "train").is_dir():
        raise ValueError(f"Dataset '{name}' must contain a train directory.")
    if not (dataset / "val").is_dir():
        raise ValueError(f"Dataset '{name}' must contain a val directory.")
    return dataset


def resolve_caption_for_image(image_path: Path, fallback_caption: str) -> TrainingExample:
    caption_result = _read_caption(image_path)
    if caption_result is not None:
        caption, caption_path = caption_result
        return TrainingExample(
            image_path=str(image_path),
            caption=caption,
            caption_path=str(caption_path),
            uses_caption=True,
        )

    caption = fallback_caption.strip()
    if not caption:
        raise ValueError(
            f"Image '{image_path.name}' does not have a .txt caption and no common caption was provided."
        )
    return TrainingExample(
        image_path=str(image_path),
        caption=caption,
        caption_path=None,
        uses_caption=False,
    )


def list_training_examples(dataset: Path, split: str, fallback_caption: str) -> list[TrainingExample]:
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'.")
    return [resolve_caption_for_image(image_path, fallback_caption) for image_path in _image_paths(dataset / split)]


def list_training_pairs(
    dataset: Path,
    split: str,
    fallback_caption: str,
    instruction: str,
) -> list[TrainingPair]:
    if split not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'.")

    pairs: list[TrainingPair] = []
    for input_image_path, target_image_path in _paired_image_paths(dataset / split):
        example = resolve_caption_for_image(input_image_path, fallback_caption)
        pairs.append(
            TrainingPair(
                input_image_path=str(input_image_path),
                target_image_path=str(target_image_path),
                caption=example.caption,
                instruction=instruction,
                caption_path=example.caption_path,
                uses_caption=example.uses_caption,
            )
        )
    return pairs
