from __future__ import annotations

import math
import random

from PIL import Image, ImageOps


RESOLUTION_BUCKETS: list[tuple[int, int]] = [
    (1024, 1024),
    (1152, 896),
    (896, 1152),
    (1216, 832),
    (832, 1216),
    (1344, 768),
    (768, 1344),
    (1536, 640),
    (640, 1536),
]


def nearest_bucket(image_size: tuple[int, int], buckets: list[tuple[int, int]]) -> tuple[int, int]:
    """Return the bucket whose aspect ratio is closest to image_size."""
    w, h = image_size
    if h == 0 or w == 0:
        return buckets[0]
    log_ar = math.log(w / h)
    return min(buckets, key=lambda b: abs(math.log(b[0] / b[1]) - log_ar))


def load_rgba_as_rgb(path: str) -> Image.Image:
    image = Image.open(path)
    try:
        if image.mode == "RGBA":
            bg = Image.new("RGB", image.size, (255, 255, 255))
            bg.paste(image, mask=image.split()[3])
            return bg
        return image.convert("RGB")
    finally:
        image.close()


def compute_random_tile_box(
    image_size: tuple[int, int], tile_size: int, rng: random.Random
) -> tuple[int, int, int, int] | None:
    """Return a randomly positioned tile_size×tile_size crop box.

    Returns None when the image is smaller than tile_size on both axes.
    When only one axis is smaller, the full extent of that axis is used.
    """
    w, h = image_size
    if w <= tile_size and h <= tile_size:
        return None
    x = rng.randint(0, max(0, w - tile_size))
    y = rng.randint(0, max(0, h - tile_size))
    return (x, y, min(x + tile_size, w), min(y + tile_size, h))


class FloorPlanAugmenter:
    """Apply synchronized geometric augmentations to paired floor plan images.

    Both images in each pair receive the exact same transformation so that
    spatial correspondence between input and target is preserved.

    When tile_size is set, each call randomly uses either a random crop of
    that size (50% probability) or the full image (50% probability).
    No color or brightness modifications are applied.
    """

    def __init__(self, flip: bool = True, rotation_mode: str = "cardinal", tile_size: int | None = None) -> None:
        self.flip = flip
        self.rotation_mode = rotation_mode  # "none" | "cardinal" | "full"
        self.tile_size = tile_size

    def transform_pair(
        self,
        input_img: Image.Image,
        target_img: Image.Image,
        crop_box: tuple[int, int, int, int] | None,
        rng: random.Random,
    ) -> tuple[Image.Image, Image.Image]:
        if crop_box is not None:
            input_img = input_img.crop(crop_box)
            target_img = target_img.crop(crop_box)
        elif self.tile_size is not None and rng.random() < 0.5:
            box = compute_random_tile_box(input_img.size, self.tile_size, rng)
            if box is not None:
                input_img = input_img.crop(box)
                target_img = target_img.crop(box)

        if self.flip and rng.random() < 0.5:
            input_img = ImageOps.mirror(input_img)
            target_img = ImageOps.mirror(target_img)

        if self.rotation_mode == "full":
            angle = rng.uniform(0.0, 360.0)
            input_img = input_img.rotate(angle, resample=Image.Resampling.NEAREST, expand=False, fillcolor=(255, 255, 255))
            target_img = target_img.rotate(angle, resample=Image.Resampling.NEAREST, expand=False, fillcolor=(255, 255, 255))
        elif self.rotation_mode == "cardinal":
            angle = rng.choice([0, 90, 180, 270])
            if angle != 0:
                input_img = input_img.rotate(angle, resample=Image.Resampling.NEAREST, expand=False, fillcolor=(255, 255, 255))
                target_img = target_img.rotate(angle, resample=Image.Resampling.NEAREST, expand=False, fillcolor=(255, 255, 255))

        return input_img, target_img
