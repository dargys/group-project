"""Shared image preprocessing for the classification baseline."""

import numpy as np
from PIL import Image


def load_image(image_path, image_width=64, image_height=48):
    """Return a resized RGB image as a float32 array on a 0-1 scale."""
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        resized_image = rgb_image.resize(
            (image_width, image_height),
            Image.Resampling.LANCZOS
        )

    pixel_values = np.asarray(resized_image)
    normalized_pixels = pixel_values.astype("float32") / 255.0

    return normalized_pixels
