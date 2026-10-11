import io
from PIL import Image
import numpy as np
import tensorflow as tf
from typing import Tuple

# Ensure consistent image dimensions with config
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

def validate_and_load_image(image_bytes: bytes) -> Image.Image:
    """
    Validate that raw bytes represent a readable, supported image.
    Converts image to RGB mode.
    """
    if not image_bytes or len(image_bytes) == 0:
        raise ValueError("Empty image data received.")

    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img.verify()
    except Exception as e:
        raise ValueError(f"Corrupt or unsupported image format: {str(e)}")

    # Re-open after verify() to load pixel data
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return pil_img

def preprocess_image_for_model(pil_img: Image.Image) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Resizes image to 224x224 and applies MobileNetV2 preprocessing.
    Returns:
        (preprocessed_batch, (orig_width, orig_height))
    """
    orig_dims = pil_img.size # (width, height)
    resized_img = pil_img.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    img_array = np.array(resized_img, dtype=np.float32)

    # Expand batch dimension: (1, 224, 224, 3)
    batch_array = np.expand_dims(img_array, axis=0)

    # MobileNetV2 preprocess_input scales values from [0, 255] to [-1, 1]
    preprocessed_batch = tf.keras.applications.mobilenet_v2.preprocess_input(batch_array)
    return preprocessed_batch, orig_dims

def preprocess_image_from_path(filepath: str) -> Tuple[np.ndarray, Image.Image]:
    """Load image from filesystem path and preprocess for inference."""
    with open(filepath, "rb") as f:
        bytes_data = f.read()
    pil_img = validate_and_load_image(bytes_data)
    preprocessed_batch, _ = preprocess_image_for_model(pil_img)
    return preprocessed_batch, pil_img
