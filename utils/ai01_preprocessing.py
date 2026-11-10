"""Validate the shared 11-Z preprocessing contract before strict SSL transfer."""

import numpy as np

from .ai01_data import DEFAULT_PREPROCESSING


def validate_preprocessing(saved, image_processor=None):
    expected = dict(DEFAULT_PREPROCESSING)
    # Crop size may differ; Z order and intensity normalization must match.
    image_size = saved.get("image_size")
    if not isinstance(image_size, int) or image_size < 32 or image_size % 32:
        raise ValueError("Preprocessing image_size must be a positive integer multiple of 32")
    del expected["image_size"]
    if image_processor is not None:
        if image_processor.do_resize:
            raise ValueError("AI01 processor must have do_resize=False (dataset performs center crop)")
        if not image_processor.do_rescale or not image_processor.do_normalize:
            raise ValueError("AI01 processor must rescale and normalize")
        expected.update({
            "rescale_factor": float(image_processor.rescale_factor),
            "mean": float(np.mean(image_processor.image_mean)),
            "std": float(np.mean(image_processor.image_std)),
        })
    for key, value in expected.items():
        actual = saved.get(key)
        if isinstance(value, float):
            matches = actual is not None and np.isclose(actual, value, rtol=1e-6, atol=1e-8)
        else:
            matches = actual == value
        if not matches:
            raise ValueError(f"Preprocessing mismatch for {key}: SSL={actual}, AI01={value}")
