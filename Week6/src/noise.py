"""
Noise Generation Module
"""

import numpy as np
from .config import NOISE_FACTOR


def add_gaussian_noise(images):
    """
    Adds Gaussian noise to images.
    """

    noisy_images = images + NOISE_FACTOR * np.random.normal(
        loc=0.0,
        scale=1.0,
        size=images.shape
    )

    noisy_images = np.clip(noisy_images, 0.0, 1.0)

    return noisy_images