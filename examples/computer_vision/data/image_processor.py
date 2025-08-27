"""Advanced image preprocessing and augmentation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class ImageProcessor:
    """Performs basic preprocessing and augmentation steps.

    The implementation intentionally avoids heavyweight image libraries to
    keep the example lightweight.  Images are assumed to be NumPy arrays in
    HWC format with values in the 0-255 range.
    """

    size: tuple[int, int] = (224, 224)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Resize (via simple cropping) and normalise the image."""
        h, w = image.shape[:2]
        cropped = image[: self.size[0], : self.size[1]]
        return cropped.astype("float32") / 255.0

    def augment(self, image: np.ndarray) -> np.ndarray:
        """Apply random horizontal flip and Gaussian noise."""
        aug = np.copy(image)
        if np.random.rand() > 0.5:
            aug = np.fliplr(aug)
        noise = np.random.normal(0, 0.01, aug.shape)
        aug = np.clip(aug + noise * 255, 0, 255)
        return aug.astype(image.dtype)

    def batch(self, images: Iterable[np.ndarray]) -> np.ndarray:
        """Preprocess a collection of images into a single batch tensor."""
        return np.stack([self.preprocess(img) for img in images])
