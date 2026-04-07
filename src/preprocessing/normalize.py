from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import TRAIN_MEAN_PATH, TRAIN_STD_PATH


def load_normalization_stats(
    mean_path: str | Path = TRAIN_MEAN_PATH,
    std_path: str | Path = TRAIN_STD_PATH,
) -> tuple[np.ndarray, np.ndarray]:
    mean_path = Path(mean_path)
    std_path = Path(std_path)

    if not mean_path.exists():
        raise FileNotFoundError(f"Train mean file not found: {mean_path}")

    if not std_path.exists():
        raise FileNotFoundError(f"Train std file not found: {std_path}")

    mean = np.load(mean_path).astype(np.float32)
    std = np.load(std_path).astype(np.float32)

    if mean.ndim != 1 or std.ndim != 1:
        raise ValueError("Normalization stats must be 1D arrays.")

    if mean.shape != std.shape:
        raise ValueError(
            f"Mean/std shape mismatch: {mean.shape} vs {std.shape}"
        )

    if np.any(std <= 0):
        raise ValueError("Standard deviation contains non-positive values.")

    return mean, std


def apply_channelwise_zscore(
    windows: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """
    windows: (n_windows, n_channels, n_timepoints)
    mean/std: (n_channels,)
    """
    if windows.ndim != 3:
        raise ValueError(f"Expected 3D windows array, got shape {windows.shape}")

    if windows.shape[1] != mean.shape[0]:
        raise ValueError(
            f"Channel mismatch between windows and normalization stats: "
            f"{windows.shape[1]} vs {mean.shape[0]}"
        )

    normalized = (windows - mean[None, :, None]) / std[None, :, None]
    normalized = normalized.astype(np.float32)

    if not np.all(np.isfinite(normalized)):
        raise ValueError("Non-finite values found after normalization.")

    return normalized