from __future__ import annotations

from typing import List, Tuple

import numpy as np

from src.config import STEP_SAMPLES, WINDOW_SAMPLES


def create_windows(
    data: np.ndarray,
    window_samples: int = WINDOW_SAMPLES,
    step_samples: int = STEP_SAMPLES,
) -> tuple[np.ndarray, List[Tuple[float, float]]]:
    """
    Split EEG into fixed windows.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_channels, n_samples)

    Returns
    -------
    windows : np.ndarray
        Shape (n_windows, n_channels, window_samples)
    window_times : list[tuple[float, float]]
        Placeholder timing list in samples converted later if needed
    """
    n_channels, n_samples = data.shape

    if n_samples < window_samples:
        raise ValueError(
            f"Signal too short for one full window. "
            f"Needed {window_samples} samples, got {n_samples}."
        )

    windows = []
    window_ranges = []

    for start in range(0, n_samples - window_samples + 1, step_samples):
        end = start + window_samples
        windows.append(data[:, start:end])
        window_ranges.append((start, end))

    windows = np.stack(windows).astype(np.float32)
    return windows, window_ranges