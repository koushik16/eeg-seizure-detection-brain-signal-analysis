from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def _max_consecutive_positives(binary_preds: np.ndarray) -> int:
    max_run = 0
    current_run = 0

    for value in binary_preds:
        if value == 1:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0

    return int(max_run)


def summarize_predictions(
    window_probs: np.ndarray,
    window_ranges: List[Tuple[int, int]],
    threshold: float,
    sfreq: float,
    min_consecutive_positive_windows: int = 3,
) -> Dict[str, Any]:
    if window_probs.ndim != 1:
        raise ValueError(f"Expected 1D probability array, got {window_probs.shape}")

    if len(window_probs) == 0:
        raise ValueError("Empty probability array provided.")

    binary_preds = (window_probs >= threshold).astype(int)

    positive_indices = np.where(binary_preds == 1)[0].tolist()
    max_prob = float(np.max(window_probs))
    mean_prob = float(np.mean(window_probs))
    num_positive = int(binary_preds.sum())
    max_consecutive = _max_consecutive_positives(binary_preds)

    positive_windows = []
    for idx in positive_indices:
        start_sample, end_sample = window_ranges[idx]
        positive_windows.append(
            {
                "window_index": int(idx),
                "start_sec": float(start_sample / sfreq),
                "end_sec": float(end_sample / sfreq),
                "probability": float(window_probs[idx]),
            }
        )

    file_pred = "YES" if max_consecutive >= min_consecutive_positive_windows else "NO"

    summary = {
        "prediction": file_pred,
        "seizure_probability": max_prob,
        "mean_window_probability": mean_prob,
        "threshold": float(threshold),
        "num_windows": int(len(window_probs)),
        "num_positive_windows": num_positive,
        "max_consecutive_positive_windows": int(max_consecutive),
        "min_consecutive_positive_windows": int(min_consecutive_positive_windows),
        "positive_windows": positive_windows,
    }

    return summary