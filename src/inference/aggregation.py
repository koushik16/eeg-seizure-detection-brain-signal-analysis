from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np


def _find_positive_runs(binary_preds: np.ndarray) -> List[Tuple[int, int]]:
    """
    Returns inclusive runs of consecutive positive windows as:
    [(start_idx, end_idx), ...]
    """
    runs: List[Tuple[int, int]] = []
    start = None

    for i, value in enumerate(binary_preds):
        if value == 1 and start is None:
            start = i
        elif value == 0 and start is not None:
            runs.append((start, i - 1))
            start = None

    if start is not None:
        runs.append((start, len(binary_preds) - 1))

    return runs


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

    max_prob = float(np.max(window_probs))
    mean_prob = float(np.mean(window_probs))
    num_positive = int(binary_preds.sum())

    runs = _find_positive_runs(binary_preds)

    qualifying_runs = []
    max_consecutive = 0

    for start_idx, end_idx in runs:
        run_length = end_idx - start_idx + 1
        max_consecutive = max(max_consecutive, run_length)

        if run_length >= min_consecutive_positive_windows:
            start_sample = window_ranges[start_idx][0]
            end_sample = window_ranges[end_idx][1]

            qualifying_runs.append(
                {
                    "start_window_index": int(start_idx),
                    "end_window_index": int(end_idx),
                    "num_windows": int(run_length),
                    "start_sec": float(start_sample / sfreq),
                    "end_sec": float(end_sample / sfreq),
                    "max_probability_in_run": float(np.max(window_probs[start_idx:end_idx + 1])),
                    "mean_probability_in_run": float(np.mean(window_probs[start_idx:end_idx + 1])),
                }
            )

    file_pred = "YES" if len(qualifying_runs) > 0 else "NO"

    summary = {
        "prediction": file_pred,
        "seizure_probability": max_prob,
        "mean_window_probability": mean_prob,
        "threshold": float(threshold),
        "num_windows": int(len(window_probs)),
        "num_positive_windows": num_positive,
        "max_consecutive_positive_windows": int(max_consecutive),
        "min_consecutive_positive_windows": int(min_consecutive_positive_windows),
        "decision_runs": qualifying_runs,
    }

    return summary