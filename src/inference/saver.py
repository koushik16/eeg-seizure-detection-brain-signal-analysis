from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.config import (
    CHANNELS_NAME,
    NORMALIZED_WINDOWS_NAME,
    OUTPUTS_DIR,
    PREPROCESSED_SIGNAL_NAME,
    SUMMARY_NAME,
    WINDOWS_NAME,
    WINDOW_PROBS_NAME,
)


def _safe_stem(file_path: str | Path) -> str:
    return Path(file_path).stem.replace(" ", "_")


def prepare_output_dir(input_path: str | Path, base_dir: str | Path = OUTPUTS_DIR) -> Path:
    input_stem = _safe_stem(input_path)
    out_dir = Path(base_dir) / input_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_intermediate_outputs(
    input_path: str | Path,
    channels_used: List[str],
    preprocessed_signal: np.ndarray,
    windows: np.ndarray,
    normalized_windows: np.ndarray,
    window_probabilities: np.ndarray,
    summary: Dict[str, Any],
) -> Path:
    out_dir = prepare_output_dir(input_path)

    np.save(out_dir / PREPROCESSED_SIGNAL_NAME, preprocessed_signal)
    np.save(out_dir / WINDOWS_NAME, windows)
    np.save(out_dir / NORMALIZED_WINDOWS_NAME, normalized_windows)
    np.save(out_dir / WINDOW_PROBS_NAME, window_probabilities)

    with open(out_dir / CHANNELS_NAME, "w", encoding="utf-8") as f:
        json.dump(channels_used, f, indent=2)

    with open(out_dir / SUMMARY_NAME, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return out_dir