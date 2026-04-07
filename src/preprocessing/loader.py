from __future__ import annotations

from pathlib import Path
from typing import Tuple

import mne
import numpy as np


def load_edf(edf_path: str | Path) -> Tuple[np.ndarray, float, list[str]]:
    """
    Load EDF file and return raw EEG data.

    Returns
    -------
    data : np.ndarray
        Shape (n_channels, n_samples), in volts
    sfreq : float
        Sampling frequency
    ch_names : list[str]
        Original channel names from EDF
    """
    edf_path = Path(edf_path)

    if not edf_path.exists():
        raise FileNotFoundError(f"EDF file not found: {edf_path}")

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    data = raw.get_data().astype(np.float32)
    sfreq = float(raw.info["sfreq"])
    ch_names = list(raw.ch_names)

    return data, sfreq, ch_names