from __future__ import annotations

import numpy as np
import mne

from src.config import (
    APPLY_NOTCH,
    BANDPASS_HIGH,
    BANDPASS_LOW,
    EXPECTED_SFREQ,
    NOTCH_FREQ,
)


def assert_expected_sampling_rate(sfreq: float) -> None:
    if int(round(sfreq)) != EXPECTED_SFREQ:
        raise ValueError(
            f"Expected sampling rate {EXPECTED_SFREQ} Hz, but got {sfreq} Hz"
        )


def apply_filters(data: np.ndarray, sfreq: float) -> np.ndarray:
    """
    Apply the same filtering logic used in the notebooks:
    - bandpass 0.5 to 40 Hz
    - notch 60 Hz
    """
    assert_expected_sampling_rate(sfreq)

    filtered = data.astype(np.float64, copy=True)

    filtered = mne.filter.filter_data(
        filtered,
        sfreq=sfreq,
        l_freq=BANDPASS_LOW,
        h_freq=BANDPASS_HIGH,
        verbose=False,
    )

    if APPLY_NOTCH:
        filtered = mne.filter.notch_filter(
            filtered,
            Fs=sfreq,
            freqs=NOTCH_FREQ,
            verbose=False,
        )

    return filtered.astype(np.float32)