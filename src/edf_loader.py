import mne
import numpy as np

from src.channel_utils import build_channel_index_map
from src.filtering import apply_filters


def load_and_prepare_edf(file_path, target_channels, l_freq, h_freq, notch_freq):
    """
    Load EDF, apply filtering, select channels in fixed order.
    Returns:
        data: np.ndarray of shape (n_channels, n_samples)
        sfreq: float
    """
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    raw = apply_filters(raw, l_freq=l_freq, h_freq=h_freq, notch_freq=notch_freq)

    ordered_idx = build_channel_index_map(raw.ch_names, target_channels)
    data = raw.get_data()[ordered_idx, :]
    sfreq = raw.info["sfreq"]

    return data.astype(np.float32), sfreq