# preprocess.py

import mne
from collections import Counter


def load_edf(edf_path: str, preload: bool = True):
    raw = mne.io.read_raw_edf(edf_path, preload=preload)
    return raw


def check_duplicate_channels(raw):
    counts = Counter(raw.ch_names)
    duplicates = {name: count for name, count in counts.items() if count > 1}

    if duplicates:
        print("Duplicate channel names found:", duplicates)
    else:
        print("All channel names are unique.")

    return duplicates


def apply_bandpass_filter(raw, l_freq: float = 0.5, h_freq: float = 40.0):
    return raw.copy().filter(l_freq=l_freq, h_freq=h_freq)


def apply_notch_filter(raw, freqs=60):
    return raw.copy().notch_filter(freqs=freqs)


def preprocess_raw(edf_path: str, l_freq: float = 0.5, h_freq: float = 40.0, notch_freq=60):
    raw = load_edf(edf_path)
    check_duplicate_channels(raw)

    raw_processed = apply_notch_filter(raw, freqs=notch_freq)
    raw_processed = apply_bandpass_filter(raw_processed, l_freq=l_freq, h_freq=h_freq)

    return raw, raw_processed