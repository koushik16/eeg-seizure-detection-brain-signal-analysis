import mne


def apply_filters(raw, l_freq=0.5, h_freq=40.0, notch_freq=60.0):
    """
    Apply notch filter and bandpass filter.
    """
    raw = raw.copy()
    raw.notch_filter(freqs=notch_freq, verbose=False)
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
    return raw