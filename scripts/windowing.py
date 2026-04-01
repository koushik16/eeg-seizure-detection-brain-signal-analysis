# windowing.py

import numpy as np


def create_windows(data, sfreq, window_sec=5, step_sec=5):
    """
    Split EEG data into sliding windows.

    Parameters
    ----------
    data : np.ndarray
        EEG data of shape (n_channels, n_samples)
    sfreq : float
        Sampling frequency in Hz
    window_sec : float
        Window length in seconds
    step_sec : float
        Step size in seconds

    Returns
    -------
    windows : np.ndarray
        Array of shape (n_windows, n_channels, window_samples)
    window_times : list[tuple[float, float]]
        List of (start_time_sec, end_time_sec) for each window
    """
    window_samples = int(window_sec * sfreq)
    step_samples = int(step_sec * sfreq)

    n_channels, n_samples = data.shape
    windows = []
    window_times = []

    for start in range(0, n_samples - window_samples + 1, step_samples):
        end = start + window_samples
        window = data[:, start:end]

        windows.append(window)
        window_times.append((start / sfreq, end / sfreq))

    return np.array(windows), window_times


def compute_overlap(window_start, window_end, seizure_start, seizure_end):
    """
    Compute overlap duration between a window and seizure interval.

    Parameters
    ----------
    window_start : float
    window_end : float
    seizure_start : float
    seizure_end : float

    Returns
    -------
    overlap : float
        Overlap duration in seconds
    """
    overlap_start = max(window_start, seizure_start)
    overlap_end = min(window_end, seizure_end)
    return max(0.0, overlap_end - overlap_start)


def label_windows(window_times, seizure_start, seizure_end):
    """
    Label windows using binary seizure labeling.

    Rule:
    - 1 if the window has any overlap with seizure interval
    - 0 otherwise

    Parameters
    ----------
    window_times : list[tuple[float, float]]
        List of (start_time_sec, end_time_sec)
    seizure_start : float
        Seizure start time in seconds
    seizure_end : float
        Seizure end time in seconds

    Returns
    -------
    labels : np.ndarray
        Binary labels of shape (n_windows,)
    """
    labels = []

    for start, end in window_times:
        overlap = (start < seizure_end) and (end > seizure_start)
        labels.append(1 if overlap else 0)

    return np.array(labels)


def label_windows_by_threshold(window_times, seizure_start, seizure_end, min_overlap_ratio=0.5):
    """
    Label windows as seizure only if seizure overlap ratio reaches a threshold.

    Parameters
    ----------
    window_times : list[tuple[float, float]]
        List of (start_time_sec, end_time_sec)
    seizure_start : float
        Seizure start time in seconds
    seizure_end : float
        Seizure end time in seconds
    min_overlap_ratio : float
        Minimum fraction of the window that must overlap seizure interval
        to assign label 1

    Returns
    -------
    labels : np.ndarray
        Binary labels of shape (n_windows,)
    """
    labels = []

    for start, end in window_times:
        overlap = compute_overlap(start, end, seizure_start, seizure_end)
        window_duration = end - start
        overlap_ratio = overlap / window_duration if window_duration > 0 else 0.0
        labels.append(1 if overlap_ratio >= min_overlap_ratio else 0)

    return np.array(labels)


def summarize_labels(labels):
    """
    Print simple label distribution summary.

    Parameters
    ----------
    labels : np.ndarray
        Binary labels

    Returns
    -------
    summary : dict
        Counts of seizure and non-seizure windows
    """
    seizure_count = int(np.sum(labels == 1))
    non_seizure_count = int(np.sum(labels == 0))

    summary = {
        "total_windows": len(labels),
        "seizure_windows": seizure_count,
        "non_seizure_windows": non_seizure_count,
    }

    print("Total windows:", summary["total_windows"])
    print("Seizure windows:", summary["seizure_windows"])
    print("Non-seizure windows:", summary["non_seizure_windows"])

    return summary