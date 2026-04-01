import numpy as np

from src.labeling import assign_label


def create_windows(data, sfreq, seizure_intervals, window_size_sec, stride_sec, overlap_threshold):
    """
    Create fixed-size windows and labels.

    Returns:
        X: np.ndarray, shape (num_windows, n_channels, window_samples)
        y: np.ndarray, shape (num_windows,)
        meta: list[dict]
    """
    n_channels, n_samples = data.shape
    window_samples = int(window_size_sec * sfreq)
    stride_samples = int(stride_sec * sfreq)

    windows = []
    labels = []
    metadata = []

    for start_idx in range(0, n_samples - window_samples + 1, stride_samples):
        end_idx = start_idx + window_samples

        window = data[:, start_idx:end_idx]

        start_sec = start_idx / sfreq
        end_sec = end_idx / sfreq

        label = assign_label(
            window_start=start_sec,
            window_end=end_sec,
            seizure_intervals=seizure_intervals,
            threshold=overlap_threshold
        )

        windows.append(window)
        labels.append(label)
        metadata.append({
            "start_sec": start_sec,
            "end_sec": end_sec,
            "label": label
        })

    X = np.stack(windows).astype(np.float32)
    y = np.array(labels, dtype=np.int64)

    return X, y, metadata