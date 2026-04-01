def compute_overlap_fraction(window_start, window_end, seizure_intervals):
    """
    Compute maximum overlap fraction between a window and any seizure interval.
    """
    window_len = window_end - window_start
    max_fraction = 0.0

    for sz_start, sz_end in seizure_intervals:
        overlap_start = max(window_start, sz_start)
        overlap_end = min(window_end, sz_end)
        overlap = max(0.0, overlap_end - overlap_start)
        frac = overlap / window_len
        max_fraction = max(max_fraction, frac)

    return max_fraction


def assign_label(window_start, window_end, seizure_intervals, threshold=0.2):
    """
    Label window as seizure (1) if overlap fraction >= threshold, else 0.
    """
    overlap_fraction = compute_overlap_fraction(window_start, window_end, seizure_intervals)
    return 1 if overlap_fraction >= threshold else 0