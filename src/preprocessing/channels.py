from __future__ import annotations

from typing import Dict, List, Tuple

from src.config import EXPECTED_CHANNELS


def _normalize_channel_name(name: str) -> str:
    """
    Normalize small naming inconsistencies seen in CHB-MIT EDF files.
    """
    name = name.strip().upper()

    fixes = {
        "T8-P8-0": "T8-P8",
    }

    return fixes.get(name, name)


def build_channel_mapping(raw_channel_names: List[str]) -> Tuple[Dict[str, int], List[str], List[str]]:
    """
    Build mapping from expected channel name -> original raw index.

    Returns:
        channel_to_index: dict mapping standardized channel name to raw index
        missing_channels: expected channels not found
        normalized_raw_names: normalized raw channel names in original order
    """
    normalized_raw_names = [_normalize_channel_name(ch) for ch in raw_channel_names]

    channel_to_index: Dict[str, int] = {}
    for idx, ch in enumerate(normalized_raw_names):
        if ch not in channel_to_index:
            channel_to_index[ch] = idx

    missing_channels = [ch for ch in EXPECTED_CHANNELS if ch not in channel_to_index]
    return channel_to_index, missing_channels, normalized_raw_names


def validate_required_channels(raw_channel_names: List[str]) -> None:
    """
    Raise error if any required channel is missing.
    """
    _, missing_channels, _ = build_channel_mapping(raw_channel_names)

    if missing_channels:
        raise ValueError(
            f"Missing required EEG channels: {missing_channels}"
        )


def reorder_signal_channels(data, raw_channel_names: List[str]):
    """
    Reorder EEG data into EXPECTED_CHANNELS order.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_channels, n_samples)
    raw_channel_names : list[str]

    Returns
    -------
    reordered_data : np.ndarray
        Shape (len(EXPECTED_CHANNELS), n_samples)
    ordered_channel_names : list[str]
    """
    channel_to_index, missing_channels, _ = build_channel_mapping(raw_channel_names)

    if missing_channels:
        raise ValueError(
            f"Cannot reorder EEG channels. Missing channels: {missing_channels}"
        )

    ordered_indices = [channel_to_index[ch] for ch in EXPECTED_CHANNELS]
    reordered_data = data[ordered_indices, :]

    return reordered_data, EXPECTED_CHANNELS.copy()