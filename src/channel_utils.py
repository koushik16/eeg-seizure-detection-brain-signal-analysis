def normalize_channel_name(ch: str) -> str:
    """
    Normalize channel names to a consistent format.
    Example: T8-P8-0 -> T8-P8
    """
    ch = ch.strip().upper()
    ch = ch.replace("EEG ", "")
    ch = ch.replace("--", "-")

    if ch.endswith("-0"):
        ch = ch[:-2]

    return ch


def build_channel_index_map(raw_ch_names, target_channels):
    """
    Build mapping from target channel names to actual raw channel indices.
    Returns a list of indices in target channel order.
    """
    normalized = [normalize_channel_name(ch) for ch in raw_ch_names]
    name_to_idx = {name: idx for idx, name in enumerate(normalized)}

    missing = [ch for ch in target_channels if ch not in name_to_idx]
    if missing:
        raise ValueError(f"Missing required channels: {missing}")

    return [name_to_idx[ch] for ch in target_channels]