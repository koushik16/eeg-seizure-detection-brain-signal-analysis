from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import BATCH_SIZE


def predict_window_probabilities(
    model: torch.nn.Module,
    windows: np.ndarray,
    device: torch.device,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """
    Parameters
    ----------
    windows : np.ndarray
        Shape (n_windows, n_channels, n_samples)

    Returns
    -------
    probs : np.ndarray
        Shape (n_windows,), sigmoid probabilities
    """
    if windows.ndim != 3:
        raise ValueError(f"Expected windows shape (n_windows, n_channels, n_samples), got {windows.shape}")

    x_tensor = torch.from_numpy(windows).float()
    dataset = TensorDataset(x_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_probs = []

    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())

    if not all_probs:
        raise ValueError("No predictions were generated. Check input windows.")

    return np.concatenate(all_probs, axis=0).astype(np.float32)