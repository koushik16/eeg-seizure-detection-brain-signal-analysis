from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch

from src.config import MODEL_PATH
from src.inference.model import EEGCNNLSTM


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path: str | Path = MODEL_PATH) -> Tuple[EEGCNNLSTM, torch.device]:
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    device = get_device()
    model = EEGCNNLSTM().to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, device