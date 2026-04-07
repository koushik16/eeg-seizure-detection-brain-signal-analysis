from __future__ import annotations

import torch
import torch.nn as nn


class EEGCNNLSTM(nn.Module):
    """
    Exact Phase 4 model architecture from the notebook.
    Input shape: (batch, 18, 1280)
    """

    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=18, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)   # 1280 -> 640
        self.dropout1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)   # 640 -> 320
        self.dropout2 = nn.Dropout(0.3)

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.dropout_lstm = nn.Dropout(0.3)

        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = x.transpose(1, 2)  # (batch, 64, 320) -> (batch, 320, 64)

        _, (hidden, _) = self.lstm(x)
        x = hidden[-1]  # (batch, 64)

        x = self.dropout_lstm(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x.squeeze(1)  # (batch,)