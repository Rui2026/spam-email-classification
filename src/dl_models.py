"""PyTorch models: MLP on bag-of-words style vectors, LSTM on token ids."""

from __future__ import annotations

import torch
import torch.nn as nn

from . import config


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        h = config.DL_HIDDEN_DIM
        self.net = nn.Sequential(
            nn.Linear(input_dim, h),
            nn.ReLU(),
            nn.Dropout(config.DL_DROPOUT),
            nn.Linear(h, h // 2),
            nn.ReLU(),
            nn.Dropout(config.DL_DROPOUT),
            nn.Linear(h // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int = 2) -> None:
        super().__init__()
        self.bidirectional = config.LSTM_BIDIRECTIONAL
        self.num_layers = config.LSTM_NUM_LAYERS
        self.embedding = nn.Embedding(vocab_size, config.LSTM_EMBED_DIM, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=config.LSTM_EMBED_DIM,
            hidden_size=config.LSTM_HIDDEN_DIM,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
        )
        self.dropout = nn.Dropout(config.DL_DROPOUT)
        out_dim = config.LSTM_HIDDEN_DIM * (2 if self.bidirectional else 1)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        _, (h_n, _) = self.lstm(emb)
        # h_n shape: (num_layers * num_directions, batch, hidden)
        if self.bidirectional:
            # concat the last layer's forward + backward final hidden states
            last = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            last = h_n[-1]
        last = self.dropout(last)
        return self.fc(last)

