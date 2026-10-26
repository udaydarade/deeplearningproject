"""Sequence-model zoo and shared trainers for the crash-prediction benchmark.

Every torch model shares the same contract: input ``(batch, seq_len,
n_features)`` float32, output one raw logit per sample, plus a compact
classification head (LayerNorm -> Dropout -> Linear). Keeping capacity small
(hidden 24-32) was a deliberate choice: crash labels occur at ~2% prevalence,
so large nets overfit the rare positives quickly on CPU-budget training.

``train_sequence_model`` replicates the leakage discipline of
``crash_forecasting.py``: the StandardScaler is fitted on train rows only,
class imbalance is handled with a capped pos_weight, early stopping tracks
validation average precision with patience 4, and thresholds are never tuned
on test data. Determinism comes from a per-task string-derived seed.
"""
from __future__ import annotations

import random

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cpu")


# ------------------------------------------------------------------- models
class LSTMModel(nn.Module):
    def __init__(self, n_features: int, hidden: int = 24):
        super().__init__()
        self.rnn = nn.LSTM(n_features, hidden, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(0.2), nn.Linear(hidden, 1))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class BiLSTMModel(nn.Module):
    def __init__(self, n_features: int, hidden: int = 24):
        super().__init__()
        self.rnn = nn.LSTM(n_features, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.LayerNorm(2 * hidden), nn.Dropout(0.2), nn.Linear(2 * hidden, 1))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class AttnBiLSTMModel(nn.Module):
    """BiLSTM with additive (Bahdanau-style) temporal attention pooling.

    Instead of only reading the last hidden state, the model learns a
    normalised attention weight over all 30 BiLSTM outputs and pools them -
    a more general read-out that often handles regimes drifting in time.
    """

    def __init__(self, n_features: int, hidden: int = 24):
        super().__init__()
        self.rnn = nn.LSTM(n_features, hidden, batch_first=True, bidirectional=True)
        self.score = nn.Linear(2 * hidden, 1)
        self.head = nn.Sequential(nn.LayerNorm(2 * hidden), nn.Dropout(0.2), nn.Linear(2 * hidden, 1))

    def forward(self, x):
        out, _ = self.rnn(x)
        weights = torch.softmax(self.score(out).squeeze(-1), dim=1)         # (B, T)
        context = torch.einsum("bt,btd->bd", weights, out)                  # (B, 2H)
        return self.head(context).squeeze(-1)


class GRUModel(nn.Module):
    def __init__(self, n_features: int, hidden: int = 24):
        super().__init__()
        self.rnn = nn.GRU(n_features, hidden, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(0.2), nn.Linear(hidden, 1))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class CNNModel(nn.Module):
    """Two 1-D conv layers followed by global max-pooling."""

    def __init__(self, n_features: int, channels: int = 24):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(n_features, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.head = nn.Sequential(nn.LayerNorm(channels), nn.Dropout(0.2), nn.Linear(channels, 1))

    def forward(self, x):
        h = self.body(x.transpose(1, 2))
        pooled, _ = h.max(dim=2)
        return self.head(pooled).squeeze(-1)


class TCNModel(nn.Module):
    """Dilated causal convolutions with residual connections."""

    def __init__(self, n_features: int, hidden: int = 24, dilations=(1, 2, 4)):
        super().__init__()
        self.stem = nn.Conv1d(n_features, hidden, 1)
        blocks = []
        for dilation in dilations:
            blocks.append(TCNBlock(hidden, hidden, dilation))
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(0.2), nn.Linear(hidden, 1))

    def forward(self, x):
        h = self.stem(x.transpose(1, 2))
        for block in self.blocks:
            h = block(h)
        return self.head(h[:, :, -1]).squeeze(-1)


class TCNBlock(nn.Module):
    def __init__(self, channels: int, out: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, out, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm = nn.GroupNorm(1, out)
        self.residual = nn.Conv1d(channels, out, 1) if channels != out else nn.Identity()

    def forward(self, x):
        y = torch.relu(self.norm(self.conv(x)))[:, :, : x.shape[-1]]
        return torch.relu(y + self.residual(x))


class TransformerModel(nn.Module):
    def __init__(self, n_features: int, d_model: int = 32, heads: int = 4, layers: int = 2):
        super().__init__()
        self.project = nn.Linear(n_features, d_model)
        self.position = nn.Parameter(torch.randn(1, 64, d_model).mul_(0.02))
        layer = nn.TransformerEncoderLayer(
            d_model, heads, dim_feedforward=4 * d_model, dropout=0.1,
            activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(0.2), nn.Linear(d_model, 1))

    def forward(self, x):
        h = self.encoder(self.project(x) + self.position[:, : x.size(1)])
        return self.head(h.mean(dim=1)).squeeze(-1)


def build_model(name: str, n_features: int) -> nn.Module:
    factories = {
        "lstm": LSTMModel,
        "bilstm": BiLSTMModel,
        "attn_bilstm": AttnBiLSTMModel,
        "gru": GRUModel,
        "cnn": CNNModel,
        "tcn": TCNModel,
        "transformer": TransformerModel,
    }
    return factories[name](n_features).to(DEVICE)


# ------------------------------------------------------------------ trainers
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def standardise(x, train_idx):
    """Fit a StandardScaler on train rows only, then transform everything."""
    scaler = StandardScaler()
    scaler.fit(x[train_idx].reshape(-1, x.shape[-1]))
    scaled = scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
    return scaled.astype(np.float32), scaler


def train_sequence_model(x, y, train_idx, val_idx, test_idx, *,
                         model_name: str, epochs: int = 12,
                         batch_size: int = 512, lr: float = 8e-4,
                         weight_decay: float = 1e-3, patience: int = 4,
                         seed: int = 0):
    """Train a sequence model; return (model, scaler, val_p, test_p)."""
    set_seed(seed)
    z, scaler = standardise(x, train_idx)
    positives = max(float(y[train_idx].sum()), 1.0)
    negatives = max(float(len(train_idx) - positives), 1.0)
    model = build_model(model_name, x.shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(min(negatives / positives, 20.0)))
    loader = DataLoader(
        TensorDataset(torch.from_numpy(z[train_idx]), torch.from_numpy(y[train_idx])),
        batch_size=batch_size, shuffle=True,
    )
    best_ap, best_state, stale = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_p = torch.sigmoid(model(torch.from_numpy(z[val_idx]))).numpy()
        ap = average_precision_score(y[val_idx], val_p) if y[val_idx].sum() else 0.0
        if ap > best_ap + 1e-5:
            best_ap, stale = ap, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_p = torch.sigmoid(model(torch.from_numpy(z[val_idx]))).numpy()
        test_p = torch.sigmoid(model(torch.from_numpy(z[test_idx]))).numpy()
    return model, scaler, val_p, test_p


def train_logit(x, y, train_idx, val_idx, test_idx, seed: int = 0):
    """Flatten the sequences and fit a convergent logistic-regression baseline.

    Serves as the sanity floor: if deep sequence models cannot beat this on
    average precision, the temporal modelling is not adding anything.
    """
    set_seed(seed)
    flattened = x.reshape(len(x), -1).astype(np.float32)
    z, scaler = standardise(flattened, train_idx)
    model = LogisticRegression(
        solver="liblinear", class_weight="balanced", C=1.0, random_state=seed,
    )
    model.fit(z[train_idx], y[train_idx])
    val_p = model.predict_proba(z[val_idx])[:, 1]
    test_p = model.predict_proba(z[test_idx])[:, 1]
    return model, scaler, val_p, test_p
