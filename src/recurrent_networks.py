"""Two "simple" RNN architectures from Bucci (2020): Elman and Jordan.

The Elman network (Elman 1990) feeds the hidden state back into itself
at the next timestep; the Jordan network (Jordan 1986) feeds the *output*
back. Both predate LSTM and are the canonical examples of the vanishing-
gradient problem in deep learning literature — they tend to lose memory
of inputs more than ~10 steps in the past. Including them here lets the
project show *why* Bucci moves on to LSTM/NARX.

Both classes plug into the same rolling-window engine as the rest of the
project (``fit(y_train) / forecast(y_history) -> float``).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from src.neural_models import _BaseNNForecaster

ArrayLike = Sequence[float] | np.ndarray


# ---------------------------------------------------------------------------
# Elman (simple recurrent) cell
# ---------------------------------------------------------------------------

class _ElmanCell(nn.Module):
    """h_t = σ(W_x x_t + W_h h_{t-1} + b_h).

    PyTorch's ``nn.RNN`` only supports ``tanh`` or ``relu`` non-linearities,
    so we write the cell explicitly to keep Bucci's sigmoid activation.
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_x = nn.Linear(input_size, hidden_size, bias=True)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: (B, T, F).  Returns h at the final timestep, shape (B, H).
        B, T, _ = x_seq.shape
        h = torch.zeros(B, self.hidden_size, device=x_seq.device, dtype=x_seq.dtype)
        for t in range(T):
            h = torch.sigmoid(self.W_x(x_seq[:, t, :]) + self.W_h(h))
        return h


class _ElmanNet(nn.Module):
    """Elman feedforward head: linear read-out from the final hidden state."""

    def __init__(self, hidden_units: int) -> None:
        super().__init__()
        self.cell = _ElmanCell(input_size=1, hidden_size=hidden_units)
        self.fc = nn.Linear(hidden_units, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.cell(x))


class ElmanForecaster(_BaseNNForecaster):
    """Bucci-style Elman RNN: q sigmoid hidden units, identity output.

    Inputs are arranged exactly like the project's LSTM forecaster — a
    length-``n_lags`` sequence of standardised log-RV values, oldest
    first — so the rolling-window engine drives this class without
    modification.
    """

    def __init__(
        self,
        n_lags: int = 3,
        hidden_units: int = 5,
        epochs: int = 100,
        early_stop_patience: int = 10,
        batch_size: int = 32,
        lr: float = 1e-3,
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            n_lags=n_lags,
            hidden_units=hidden_units,
            epochs=epochs,
            early_stop_patience=early_stop_patience,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            device=device,
        )

    def _build_dataset(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_lags, T = self.n_lags, len(z)
        X = np.stack([z[i: i + n_lags] for i in range(T - n_lags)], axis=0)
        X = X.reshape(-1, n_lags, 1).astype(np.float32)
        Y = z[n_lags:T].reshape(-1, 1).astype(np.float32)
        return X, Y

    def _build_input_for_forecast(self, z: np.ndarray) -> np.ndarray:
        return z[-self.n_lags:].reshape(self.n_lags, 1).astype(np.float32)

    def _build_model(self) -> nn.Module:
        return _ElmanNet(self.hidden_units)


# ---------------------------------------------------------------------------
# Jordan cell — output feedback (no teacher forcing in this implementation)
# ---------------------------------------------------------------------------

class _JordanCell(nn.Module):
    """Recurrent unit with feedback from the *previous output*.

        h_t = σ(W_x x_t + W_c c_t + b_h),
        o_t = W_o h_t + b_o,
        c_{t+1} = o_t.

    We initialise the context unit to zero (mean of the standardised data).
    With ``n_lags`` typically small (= 3 here) the cell is unrolled
    explicitly in Python — fast enough on CPU.
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_x = nn.Linear(input_size, hidden_size, bias=True)
        self.W_c = nn.Linear(1, hidden_size, bias=False)
        self.W_o = nn.Linear(hidden_size, 1, bias=True)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        B, T, _ = x_seq.shape
        c = torch.zeros(B, 1, device=x_seq.device, dtype=x_seq.dtype)
        o = None
        for t in range(T):
            h = torch.sigmoid(self.W_x(x_seq[:, t, :]) + self.W_c(c))
            o = self.W_o(h)
            c = o
        assert o is not None
        return o  # (B, 1)


class _JordanNet(nn.Module):
    def __init__(self, hidden_units: int) -> None:
        super().__init__()
        self.cell = _JordanCell(input_size=1, hidden_size=hidden_units)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cell(x)


class JordanForecaster(_BaseNNForecaster):
    """Bucci-style Jordan RNN with output feedback."""

    def __init__(
        self,
        n_lags: int = 3,
        hidden_units: int = 5,
        epochs: int = 100,
        early_stop_patience: int = 10,
        batch_size: int = 32,
        lr: float = 1e-3,
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            n_lags=n_lags,
            hidden_units=hidden_units,
            epochs=epochs,
            early_stop_patience=early_stop_patience,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            device=device,
        )

    def _build_dataset(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_lags, T = self.n_lags, len(z)
        X = np.stack([z[i: i + n_lags] for i in range(T - n_lags)], axis=0)
        X = X.reshape(-1, n_lags, 1).astype(np.float32)
        Y = z[n_lags:T].reshape(-1, 1).astype(np.float32)
        return X, Y

    def _build_input_for_forecast(self, z: np.ndarray) -> np.ndarray:
        return z[-self.n_lags:].reshape(self.n_lags, 1).astype(np.float32)

    def _build_model(self) -> nn.Module:
        return _JordanNet(self.hidden_units)


# ---------------------------------------------------------------------------
# Convenient aliases matching the Bucci-paper naming
# ---------------------------------------------------------------------------
ENNForecaster = ElmanForecaster
JNNForecaster = JordanForecaster


__all__ = [
    "ElmanForecaster",
    "JordanForecaster",
    "ENNForecaster",
    "JNNForecaster",
]
