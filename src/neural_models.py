"""Neural-network one-step-ahead forecasters for daily log-realized-volatility.

Implements the three architectures used in Bucci (2020) — FNN (Section 2),
LSTM (eqs. 6–12) and NAR (eq. 14 without exogenous regressors) — adapted
to daily data. All three share the same training loop (chronological 80/20
validation split, Adam, MSE, early stopping) and the same per-call
``fit / forecast`` interface as :mod:`src.econometric_models`, so the
rolling-window engine can drive them uniformly.

Reproducibility: every ``fit`` call reseeds NumPy and PyTorch with
``seed``; ``torch.use_deterministic_algorithms(False)`` is left at the
default to avoid spurious failures on operators that don't have
deterministic kernels on CPU. The interaction of fresh seeds with rolling
windows means each refit produces the same initialisation but converges
to slightly different weights driven by the different training data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

ArrayLike = Sequence[float] | np.ndarray


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


class _StandardScaler:
    """Plain z-score scaler. Fitted on training only — never sees test data."""

    mu: float = 0.0
    sd: float = 1.0

    def fit(self, x: np.ndarray) -> "_StandardScaler":
        self.mu = float(np.mean(x))
        self.sd = float(np.std(x))
        if self.sd < 1e-8:
            self.sd = 1e-8
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float32) - self.mu) / self.sd

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=np.float32) * self.sd + self.mu


@dataclass
class TrainingHistory:
    train: list[float] = field(default_factory=list)
    val: list[float] = field(default_factory=list)
    best_epoch: int = 0


# ---------------------------------------------------------------------------
# Base forecaster — shared fit/forecast pipeline
# ---------------------------------------------------------------------------

class _BaseNNForecaster:
    """Shared scaffolding: dataset building, training loop, forecast head.

    Subclasses must override :meth:`_build_dataset`,
    :meth:`_build_input_for_forecast` and :meth:`_build_model`.
    """

    def __init__(
        self,
        n_lags: int,
        hidden_units: int,
        epochs: int = 100,
        early_stop_patience: int = 10,
        batch_size: int = 32,
        lr: float = 1e-3,
        seed: int = 42,
        device: str = "cpu",
        val_frac: float = 0.20,
    ) -> None:
        self.n_lags = n_lags
        self.hidden_units = hidden_units
        self.epochs = epochs
        self.early_stop_patience = early_stop_patience
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.device = torch.device(device)
        self.val_frac = val_frac
        self.scaler = _StandardScaler()
        self.model: nn.Module | None = None
        self.history = TrainingHistory()

    # -- subclass hooks ----------------------------------------------------
    def _build_dataset(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def _build_input_for_forecast(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _build_model(self) -> nn.Module:
        raise NotImplementedError

    # -- public API --------------------------------------------------------
    def fit(self, y_train: ArrayLike) -> "_BaseNNForecaster":
        _seed_all(self.seed)
        y = np.asarray(y_train, dtype=np.float32).ravel()
        if len(y) <= self.n_lags + 10:
            raise ValueError(f"need > {self.n_lags + 10} train obs, got {len(y)}")
        self.scaler.fit(y)
        z = self.scaler.transform(y)
        X, Y = self._build_dataset(z)
        # Chronological 80/20 split — validation is the most recent slice.
        cut = int((1.0 - self.val_frac) * len(X))
        X_tr, Y_tr = X[:cut], Y[:cut]
        X_va, Y_va = X[cut:], Y[cut:]

        model = self._build_model().to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        Xtr_t = torch.from_numpy(X_tr).to(self.device)
        Ytr_t = torch.from_numpy(Y_tr).to(self.device)
        Xva_t = torch.from_numpy(X_va).to(self.device)
        Yva_t = torch.from_numpy(Y_va).to(self.device)

        history = TrainingHistory()
        best_val = float("inf")
        best_state: dict | None = None
        patience = 0
        n_train = len(X_tr)
        generator = torch.Generator(device="cpu").manual_seed(self.seed)

        for epoch in range(self.epochs):
            model.train()
            perm = torch.randperm(n_train, generator=generator)
            batch_losses = []
            for i in range(0, n_train, self.batch_size):
                idx = perm[i: i + self.batch_size]
                opt.zero_grad()
                pred = model(Xtr_t[idx])
                loss = loss_fn(pred, Ytr_t[idx])
                loss.backward()
                opt.step()
                batch_losses.append(float(loss.detach()))
            train_loss = float(np.mean(batch_losses))

            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(Xva_t), Yva_t))

            history.train.append(train_loss)
            history.val.append(val_loss)

            if val_loss < best_val - 1e-8:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                history.best_epoch = epoch
                patience = 0
            else:
                patience += 1
                if patience >= self.early_stop_patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self.model = model
        self.history = history
        return self

    def forecast(self, y_history: ArrayLike) -> float:
        if self.model is None:
            raise RuntimeError("call fit() before forecast()")
        y = np.asarray(y_history, dtype=np.float32).ravel()
        z = self.scaler.transform(y)
        x = self._build_input_for_forecast(z)
        with torch.no_grad():
            x_t = torch.from_numpy(np.expand_dims(x, 0)).to(self.device)
            z_hat = float(self.model(x_t).item())
        return float(self.scaler.inverse_transform(np.array([z_hat], dtype=np.float32))[0])


# ---------------------------------------------------------------------------
# FNN (Bucci eq. 3) — 1 hidden sigmoid layer, 5 units, 3 lags
# ---------------------------------------------------------------------------

class FNNForecaster(_BaseNNForecaster):
    """Feed-forward NN: 3 lag inputs → 5 sigmoid hidden → 1 linear out."""

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
        # Column 0 = y_{t-1}, column 1 = y_{t-2}, …, column n_lags-1 = y_{t-n_lags}
        cols = [z[n_lags - 1 - j: T - 1 - j] for j in range(n_lags)]
        X = np.stack(cols, axis=1).astype(np.float32)
        Y = z[n_lags:T].reshape(-1, 1).astype(np.float32)
        return X, Y

    def _build_input_for_forecast(self, z: np.ndarray) -> np.ndarray:
        # Most recent first to match training column order.
        return z[-self.n_lags:][::-1].copy().astype(np.float32)

    def _build_model(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.n_lags, self.hidden_units),
            nn.Sigmoid(),
            nn.Linear(self.hidden_units, 1),
        )


# ---------------------------------------------------------------------------
# NAR — Bucci eq. 14 minus the exogenous block. Same shape as FNN but
# 7 lag inputs, tanh activation, 7 hidden units (Bucci Table 2 best NAR).
# ---------------------------------------------------------------------------

class NARForecaster(FNNForecaster):
    """NAR (Bucci eq. 14, no X): 7 lag inputs → 7 tanh hidden → 1 linear out."""

    def __init__(
        self,
        n_lags: int = 7,
        hidden_units: int = 7,
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

    def _build_model(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.n_lags, self.hidden_units),
            nn.Tanh(),
            nn.Linear(self.hidden_units, 1),
        )


# ---------------------------------------------------------------------------
# LSTM (Bucci eqs. 6–12) — 50 hidden units, 3-step sequence, dropout 0.2
# ---------------------------------------------------------------------------

class _LSTMNet(nn.Module):
    def __init__(self, hidden_units: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_units, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_units, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, 1)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(self.dropout(last))


class LSTMForecaster(_BaseNNForecaster):
    """LSTM forecaster: 3-step sequence → LSTM(50, dropout=0.2) → linear out."""

    def __init__(
        self,
        n_lags: int = 3,
        hidden_units: int = 50,
        epochs: int = 100,
        early_stop_patience: int = 10,
        dropout: float = 0.2,
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
        self.dropout = dropout

    def _build_dataset(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_lags, T = self.n_lags, len(z)
        # Standard time-major sequence: position 0 is the OLDEST element.
        X = np.stack([z[i: i + n_lags] for i in range(T - n_lags)], axis=0)
        X = X.reshape(-1, n_lags, 1).astype(np.float32)
        Y = z[n_lags:T].reshape(-1, 1).astype(np.float32)
        return X, Y

    def _build_input_for_forecast(self, z: np.ndarray) -> np.ndarray:
        return z[-self.n_lags:].reshape(self.n_lags, 1).astype(np.float32)

    def _build_model(self) -> nn.Module:
        return _LSTMNet(self.hidden_units, self.dropout)


# ---------------------------------------------------------------------------
# NARX — Nonlinear AutoRegressive with eXogenous inputs (Bucci eq. 14)
# ---------------------------------------------------------------------------

class _ColumnScaler:
    """Per-column z-score scaler for the NARX feature matrix."""

    def fit(self, X: np.ndarray) -> "_ColumnScaler":
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0)
        self.sd[self.sd < 1e-8] = 1e-8
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mu) / self.sd).astype(np.float32)


class NARXForecaster:
    """NARX: lagged log-RV + lagged exogenous macro → 1 hidden sigmoid layer.

    Architecture follows Bucci eq. 14: input is
    ``[y_{t-1}, …, y_{t-p}, x_{t-1}]`` (the exogenous block is the 1-day-lagged
    macro vector), one hidden layer of ``hidden_units`` sigmoid neurons, and a
    linear output. Defaults (7 lags, 7 hidden, sigmoid) match the project's
    NAR so that NARX is *exactly* NAR + exogenous block.

    Training mirrors :class:`_BaseNNForecaster`: per-fit reseed, chronological
    80/20 validation split, Adam, MSE on the standardised target, early
    stopping on validation loss.
    """

    def __init__(
        self,
        n_lags: int = 7,
        hidden_units: int = 7,
        epochs: int = 100,
        early_stop_patience: int = 10,
        batch_size: int = 32,
        lr: float = 1e-3,
        seed: int = 42,
        device: str = "cpu",
        val_frac: float = 0.20,
    ) -> None:
        self.n_lags = n_lags
        self.hidden_units = hidden_units
        self.epochs = epochs
        self.early_stop_patience = early_stop_patience
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.device = torch.device(device)
        self.val_frac = val_frac
        self.x_scaler = _ColumnScaler()
        self.y_scaler = _StandardScaler()
        self.model: nn.Module | None = None
        self.history = TrainingHistory()

    def _build_dataset(self, y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Feature row t (for t = n_lags .. T-1): [y_{t-1}..y_{t-n_lags}, X[t]].
        # X is assumed ALREADY 1-day-lagged and aligned to y (X[t] is the macro
        # state known before y_t is realised), so it is used directly — no
        # second lag here. This keeps NARX and ARFIMAX on an identical exog
        # convention.
        n = len(y)
        feats, targs = [], []
        for t in range(self.n_lags, n):
            y_lags = y[t - self.n_lags: t][::-1]   # most recent first
            feats.append(np.concatenate([y_lags, X[t]]))
            targs.append(y[t])
        return np.asarray(feats, dtype=np.float64), np.asarray(targs, dtype=np.float64)

    def _build_model(self, input_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_units),
            nn.Sigmoid(),
            nn.Linear(self.hidden_units, 1),
        )

    def fit(self, y_train: ArrayLike, X_train: np.ndarray) -> "NARXForecaster":
        _seed_all(self.seed)
        y = np.asarray(y_train, dtype=np.float64).ravel()
        X = np.asarray(X_train, dtype=np.float64)
        if len(y) <= self.n_lags + 10:
            raise ValueError(f"need > {self.n_lags + 10} train obs, got {len(y)}")

        feats, targs = self._build_dataset(y, X)
        # Standardise features (per-column) and target on the training rows only.
        self.x_scaler.fit(feats)
        self.y_scaler.fit(targs)
        Xz = self.x_scaler.transform(feats)
        Yz = self.y_scaler.transform(targs).reshape(-1, 1).astype(np.float32)

        cut = int((1.0 - self.val_frac) * len(Xz))
        Xtr, Ytr = Xz[:cut], Yz[:cut]
        Xva, Yva = Xz[cut:], Yz[cut:]

        model = self._build_model(Xz.shape[1]).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        Xtr_t = torch.from_numpy(Xtr).to(self.device)
        Ytr_t = torch.from_numpy(Ytr).to(self.device)
        Xva_t = torch.from_numpy(Xva).to(self.device)
        Yva_t = torch.from_numpy(Yva).to(self.device)

        history = TrainingHistory()
        best_val, best_state, patience = float("inf"), None, 0
        n_train = len(Xtr)
        generator = torch.Generator(device="cpu").manual_seed(self.seed)

        for epoch in range(self.epochs):
            model.train()
            perm = torch.randperm(n_train, generator=generator)
            batch_losses = []
            for i in range(0, n_train, self.batch_size):
                idx = perm[i: i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(model(Xtr_t[idx]), Ytr_t[idx])
                loss.backward()
                opt.step()
                batch_losses.append(float(loss.detach()))
            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(Xva_t), Yva_t)) if len(Xva) else float(np.mean(batch_losses))
            history.train.append(float(np.mean(batch_losses)))
            history.val.append(val_loss)
            if val_loss < best_val - 1e-8:
                best_val, history.best_epoch, patience = val_loss, epoch, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= self.early_stop_patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self.model = model
        self.history = history
        return self

    def forecast(self, y_history: ArrayLike, X_history: np.ndarray, x_next: np.ndarray) -> float:
        if self.model is None:
            raise RuntimeError("call fit() before forecast()")
        y = np.asarray(y_history, dtype=np.float64).ravel()
        x_next = np.asarray(x_next, dtype=np.float64).ravel()
        y_lags = y[-self.n_lags:][::-1]
        feat = np.concatenate([y_lags, x_next]).reshape(1, -1)
        feat_z = self.x_scaler.transform(feat)
        with torch.no_grad():
            z_hat = float(self.model(torch.from_numpy(feat_z).to(self.device)).item())
        return float(self.y_scaler.inverse_transform(np.array([z_hat], dtype=np.float32))[0])


__all__ = [
    "FNNForecaster",
    "LSTMForecaster",
    "NARForecaster",
    "NARXForecaster",
    "TrainingHistory",
]
