"""Logistic Smooth Transition AutoRegressive (LSTAR) model.

Two-regime LSTAR (Teräsvirta 1994; van Dijk, Teräsvirta, Franses 2002):

    y_t = (1 − F(z_t; γ, c)) · (φ_1' x_t)
          + F(z_t; γ, c)     · (φ_2' x_t) + ε_t,

with logistic transition

    F(z_t; γ, c) = 1 / (1 + exp(−γ_std · (z_t − c))),

where ``γ_std = γ / std(z)`` removes the scale dependence of γ (Teräsvirta
1994 §3). The transition variable here is ``z_t = y_{t-1}``; both regimes
share the same AR(p) lag structure ``x_t = [1, y_{t-1}, …, y_{t-p}]``.

Estimation is by conditional least squares with multiple random starts —
LSTAR likelihoods are famously multi-modal, so the 20-start protocol from
the project spec is required to get reproducible solutions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import minimize


ArrayLike = Sequence[float] | np.ndarray


def _logistic(z: np.ndarray, gamma: float, c: float, scale: float) -> np.ndarray:
    """F(z; γ, c) on a vector ``z``; ``scale = std(z)`` to make γ scale-free."""
    return 1.0 / (1.0 + np.exp(-(gamma / scale) * (z - c)))


@dataclass
class LSTAR:
    """2-regime LSTAR forecaster with the project's unified API."""

    p: int = 1
    n_starts: int = 20
    max_iter: int = 200
    seed: int = 42

    # learned attributes
    phi1_: np.ndarray = field(default_factory=lambda: np.zeros(0))
    phi2_: np.ndarray = field(default_factory=lambda: np.zeros(0))
    gamma_: float = 0.0
    c_: float = 0.0
    z_scale_: float = 1.0
    sse_: float = float("inf")
    converged_: bool = False
    n_successful_starts_: int = 0

    # -- helpers -----------------------------------------------------------
    def _build_design(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (Y, X, z) where Y = y_t, X = [1, y_{t-1}, …, y_{t-p}], z = y_{t-1}."""
        T = len(y)
        if T <= self.p + 5:
            raise ValueError(f"need > {self.p + 5} observations, got {T}")
        Y = y[self.p: T].astype(float)
        cols = [np.ones(T - self.p)]
        for k in range(1, self.p + 1):
            cols.append(y[self.p - k: T - k])
        X = np.column_stack(cols).astype(float)
        z = y[self.p - 1: T - 1].astype(float)  # transition variable = y_{t-1}
        return Y, X, z

    def _sse(self, params: np.ndarray, Y: np.ndarray, X: np.ndarray, z: np.ndarray) -> float:
        n_phi = self.p + 1
        phi1 = params[:n_phi]
        phi2 = params[n_phi: 2 * n_phi]
        gamma = np.exp(params[-2])  # enforce γ > 0
        c = params[-1]
        F = _logistic(z, gamma, c, self.z_scale_)
        y_hat = (1.0 - F) * (X @ phi1) + F * (X @ phi2)
        resid = Y - y_hat
        return float(resid @ resid)

    # -- public API --------------------------------------------------------
    def fit(self, y_train: ArrayLike) -> "LSTAR":
        rng = np.random.default_rng(self.seed)
        y = np.asarray(y_train, dtype=float).ravel()
        Y, X, z = self._build_design(y)
        self.z_scale_ = max(float(np.std(z)), 1e-6)
        n_phi = self.p + 1
        n_params = 2 * n_phi + 2
        ols_init, *_ = np.linalg.lstsq(X, Y, rcond=None)

        best_sse = float("inf")
        best_params: np.ndarray | None = None
        n_success = 0

        for _ in range(self.n_starts):
            phi1_init = ols_init + 0.1 * rng.standard_normal(n_phi)
            phi2_init = ols_init + 0.1 * rng.standard_normal(n_phi)
            log_gamma_init = rng.uniform(np.log(1.0), np.log(20.0))  # γ ∈ [1, 20]
            c_init = float(np.quantile(z, rng.uniform(0.25, 0.75)))
            x0 = np.concatenate([phi1_init, phi2_init, [log_gamma_init, c_init]])

            # Bounds: only the log-γ and c parameters are constrained.
            # log γ ∈ [log 0.1, log 50] keeps the transition strictly smooth
            # (the bound at 50 is loose enough that a near-threshold fit
            # still hits but well-behaved logistics dominate).
            bounds = (
                [(None, None)] * (2 * n_phi)
                + [(np.log(0.1), np.log(50.0))]
                + [(float(z.min()), float(z.max()))]
            )
            try:
                res = minimize(
                    self._sse, x0,
                    args=(Y, X, z),
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": self.max_iter, "ftol": 1e-9},
                )
            except Exception:
                continue
            if not res.success:
                continue
            n_success += 1
            if res.fun < best_sse:
                best_sse = float(res.fun)
                best_params = res.x

        self.n_successful_starts_ = n_success
        if best_params is None:
            # Fall back to a degenerate linear AR(p) — all weight on regime 1.
            self.phi1_ = ols_init
            self.phi2_ = ols_init
            self.gamma_ = 1.0
            self.c_ = float(np.median(z))
            self.sse_ = self._sse(
                np.concatenate([ols_init, ols_init, [np.log(1.0), self.c_]]),
                Y, X, z,
            )
            self.converged_ = False
        else:
            self.phi1_ = best_params[:n_phi]
            self.phi2_ = best_params[n_phi: 2 * n_phi]
            self.gamma_ = float(np.exp(best_params[-2]))
            self.c_ = float(best_params[-1])
            self.sse_ = best_sse
            self.converged_ = True
        return self

    def transition_value(self, z: ArrayLike) -> np.ndarray:
        """F(z; γ, c) for diagnostics / plotting."""
        z_arr = np.asarray(z, dtype=float).ravel()
        return _logistic(z_arr, self.gamma_, self.c_, self.z_scale_)

    def forecast(self, y_history: ArrayLike, h: int = 1) -> float | np.ndarray:
        """Iterative h-step-ahead forecast."""
        y = list(np.asarray(y_history, dtype=float).ravel())
        out = np.empty(h, dtype=float)
        for step in range(h):
            x = np.concatenate([[1.0], [y[-k] for k in range(1, self.p + 1)]])
            z_t = y[-1]
            F = float(_logistic(np.array([z_t]), self.gamma_, self.c_, self.z_scale_)[0])
            y_hat = (1.0 - F) * float(x @ self.phi1_) + F * float(x @ self.phi2_)
            out[step] = y_hat
            y.append(y_hat)
        return float(out[0]) if h == 1 else out


class LSTARForecaster:
    """Thin wrapper exposing the project's ``fit / forecast`` API.

    The default ``p=1`` matches the spec ("p — AR order для каждого
    режима"); ``n_starts=20`` is the multi-start count. The wrapper is
    independent of :class:`LSTAR` only in name — both fit/forecast calls
    delegate straight through.
    """

    def __init__(self, p: int = 1, n_starts: int = 20, seed: int = 42) -> None:
        self.model = LSTAR(p=p, n_starts=n_starts, seed=seed)

    def fit(self, y_train: ArrayLike) -> "LSTARForecaster":
        self.model.fit(y_train)
        return self

    def forecast(self, y_history: ArrayLike) -> float:
        return float(self.model.forecast(y_history, h=1))

    @property
    def converged(self) -> bool:
        return self.model.converged_

    @property
    def params(self) -> dict:
        return {
            "model": f"LSTAR(p={self.model.p})",
            "phi1": self.model.phi1_.tolist(),
            "phi2": self.model.phi2_.tolist(),
            "gamma": self.model.gamma_,
            "c": self.model.c_,
            "n_successful_starts": self.model.n_successful_starts_,
            "converged": self.model.converged_,
        }


__all__ = ["LSTAR", "LSTARForecaster"]
