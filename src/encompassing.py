"""Forecast-encompassing tests (Fair & Shiller 1989; Chong & Hendry 1986).

For two competing one-step-ahead forecasts ``f1`` and ``f2`` we run the
encompassing regression with HAC (Newey–West) standard errors:

    y_t = α_0 + α_1 · f1_t + α_2 · f2_t + u_t.

Interpretation:
  * if α_1 ≈ 1 and α_2 ≈ 0 ⇒ ``f1`` *encompasses* ``f2`` (``f2`` adds
    nothing once ``f1`` is in the regression);
  * if both 0 < α_1, α_2 < 1 with t-stats > 2 ⇒ each forecast contains
    information the other lacks (Bates–Granger 1969 combination is useful);
  * if both insignificant ⇒ neither model captures the target well.

The verdict thresholds follow the project spec (α_1 > 0.7 significant and
α_2 < 0.3 ⇒ "encompasses"; both in (0.3, 0.7) ⇒ "mixed").
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


_ENCOMPASS_HIGH = 0.70
_ENCOMPASS_LOW = 0.30
_NW_LAGS = 5  # Newey-West lag length; Bucci/Hansen use a few lags for h=1 daily data


def encompassing_test(
    y_actual,
    forecast_1,
    forecast_2,
    use_hac: bool = True,
    nw_lags: int = _NW_LAGS,
) -> dict:
    """OLS of ``y`` on ``[1, f1, f2]`` with HAC errors.

    Returns
    -------
    dict
        ``alpha_0, alpha_1, alpha_2`` — point estimates.
        ``se_1, se_2`` — HAC (or homoscedastic) standard errors of α_1, α_2.
        ``t_stat_1, t_stat_2`` — corresponding t-statistics.
        ``p_value_1, p_value_2`` — two-sided p-values from ``H_0: α_i = 0``.
        ``r2`` — R² of the encompassing regression.
        ``verdict`` — one of
            ``"model_1_encompasses_2"``, ``"model_2_encompasses_1"``,
            ``"mixed"`` (both informative), ``"neither"`` (both insignificant),
            ``"model_1_dominant"`` / ``"model_2_dominant"`` (one strongly
            significant, the other not, but coefficients don't cross the
            0.7/0.3 thresholds — a softer ranking).
    """
    y = np.asarray(y_actual, dtype=float).ravel()
    f1 = np.asarray(forecast_1, dtype=float).ravel()
    f2 = np.asarray(forecast_2, dtype=float).ravel()
    if not (y.shape == f1.shape == f2.shape):
        raise ValueError(f"shape mismatch: {y.shape}, {f1.shape}, {f2.shape}")

    X = sm.add_constant(np.column_stack([f1, f2]), has_constant="add")
    if use_hac:
        ols = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags})
    else:
        ols = sm.OLS(y, X).fit()

    a0, a1, a2 = ols.params
    se_a0, se_a1, se_a2 = ols.bse
    p0, p1, p2 = ols.pvalues
    t0, t1, t2 = ols.tvalues
    r2 = float(ols.rsquared)

    sig1 = p1 < 0.05
    sig2 = p2 < 0.05
    if sig1 and a1 > _ENCOMPASS_HIGH and (not sig2 or a2 < _ENCOMPASS_LOW):
        verdict = "model_1_encompasses_2"
    elif sig2 and a2 > _ENCOMPASS_HIGH and (not sig1 or a1 < _ENCOMPASS_LOW):
        verdict = "model_2_encompasses_1"
    elif sig1 and sig2 and (_ENCOMPASS_LOW < a1 < _ENCOMPASS_HIGH) and (_ENCOMPASS_LOW < a2 < _ENCOMPASS_HIGH):
        verdict = "mixed"
    elif sig1 and not sig2:
        verdict = "model_1_dominant"
    elif sig2 and not sig1:
        verdict = "model_2_dominant"
    elif not sig1 and not sig2:
        verdict = "neither"
    else:
        verdict = "mixed"

    return {
        "alpha_0": float(a0),
        "alpha_1": float(a1),
        "alpha_2": float(a2),
        "se_1": float(se_a1),
        "se_2": float(se_a2),
        "t_stat_1": float(t1),
        "t_stat_2": float(t2),
        "p_value_1": float(p1),
        "p_value_2": float(p2),
        "r2": r2,
        "verdict": verdict,
    }


def pairwise_encompassing_matrix(
    forecasts_dict: dict[str, np.ndarray],
    y_actual: np.ndarray,
    use_hac: bool = True,
    nw_lags: int = _NW_LAGS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Encompassing matrices over all ordered pairs of models.

    Cell ``(i, j)`` contains the test result with ``f1 = model_i`` and
    ``f2 = model_j``. The α_1 coefficient at row i, column j answers
    *"how much weight does model i carry after also accounting for model j"*.

    Returns
    -------
    coef_matrix : pd.DataFrame
        α_1 coefficients. Diagonal = 1 (by construction we never test
        a model against itself; entries are NaN).
    pvalue_matrix : pd.DataFrame
        Two-sided p-value for ``H_0: α_1 = 0``.
    verdict_matrix : pd.DataFrame
        Categorical verdict per the :func:`encompassing_test` rules.
    alpha2_matrix : pd.DataFrame
        α_2 coefficients (the *competitor*'s weight). Useful for the
        heatmap of "who is being absorbed by whom".
    """
    names = list(forecasts_dict.keys())
    coef = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
    alpha2 = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
    pval = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
    verdict = pd.DataFrame("self", index=names, columns=names, dtype=object)
    for i in names:
        for j in names:
            if i == j:
                continue
            res = encompassing_test(
                y_actual,
                forecasts_dict[i],
                forecasts_dict[j],
                use_hac=use_hac,
                nw_lags=nw_lags,
            )
            coef.loc[i, j] = res["alpha_1"]
            alpha2.loc[i, j] = res["alpha_2"]
            pval.loc[i, j] = res["p_value_1"]
            verdict.loc[i, j] = res["verdict"]
    return coef, pval, verdict, alpha2


__all__ = ["encompassing_test", "pairwise_encompassing_matrix"]
