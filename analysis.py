from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy.stats import (
    chi2_contingency,
    fisher_exact,
)


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

def _fit_logit(
    df: pd.DataFrame,
    predictors: list[str],
) -> dict | None:
    """
    Fit a logistic regression and return the effect estimate
    for the first predictor.
    """

    required_columns = (
        predictors
        + ["outcome"]
    )

    data = (
        df[required_columns]
        .dropna()
        .copy()
    )

    minimum_n = max(
        30,
        10 * len(predictors),
    )

    if len(data) < minimum_n:
        return None

    if data["outcome"].nunique() < 2:
        return None

    for predictor in predictors:

        if data[predictor].nunique() < 2:
            return None

    try:

        x = sm.add_constant(
            data[predictors],
            has_constant="add",
        )

        model = sm.Logit(
            data["outcome"],
            x,
        ).fit(
            disp=0,
            maxiter=200,
        )

        term = predictors[0]

        coefficient = float(
            model.params[term]
        )

        confidence_interval = (
            model.conf_int()
            .loc[term]
            .to_numpy(
                dtype=float
            )
        )

        return {
            "estimate": float(
                np.exp(
                    coefficient
                )
            ),
            "ci_low": float(
                np.exp(
                    confidence_interval[0]
                )
            ),
            "ci_high": float(
                np.exp(
                    confidence_interval[1]
                )
            ),
            "p_value": float(
                model.pvalues[term]
            ),
            "n": int(
                len(data)
            ),
        }

    except (
        ValueError,
        np.linalg.LinAlgError,
        FloatingPointError,
    ):

        return None


# ============================================================
# 2 x 2 ODDS RATIO
# ============================================================

def odds_ratio_2x2(
    df: pd.DataFrame,
    exposure: str = "exposure",
    outcome: str = "outcome",
) -> dict | None:
    """
    Calculate an odds ratio and approximate 95% CI
    from a 2x2 table.
    """

    data = (
        df[
            [
                exposure,
                outcome,
            ]
        ]
        .dropna()
        .copy()
    )

    if len(data) == 0:
        return None

    if data[exposure].nunique() < 2:
        return None

    if data[outcome].nunique() < 2:
        return None

    table = (
        pd.crosstab(
            data[exposure],
            data[outcome],
        )
        .reindex(
            index=[0, 1],
            columns=[0, 1],
            fill_value=0,
        )
    )

    a = int(
        table.loc[1, 1]
    )

    b = int(
        table.loc[1, 0]
    )

    c = int(
        table.loc[0, 1]
    )

    d = int(
        table.loc[0, 0]
    )

    # If any cell is zero, the regular
    # Wald CI is not defined.
    if (
        a == 0
        or b == 0
        or c == 0
        or d == 0
    ):

        odds_ratio = np.nan
        ci_low = np.nan
        ci_high = np.nan

        _, p_value = fisher_exact(
            table
        )

    else:

        odds_ratio = (
            a * d
        ) / (
            b * c
        )

        standard_error = np.sqrt(
            1 / a
            + 1 / b
            + 1 / c
            + 1 / d
        )

        log_or = np.log(
            odds_ratio
        )

        ci_low = np.exp(
            log_or
            - 1.96
            * standard_error
        )

        ci_high = np.exp(
            log_or
            + 1.96
            * standard_error
        )

        _, p_value, _, _ = (
            chi2_contingency(
                table,
                correction=False,
            )
        )

    return {
        "estimate": float(
            odds_ratio
        ),
        "ci_low": float(
            ci_low
        ),
        "ci_high": float(
            ci_high
        ),
        "p_value": float(
            p_value
        ),
        "n": int(
            len(data)
        ),
    }


# ============================================================
# MISSING DATA ANALYSIS
# ============================================================

def analyse_missing_data(
    df_obs: pd.DataFrame,
) -> list[dict]:
    """
    Run simple baseline analyses on data with missing values.

    Methods:
        1. Complete-case analysis
        2. Naive mode imputation
    """

    results: list[dict] = []

    complete_case = odds_ratio_2x2(
        df_obs
    )

    if complete_case:

        results.append(
            {
                "Method": "Complete-case",
                **complete_case,
            }
        )

    imputed = (
        df_obs.copy()
    )

    for column in [
        "exposure",
        "outcome",
    ]:

        if imputed[column].isna().any():

            mode = (
                imputed[column]
                .mode(
                    dropna=True
                )
            )

            if mode.empty:
                continue

            imputed[column] = (
                imputed[column]
                .fillna(
                    mode.iloc[0]
                )
            )

    naive = odds_ratio_2x2(
        imputed
    )

    if naive:

        results.append(
            {
                "Method":
                    "Naive mode imputation",
                **naive,
            }
        )

    return results


# ============================================================
# BIAS METRICS
# ============================================================

def add_bias_metrics(
    results: pd.DataFrame,
    true_value: float,
) -> pd.DataFrame:
    """
    Add relative bias, absolute bias,
    CI width, and truth coverage.
    """

    out = (
        results.copy()
    )

    out["Bias %"] = (
        (
            out["estimate"]
            - true_value
        )
        / true_value
        * 100.0
    )

    out["Absolute Bias %"] = (
        out["Bias %"].abs()
    )

    out["CI Width"] = (
        out["ci_high"]
        - out["ci_low"]
    )

    out["Covers Truth"] = (
        (out["ci_low"] <= true_value)
        & (
            out["ci_high"]
            >= true_value
        )
    )

    return out


# ============================================================
# CONFOUNDING
# ============================================================

def run_confounding_analysis(
    df: pd.DataFrame,
    true_or: float,
) -> pd.DataFrame:
    """
    Compare crude and confounder-adjusted logistic models.
    """

    rows = []

    crude = _fit_logit(
        df,
        ["exposure"],
    )

    if crude:

        rows.append(
            {
                "Method":
                    "Crude logistic regression",
                **crude,
            }
        )

    adjusted = _fit_logit(
        df,
        [
            "exposure",
            "confounder",
        ],
    )

    if adjusted:

        rows.append(
            {
                "Method":
                    "Adjusted for confounder",
                **adjusted,
            }
        )

    output = pd.DataFrame(
        rows
    )

    if output.empty:
        return output

    return add_bias_metrics(
        output,
        true_or,
    )


# ============================================================
# SELECTION BIAS
# ============================================================

def run_selection_analysis(
    df: pd.DataFrame,
    true_or: float,
) -> pd.DataFrame:
    """
    Compare the full population association
    with the selected-sample association.
    """

    rows = []

    full_population = odds_ratio_2x2(
        df
    )

    selected_sample = odds_ratio_2x2(
        df[df["selected"] == 1]
    )

    if full_population:

        rows.append(
            {
                "Method":
                    "Full population",
                **full_population,
            }
        )

    if selected_sample:

        rows.append(
            {
                "Method":
                    "Selected sample",
                **selected_sample,
            }
        )

    output = pd.DataFrame(
        rows
    )

    if output.empty:
        return output

    return add_bias_metrics(
        output,
        true_or,
    )


# ============================================================
# MEASUREMENT ERROR
# ============================================================

def run_measurement_analysis(
    df: pd.DataFrame,
    true_or: float,
) -> pd.DataFrame:
    """
    Compare the association using the true exposure
    with the association using the misclassified exposure.
    """

    rows = []

    true_result = odds_ratio_2x2(
        df,
        exposure="exposure",
        outcome="outcome",
    )

    observed_result = odds_ratio_2x2(
        df,
        exposure="exposure_observed",
        outcome="outcome",
    )

    if true_result:

        rows.append(
            {
                "Method":
                    "True exposure",
                **true_result,
            }
        )

    if observed_result:

        rows.append(
            {
                "Method":
                    "Misclassified exposure",
                **observed_result,
            }
        )

    output = pd.DataFrame(
        rows
    )

    if output.empty:
        return output

    return add_bias_metrics(
        output,
        true_or,
    )
