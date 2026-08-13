from __future__ import annotations

import numpy as np
import pandas as pd


def _logit(p: float) -> float:
    """
    Convert a probability to log-odds.
    """
    p = float(
        np.clip(
            p,
            1e-8,
            1 - 1e-8,
        )
    )

    return np.log(
        p / (1.0 - p)
    )


def _sigmoid(
    x: np.ndarray | float,
) -> np.ndarray | float:
    """
    Numerically stable logistic transformation.
    """
    x = np.clip(
        x,
        -35,
        35,
    )

    return 1.0 / (
        1.0 + np.exp(-x)
    )


def _solve_intercept(
    target_rate: float,
    linear_term: np.ndarray,
) -> float:
    """
    Find an intercept such that the mean predicted probability
    is approximately equal to target_rate.
    """

    target_rate = float(
        np.clip(
            target_rate,
            1e-6,
            1 - 1e-6,
        )
    )

    lo = -30.0
    hi = 30.0

    for _ in range(70):

        mid = (
            lo + hi
        ) / 2.0

        mean_probability = float(
            np.mean(
                _sigmoid(
                    mid + linear_term
                )
            )
        )

        if mean_probability < target_rate:
            lo = mid
        else:
            hi = mid

    return (
        lo + hi
    ) / 2.0


# ============================================================
# BASIC POPULATION
# ============================================================

def generate_population(
    n: int = 1000,
    exposure_prev: float = 0.30,
    outcome_prev_unexp: float = 0.10,
    true_or: float = 2.0,
    age_mean: float = 50.0,
    age_sd: float = 10.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic cross-sectional health population.

    Variables:
        age
        exposure
        outcome

    The outcome is generated from a logistic model where:
        true_or = exp(beta_exposure)
    """

    if n <= 0:
        raise ValueError(
            "n must be positive."
        )

    rng = np.random.default_rng(
        seed
    )

    age = np.clip(
        rng.normal(
            age_mean,
            age_sd,
            n,
        ),
        18,
        100,
    )

    exposure = rng.binomial(
        1,
        exposure_prev,
        n,
    )

    beta0 = _logit(
        outcome_prev_unexp
    )

    beta_exposure = np.log(
        true_or
    )

    logit_probability = (
        beta0
        + beta_exposure * exposure
    )

    probability = _sigmoid(
        logit_probability
    )

    outcome = rng.binomial(
        1,
        probability,
        n,
    )

    return pd.DataFrame(
        {
            "age": age,
            "exposure": exposure,
            "outcome": outcome,
        }
    )


# ============================================================
# CONFOUNDING POPULATION
# ============================================================

def generate_confounding_population(
    n: int = 2000,
    confounder_prev: float = 0.40,
    exposure_prev_target: float = 0.30,
    outcome_prev_unexposed: float = 0.10,
    true_or: float = 2.0,
    confounder_exposure_or: float = 2.2,
    confounder_outcome_or: float = 2.5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a population with a binary confounder C.

    Structure:

        C -> Exposure
        C -> Outcome
        Exposure -> Outcome

    true_or represents the causal Exposure -> Outcome odds ratio
    in the data-generating model.
    """

    rng = np.random.default_rng(
        seed
    )

    confounder = rng.binomial(
        1,
        confounder_prev,
        n,
    )

    beta_conf_exposure = np.log(
        confounder_exposure_or
    )

    exposure_intercept = _solve_intercept(
        exposure_prev_target,
        beta_conf_exposure * confounder,
    )

    p_exposure = _sigmoid(
        exposure_intercept
        + beta_conf_exposure * confounder
    )

    exposure = rng.binomial(
        1,
        p_exposure,
        n,
    )

    beta_exposure = np.log(
        true_or
    )

    beta_conf_outcome = np.log(
        confounder_outcome_or
    )

    baseline = _logit(
        outcome_prev_unexposed
    )

    p_outcome = _sigmoid(
        baseline
        + beta_exposure * exposure
        + beta_conf_outcome * confounder
    )

    outcome = rng.binomial(
        1,
        p_outcome,
        n,
    )

    return pd.DataFrame(
        {
            "confounder": confounder,
            "exposure": exposure,
            "outcome": outcome,
        }
    )


# ============================================================
# SELECTION-BIAS POPULATION
# ============================================================

def generate_selection_population(
    n: int = 5000,
    exposure_prev: float = 0.30,
    outcome_prev_unexp: float = 0.10,
    true_or: float = 2.0,
    selection_rate: float = 0.40,
    exposure_selection_log_or: float = 1.0,
    outcome_selection_log_or: float = 1.2,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a population where selection depends on both exposure
    and outcome.

    This provides a controlled selection/collider demonstration.
    """

    rng = np.random.default_rng(
        seed
    )

    exposure = rng.binomial(
        1,
        exposure_prev,
        n,
    )

    beta_outcome = np.log(
        true_or
    )

    baseline = _logit(
        outcome_prev_unexp
    )

    p_outcome = _sigmoid(
        baseline
        + beta_outcome * exposure
    )

    outcome = rng.binomial(
        1,
        p_outcome,
        n,
    )

    selection_linear = (
        exposure_selection_log_or * exposure
        + outcome_selection_log_or * outcome
    )

    selection_intercept = _solve_intercept(
        selection_rate,
        selection_linear,
    )

    p_selected = _sigmoid(
        selection_intercept
        + selection_linear
    )

    selected = rng.binomial(
        1,
        p_selected,
        n,
    )

    return pd.DataFrame(
        {
            "exposure": exposure,
            "outcome": outcome,
            "selected": selected,
        }
    )


# ============================================================
# MEASUREMENT ERROR
# ============================================================

def apply_exposure_misclassification(
    df: pd.DataFrame,
    sensitivity: float = 0.90,
    specificity: float = 0.90,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate an observed exposure variable subject to
    nondifferential misclassification.
    """

    if not 0 < sensitivity <= 1:
        raise ValueError(
            "Sensitivity must be between 0 and 1."
        )

    if not 0 < specificity <= 1:
        raise ValueError(
            "Specificity must be between 0 and 1."
        )

    rng = np.random.default_rng(
        seed
    )

    out = df.copy()

    true_exposure = (
        out["exposure"]
        .astype(int)
        .to_numpy()
    )

    observed_exposure = np.where(
        true_exposure == 1,
        rng.binomial(
            1,
            sensitivity,
            len(out),
        ),
        1
        - rng.binomial(
            1,
            specificity,
            len(out),
        ),
    )

    out[
        "exposure_observed"
    ] = observed_exposure

    return out
