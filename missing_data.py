from __future__ import annotations

import numpy as np
import pandas as pd

from simulation import (
    _sigmoid,
    _solve_intercept,
)


def introduce_missingness(
    df: pd.DataFrame,
    mechanism: str,
    missing_rate: float,
    missing_variable: str,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Introduce controlled MCAR, MAR, or MNAR missingness.

    MCAR:
        Missingness is independent of the variables.

    MAR:
        Missingness depends on an observed variable (age).

    MNAR:
        Missingness depends on the value of the variable
        being made missing.

    The intercept is calibrated so the expected overall
    missingness is approximately equal to missing_rate.
    """

    if not 0 < missing_rate < 1:
        raise ValueError(
            "missing_rate must be between 0 and 1."
        )

    mechanism = mechanism.upper()

    valid_mechanisms = {
        "MCAR",
        "MAR",
        "MNAR",
    }

    if mechanism not in valid_mechanisms:
        raise ValueError(
            "mechanism must be MCAR, MAR, or MNAR."
        )

    if missing_variable == "exposure":

        columns = [
            "exposure"
        ]

    elif missing_variable == "outcome":

        columns = [
            "outcome"
        ]

    elif missing_variable == "both":

        columns = [
            "exposure",
            "outcome",
        ]

    else:

        raise ValueError(
            "missing_variable must be "
            "exposure, outcome, or both."
        )

    rng = np.random.default_rng(
        seed
    )

    output = df.copy()

    for column in columns:

        if mechanism == "MCAR":

            missing_probability = np.full(
                len(output),
                missing_rate,
            )

        elif mechanism == "MAR":

            # Missingness depends on age,
            # which remains observed.
            age_sd = max(
                float(df["age"].std()),
                1e-8,
            )

            age_z = (
                df["age"]
                - df["age"].mean()
            ) / age_sd

            linear_component = (
                1.1 * age_z
            )

            intercept = _solve_intercept(
                missing_rate,
                linear_component.to_numpy(),
            )

            missing_probability = _sigmoid(
                intercept
                + linear_component
            )

        else:  # MNAR

            # Missingness depends on
            # the value that is itself
            # being hidden.
            source = (
                df[column]
                .astype(float)
                .to_numpy()
            )

            centered = (
                source
                - np.mean(source)
            )

            linear_component = (
                1.5 * centered
            )

            intercept = _solve_intercept(
                missing_rate,
                linear_component,
            )

            missing_probability = _sigmoid(
                intercept
                + linear_component
            )

        missing_indicator = (
            rng.random(len(output))
            < missing_probability
        )

        output.loc[
            missing_indicator,
            column,
        ] = np.nan

    return output
