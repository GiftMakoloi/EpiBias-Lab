import sys
from pathlib import Path

import numpy as np


ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)


from analysis import (  # noqa: E402
    analyse_missing_data,
    run_confounding_analysis,
    run_measurement_analysis,
    run_selection_analysis,
)

from missing_data import introduce_missingness  # noqa: E402

from simulation import (  # noqa: E402
    apply_exposure_misclassification,
    generate_confounding_population,
    generate_population,
    generate_selection_population,
)


def test_population_reproducibility():

    a = generate_population(
        seed=123
    )

    b = generate_population(
        seed=123
    )

    assert a.equals(b)


def test_mcar_missingness():

    df = generate_population(
        seed=1
    )

    out = introduce_missingness(
        df,
        "MCAR",
        0.20,
        "exposure",
        seed=2,
    )

    observed_rate = (
        out["exposure"]
        .isna()
        .mean()
    )

    assert (
        0.12
        < observed_rate
        < 0.28
    )


def test_mar_missingness():

    df = generate_population(
        seed=1
    )

    out = introduce_missingness(
        df,
        "MAR",
        0.20,
        "exposure",
        seed=2,
    )

    observed_rate = (
        out["exposure"]
        .isna()
        .mean()
    )

    assert (
        0.12
        < observed_rate
        < 0.28
    )


def test_mnar_missingness():

    df = generate_population(
        seed=1
    )

    out = introduce_missingness(
        df,
        "MNAR",
        0.20,
        "exposure",
        seed=2,
    )

    observed_rate = (
        out["exposure"]
        .isna()
        .mean()
    )

    assert (
        0.12
        < observed_rate
        < 0.28
    )


def test_confounding_analysis():

    df = generate_confounding_population(
        n=3000,
        seed=7,
    )

    result = run_confounding_analysis(
        df,
        true_or=2.0,
    )

    assert set(
        result["Method"]
    ) == {
        "Crude logistic regression",
        "Adjusted for confounder",
    }

    assert np.all(
        np.isfinite(
            result["estimate"]
        )
    )


def test_selection_bias():

    df = generate_selection_population(
        n=3000,
        seed=9,
    )

    result = run_selection_analysis(
        df,
        true_or=2.0,
    )

    assert set(
        result["Method"]
    ) == {
        "Full population",
        "Selected sample",
    }

    assert (
        df["selected"]
        .mean()
        > 0
    )


def test_measurement_error():

    df = generate_population(
        n=3000,
        seed=4,
    )

    out = apply_exposure_misclassification(
        df,
        sensitivity=0.90,
        specificity=0.90,
        seed=5,
    )

    assert (
        "exposure_observed"
        in out.columns
    )

    assert set(
        out["exposure_observed"]
        .unique()
    ).issubset(
        {
            0,
            1,
        }
    )

    result = run_measurement_analysis(
        out,
        true_or=2.0,
    )

    assert len(result) == 2


def test_missing_data_analysis():

    df = generate_population(
        seed=5
    )

    out = introduce_missingness(
        df,
        "MCAR",
        0.20,
        "both",
        seed=6,
    )

    result = analyse_missing_data(
        out
    )

    method_names = {
        row["Method"]
        for row in result
    }

    assert method_names <= {
        "Complete-case",
        "Naive mode imputation",
    }

    assert len(result) >= 1
