from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from analysis import (
    add_bias_metrics,
    analyse_missing_data,
    run_confounding_analysis,
    run_measurement_analysis,
    run_selection_analysis,
)
from missing_data import introduce_missingness
from simulation import (
    apply_exposure_misclassification,
    generate_confounding_population,
    generate_population,
    generate_selection_population,
)
from visualizations import (
    plot_bias_by_method,
    plot_bias_curve,
    plot_dag,
    plot_estimate_comparison,
    plot_missingness_map,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EpiBias Lab",
    page_icon="🧪",
    layout="wide",
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🧪 EpiBias Lab")
st.sidebar.caption(
    "Interactive epidemiological simulation platform"
)

module = st.sidebar.radio(
    "Select Module",
    [
        "Home",
        "Missing Data Lab",
        "Confounding Lab",
        "Selection Bias Lab",
        "Measurement Error Lab",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "All records are synthetic. No real patient information is used."
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def display_results_table(
    results: pd.DataFrame,
    true_value: float,
) -> pd.DataFrame:
    """
    Prepare a consistent results table for display.
    """
    if results.empty:
        return results

    out = add_bias_metrics(results, true_value).copy()

    out["Coverage"] = out["Covers Truth"].map(
        {
            True: "Yes",
            False: "No",
        }
    )

    out = out.rename(
        columns={
            "estimate": "Odds Ratio",
            "ci_low": "95% CI Lower",
            "ci_high": "95% CI Upper",
            "p_value": "p-value",
            "n": "N",
        }
    )

    columns = [
        "Method",
        "Odds Ratio",
        "95% CI Lower",
        "95% CI Upper",
        "p-value",
        "N",
        "Bias %",
        "Absolute Bias %",
        "CI Width",
        "Coverage",
    ]

    return out[[c for c in columns if c in out.columns]]


def show_interpretation(
    results: pd.DataFrame,
    true_value: float,
    heading: str = "What happened?",
) -> None:
    """
    Display a plain-language interpretation of the results.
    """
    if results.empty:
        st.warning(
            "There are not enough valid observations to perform the requested analysis."
        )
        return

    out = add_bias_metrics(results, true_value)

    st.markdown(f"### {heading}")

    for _, row in out.iterrows():
        estimate = row["estimate"]
        bias = row["Bias %"]

        if pd.isna(estimate):
            continue

        direction = "above" if bias > 0 else "below"

        st.write(
            f"**{row['Method']}** produced an odds ratio of "
            f"**{estimate:.2f}**, compared with the generating value "
            f"of **{true_value:.2f}**. "
            f"The estimate was {abs(bias):.1f}% {direction} the generating value."
        )

    st.info(
        "EpiBias uses a controlled simulation, so the underlying truth is known. "
        "In real-world epidemiology, the true population parameter is usually unknown."
    )


# ============================================================
# HOME PAGE
# ============================================================

if module == "Home":

    st.title("🧪 EpiBias Lab")

    st.subheader(
        "Interactive Epidemiological Bias and Missing Data Simulation Platform"
    )

    st.markdown(
        """
        **EpiBias is a free, no-data-needed Streamlit app that simulates incomplete
        and imperfect health records to demonstrate how missing data, confounding,
        selection bias, measurement error, and other data problems can distort
        epidemiological results—and how appropriate biostatistical methods can
        help detect, quantify, and reduce those problems.**
        """
    )

    st.info(
        """
        ### Why synthetic data?

        Real patient-level health data can be restricted, delayed, unavailable,
        or unsuitable for experimentation.

        EpiBias instead generates synthetic health populations with a known
        data-generating process. This allows the user to deliberately introduce
        problems and compare the observed result with the underlying truth.
        """
    )

    st.markdown("### Core experiment")

    cols = st.columns(5)

    steps = [
        "1. Generate truth",
        "2. Introduce problem",
        "3. Analyse",
        "4. Apply method",
        "5. Compare with truth",
    ]

    for col, step in zip(cols, steps):
        col.success(step)

    st.markdown("---")

    st.markdown("### What makes EpiBias different?")

    st.write(
        """
        EpiBias is not designed primarily as a collection of statistical
        calculators. Its defining experiment is to create a known epidemiological
        truth, deliberately introduce a data problem, estimate the effect from
        the imperfect data, and quantify how far the estimate moved away from
        the truth.
        """
    )

    st.markdown("### Available laboratories")

    modules_df = pd.DataFrame(
        {
            "Laboratory": [
                "Missing Data Lab",
                "Confounding Lab",
                "Selection Bias Lab",
                "Measurement Error Lab",
            ],
            "Main concept": [
                "MCAR, MAR, MNAR",
                "Crude vs adjusted effect",
                "Selection and collider distortion",
                "Exposure misclassification",
            ],
        }
    )

    st.dataframe(
        modules_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# MISSING DATA LAB
# ============================================================

elif module == "Missing Data Lab":

    st.title("🔍 Missing Data Lab")

    st.write(
        """
        Explore how MCAR, MAR, and MNAR missingness can change epidemiological
        effect estimates.
        """
    )

    # --------------------------------------------------------
    # SETTINGS
    # --------------------------------------------------------

    with st.sidebar:

        st.subheader("Simulation settings")

        n = st.slider(
            "Sample size",
            200,
            10000,
            1000,
            100,
        )

        exposure_prev = st.slider(
            "Exposure prevalence",
            0.10,
            0.90,
            0.30,
            0.05,
        )

        outcome_prev_unexp = st.slider(
            "Outcome prevalence (unexposed)",
            0.02,
            0.50,
            0.10,
            0.01,
        )

        true_or = st.slider(
            "True generating odds ratio",
            1.0,
            5.0,
            2.0,
            0.1,
        )

        age_mean = st.slider(
            "Age mean",
            18,
            80,
            50,
            1,
        )

        age_sd = st.slider(
            "Age SD",
            5,
            20,
            10,
            1,
        )

        mechanism = st.selectbox(
            "Missingness mechanism",
            ["MCAR", "MAR", "MNAR"],
        )

        missing_rate = st.slider(
            "Target missingness",
            0.05,
            0.60,
            0.20,
            0.05,
        )

        missing_variable = st.selectbox(
            "Variable affected",
            ["exposure", "outcome", "both"],
        )

        seed = st.number_input(
            "Random seed",
            min_value=1,
            max_value=999999,
            value=42,
        )

        reps = st.slider(
            "Repeated simulation runs",
            10,
            500,
            100,
            10,
        )

    # --------------------------------------------------------
    # GENERATE ONE DATASET
    # --------------------------------------------------------

    true_df = generate_population(
        n=n,
        exposure_prev=exposure_prev,
        outcome_prev_unexp=outcome_prev_unexp,
        true_or=true_or,
        age_mean=age_mean,
        age_sd=age_sd,
        seed=int(seed),
    )

    observed_df = introduce_missingness(
        true_df,
        mechanism=mechanism,
        missing_rate=missing_rate,
        missing_variable=missing_variable,
        seed=int(seed) + 1,
    )

    if missing_variable == "both":
        measured_columns = ["exposure", "outcome"]
    else:
        measured_columns = [missing_variable]

    observed_missing_pct = (
        observed_df[measured_columns].isna().mean().mean() * 100
    )

    # --------------------------------------------------------
    # SUMMARY CARDS
    # --------------------------------------------------------

    st.markdown("### 1. Single simulation")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Generating OR",
        f"{true_or:.2f}",
    )

    c2.metric(
        "Observed missingness",
        f"{observed_missing_pct:.1f}%",
    )

    c3.metric(
        "Population N",
        f"{len(true_df):,}",
    )

    complete_records = int(
        observed_df[["exposure", "outcome"]]
        .dropna()
        .shape[0]
    )

    c4.metric(
        "Complete exposure/outcome records",
        f"{complete_records:,}",
    )

    # --------------------------------------------------------
    # DATA PREVIEW
    # --------------------------------------------------------

    left, right = st.columns(2)

    with left:

        st.markdown("#### True synthetic data")

        st.dataframe(
            true_df.head(10),
            use_container_width=True,
        )

    with right:

        st.markdown("#### Observed data after missingness")

        st.dataframe(
            observed_df.head(10),
            use_container_width=True,
        )

    # --------------------------------------------------------
    # MISSINGNESS VISUALISATION
    # --------------------------------------------------------

    st.plotly_chart(
        plot_missingness_map(
            observed_df[["age", "exposure", "outcome"]]
        ),
        use_container_width=True,
    )

    # --------------------------------------------------------
    # SINGLE-SIMULATION ANALYSIS
    # --------------------------------------------------------

    one_result = pd.DataFrame(
        analyse_missing_data(observed_df)
    )

    if one_result.empty:

        st.error(
            "The observed dataset is too sparse or degenerate "
            "for the requested odds-ratio analysis. "
            "Try reducing missingness or increasing the sample size."
        )

    else:

        display_table = display_results_table(
            one_result,
            true_or,
        )

        st.markdown("### Effect estimates")

        st.dataframe(
            display_table,
            use_container_width=True,
            hide_index=True,
        )

        st.plotly_chart(
            plot_estimate_comparison(
                one_result,
                true_or,
                "True vs Observed Effect Estimates",
            ),
            use_container_width=True,
        )

        st.plotly_chart(
            plot_bias_by_method(
                add_bias_metrics(
                    one_result,
                    true_or,
                ),
                "Absolute Bias by Method",
            ),
            use_container_width=True,
        )

        show_interpretation(
            one_result,
            true_or,
        )

    # --------------------------------------------------------
    # REPEATED SIMULATION
    # --------------------------------------------------------

    st.markdown("---")

    st.markdown("### 2. Repeated simulation")

    st.write(
        """
        One simulation can be affected by random variation. Repeated simulations
        estimate the average behaviour of each analysis method across many
        synthetic datasets.
        """
    )

    progress = st.progress(0)

    curve_rows = []

    rate_grid = np.linspace(
        max(0.05, missing_rate / 4),
        min(0.60, max(missing_rate, 0.10) * 2),
        7,
    )

    for idx, rate in enumerate(rate_grid):

        method_values = {
            "Complete-case": [],
            "Naive mode imputation": [],
        }

        for r in range(reps):

            sim_seed = (
                int(seed)
                + 10000 * idx
                + r
            )

            sim_df = generate_population(
                n=n,
                exposure_prev=exposure_prev,
                outcome_prev_unexp=outcome_prev_unexp,
                true_or=true_or,
                age_mean=age_mean,
                age_sd=age_sd,
                seed=sim_seed,
            )

            sim_df = introduce_missingness(
                sim_df,
                mechanism=mechanism,
                missing_rate=float(rate),
                missing_variable=missing_variable,
                seed=sim_seed + 1,
            )

            sim_results = pd.DataFrame(
                analyse_missing_data(sim_df)
            )

            if sim_results.empty:
                continue

            for _, row in sim_results.iterrows():

                estimate = row["estimate"]

                if pd.isna(estimate):
                    continue

                absolute_bias = (
                    abs(
                        (estimate - true_or)
                        / true_or
                    )
                    * 100
                )

                method_values.setdefault(
                    row["Method"],
                    [],
                ).append(
                    absolute_bias
                )

        for method, values in method_values.items():

            if values:

                curve_rows.append(
                    {
                        "Missingness %": rate * 100,
                        "Method": method,
                        "Absolute Bias %": float(
                            np.mean(values)
                        ),
                    }
                )

        progress.progress(
            (idx + 1)
            / len(rate_grid)
        )

    progress.empty()

    curve_df = pd.DataFrame(curve_rows)

    if not curve_df.empty:

        st.plotly_chart(
            plot_bias_curve(curve_df),
            use_container_width=True,
        )

        st.caption(
            """
            Each point represents mean absolute bias across repeated synthetic
            datasets. Because the generating value is known, EpiBias can directly
            quantify the distortion introduced by missingness.
            """
        )

    st.warning(
        """
        **Important:** MNAR is not generally identifiable from the observed data
        alone. In EpiBias it is treated as a controlled simulation scenario so
        that users can study what happens when missingness depends on the
        unobserved value.
        """
    )


# ============================================================
# CONFOUNDING LAB
# ============================================================

elif module == "Confounding Lab":

    st.title("🧩 Confounding Lab")

    st.write(
        """
        Generate a population with a known confounder and compare the crude
        association with the association obtained after adjustment.
        """
    )

    with st.sidebar:

        st.subheader("Simulation settings")

        n = st.slider(
            "Sample size",
            500,
            10000,
            3000,
            250,
        )

        conf_prev = st.slider(
            "Confounder prevalence",
            0.10,
            0.90,
            0.40,
            0.05,
        )

        exp_prev = st.slider(
            "Target exposure prevalence",
            0.10,
            0.80,
            0.30,
            0.05,
        )

        true_or = st.slider(
            "True causal OR",
            1.0,
            5.0,
            2.0,
            0.1,
        )

        c_to_e = st.slider(
            "Confounder → Exposure OR",
            1.0,
            5.0,
            2.2,
            0.1,
        )

        c_to_y = st.slider(
            "Confounder → Outcome OR",
            1.0,
            5.0,
            2.5,
            0.1,
        )

        seed = st.number_input(
            "Random seed",
            1,
            999999,
            42,
        )

    df = generate_confounding_population(
        n=n,
        confounder_prev=conf_prev,
        exposure_prev_target=exp_prev,
        outcome_prev_unexposed=0.10,
        true_or=true_or,
        confounder_exposure_or=c_to_e,
        confounder_outcome_or=c_to_y,
        seed=int(seed),
    )

    results = run_confounding_analysis(
        df,
        true_or,
    )

    st.plotly_chart(
        plot_dag(),
        use_container_width=True,
    )

    display_table = display_results_table(
        results,
        true_or,
    )

    st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True,
    )

    if not results.empty:

        st.plotly_chart(
            plot_estimate_comparison(
                results,
                true_or,
                "Crude vs Adjusted Effect",
            ),
            use_container_width=True,
        )

        st.plotly_chart(
            plot_bias_by_method(
                results,
                "Confounding Distortion",
            ),
            use_container_width=True,
        )

        show_interpretation(
            results,
            true_or,
        )

    st.info(
        """
        The generating exposure coefficient represents the causal effect specified
        by the simulation. The crude model omits the confounder, while the adjusted
        model includes the confounder.
        """
    )


# ============================================================
# SELECTION BIAS LAB
# ============================================================

elif module == "Selection Bias Lab":

    st.title("🎯 Selection Bias Lab")

    st.write(
        """
        Simulate a study in which selection depends on both exposure and outcome.
        This creates a controlled collider-selection experiment.
        """
    )

    with st.sidebar:

        st.subheader("Simulation settings")

        n = st.slider(
            "Population size",
            1000,
            20000,
            5000,
            500,
        )

        exp_prev = st.slider(
            "Exposure prevalence",
            0.10,
            0.80,
            0.30,
            0.05,
        )

        outcome_prev = st.slider(
            "Outcome prevalence (unexposed)",
            0.02,
            0.40,
            0.10,
            0.01,
        )

        true_or = st.slider(
            "True population OR",
            1.0,
            5.0,
            2.0,
            0.1,
        )

        selection_rate = st.slider(
            "Target selection rate",
            0.10,
            0.90,
            0.40,
            0.05,
        )

        e_sel = st.slider(
            "Effect of exposure on selection",
            0.0,
            3.0,
            1.0,
            0.1,
        )

        y_sel = st.slider(
            "Effect of outcome on selection",
            0.0,
            3.0,
            1.2,
            0.1,
        )

        seed = st.number_input(
            "Random seed",
            1,
            999999,
            42,
        )

    df = generate_selection_population(
        n=n,
        exposure_prev=exp_prev,
        outcome_prev_unexp=outcome_prev,
        true_or=true_or,
        selection_rate=selection_rate,
        exposure_selection_log_or=e_sel,
        outcome_selection_log_or=y_sel,
        seed=int(seed),
    )

    results = run_selection_analysis(
        df,
        true_or,
    )

    selected_rate = (
        df["selected"].mean()
        * 100
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Population N",
        f"{len(df):,}",
    )

    c2.metric(
        "Selected participants",
        f"{int(df['selected'].sum()):,}",
    )

    c3.metric(
        "Observed selection rate",
        f"{selected_rate:.1f}%",
    )

    st.dataframe(
        display_results_table(
            results,
            true_or,
        ),
        use_container_width=True,
        hide_index=True,
    )

    if not results.empty:

        st.plotly_chart(
            plot_estimate_comparison(
                results,
                true_or,
                "Population vs Selected Sample",
            ),
            use_container_width=True,
        )

        st.plotly_chart(
            plot_bias_by_method(
                results,
                "Selection-Induced Distortion",
            ),
            use_container_width=True,
        )

        show_interpretation(
            results,
            true_or,
        )

    st.warning(
        """
        This is a controlled selection-bias demonstration. Selection depends on
        both exposure and outcome. Conditioning on this selection process can
        change the observed association.
        """
    )


# ============================================================
# MEASUREMENT ERROR LAB
# ============================================================

elif module == "Measurement Error Lab":

    st.title("📏 Measurement Error Lab")

    st.write(
        """
        Introduce exposure misclassification and observe how sensitivity and
        specificity can affect the estimated association.
        """
    )

    with st.sidebar:

        st.subheader("Simulation settings")

        n = st.slider(
            "Sample size",
            500,
            10000,
            3000,
            250,
        )

        exp_prev = st.slider(
            "True exposure prevalence",
            0.10,
            0.80,
            0.30,
            0.05,
        )

        outcome_prev = st.slider(
            "Outcome prevalence (unexposed)",
            0.02,
            0.40,
            0.10,
            0.01,
        )

        true_or = st.slider(
            "True OR",
            1.0,
            5.0,
            2.0,
            0.1,
        )

        sens = st.slider(
            "Exposure sensitivity",
            0.50,
            1.00,
            0.90,
            0.01,
        )

        spec = st.slider(
            "Exposure specificity",
            0.50,
            1.00,
            0.90,
            0.01,
        )

        seed = st.number_input(
            "Random seed",
            1,
            999999,
            42,
        )

    true_df = generate_population(
        n=n,
        exposure_prev=exp_prev,
        outcome_prev_unexp=outcome_prev,
        true_or=true_or,
        age_mean=50,
        age_sd=10,
        seed=int(seed),
    )

    observed_df = apply_exposure_misclassification(
        true_df,
        sensitivity=sens,
        specificity=spec,
        seed=int(seed) + 1,
    )

    results = run_measurement_analysis(
        observed_df,
        true_or,
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Sensitivity",
        f"{sens:.2f}",
    )

    c2.metric(
        "Specificity",
        f"{spec:.2f}",
    )

    c3.metric(
        "Generating OR",
        f"{true_or:.2f}",
    )

    st.dataframe(
        display_results_table(
            results,
            true_or,
        ),
        use_container_width=True,
        hide_index=True,
    )

    if not results.empty:

        st.plotly_chart(
            plot_estimate_comparison(
                results,
                true_or,
                "True vs Misclassified Exposure",
            ),
            use_container_width=True,
        )

        st.plotly_chart(
            plot_bias_by_method(
                results,
                "Measurement-Error Distortion",
            ),
            use_container_width=True,
        )

        show_interpretation(
            results,
            true_or,
        )

    st.info(
        """
        The observed exposure is an imperfect measurement of the true exposure.
        Changing sensitivity and specificity lets the user explore how
        misclassification can move the estimated association away from the
        generating value.
        """
    )
