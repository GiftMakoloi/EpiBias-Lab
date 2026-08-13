from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_missingness_map(
    df: pd.DataFrame,
):
    """
    Heatmap showing missing-value locations.
    """

    missing = (
        df.isna()
        .astype(int)
    )

    fig = px.imshow(
        missing,
        aspect="auto",
        title="Missingness Pattern (1 = missing)",
    )

    fig.update_layout(
        height=360,
    )

    return fig


def plot_estimate_comparison(
    results_df: pd.DataFrame,
    true_value: float,
    title: str = "Effect Estimates",
):
    """
    Plot odds-ratio estimates with 95% confidence intervals
    and a vertical reference line showing the generating value.
    """

    df = results_df.dropna(
        subset=[
            "estimate",
            "ci_low",
            "ci_high",
        ]
    ).copy()

    fig = go.Figure()

    for _, row in df.iterrows():

        fig.add_trace(
            go.Scatter(
                x=[
                    row["estimate"]
                ],
                y=[
                    row["Method"]
                ],
                mode="markers",
                name=row["Method"],
                error_x={
                    "type": "data",
                    "array": [
                        row["ci_high"]
                        - row["estimate"]
                    ],
                    "arrayminus": [
                        row["estimate"]
                        - row["ci_low"]
                    ],
                    "visible": True,
                },
            )
        )

    fig.add_vline(
        x=true_value,
        line_dash="dash",
        annotation_text=(
            f"True generating OR = "
            f"{true_value:.2f}"
        ),
    )

    fig.update_layout(
        title=title,
        xaxis_title="Odds Ratio",
        yaxis_title="",
        height=420,
        showlegend=False,
    )

    return fig


def plot_bias_by_method(
    results_df: pd.DataFrame,
    title: str = "Absolute Bias by Method",
):
    """
    Bar chart of absolute relative bias.
    """

    df = results_df.copy()

    if "Absolute Bias %" not in df.columns:
        return go.Figure()

    fig = px.bar(
        df,
        x="Method",
        y="Absolute Bias %",
        title=title,
        text_auto=".1f",
    )

    fig.update_yaxes(
        title="Absolute bias (%)"
    )

    fig.update_layout(
        showlegend=False,
        height=360,
    )

    return fig


def plot_bias_curve(
    curve_df: pd.DataFrame,
):
    """
    Plot average absolute bias as missingness increases.
    """

    fig = px.line(
        curve_df,
        x="Missingness %",
        y="Absolute Bias %",
        color="Method",
        markers=True,
        title="Bias as Missingness Increases",
    )

    fig.update_layout(
        height=420,
    )

    return fig


def plot_distribution(
    df: pd.DataFrame,
    column: str,
    color: str | None = None,
):
    """
    Histogram for a selected variable.
    """

    fig = px.histogram(
        df,
        x=column,
        color=color,
        barmode="overlay",
        title=f"Distribution of {column}",
    )

    return fig


def plot_dag():
    """
    Simple DAG for the confounding laboratory.
    """

    fig = go.Figure()

    nodes = {
        "C": (0, 1),
        "E": (1, 0),
        "Y": (2, 1),
    }

    labels = {
        "C": "Confounder",
        "E": "Exposure",
        "Y": "Outcome",
    }

    edges = [
        ("C", "E"),
        ("C", "Y"),
        ("E", "Y"),
    ]

    for source, target in edges:

        x0, y0 = nodes[source]
        x1, y1 = nodes[target]

        fig.add_annotation(
            x=x1,
            y=y1,
            ax=x0,
            ay=y0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.2,
            arrowwidth=2,
        )

    for key, (x, y) in nodes.items():

        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text",
                marker_size=34,
                text=[labels[key]],
                textposition="bottom center",
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_xaxes(
        visible=False,
        range=[
            -0.5,
            2.5,
        ],
    )

    fig.update_yaxes(
        visible=False,
        range=[
            -0.5,
            1.5,
        ],
    )

    fig.update_layout(
        title="Confounding Structure",
        height=360,
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
    )

    return fig
