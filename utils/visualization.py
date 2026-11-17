"""
Visualization utilities for sentiment analysis results using Plotly.
"""

from __future__ import annotations

import plotly.graph_objects as go

SENTIMENT_COLORS = {
    "POSITIVE": "#22c55e",
    "NEGATIVE": "#ef4444",
    "NEUTRAL": "#6b7280",
}

DEFAULT_COLOR = "#3b82f6"


def create_probability_chart(probabilities: dict[str, float]) -> go.Figure:
    """
    Create a horizontal bar chart showing class probabilities.

    Args:
        probabilities: Mapping of label -> probability (0–1).

    Returns:
        Plotly Figure object.
    """
    labels = list(probabilities.keys())
    values = [v * 100 for v in probabilities.values()]
    colors = [SENTIMENT_COLORS.get(l, DEFAULT_COLOR) for l in labels]

    fig = go.Figure(
        data=[
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker_color=colors,
                text=[f"{v:.1f}%" for v in values],
                textposition="auto",
                width=0.5,
            )
        ]
    )

    fig.update_layout(
        title="Class Probability Distribution",
        xaxis_title="Probability (%)",
        xaxis_range=[0, 100],
        yaxis_title="",
        showlegend=False,
        height=300,
        margin=dict(l=20, r=20, t=50, b=40),
        font=dict(size=13),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb")
    return fig


def create_sentiment_gauge(label: str, confidence: float) -> go.Figure:
    """
    Create a semi-circular gauge showing confidence for the predicted sentiment.

    Args:
        label: Predicted sentiment label.
        confidence: Confidence score (0–1).

    Returns:
        Plotly Figure object.
    """
    color = SENTIMENT_COLORS.get(label, DEFAULT_COLOR)
    pct = confidence * 100

    fig = go.Figure(
        data=[
            go.Indicator(
                mode="gauge+number",
                value=pct,
                number={"suffix": "%", "font": {"size": 36}},
                title={
                    "text": f"<b>{label}</b>",
                    "font": {"size": 20},
                },
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": color, "thickness": 0.3},
                    "bgcolor": "#f3f4f6",
                    "borderwidth": 2,
                    "bordercolor": "#d1d5db",
                    "steps": [
                        {"range": [0, 50], "color": "#f9fafb"},
                        {"range": [50, 100], "color": "#f3f4f6"},
                    ],
                },
            )
        ]
    )

    fig.update_layout(
        height=250,
        margin=dict(l=30, r=30, t=60, b=20),
        paper_bgcolor="white",
    )
    return fig


def create_batch_distribution_chart(labels: list[str]) -> go.Figure:
    """
    Create a pie chart showing the distribution of sentiments in a batch.

    Args:
        labels: List of predicted sentiment labels.

    Returns:
        Plotly Figure object.
    """
    counts: dict[str, int] = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1

    sorted_items = sorted(counts.items(), key=lambda x: -x[1])
    pie_labels = [k for k, _ in sorted_items]
    pie_values = [v for _, v in sorted_items]
    pie_colors = [SENTIMENT_COLORS.get(l, DEFAULT_COLOR) for l in pie_labels]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=pie_labels,
                values=pie_values,
                marker=dict(colors=pie_colors),
                textinfo="label+percent",
                textfont=dict(size=14),
                hole=0.4,
            )
        ]
    )

    fig.update_layout(
        title="Batch Sentiment Distribution",
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1),
        paper_bgcolor="white",
    )
    return fig