"""Reusable Plotly chart components for AlgoTrade dashboard."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.dashboard.theme import (
    AMBER, GREEN, RED, BASE, SURFACE, BORDER, TEXT, PLOTLY_TEMPLATE, SIGNAL_COLORS
)


def equity_curve(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    title: str = "Portfolio Equity Curve",
) -> go.Figure:
    equity = (1 + returns).cumprod() * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values,
        mode="lines", name="Strategy",
        line=dict(color=AMBER, width=2),
    ))
    if benchmark_returns is not None:
        bm = (1 + benchmark_returns).cumprod() * 100
        fig.add_trace(go.Scatter(
            x=bm.index, y=bm.values,
            mode="lines", name="Benchmark",
            line=dict(color=TEXT, width=1, dash="dash"),
        ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE, title=title,
        xaxis_title="Date", yaxis_title="Value (Base=100)",
        height=350,
    )
    return fig


def drawdown_chart(returns: pd.Series, title: str = "Drawdown") -> go.Figure:
    equity = (1 + returns).cumprod()
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        mode="lines", fill="tozeroy",
        name="Drawdown",
        line=dict(color=RED, width=1),
        fillcolor="rgba(255,23,68,0.15)",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE, title=title,
        yaxis_title="Drawdown %", height=200,
    )
    return fig


def signal_overlay(
    close: pd.Series,
    signals: pd.Series,
    title: str = "Price + Signals",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=close.index, y=close.values,
        mode="lines", name="Price",
        line=dict(color=TEXT, width=1),
    ))
    for sig_val, color, name in [(1, GREEN, "Long"), (-1, RED, "Short")]:
        mask = signals == sig_val
        if mask.any():
            fig.add_trace(go.Scatter(
                x=close.index[mask], y=close.values[mask],
                mode="markers", name=name,
                marker=dict(color=color, size=6, symbol="circle"),
            ))
    fig.update_layout(template=PLOTLY_TEMPLATE, title=title, height=300)
    return fig


def feature_importance_bar(
    importance: pd.Series,
    title: str = "Feature Importance",
    n: int = 15,
) -> go.Figure:
    top = importance.head(n).sort_values()
    fig = go.Figure(go.Bar(
        x=top.values, y=top.index, orientation="h",
        marker_color=AMBER,
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE, title=title,
        xaxis_title="Importance", height=max(250, n * 22),
    )
    return fig


def regime_timeline(
    regime_series: pd.Series,
    title: str = "Market Regime",
) -> go.Figure:
    color_map = {0: GREEN, 1: AMBER, 2: RED}
    label_map = {0: "Low-Vol", 1: "Mid-Vol", 2: "High-Vol"}
    fig = go.Figure()
    for r_val in [0, 1, 2]:
        mask = regime_series == r_val
        if mask.any():
            fig.add_trace(go.Scatter(
                x=regime_series.index[mask],
                y=[r_val] * mask.sum(),
                mode="markers",
                name=label_map[r_val],
                marker=dict(color=color_map[r_val], size=4, symbol="square"),
            ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE, title=title,
        yaxis=dict(tickvals=[0, 1, 2], ticktext=list(label_map.values())),
        height=150,
    )
    return fig


def correlation_heatmap(returns_df: pd.DataFrame, title: str = "Correlation Matrix") -> go.Figure:
    corr = returns_df.corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale=[[0, RED], [0.5, SURFACE], [1, GREEN]],
        zmid=0, text=corr.round(2).values,
        texttemplate="%{text}", showscale=True,
    ))
    fig.update_layout(template=PLOTLY_TEMPLATE, title=title, height=400)
    return fig
