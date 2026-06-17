"""Bloomberg-inspired color palette and Plotly theme for AlgoTrade."""
from __future__ import annotations

import plotly.graph_objects as go

# ── Color System ──────────────────────────────────────────────────────────
AMBER   = "#FFB300"   # Labels, structure, borders
GREEN   = "#00E676"   # Positive / up / long
RED     = "#FF1744"   # Negative / down / short
BASE    = "#0A0A0A"   # Near-black background
SURFACE = "#111111"   # Card/panel surface
BORDER  = "#1E1E1E"   # Panel borders
TEXT    = "#E0E0E0"   # Primary text
MUTED   = "#757575"   # Secondary text
AMBER_DIM = "#B37A00" # Dimmed amber for subtle labels

# Signal color map
SIGNAL_COLORS = {1: GREEN, 0: AMBER, -1: RED}
REGIME_COLORS = {0: GREEN, 1: AMBER, 2: RED}

# ── Plotly Layout Template ────────────────────────────────────────────────
PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=BASE,
        plot_bgcolor=SURFACE,
        font=dict(family="Consolas, monospace", color=TEXT, size=11),
        title=dict(font=dict(color=AMBER, size=14)),
        xaxis=dict(
            gridcolor=BORDER,
            linecolor=BORDER,
            tickcolor=TEXT,
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor=BORDER,
            linecolor=BORDER,
            tickcolor=TEXT,
            zeroline=False,
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=BORDER,
            font=dict(color=TEXT),
        ),
        colorway=[AMBER, GREEN, RED, "#29B6F6", "#CE93D8", "#FFCC02", "#80DEEA"],
    )
)

AMBER_STYLE = f"color:{AMBER};font-weight:700;"
GREEN_STYLE = f"color:{GREEN};font-weight:700;"
RED_STYLE   = f"color:{RED};font-weight:700;"


def colored_metric(value: float, positive_is_good: bool = True) -> str:
    """Return HTML-styled metric string (green if good, red if bad)."""
    if value > 0:
        color = GREEN if positive_is_good else RED
    elif value < 0:
        color = RED if positive_is_good else GREEN
    else:
        color = AMBER
    return f'<span style="color:{color};font-weight:700">{value:+.2f}</span>'
