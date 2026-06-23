"""
Visualization utilities for L-C Fusion analysis results.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional

# Use a non-interactive backend when running headless (e.g. in CI / scripts).
# Must be set before any pyplot calls.
matplotlib.use("Agg")

# Stage boundary defaults derived from the Mirror Labyrinth case studies:
# Messages 1-4: Logic, 5-12: Pseudo-Dialectics, 13+: Fusion
DEFAULT_STAGES: Dict[str, tuple] = {
    "Logic": (1, 4),
    "Pseudo-Dialectics": (5, 12),
    "Fusion": (13, None),
}

STAGE_COLORS: Dict[str, str] = {
    "Logic": "#d0e8ff",
    "Pseudo-Dialectics": "#ffe8b0",
    "Fusion": "#ffd0d0",
}


def plot_fusion_dynamics(
    similarity_dynamics: List[float],
    title: str = "Lexico-Conceptual Fusion Dynamics",
    stages: Optional[Dict[str, tuple]] = None,
    relevance_dynamics: Optional[List[float]] = None,
    model_name: str = "",
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot the similarity_dynamics time series with optional stage annotations.

    Args:
        similarity_dynamics: Output of analyze_dialogue_fusion() — general similarity
                             (no prefix, symmetric mode).
        title:               Plot title.
        stages:              Dict mapping stage name → (start_turn, end_turn) 1-indexed.
                             Pass None to skip stage shading. End turn is inclusive;
                             pass None as end_turn to extend to the last dialogue turn.
        relevance_dynamics:  Optional second series (prefix mode — asymmetric relevance).
                             When provided, both series are plotted for the gap analysis
                             described in section 6.3 of the paper.
        model_name:          Appended to title if provided (e.g. "Gemini 2.5 Pro").
        save_path:           If provided, save figure to this path (PNG, PDF, SVG …).
        show:                If True (and save_path is None), call plt.show().
                             Set to False when running in scripts or notebooks that
                             manage display themselves.

    Returns:
        matplotlib Figure object.
    """
    # Apply style before creating the figure
    plt.style.use("seaborn-v0_8-whitegrid")

    n = len(similarity_dynamics)
    turns = list(range(1, n + 1))

    fig, ax = plt.subplots(figsize=(13, 7))

    # Stage shading
    if stages:
        for stage_name, (start, end) in stages.items():
            end_plot = end if end is not None else n
            ax.axvspan(
                start - 0.5,
                end_plot + 0.5,
                alpha=0.25,
                color=STAGE_COLORS.get(stage_name, "#eeeeee"),
                label=f"Stage: {stage_name}",
                zorder=0,
            )

    # General similarity (no-prefix, symmetric)
    ax.plot(
        turns,
        similarity_dynamics,
        marker="o",
        linestyle="-",
        color="#1a6bb5",
        linewidth=2,
        label="General similarity (no prefix)",
        zorder=3,
    )
    for i, val in enumerate(similarity_dynamics):
        ax.annotate(
            f"{val:.3f}",
            (turns[i], val),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=8,
            color="#1a6bb5",
        )

    # Relevance series (prefix, asymmetric) — optional
    if relevance_dynamics is not None:
        rel_turns = turns[: len(relevance_dynamics)]
        ax.plot(
            rel_turns,
            relevance_dynamics,
            marker="s",
            linestyle="--",
            color="#c0392b",
            linewidth=2,
            label="Relevance (with prefix)",
            zorder=3,
        )
        for i, val in enumerate(relevance_dynamics):
            ax.annotate(
                f"{val:.3f}",
                (rel_turns[i], val),
                textcoords="offset points",
                xytext=(0, -14),
                ha="center",
                fontsize=8,
                color="#c0392b",
            )

    full_title = f"{title}\n{model_name}" if model_name else title
    ax.set_title(full_title, fontsize=15, fontweight="bold")
    ax.set_xlabel("LLM Response Number in Dialogue", fontsize=12)
    ax.set_ylabel("Cosine Similarity", fontsize=12)
    ax.set_xticks(turns)

    all_vals = list(similarity_dynamics) + (list(relevance_dynamics) if relevance_dynamics else [])
    margin = 0.02
    ax.set_ylim(
        max(0.0, min(all_vals) - margin),
        min(1.0, max(all_vals) + margin + 0.03),
    )

    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, zorder=1)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    elif show:
        plt.show()

    return fig
