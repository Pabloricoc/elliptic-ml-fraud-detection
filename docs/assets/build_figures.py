"""Render README figures from the three local aggregate reports."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

INK = "#172940"
MUTED = "#435972"
PAPER = "#f1f5fa"
GRID = "#d8e3ee"
TEAL = "#087970"
BLUE = "#3d72cc"
ORANGE = "#a94a26"


def style_axis(axis: Axes) -> None:
    """Apply the same typography, grid and background to each plot."""
    axis.set_axisbelow(True)
    axis.grid(axis="y", color=GRID)
    axis.tick_params(colors=MUTED, length=0, pad=6)
    axis.xaxis.label.set_color(MUTED)
    axis.yaxis.label.set_color(MUTED)
    for spine in axis.spines.values():
        spine.set_visible(False)


def heading(figure: Figure, title: str, subtitle: str) -> None:
    """Add a title and the experimental scope to the plot card."""
    figure.text(0.06, 0.94, title, color=INK, fontsize=19, weight="bold", va="top")
    figure.text(0.06, 0.86, subtitle, color=MUTED, fontsize=10.5, va="top")


def save(figure: Figure, path: Path) -> None:
    figure.savefig(path, dpi=165, facecolor=figure.get_facecolor())
    plt.close(figure)


def temporal_plot(periods: pd.DataFrame, windows: dict, path: Path) -> None:
    """Visualize overall volume and known label coverage by time step."""
    figure, (volume, coverage) = plt.subplots(
        2, 1, figsize=(9, 4.9), sharex=True, facecolor=PAPER
    )
    figure.subplots_adjust(left=0.105, right=0.96, top=0.72, bottom=0.14, hspace=0.23)
    heading(
        figure,
        "The Elliptic graph across time",
        "Transaction volume and known-label coverage across the 49 release steps",
    )
    for axis in (volume, coverage):
        style_axis(axis)
        axis.set_facecolor(PAPER)
    for name, colour in zip(
        ("train", "validation", "test"), (BLUE, TEAL, "#6b609b"), strict=True
    ):
        window = windows[name]
        start = window["first_step"]
        end = window["last_step"]
        selected = periods.loc[periods.index.to_series().between(start, end)]
        volume.bar(selected.index, selected["transactions"], width=0.78, color=colour)
        coverage.axvspan(start - 0.5, end + 0.5, color=colour, alpha=0.075, lw=0)
        volume.text(
            (start + end) / 2,
            1.06,
            f"{name.title()} {start}–{end}",
            color=colour,
            ha="center",
            va="bottom",
            transform=volume.get_xaxis_transform(),
            fontsize=10,
            weight="bold",
        )
    coverage.plot(
        periods.index, periods["label_coverage"], color=TEAL, lw=1.8, marker="o", ms=2.8
    )
    volume.set(ylabel="Transactions", xlim=(0.5, 49.5))
    coverage.set(
        ylabel="Labelled share",
        xlabel="Time step",
        xticks=[1, 10, 20, 29, 39, 49],
        ylim=(0, 0.5),
    )
    coverage.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    save(figure, path)


def review_plot(queue: pd.DataFrame, budget: int, path: Path) -> None:
    """Show known outcomes and unlabelled selections at the fixed capacity."""
    figure, axis = plt.subplots(figsize=(9, 3.9), facecolor=PAPER)
    figure.subplots_adjust(left=0.105, right=0.96, top=0.66, bottom=0.16)
    heading(
        figure,
        "What the review queue contained",
        f"Frozen model ranking • {budget:,} review slots per step",
    )
    axis.set_facecolor(PAPER)
    style_axis(axis)
    base = pd.Series(0, index=queue.index)
    for column, label, colour in (
        ("confirmed_illicit", "Known illicit", ORANGE),
        ("confirmed_licit", "Known licit", TEAL),
        ("reviewed_unknown", "Unlabelled", "#a8bad0"),
    ):
        values = queue[column]
        axis.bar(
            queue.index,
            values,
            bottom=base,
            color=colour,
            width=0.6,
            label=f"{label} ({values.sum():,})",
        )
        base += values
    axis.set(xticks=queue.index, xlabel="Test time step", ylabel="Reviews")
    axis.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.025),
        frameon=False,
        ncol=3,
        labelcolor=INK,
        borderaxespad=0,
    )
    save(figure, path)


def shap_plot(summary: dict, path: Path) -> None:
    """Render mean absolute log-odds contributions from the scored batch."""
    labels = {
        "observed_parent_count": "Observed parent count",
        "observed_ancestor_depth": "Observed ancestor depth",
        "max_parent_indegree": "Max. parent indegree",
        "mean_parent_indegree": "Mean parent indegree",
    }
    ordered = pd.Series(summary["mean_abs_shap"]).sort_values()
    figure, axis = plt.subplots(figsize=(9, 3.6), facecolor=PAPER)
    figure.subplots_adjust(left=0.285, right=0.94, top=0.74, bottom=0.2)
    heading(
        figure,
        "What moved the model score",
        f"SHAP on {summary['scored']:,} scored transactions in steps "
        f"{summary['first_step']}–{summary['last_step']}",
    )
    axis.set_facecolor(PAPER)
    style_axis(axis)
    axis.grid(axis="y", visible=False)
    axis.grid(axis="x", color=GRID)
    bars = axis.barh(
        [labels[key] for key in ordered.index], ordered, color=TEAL, height=0.58
    )
    axis.bar_label(bars, fmt="%.3f", padding=5, color=INK, fontsize=10)
    axis.set(
        xlabel="Mean absolute SHAP contribution (log-odds)",
        xlim=(0, ordered.max() * 1.18),
    )
    save(figure, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eda_run", type=Path)
    parser.add_argument("model_run", type=Path)
    parser.add_argument("score_run", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent
    )
    args = parser.parse_args()
    metrics = json.loads((args.model_run / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (args.model_run / "manifest.json").read_text(encoding="utf-8")
    )
    summary = json.loads((args.score_run / "summary.json").read_text(encoding="utf-8"))
    if summary["model_sha256"] != manifest["model_sha256"]:
        parser.error(
            "The scoring report and training report must refer to the same model."
        )
    periods = pd.read_csv(args.eda_run / "time_steps.csv", index_col="time_step")
    queue = pd.read_csv(args.model_run / "test_by_step.csv", index_col="time_step")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporal_plot(
        periods, metrics["windows"], args.output_dir / "temporal-coverage.png"
    )
    review_plot(
        queue,
        metrics["test"]["review_queue"]["budget_per_step"],
        args.output_dir / "review-queue.png",
    )
    shap_plot(summary, args.output_dir / "shap-importance.png")


if __name__ == "__main__":
    main()
