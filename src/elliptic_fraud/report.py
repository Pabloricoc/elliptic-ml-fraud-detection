"""Write a complete, aggregate-only EDA report to a new local run directory."""

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .analysis import Summary, feature_statistics, summarize
from .ingestion import EllipticDataset
from .runs import new_run


def _plot_timeline(periods: pd.DataFrame, path: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import PercentFormatter

    figure = Figure(figsize=(10, 6), layout="constrained")
    FigureCanvasAgg(figure)
    counts, ratios = figure.subplots(2, 1, sharex=True)
    observed = periods["transactions"] > 0
    counts.bar(
        periods.index,
        periods["transactions"],
        color="#d7dce0",
        label="All transactions",
    )
    for name, colour in (("licit", "#006b86"), ("illicit", "#ae421c")):
        counts.plot(
            periods.index,
            periods[name].where(observed),
            ".-",
            color=colour,
            label=name.title(),
        )
    counts.set(ylabel="Transactions", title="Transaction volume and known classes")
    counts.legend()
    for column, label, colour in (
        ("label_coverage", "Labelled / all transactions", "#006b86"),
        ("illicit_share_labelled", "Illicit / labelled transactions", "#ae421c"),
    ):
        ratios.plot(periods.index, periods[column], ".-", color=colour, label=label)
    ratios.set(
        xlabel="Time step",
        ylabel="Share",
        ylim=(-0.025, 1.025),
        title="Label coverage and class balance",
    )
    ratios.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ratios.legend()
    figure.savefig(path, dpi=140)


def write_report(dataset: EllipticDataset, output_dir: Path) -> tuple[Path, Summary]:
    """Publish all four report files together, preserving earlier successful runs."""
    summary, periods = summarize(dataset)
    with new_run(output_dir, "eda") as (staging, destination):
        (staging / "summary.json").write_text(
            json.dumps(asdict(summary), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        periods.to_csv(staging / "time_steps.csv")
        feature_statistics(dataset.transactions).to_csv(staging / "features.csv")
        _plot_timeline(periods, staging / "timeline.png")
    return destination, summary
