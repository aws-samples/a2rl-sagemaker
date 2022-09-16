from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from smallmatter.ds import SimpleMatrixPlotter

from .flight_sales_gym import fsigmoid, reward_functions


def plot_contours(save: bool = False, output_dir: Path = Path()) -> None:
    # fmt: off
    F = (
        "revenue_0_20_no_jitter", "revenue_0_20",
        "revenue_0_10_no_jitter", "revenue_0_10",
        "revenue_0_05_no_jitter", "revenue_0_05",
        "revenue_0_02_no_jitter", "revenue_0_02",
        "revenue_0_01_no_jitter", "revenue_0_01",
        "profit_no_jitter"      , "profit",
    )
    # fmt: on

    plot_kwargs = []
    for f in F:
        for day in (None, 1, 183):
            plot_kwargs.append({"f": reward_functions[f], "day": day})

    smp = SimpleMatrixPlotter(3, init_figcount=len(plot_kwargs), figsize=(4, 2.5))
    for kwargs in plot_kwargs:
        plot_contour(ax=smp.pop(), **kwargs)  # type: ignore[arg-type]

    smp.trim()
    smp.fig.tight_layout()
    if save:
        smp.fig.savefig(output_dir / "reward-contour.png", dpi=150)


def plot_contour(f, ax, day: int | None = None) -> None:
    """Add countour to the given axes."""
    max_fare = 20
    daily_seats_quota = 20
    prices = [round(i * max_fare, 1) for i in np.linspace(0.0, 1.0, 11)]

    X, Y = np.meshgrid(np.arange(daily_seats_quota + 1), prices)
    XY = np.column_stack((X.ravel(), Y.ravel()))
    if day is None:
        rewards = [f(*xy)[0] for xy in XY]
        title = f"Reward function: {f.__name__}()"
    else:
        rewards = [f(*xy)[0] * fsigmoid(xy[1], day) for xy in XY]
        title = f"Reward function: {f.__name__}(), propensity on day {day}"

    Z = np.array(rewards).reshape(X.shape)

    cs = ax.contourf(X, Y, Z, cmap=plt.cm.Spectral, alpha=0.8)
    ax.set_xlabel("Sold")
    ax.set_xlim(0, daily_seats_quota)
    ax.set_xticks(range(0, daily_seats_quota + 1))
    ax.set_ylabel("Price")
    ax.set_ylim(prices[0], prices[-1])
    ax.tick_params(axis="both", which="major")
    ax.set_title(title)

    # Make a colorbar for the ContourSet returned by the contourf call.
    cbar = plt.colorbar(cs, ax=ax)
    cbar.ax.set_ylabel("Reward")
    cbar.ax.tick_params(axis="both", which="major")


if __name__ == "__main__":
    DEFAULT_OUTPUT_DIR = os.environ.get("SM_OUTPUT_DATA_DIR", "results")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=f"Where to write the output (default: {DEFAULT_OUTPUT_DIR}/)",
        default=DEFAULT_OUTPUT_DIR,
    )
    args = parser.parse_args()
    plot_contours(save=True, output_dir=args.output_dir)
