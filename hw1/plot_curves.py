"""Plot training curves (steps vs. loss and eval reward) for the HW1 report.

Usage:
    uv run --with matplotlib plot_curves.py exp/<run_dir> [--out mse_curves.png]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

SERIES = "#2a78d6"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def load_metrics(run_dir: Path) -> list[dict]:
    with (run_dir / "metrics.jsonl").open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--title", default="MSE policy")
    args = parser.parse_args()

    rows = load_metrics(args.run_dir)
    train = [r for r in rows if "train/loss" in r]
    evals = [r for r in rows if "eval/mean_reward" in r]

    plt.rcParams.update(
        {
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "axes.edgecolor": GRID,
        }
    )
    fig, (ax_loss, ax_rew) = plt.subplots(
        2, 1, figsize=(7, 6), sharex=True, facecolor=SURFACE
    )

    ax_loss.plot(
        [r["step"] for r in train], [r["train/loss"] for r in train],
        color=SERIES, linewidth=2,
    )
    ax_loss.set_yscale("log")
    ax_loss.set_ylabel("Training loss (log scale)")
    ax_loss.set_title(f"{args.title}: training loss", loc="left")

    steps = [r["step"] for r in evals]
    rewards = [r["eval/mean_reward"] for r in evals]
    ax_rew.plot(steps, rewards, color=SERIES, linewidth=2, marker="o", markersize=7)
    ax_rew.axhline(0.5, color=INK_SECONDARY, linewidth=1, linestyle="--")
    ax_rew.text(
        steps[0] if steps else 0, 0.51, "target 0.5",
        color=INK_SECONDARY, fontsize=9, va="bottom",
    )
    if steps:
        ax_rew.annotate(
            f"{rewards[-1]:.2f}", (steps[-1], rewards[-1]),
            textcoords="offset points", xytext=(0, 8), ha="center", color=INK,
        )
    ax_rew.set_ylim(0, 1)
    ax_rew.set_ylabel("Eval mean reward (100 episodes)")
    ax_rew.set_xlabel("Training step")
    ax_rew.set_title(f"{args.title}: evaluation reward", loc="left")

    for ax in (ax_loss, ax_rew):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = args.out or args.run_dir / "training_curves.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
