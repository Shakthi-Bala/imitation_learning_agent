"""Figures for self-improvement runs: per-round evaluation and per-round training curves."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import tyro

COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
INK, INK_SECONDARY, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"


@dataclass
class PlotConfig:
    runs: tuple[Path, ...]
    labels: tuple[str, ...]
    out: Path = Path("media")
    reference_reward: float | None = None
    reference_label: str = "2D policy in 2D sim"


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY)


def plot_rounds(config: PlotConfig) -> Path:
    fig, (ax_success, ax_reward) = plt.subplots(1, 2, figsize=(10, 4), facecolor=SURFACE)
    for run, label, color in zip(config.runs, config.labels, COLORS):
        history = json.loads((run / "history.json").read_text())
        rounds = [h["round"] for h in history]
        success = [100 * h["eval"]["success_rate"] for h in history]
        reward = [h["eval"]["mean_reward"] for h in history]
        for ax, values, fmt in ((ax_success, success, "{:.0f}%"), (ax_reward, reward, "{:.2f}")):
            ax.plot(rounds, values, color=color, linewidth=2, marker="o", markersize=7, label=label)
            ax.annotate(
                fmt.format(values[-1]),
                (rounds[-1], values[-1]),
                textcoords="offset points",
                xytext=(8, 0),
                va="center",
                color=INK,
            )

    if config.reference_reward is not None:
        ax_reward.axhline(config.reference_reward, color=INK_SECONDARY, linewidth=1, linestyle="--")
        ax_reward.text(
            0.05, config.reference_reward + 0.01, config.reference_label, color=INK_SECONDARY
        )
    ax_success.set_title("Episodes solved (coverage > 95%)", loc="left", color=INK)
    ax_reward.set_title("Mean reward", loc="left", color=INK)
    ax_success.set_ylabel("% of 100 evaluation seeds", color=INK_SECONDARY)
    ax_reward.set_ylim(0, 1)
    for ax in (ax_success, ax_reward):
        ax.set_xlabel("Self-improvement round (0 = zero-shot 2D policy)", color=INK_SECONDARY)
        ax.set_xticks(range(len(rounds)))
        _style(ax)
    ax_success.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    path = config.out / "self_improvement.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    return path


def plot_training(run: Path, label: str, out: Path) -> Path:
    rounds = sorted(run.glob("round*/metrics.jsonl"))
    fig, (ax_loss, ax_success) = plt.subplots(1, 2, figsize=(10, 4), facecolor=SURFACE)
    for index, path in enumerate(rounds):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        train = [r for r in rows if "train/loss" in r]
        evals = [r for r in rows if "eval/success_rate" in r]
        alpha = 0.35 + 0.65 * (index + 1) / len(rounds)
        shade = {"color": COLORS[0], "alpha": alpha, "linewidth": 2}
        name = path.parent.name.replace("round", "round ")
        ax_loss.plot(
            [r["step"] for r in train], [r["train/loss"] for r in train], **shade, label=name
        )
        ax_success.plot(
            [r["step"] for r in evals],
            [100 * r["eval/success_rate"] for r in evals],
            marker="o",
            **shade,
            label=name,
        )
    ax_loss.set_yscale("log")
    ax_loss.set_title(f"{label}: training loss", loc="left", color=INK)
    ax_success.set_title(f"{label}: episodes solved", loc="left", color=INK)
    ax_success.set_ylabel("% of 100 evaluation seeds", color=INK_SECONDARY)
    for ax in (ax_loss, ax_success):
        ax.set_xlabel("Training step", color=INK_SECONDARY)
        _style(ax)
    ax_loss.legend(frameon=False)
    fig.tight_layout()
    path = out / f"training_{run.name}.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    return path


def main(config: PlotConfig) -> None:
    config.out.mkdir(parents=True, exist_ok=True)
    print(plot_rounds(config))
    for run, label in zip(config.runs, config.labels):
        print(plot_training(run, label, config.out))


if __name__ == "__main__":
    main(tyro.cli(PlotConfig))
