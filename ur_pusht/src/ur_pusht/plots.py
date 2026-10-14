"""Figures: self-improvement rounds, their training curves, and RL fine-tuning curves."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro

COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
INK, INK_SECONDARY, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"


@dataclass
class PlotConfig:
    runs: tuple[Path, ...]
    labels: tuple[str, ...]
    out: Path = Path("media")
    baselines: tuple[Path, ...] = ()
    baseline_label: str = "trained on human demos only"


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY)


def _final_eval(run: Path) -> dict[str, float]:
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    last = [r for r in rows if "eval/mean_reward" in r][-1]
    return {k.removeprefix("eval/"): v for k, v in last.items() if k.startswith("eval/")}


def plot_rounds(config: PlotConfig) -> Path:
    fig, (ax_success, ax_reward) = plt.subplots(1, 2, figsize=(10, 4.6), facecolor=SURFACE)
    for run, label, color in zip(config.runs, config.labels, COLORS):
        history = json.loads((run / "history.json").read_text())
        rounds = [h["round"] for h in history]
        success = [100 * h["eval"]["success_rate"] for h in history]
        reward = [h["eval"]["mean_reward"] for h in history]
        episodes = history[0].get("episodes", 100)
        ci = [196 * np.sqrt(s / 100 * (1 - s / 100) / episodes) for s in success]
        ax_success.errorbar(
            rounds, success, yerr=ci, fmt="none", ecolor=color, alpha=0.35, capsize=4
        )
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

    if config.baselines:
        finals = [_final_eval(run) for run in config.baselines]
        label = f"{config.baseline_label} ({len(finals)} seeds)"
        for ax, key, scale in ((ax_success, "success_rate", 100), (ax_reward, "mean_reward", 1)):
            values = [scale * f[key] for f in finals]
            ax.axhspan(min(values), max(values), color=GRID, alpha=0.8, zorder=0, label=label)
    ax_success.set_title("Episodes solved (coverage > 95%)", loc="left", color=INK)
    ax_reward.set_title("Mean reward", loc="left", color=INK)
    ax_success.set_ylabel("% of 100 evaluation seeds", color=INK_SECONDARY)
    ax_success.set_ylim(0, 35)
    ax_reward.set_ylim(0, 1)
    for ax in (ax_success, ax_reward):
        ax.set_xlabel("Self-improvement round (0 = zero-shot 2D policy)", color=INK_SECONDARY)
        ax.set_xticks(range(len(rounds)))
        _style(ax)
    handles, labels = ax_success.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
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


@dataclass
class RLPlotConfig:
    runs: tuple[Path, ...]
    labels: tuple[str, ...]
    offsets: tuple[int, ...] = ()
    out: Path = Path("media")


def plot_rl(config: RLPlotConfig) -> Path:
    fig, (ax_reward, ax_success) = plt.subplots(1, 2, figsize=(10, 4.6), facecolor=SURFACE)
    offsets = config.offsets or (0,) * len(config.runs)
    starts = []
    for run, label, offset, color in zip(config.runs, config.labels, offsets, COLORS):
        rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        evals = [r for r in rows if "eval/mean_reward" in r]
        if offset == 0:
            starts.append(evals[0])
        x = [offset + r["iteration"] for r in evals]
        for ax, key, scale in ((ax_reward, "mean_reward", 1), (ax_success, "success_rate", 100)):
            y = [scale * r[f"eval/{key}"] for r in evals]
            ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=6, label=label)

    for ax, key, scale in ((ax_reward, "mean_reward", 1), (ax_success, "success_rate", 100)):
        values = [scale * s[f"eval/{key}"] for s in starts]
        ax.axhspan(
            min(values),
            max(values),
            color=GRID,
            alpha=0.8,
            zorder=0,
            label="starting policy, repeated evaluations",
        )
        _style(ax)
        ax.set_xlabel("RL iteration (256 episodes each)", color=INK_SECONDARY)
    ax_reward.set_title("Mean reward (100 evaluation seeds)", loc="left", color=INK)
    ax_success.set_title("Episodes solved (coverage > 95%)", loc="left", color=INK)
    ax_success.set_ylabel("% of 100 evaluation seeds", color=INK_SECONDARY)
    handles, labels = ax_reward.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    path = config.out / "rl_finetuning.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    return path


def rounds(config: PlotConfig) -> None:
    config.out.mkdir(parents=True, exist_ok=True)
    print(plot_rounds(config))
    for run, label in zip(config.runs, config.labels):
        print(plot_training(run, label, config.out))


def rl(config: RLPlotConfig) -> None:
    config.out.mkdir(parents=True, exist_ok=True)
    print(plot_rl(config))


if __name__ == "__main__":
    config = tyro.extras.subcommand_cli_from_dict({"rounds": PlotConfig, "rl": RLPlotConfig})
    rl(config) if isinstance(config, RLPlotConfig) else rounds(config)
