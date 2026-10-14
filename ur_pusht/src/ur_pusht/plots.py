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
    start_eval: Path
    final_evals: tuple[Path, ...]
    final_labels: tuple[str, ...]
    out: Path = Path("media")
    bootstrap: int = 5000


def _paired_change(start: Path, final: Path, metric, samples: int) -> tuple[float, float, float]:
    """Mean per-seed change from the start policy, with a 95% bootstrap interval."""
    before = {e["seed"]: e for e in json.loads(start.read_text())["per_episode"]}
    after = {e["seed"]: e for e in json.loads(final.read_text())["per_episode"]}
    change = np.array([metric(after[s]) - metric(before[s]) for s in sorted(before)])
    rng = np.random.default_rng(0)
    means = [change[rng.integers(0, len(change), len(change))].mean() for _ in range(samples)]
    low, high = np.percentile(means, [2.5, 97.5])
    return change.mean(), low, high


def plot_rl(config: RLPlotConfig) -> Path:
    fig, axes = plt.subplots(
        1, 3, figsize=(13, 4.4), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.6, 1, 1]}
    )
    ax_curve, ax_reward, ax_failure = axes
    starts = []
    for run, label, color in zip(config.runs, config.labels, COLORS):
        rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        evals = [r for r in rows if "eval/mean_reward" in r]
        starts.append(evals[0]["eval/mean_reward"])
        ax_curve.plot(
            [r["iteration"] for r in evals],
            [r["eval/mean_reward"] for r in evals],
            color=color,
            linewidth=2,
            marker="o",
            markersize=5,
            label=label,
        )
    ax_curve.axhspan(min(starts), max(starts), color=GRID, alpha=0.8, zorder=0)
    ax_curve.set_title("Mean reward during training (100 seeds)", loc="left", color=INK)
    ax_curve.set_xlabel("RL iteration (256 episodes each)", color=INK_SECONDARY)

    metrics = (
        (ax_reward, lambda e: e["max_reward"], 1, "{:+.3f}", "Change in mean reward"),
        (
            ax_failure,
            lambda e: float(e["max_reward"] < 0.3),
            100,
            "{:+.1f}",
            "Change in clear failures (pts)",
        ),
    )
    rows = range(len(config.final_evals))
    for ax, metric, scale, fmt, title in metrics:
        for row, path in zip(rows, config.final_evals):
            mean, low, high = (
                scale * v for v in _paired_change(config.start_eval, path, metric, config.bootstrap)
            )
            ax.errorbar(
                mean,
                row,
                xerr=[[mean - low], [high - mean]],
                fmt="o",
                color=INK,
                ecolor=INK_SECONDARY,
                capsize=4,
                markersize=7,
            )
            ax.annotate(
                fmt.format(mean),
                (mean, row),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                color=INK,
            )
        ax.axvline(0, color=INK_SECONDARY, linewidth=1)
        ax.set_yticks(list(rows), config.final_labels if ax is ax_reward else [])
        ax.set_ylim(len(config.final_evals) - 0.5, -0.6)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel("vs. 2D policy, same 500 seeds (95% CI)", color=INK_SECONDARY)

    for ax in axes:
        _style(ax)
    handles, labels = ax_curve.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", ncol=2, frameon=False, bbox_to_anchor=(0.02, 0))
    fig.tight_layout(rect=(0, 0.14, 1, 1))
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
