"""Render policy rollouts: one policy with a Push-T style top view, or policies side by side."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import tyro
from PIL import Image, ImageDraw, ImageFont

from ur_pusht import geometry as geo
from ur_pusht.env import URPushTEnv
from ur_pusht.policy import Policy
from ur_pusht.rollout import Episode, run_episode

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


@dataclass
class RenderConfig:
    policy: Path
    seeds: tuple[int, ...] = (0,)
    label: str = ""
    out: Path = Path("media")
    height: int = 480
    width: int = 640
    frame_every: int = 5
    gif_width: int = 800
    gif_stride: int = 3


@dataclass
class CompareConfig:
    policies: tuple[Path, ...]
    labels: tuple[str, ...]
    seed: int
    name: str = "comparison"
    out: Path = Path("media")
    max_seconds: float | None = None
    height: int = 480
    width: int = 640
    frame_every: int = 5
    gif_width: int = 600
    gif_stride: int = 5


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def _label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, size: int) -> None:
    font = _font(size)
    left, top, right, bottom = draw.textbbox(xy, text, font=font)
    draw.rounded_rectangle(
        (left - 6, top - 4, right + 6, bottom + 4), radius=6, fill=(255, 255, 255)
    )
    draw.text(xy, text, fill=(30, 30, 30), font=font)


def _status(env: URPushTEnv) -> str:
    coverage = geo.coverage(env.observation()[2:])
    return f"coverage {coverage:4.0%}   t = {env.steps / env.config.control_hz:4.1f} s"


def front_frame(env: URPushTEnv, height: int, width: int, label: str) -> np.ndarray:
    image = Image.fromarray(env.render("front", height, width))
    draw = ImageDraw.Draw(image)
    _label(draw, (16, 12), label, 22)
    _label(draw, (16, height - 34), _status(env), 18)
    return np.asarray(image)


def compose_frame(env: URPushTEnv, height: int, width: int, label: str) -> np.ndarray:
    front = env.render("front", height, width)
    left = (width - height) // 2
    top = env.render("top", height, width, show_arm=False)[:, left : left + height]
    image = Image.fromarray(np.concatenate([front, top], axis=1))

    draw = ImageDraw.Draw(image)
    _label(draw, (16, 12), label, 22)
    _label(draw, (width + 14, 12), _status(env), 18)
    return np.asarray(image)


def render_episode(
    env: URPushTEnv,
    policy: Policy,
    seed: int,
    frame_every: int,
    make_frame: Callable[[URPushTEnv], np.ndarray],
) -> tuple[list[np.ndarray], Episode]:
    frames, substeps = [], 0

    def capture() -> None:
        nonlocal substeps
        substeps += 1
        if substeps % frame_every == 0:
            frames.append(make_frame(env))

    env.reset(seed)
    frames.append(make_frame(env))
    episode = run_episode(env, policy, seed, on_substep=capture)
    return frames, episode


def save_media(
    frames: list[np.ndarray], path: Path, fps: float, config: RenderConfig | CompareConfig
) -> None:
    frames = frames + [frames[-1]] * int(fps)
    imageio.mimsave(path.with_suffix(".mp4"), frames, fps=fps, codec="libx264", macro_block_size=1)
    height = round(frames[0].shape[0] * config.gif_width / frames[0].shape[1])
    gif_frames = [
        Image.fromarray(frame)
        .resize((config.gif_width, height), Image.LANCZOS)
        .quantize(colors=96, dither=Image.Dither.NONE)
        for frame in frames[:: config.gif_stride]
    ]
    gif_frames[0].save(
        path.with_suffix(".gif"),
        save_all=True,
        append_images=gif_frames[1:],
        duration=round(1000 * config.gif_stride / fps),
        loop=0,
        optimize=True,
    )


def _outcome(episode: Episode) -> str:
    return "success" if episode.success else f"max reward {episode.max_reward:.2f}"


def episodes(config: RenderConfig) -> None:
    config.out.mkdir(parents=True, exist_ok=True)
    env, policy = URPushTEnv(), Policy.load(config.policy)
    fps = env.substep_hz / config.frame_every
    for seed in config.seeds:
        frames, episode = render_episode(
            env,
            policy,
            seed,
            config.frame_every,
            lambda e: compose_frame(e, config.height, config.width, config.label),
        )
        name = f"{config.policy.stem}_seed{seed}"
        save_media(frames, config.out / name, fps, config)
        print(f"{name}: {len(episode)} steps, {_outcome(episode)}")


def compare(config: CompareConfig) -> None:
    config.out.mkdir(parents=True, exist_ok=True)
    env = URPushTEnv()
    tracks = []
    for path, label in zip(config.policies, config.labels):
        frames, episode = render_episode(
            env,
            Policy.load(path),
            config.seed,
            config.frame_every,
            lambda e, label=label: front_frame(e, config.height, config.width, label),
        )
        tracks.append(frames)
        print(f"{label}: {len(episode)} steps, {_outcome(episode)}")
    fps = env.substep_hz / config.frame_every
    length = max(len(track) for track in tracks)
    if config.max_seconds is not None:
        length = min(length, int(config.max_seconds * fps))
    tracks = [(track + [track[-1]] * length)[:length] for track in tracks]
    frames = [np.concatenate(row, axis=1) for row in zip(*tracks)]
    save_media(
        frames,
        config.out / f"{config.name}_seed{config.seed}",
        env.substep_hz / config.frame_every,
        config,
    )


if __name__ == "__main__":
    torch.set_num_threads(1)
    config = tyro.extras.subcommand_cli_from_dict(
        {"episode": RenderConfig, "compare": CompareConfig}
    )
    compare(config) if isinstance(config, CompareConfig) else episodes(config)
