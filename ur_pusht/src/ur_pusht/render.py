"""Render policy rollouts: the arm from the front camera beside a Push-T style top view."""

from __future__ import annotations

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
    gif_width: int = 600
    gif_stride: int = 3


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def compose_frame(env: URPushTEnv, height: int, width: int, label: str) -> np.ndarray:
    front = env.render("front", height, width)
    left = (width - height) // 2
    top = env.render("top", height, width, show_arm=False)[:, left : left + height]
    image = Image.fromarray(np.concatenate([front, top], axis=1))

    draw = ImageDraw.Draw(image)
    coverage = geo.coverage(env.observation()[2:])
    seconds = env.steps / env.config.control_hz
    draw.text((14, 10), label, fill=(30, 30, 30), font=_font(22))
    draw.text(
        (width + 12, 10),
        f"coverage {coverage:4.0%}   t = {seconds:4.1f} s",
        fill=(30, 30, 30),
        font=_font(18),
    )
    return np.asarray(image)


def render_episode(
    env: URPushTEnv, policy: Policy, seed: int, config: RenderConfig
) -> tuple[list[np.ndarray], Episode]:
    frames, substeps = [], 0

    def capture() -> None:
        nonlocal substeps
        substeps += 1
        if substeps % config.frame_every == 0:
            frames.append(compose_frame(env, config.height, config.width, config.label))

    env.reset(seed)
    frames.append(compose_frame(env, config.height, config.width, config.label))
    episode = run_episode(env, policy, seed, on_substep=capture)
    fps = env.substep_hz / config.frame_every
    frames.extend([frames[-1]] * int(fps))
    return frames, episode


def save_media(frames: list[np.ndarray], path: Path, fps: float, config: RenderConfig) -> None:
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


def main(config: RenderConfig) -> None:
    config.out.mkdir(parents=True, exist_ok=True)
    env, policy = URPushTEnv(), Policy.load(config.policy)
    torch.set_num_threads(1)
    fps = env.substep_hz / config.frame_every
    for seed in config.seeds:
        frames, episode = render_episode(env, policy, seed, config)
        name = f"{config.policy.stem}_seed{seed}"
        save_media(frames, config.out / name, fps, config)
        outcome = "success" if episode.success else f"max reward {episode.max_reward:.2f}"
        print(f"{name}: {len(episode)} steps, {outcome}, {len(frames)} frames")


if __name__ == "__main__":
    main(tyro.cli(RenderConfig))
