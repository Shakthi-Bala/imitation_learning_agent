"""Push-T geometry in the original pixel frame and its placement on the robot table.

The task keeps Push-T's 512 x 512 pixel frame with 1 px = 1 mm, so states, actions and
policies are interchangeable between the 2D benchmark and the robot.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

WORKSPACE_PX = 512.0
METERS_PER_PX = 1e-3
WORKSPACE_ORIGIN = np.array([0.45 - 0.256, -0.256])

GOAL_POSE = np.array([256.0, 256.0, np.pi / 4])
SUCCESS_COVERAGE = 0.95

TEE_BAR = ((-60, 30), (60, 30), (60, 0), (-60, 0))
TEE_STEM = ((-15, 30), (-15, 120), (15, 120), (15, 30))


def px_to_world(xy_px: np.ndarray) -> np.ndarray:
    return WORKSPACE_ORIGIN + METERS_PER_PX * np.asarray(xy_px, dtype=float)


def world_to_px(xy_world: np.ndarray) -> np.ndarray:
    return (np.asarray(xy_world, dtype=float) - WORKSPACE_ORIGIN) / METERS_PER_PX


def tee_polygon(pose: np.ndarray) -> Polygon:
    x, y, theta = pose
    c, s = np.cos(theta), np.sin(theta)

    def place(vertices):
        return Polygon([(x + c * u - s * v, y + s * u + c * v) for u, v in vertices])

    return unary_union([place(TEE_BAR), place(TEE_STEM)])


_GOAL = tee_polygon(GOAL_POSE)


def coverage(block_pose: np.ndarray) -> float:
    return _GOAL.intersection(tee_polygon(block_pose)).area / _GOAL.area


def reward(block_pose: np.ndarray) -> float:
    return float(np.clip(coverage(block_pose) / SUCCESS_COVERAGE, 0.0, 1.0))
