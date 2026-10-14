"""Push-T with a UR5e arm in MuJoCo.

Observation and action follow the 2D Push-T benchmark: the state is
[tip_x, tip_y, block_x, block_y, block_angle] in pixels (1 px = 1 mm) and the action is a target
tip position in pixels. A reference point follows the target with Push-T's PD agent dynamics,
and the arm tracks the reference with differential inverse kinematics at a fixed tip height.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from ur_pusht import geometry as geo

ASSETS = Path(__file__).parent / "assets"
SCENE_XML = ASSETS / "scene.xml"
ARM_XML = ASSETS / "universal_robots_ur5e" / "ur5e.xml"

ARM_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
READY_SEED = np.array([np.pi, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])

TIP_RADIUS = 0.015
TIP_OFFSET = 0.18
ARM_MESH_GROUP = 2


@dataclass(frozen=True)
class EnvConfig:
    control_hz: float = 10.0
    reference_dt: float = 0.01
    max_steps: int = 300
    tip_height: float = 0.020
    kp: float = 100.0
    kv: float = 20.0
    ik_iters: int = 2
    ik_damping: float = 0.03


def _add_stick(arm: mujoco.MjSpec) -> None:
    # Collision bit 2 is shared only with the T block, so the stick can never touch the table.
    site = arm.site("attachment_site")
    stick = arm.body("wrist_3_link").add_body(name="stick", pos=site.pos, quat=site.quat)
    stick.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[0.032, 0.008, 0],
        pos=[0, 0, 0.008],
        rgba=[0.15, 0.15, 0.16, 1],
        contype=0,
        conaffinity=0,
        group=ARM_MESH_GROUP,
    )
    stick.add_geom(
        name="rod",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=[0, 0, 0.016, 0, 0, TIP_OFFSET - 0.012],
        size=[0.007, 0, 0],
        rgba=[0.75, 0.76, 0.78, 1],
        contype=2,
        conaffinity=2,
    )
    stick.add_geom(
        name="tip",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[TIP_RADIUS, 0.012, 0],
        pos=[0, 0, TIP_OFFSET],
        rgba=[0.255, 0.412, 0.882, 1],
        contype=2,
        conaffinity=2,
        friction=[0.5, 0.005, 0.0001],
    )
    stick.add_site(name="tip", pos=[0, 0, TIP_OFFSET])


def build_model() -> mujoco.MjModel:
    scene = mujoco.MjSpec.from_file(str(SCENE_XML))
    arm = mujoco.MjSpec.from_file(str(ARM_XML))
    _add_stick(arm)
    for body in arm.bodies:
        body.gravcomp = 1.0
    scene.attach(arm, prefix="ur5e/", frame=scene.worldbody.add_frame())
    return scene.compile()


class DifferentialIK:
    """Damped least-squares IK for the tip position at a fixed tip orientation, within joint limits."""

    def __init__(
        self,
        model: mujoco.MjModel,
        joint_ids: np.ndarray,
        site: int,
        rotation: np.ndarray,
    ):
        self.model, self.data = model, mujoco.MjData(model)
        self.qpos_ids = model.jnt_qposadr[joint_ids]
        self.dof_ids = model.jnt_dofadr[joint_ids]
        self.lower, self.upper = model.jnt_range[joint_ids].T
        self.site, self.rotation = site, rotation
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def solve(self, q: np.ndarray, position: np.ndarray, iters: int, damping: float) -> np.ndarray:
        q = q.copy()
        for _ in range(iters):
            self.data.qpos[self.qpos_ids] = q
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            rotation = self.data.site_xmat[self.site].reshape(3, 3)
            position_error = position - self.data.site_xpos[self.site]
            rotation_error = 0.5 * np.cross(rotation, self.rotation, axis=0).sum(axis=1)
            error = np.concatenate([position_error, rotation_error])

            mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.site)
            jacobian = np.vstack([self._jacp, self._jacr])[:, self.dof_ids]
            q += jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping**2 * np.eye(6), error)
            q = np.clip(q, self.lower, self.upper)
        return q


class URPushTEnv:
    def __init__(self, config: EnvConfig | None = None):
        self.config = config = config or EnvConfig()
        self.model = build_model()
        self.data = mujoco.MjData(self.model)

        joint_ids = np.array([self.model.joint(f"ur5e/{name}").id for name in ARM_JOINTS])
        self._arm_qpos = self.model.jnt_qposadr[joint_ids]
        self._tee_qpos = self.model.joint("tee").qposadr[0]
        self._tee_body = self.model.body("tee").id
        self._target_mocap = self.model.body("target").mocapid[0]
        self._tip = self.model.site("ur5e/tip").id
        self.data.qpos[self._arm_qpos] = READY_SEED
        mujoco.mj_kinematics(self.model, self.data)
        tip_down = self.data.site_xmat[self._tip].reshape(3, 3).copy()
        self._ik = DifferentialIK(self.model, joint_ids, self._tip, tip_down)

        # The UR5e position servos lag a moving target by (kv / kp) * joint velocity;
        # adding that term to the command cancels the lag.
        self._servo_lag = -self.model.actuator_biasprm[:, 2] / self.model.actuator_gainprm[:, 0]
        self._substeps = round(1.0 / (config.control_hz * config.reference_dt))
        self._physics_steps = round(config.reference_dt / self.model.opt.timestep)
        self._ready_q = self._ik.solve(READY_SEED, self._tip_world(geo.GOAL_POSE[:2]), 100, 0.05)
        self._renderer: mujoco.Renderer | None = None

    @property
    def substep_hz(self) -> float:
        return self.config.control_hz * self._substeps

    def _tip_world(self, xy_px: np.ndarray) -> np.ndarray:
        return np.append(geo.px_to_world(xy_px), self.config.tip_height)

    def _block_pose(self) -> np.ndarray:
        xy = geo.world_to_px(self.data.xpos[self._tee_body, :2])
        rotation = self.data.xmat[self._tee_body].reshape(3, 3)
        angle = np.arctan2(rotation[1, 0], rotation[0, 0]) % (2 * np.pi)
        return np.array([*xy, angle])

    def observation(self) -> np.ndarray:
        tip = geo.world_to_px(self.data.site_xpos[self._tip, :2])
        return np.concatenate([tip, self._block_pose()])

    def reset(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        state = np.array(
            [
                rng.integers(50, 450),
                rng.integers(50, 450),
                rng.integers(100, 400),
                rng.integers(100, 400),
                rng.uniform(-np.pi, np.pi),
            ],
            dtype=float,
        )
        return self.reset_to_state(state)

    def reset_to_state(self, state: np.ndarray) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        tee_xy, tee_angle = geo.px_to_world(state[2:4]), state[4]
        self.data.qpos[self._tee_qpos : self._tee_qpos + 7] = [
            *tee_xy,
            0.0,
            np.cos(tee_angle / 2),
            0.0,
            0.0,
            np.sin(tee_angle / 2),
        ]

        self._q_target = self._ik.solve(self._ready_q, self._tip_world(state[:2]), 100, 0.05)
        self.data.qpos[self._arm_qpos] = self._q_target
        self.data.ctrl[:] = self._q_target
        self._reference = state[:2].astype(float)
        self._reference_velocity = np.zeros(2)
        self.data.mocap_pos[self._target_mocap, :2] = geo.px_to_world(state[:2])
        mujoco.mj_forward(self.model, self.data)

        self.steps = 0
        return self.observation()

    def step(self, action: np.ndarray, on_substep: Callable[[], None] | None = None):
        config = self.config
        target = np.clip(action, 0.0, geo.WORKSPACE_PX)
        self.data.mocap_pos[self._target_mocap, :2] = geo.px_to_world(target)
        for _ in range(self._substeps):
            acceleration = (
                config.kp * (target - self._reference) - config.kv * self._reference_velocity
            )
            self._reference_velocity += acceleration * config.reference_dt
            self._reference += self._reference_velocity * config.reference_dt
            q_previous = self._q_target
            self._q_target = self._ik.solve(
                q_previous, self._tip_world(self._reference), config.ik_iters, config.ik_damping
            )
            joint_velocity = (self._q_target - q_previous) / config.reference_dt
            self.data.ctrl[:] = self._q_target + self._servo_lag * joint_velocity
            mujoco.mj_step(self.model, self.data, nstep=self._physics_steps)
            if on_substep is not None:
                on_substep()

        self.steps += 1
        block_pose = self._block_pose()
        coverage = geo.coverage(block_pose)
        terminated = coverage > geo.SUCCESS_COVERAGE
        truncated = self.steps >= config.max_steps
        info = {"coverage": coverage, "is_success": terminated}
        return self.observation(), geo.reward(block_pose), terminated, truncated, info

    def render(
        self, camera: str = "front", height: int = 480, width: int = 640, show_arm: bool = True
    ) -> np.ndarray:
        if self._renderer is None or (self._renderer.height, self._renderer.width) != (
            height,
            width,
        ):
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, height, width)
        options = mujoco.MjvOption()
        options.geomgroup[ARM_MESH_GROUP] = show_arm
        self._renderer.update_scene(self.data, camera=camera, scene_option=options)
        self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = show_arm
        return self._renderer.render()
