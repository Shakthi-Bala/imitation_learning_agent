# Push-T on a UR5e

Push-T with a UR5e arm and a stick end effector in MuJoCo. The task, seeds and success test match
the 2D benchmark in [`../pusht_action_chunking`](../pusht_action_chunking), so 2D policies run on the
arm unchanged. Results are in the [repository README](../README.md#push-t-on-a-ur5e-arm) and the
[report](report/ur_pusht_report.pdf).

## Setup

```bash
cd ur_pusht
scripts/fetch_ur5e.sh   # UR5e model from MuJoCo Menagerie (pinned commit, Apache-2.0)
uv sync
```

The project depends on `../pusht_action_chunking` for the policy and dataset code. Its first
run downloads the Push-T dataset (31 MB) into `../pusht_action_chunking/data/`.

## Run

```bash
# 2D policy, zero-shot on the arm (100 evaluation seeds)
uv run python -m ur_pusht.evaluate --policy checkpoints/pusht2d_flow.pt --workers 16

# Self-improvement: collect successes, retrain, repeat
uv run python -m ur_pusht.improve --out runs/finetune --finetune --lr 1e-4 --train-steps 20000

# Train on any mix of datasets (robot episodes or the 2D human demos)
uv run python -m ur_pusht.train --datasets ../pusht_action_chunking/data/pusht/pusht_cchi_v7_replay.zarr --out runs/human_only

# Figures and videos
uv run python -m ur_pusht.plots --runs runs/finetune --labels "fine-tuned" --out media
uv run python -m ur_pusht.render --policy checkpoints/pusht2d_flow.pt --seeds 7 --out media
```

Every command takes `--help`. Collection and evaluation run episodes in parallel; set `--workers`
to the number of CPU cores.

## Rendering without a display

On a desktop, MuJoCo renders through GLFW with no setup. On a headless machine, use EGL
(`MUJOCO_GL=egl`) if the GPU driver supports it, or the OSMesa software renderer:

```bash
conda create -p ~/envs/osmesa -c conda-forge mesalib=24.3   # Mesa >= 25 no longer ships OSMesa
OSMESA_PREFIX=~/envs/osmesa scripts/headless.sh uv run python -m ur_pusht.render ...
```

## Layout

| File | Contents |
|---|---|
| `assets/scene.xml` | table, T block, goal, cameras |
| `env.py` | scene assembly, IK controller, Push-T compatible `reset` / `step` |
| `geometry.py` | T shape, coverage, pixel to table coordinates |
| `policy.py` | model + normalizer bundles, import of 2D checkpoints |
| `rollout.py` | parallel episodes with open-loop action chunks |
| `collect.py`, `train.py`, `improve.py` | data collection, training, self-improvement rounds |
| `evaluate.py`, `plots.py`, `render.py` | evaluation, figures, videos |
