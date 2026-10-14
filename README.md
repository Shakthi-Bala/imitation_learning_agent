# Imitation Learning on Push-T

Teaching a robot to push a T-shaped block into a goal zone by imitating 206 human demonstrations.

<p align="center">
  <img src="pusht_action_chunking/report/flow_vs_mse.gif" width="420" alt="Same starting position: the MSE policy gets stuck beside the T, the flow matching policy brings it back to the goal">
</p>

The policy sees a 5-number state (agent position, block position and angle) and predicts the next **8 actions at once** instead of one at a time. This is called action chunking, and it is the idea behind modern robot-learning methods such as [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/).

I trained two policies with the same network and training budget:

- **MSE:** predicts the 8 actions directly in one pass.
- **Flow matching:** starts from random noise and refines it into an action chunk over 10 small steps, like a simpler form of diffusion.

This is Homework 1 of UC Berkeley's [CS 185/285 Deep Reinforcement Learning](https://rail.eecs.berkeley.edu/deeprlcourse/) (Spring 2026).

## Results

| | MSE | Flow matching |
|---|---|---|
| Final evaluation reward | 0.629 (target 0.5) | **0.817** (target 0.7) |
| Episodes fully solved, out of 100 | 7 | **17** |
| Clear failures (reward < 0.3), out of 100 | 22 | **8** |
| Training time | ~7 min, 8 CPU cores | ~17 min, 4 CPU cores |

Reward is the average over 100 test episodes of how much of the goal the block covers (1.0 = perfect).

<p align="center">
  <img src="pusht_action_chunking/report/mse_training_curves.png" width="49%" alt="MSE training loss and evaluation reward">
  <img src="pusht_action_chunking/report/flow_training_curves.png" width="49%" alt="Flow matching training loss and evaluation reward">
</p>

MSE levels off around 0.63. When the demonstrations show two different ways to push the block, MSE predicts their average, which is often neither. In the GIF above it parks beside the T and never moves again. Flow matching samples one of the options instead, so it keeps trying and recovers. The cost is that it sometimes hesitates between strategies and rarely holds perfectly still at the goal.

Reports: [MSE (PDF)](pusht_action_chunking/report/mse_report.pdf) · [Flow matching (PDF)](pusht_action_chunking/report/flow_report.pdf) · [rollout videos](pusht_action_chunking/report/videos/)

## How it works

- **Model:** an MLP with 3 hidden layers of 256 units and ReLU, about 140k parameters, for both policies.
- **MSE loss:** squared error between the predicted chunk and the human's chunk.
- **Flow matching loss:** mix the human's chunk with noise at a random level τ, and train the network to predict the direction from the noise to the chunk. At test time, start from noise and take 10 Euler steps along that direction.
- **Training:** Adam (learning rate 3e-4), batch size 128, 400 epochs, `torch.compile` for speed.
- **Evaluation:** every 10k steps the policy plays 100 episodes in the simulator, running each 8-action chunk open-loop before re-planning.

My code is in [`model.py`](pusht_action_chunking/src/hw1_imitation/model.py) (`MSEPolicy`, `FlowMatchingPolicy`), [`train.py`](pusht_action_chunking/src/hw1_imitation/train.py) (training loop) and [`plot_curves.py`](pusht_action_chunking/plot_curves.py). The data loading, environment and evaluation code is the course's starter code.

## Reproduce it

You need Linux or macOS and [uv](https://docs.astral.sh/uv/). The dataset (31 MB) downloads automatically on the first run.

```bash
git clone https://github.com/Shakthi-Bala/imitation_learning_agent.git
cd imitation_learning_agent/pusht_action_chunking

uv sync                    # install dependencies
uv run wandb login         # optional: log runs to Weights & Biases

uv run src/hw1_imitation/train.py --policy-type mse
uv run src/hw1_imitation/train.py --policy-type flow
```

No W&B account? Prefix the training command with `WANDB_MODE=offline` and everything is still saved locally.

Each run writes to `pusht_action_chunking/exp/<run_name>/`. To plot its curves:

```bash
uv run --with matplotlib plot_curves.py exp/<run_name> --title "Flow matching policy" --target 0.7
```

All defaults (seed 42) match the results above. Flow matching samples random noise, so its exact reward varies slightly from run to run. Run `uv run src/hw1_imitation/train.py --help` to see every option.

## Status

- [x] MSE action-chunking policy
- [x] Flow matching policy
