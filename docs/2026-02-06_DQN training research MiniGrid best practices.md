# DQN Training Research: MiniGrid Best Practices

## Context

After implementing subgoal-chain reward shaping, multi-door procedural levels, larger network [256,256,128], and tuned hyperparameters, the Boxworld agent can open doors in Four Rooms but performance on other levels regressed. This prompted research into how the MiniGrid community trains agents on grid-world key-door puzzles.

## Key Findings

### 1. Observation Encoding — Raw Integers Are Wrong

Our `_get_obs()` encodes cells as integers 0–5 (Floor=0, Wall=1, Door=2, Key=3, Goal=4, Lava=5). This imposes a false ordinal relationship — the network interprets "Lava (5) is five times more than Wall (1)" and "Key (3) is the average of Door and Goal." This wastes capacity learning that these values are categorical.

**What researchers do:**
- **One-hot encoding** per cell: each cell becomes a 6-element binary vector. For 6 cell types on a 10×10 grid, observation goes from 103 to 603 values
- **Multi-channel binary masks** fed to a CNN: grid becomes `(6, 10, 10)` — one binary channel per cell type
- MiniGrid's official wrappers (`OneHotPartialObsWrapper`, `ImgObsWrapper`, `FlatObsWrapper`) all avoid raw integers

### 2. CNN vs MLP — CNN Strongly Preferred

Flattening a 2D grid into an MLP destroys spatial relationships. The network must learn from scratch that cell (3,4) and cell (3,5) are adjacent, while a CNN encodes this structurally.

**MiniGrid convention:**
- Small **2×2 kernels** (not 4×4 or 8×8 from Atari — grids are too small)
- Architecture: `Conv2d(16) → Conv2d(32) → Conv2d(64)` with ReLU, then flatten → Linear(features_dim)
- SB3's `CnnPolicy` with a custom `BaseFeaturesExtractor` subclass
- SB3 Issue #809 confirms larger kernels cause errors on small grids

### 3. PPO Dominates, DQN Absent

**PPO is the standard algorithm for MiniGrid.** Evidence:
- Official MiniGrid training docs feature only PPO
- RL Zoo (SB3's tuned hyperparameter repo) has **zero DQN configs** for any MiniGrid environment, but tuned PPO for 12+
- `rl-starter-files` (official MiniGrid starter code) supports only PPO and A2C
- PPO with `n_envs=8` solves DoorKey-5x5 in **100k steps** with mean reward 0.97

**Why PPO wins here:**
- On-policy stability, less hyperparameter sensitivity
- Parallel environments (8 envs standard) provide natural exploration diversity
- Entropy regularization encourages policy diversity

**RL Zoo reference config for MiniGrid-DoorKey-5x5:**
- `n_envs=8`, `n_steps=128`, `batch_size=64`, `n_epochs=10`
- `lr=2.5e-4`, `gamma=0.99`, `gae_lambda=0.95`, `ent_coef=0.0`
- `normalize=true` (VecNormalize wrapper)
- Solves in 100k steps

### 4. Exploration — State Count Beats Intrinsic Motivation

A 2025 Springer study on DoorKey-16×16 found:

| Method | Result |
|--------|--------|
| **State Count** | Best performer, first to find task reward |
| ICM | Second best, lower sample efficiency |
| Max Entropy | Struggles |
| DIAYN | Fails completely |

State count works especially well with low-dimensional observations (like grid encodings). For RGB observations, it degrades.

### 5. Rainbow DQN Components (If Staying with DQN)

From the Rainbow ablation study, ranked by impact:
- **Multi-step learning** — Critical
- **Distributional RL** — Critical
- **Prioritized Experience Replay (PER)** — Critical for sparse rewards
- **Noisy Nets** — Significant (replaces epsilon-greedy)
- **Dueling architecture** — Marginal
- **Double DQN** — Marginal

SB3 base DQN has none of these. `sb3-contrib` provides QR-DQN (distributional).

## Comparison: Our Setup vs Best Practices

| Aspect | Our Setup | Best Practice |
|--------|-----------|---------------|
| Algorithm | DQN | PPO |
| Obs format | Flat integers (103-dim) | One-hot grid image (H,W,C) |
| Network | MLP [256,256,128] | CNN with 2×2 kernels |
| Exploration | Epsilon-greedy | State count bonus / NoisyNet |
| Parallel envs | 1 | 8 |
| Obs normalization | No | Yes (VecNormalize) |
| Training steps | 1M | 100k (DoorKey-5x5), 5–10M (harder) |

## Recommended Changes (Ranked by Impact)

### High Impact
1. **One-hot observation encoding** — eliminate false ordinal relationships
2. **CNN with spatial grid** — preserve spatial structure, use 2×2 kernels
3. **Switch to PPO** — the algorithm the entire MiniGrid community uses

### Medium Impact
4. **State count exploration bonus** — top performer on hard key-door tasks
5. **Prioritized Experience Replay** — if staying with DQN
6. **Observation normalization** — VecNormalize, used in all RL Zoo MiniGrid configs

### Lower Impact
7. **Multi-step returns** — faster reward propagation in long episodes
8. **Progressive curriculum** — start easy, increase difficulty dynamically
9. **Longer exploration / NoisyNet** — state-dependent exploration

## What We Keep

Our subgoal-chain reward shaping is theoretically sound (potential-based, preserves optimal policies). It should complement the algorithm/representation fixes, not replace them. The multi-door procedural generation and weighted level sampling are also good training diversity mechanisms.

## Sources

- [MiniGrid Training Documentation](https://minigrid.farama.org/content/training/)
- [MiniGrid Wrappers](https://minigrid.farama.org/api/wrappers/)
- [RL Baselines3 Zoo — PPO hyperparameters](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml)
- [HuggingFace sb3/ppo-MiniGrid-DoorKey-5x5-v0](https://huggingface.co/sb3/ppo-MiniGrid-DoorKey-5x5-v0)
- [SB3 Issue #809 — Kernel size with MiniGrid](https://github.com/DLR-RM/stable-baselines3/issues/809)
- [Impact of Intrinsic Rewards on Exploration (Springer 2025)](https://link.springer.com/article/10.1007/s00521-025-11340-0)
- [Rainbow DQN (AAAI 2018)](https://cdn.aaai.org/ojs/11796/11796-13-15324-1-2-20201228.pdf)
- [Exploration Strategies in Deep RL (Lilian Weng)](https://lilianweng.github.io/posts/2020-06-07-exploration-drl/)
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
