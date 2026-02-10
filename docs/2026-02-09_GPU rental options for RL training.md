# GPU Rental Options for RL Training

## TL;DR: Not Worth It for This Workload

For a [128, 128] MLP policy with 103-float observations trained with SB3 PPO across 8 vectorized CPU environments, renting a GPU would likely be a waste of money — and might actually slow things down.

## Why GPU Won't Help

The bottleneck in SB3 PPO for small MLP policies is environment stepping, which happens on CPU in Python/NumPy. During training, tensors shuttle between PyTorch (GPU) and NumPy (CPU) every rollout step. For a network this small, GPU-CPU transfer overhead actually makes GPU training slower than CPU. The SB3 docs explicitly state: "PPO is meant to be run primarily on the CPU, especially when you are not using a CNN."

1M steps on an M-series Mac with 8 parallel envs takes roughly 5-15 minutes. A GPU wouldn't meaningfully change that.

## Platforms (If You Wanted to Try Anyway)

| Platform | Cheapest GPU | Price | Interface |
|----------|-------------|-------|-----------|
| [Vast.ai](https://vast.ai/pricing) | RTX 4090 (24GB) | ~$0.24-0.60/hr | Docker container with SSH + Jupyter. Peer-to-peer marketplace (variable reliability). |
| [RunPod](https://www.runpod.io/pricing) | RTX 4090 (24GB) | ~$0.34/hr | Docker template with SSH + JupyterLab + TensorBoard. More polished UX. |
| [Lambda Labs](https://lambdalabs.com/service/gpu-cloud) | A10 (24GB) | ~$0.75/hr | SSH into a full Ubuntu VM with PyTorch pre-installed. |

The interface for all of them: pick a Docker template (or bring your own), get SSH access to a machine with CUDA drivers pre-installed, rsync your code over, run it. RunPod and Vast.ai also give JupyterLab in-browser. Billing is per-second or per-hour; no minimum commitment on most.

## Cost Estimate for 1M Steps

At $0.34/hr, if training takes 15 minutes: ~$0.09. Even 10M steps would be under a dollar. The cost is dominated by setup time (uploading code, installing deps), not compute.

## What Would Actually Speed Things Up

1. **More parallel CPU environments** — increase `n_envs` from 8 to 16 or 32. Single biggest win for this workload.
2. **Faster CPU** — cloud VMs with high single-thread perf (AMD EPYC, Intel Sapphire Rapids).
3. **GPU-native environments** — frameworks like NVIDIA Isaac Gym or Brax. Requires rewriting BoxworldEnv in JAX/CUDA — massive undertaking for marginal benefit on a 10x10 grid.

## Sources

- [SB3 PPO GPU/VecEnv optimization discussion (Issue #314)](https://github.com/DLR-RM/stable-baselines3/issues/314)
- [SB3 PPO documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [Vast.ai pricing](https://vast.ai/pricing)
- [RunPod pricing](https://www.runpod.io/pricing)
- [Top Cloud GPU Providers 2026 (Hyperstack)](https://www.hyperstack.cloud/blog/case-study/top-cloud-gpu-providers)
- [Cheapest cloud GPU providers 2026 (Northflank)](https://northflank.com/blog/cheapest-cloud-gpu-providers)
