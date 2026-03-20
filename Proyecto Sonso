# Sonso IA v12b - Bipedal Walking Agent with Reinforcement Learning

A 2D bipedal robot that learns to walk from scratch using **Soft Actor-Critic (SAC)** reinforcement learning, built with a custom Gymnasium environment powered by Pymunk physics.

## Demo

The agent learns alternating gait locomotion over obstacles with domain randomization for robustness.

| Metric | Final Value |
|---|---|
| Algorithm | SAC (Soft Actor-Critic) |
| Total Timesteps | 4,000,000 |
| Peak Episode Reward | +2,970 |
| Peak Episode Length | 643 steps (~10.7 sec) |
| Final Converged Reward | ~2,900 |
| Training Time | ~6.5 hours (CPU only) |

## Architecture

### Environment (`SonsoEnv`)

Custom Gymnasium environment featuring:

- **Physics Engine:** Pymunk 2D rigid body simulation
- **Robot:** 5-part articulated biped (torso + 2 thighs + 2 feet) with 4 motor-controlled joints
- **Observation Space:** 30-dimensional continuous (body state, joint angles, LIDAR rays, contact sensors, goal velocity)
- **Action Space:** 4-dimensional continuous (hip and knee motor rates for each leg)
- **Simulation:** 4 substeps at 240Hz per environment step (effective 60Hz control)

### Reward Design (Multi-Objective)

| Component | Weight | Description |
|---|---|---|
| **Velocity Tracking** | 1.5 | Gaussian reward centered on goal velocity |
| **Forward Progress** | 0.3 | Linear bonus for positive velocity |
| **Posture** | 0.8 + 0.4 | Upright torso angle + target height maintenance |
| **Alternating Gait** | 2.0 | Reward for left-right foot alternation |
| **Rhythmic Cadence** | 0.5 | Bonus for 20-80 step intervals between footfalls |
| **Anti-Hopping** | -0.3 | Penalty for >8 steps airborne |
| **Energy Efficiency** | -0.02 | L2 penalty on actions |
| **Jerk Penalty** | -0.01 | Smoothness penalty on action changes |
| **Safety** | -5 to -20 | Termination on torso/thigh ground contact |

### Domain Randomization

Each episode randomizes:
- **Gravity:** 850-950 m/s^2 (+-5% around standard)
- **Ground Friction:** 0.8-1.2
- **Obstacle Height:** 20-45px, randomly placed along the path

### Training Pipeline

- **Algorithm:** SAC with automatic entropy coefficient tuning
- **Parallelization:** 6 SubprocVecEnv workers (optimized for 8-core CPU)
- **Normalization:** VecNormalize (observations + rewards)
- **Replay Buffer:** 600K transitions (~250MB, optimized for 16GB RAM)
- **Gradient Steps:** 2 per environment step
- **Action Smoothing:** Exponential moving average (alpha=0.6)

## Training Curve

```
Step 0K:    reward=-64     ep_len=53     | Learning to stand
Step 200K:  reward=+1,130  ep_len=934    | First walking policy (fragile)
Step 400K:  reward=+377    ep_len=75     | Policy collapse (expected)
Step 800K:  reward=+614    ep_len=117    | Recovery phase
Step 1.2M:  reward=+999    ep_len=245    | Robust recovery
Step 1.6M:  reward=+1,530  ep_len=377    | Surpassed initial peak
Step 2.0M:  reward=+2,310  ep_len=473    | Consistent improvement
Step 2.4M:  reward=+2,720  ep_len=561    | New record
Step 3.2M:  reward=+2,970  ep_len=619    | Peak performance
Step 4.0M:  reward=+2,910  ep_len=598    | Converged (stable)
```

## Bug Fixes Applied (v12b)

12 bugs were identified and fixed from the original codebase:

1. **`ent_coef` fixed at 0.1** - Changed to `'auto'` for dynamic entropy tuning (SAC's core advantage)
2. **AABB collision missing `half_w`** - Added per-body width to geometric contact detection
3. **`GROUND_Y` off by 1px** - Corrected from 544 to 545 (Segment radius=5 at y=550)
4. **Dead imports (`torch.nn`, `math`)** - Removed, saving ~1.6GB RAM across subprocesses
5. **Raycast ShapeFilter using XOR** - Changed to AND NOT for clarity (`& ~2`)
6. **Observation height reference mismatch** - Aligned obs (was 450) with reward reference (460)
7. **`buffer_size` 1M too large for 16GB RAM** - Reduced to 600K
8. **Dual-foot contact giving double reward** - Added simultaneous landing protection
9. **Missing `device="cpu"`** - Added explicit CPU device (AMD GPU, no CUDA)
10. **Cadence reward using `elif`** - Changed to independent `if` blocks
11. **Bare `except: pass`** - Replaced with specific `(TypeError, OverflowError)`
12. **Ground friction ignoring DR** - `suelo.friction` now uses `self.friccion_dr`

## Requirements

```
gymnasium>=0.29.0
pymunk>=6.0.0
pygame>=2.1.0
stable-baselines3>=2.0.0
numpy
torch
```

## Usage

### Train from scratch
```bash
python main_ppo.py
```

### Watch trained agent
```python
from main_ppo import SonsoEnv
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

env = DummyVecEnv([lambda: SonsoEnv(render_mode="human")])
env = VecNormalize.load("sonso_v12b_gait_fixed_vecnorm.pkl", env)
env.training = False
model = SAC.load("sonso_v12b_gait_fixed", device="cpu")

obs = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
```

## Hardware Used

| Component | Spec |
|---|---|
| CPU | AMD Ryzen 7 5700X (8C/16T) |
| GPU | AMD Radeon RX 570 8GB (not used - no CUDA) |
| RAM | 16 GB DDR4 |
| Training | CPU-only, 6 parallel environments |

## Project Structure

```
.
|-- main_ppo.py              # Environment + Training script
|-- sonso_v12b_gait_fixed.zip        # Trained SAC model (4M steps)
|-- sonso_v12b_gait_fixed_vecnorm.pkl # VecNormalize statistics
|-- models/                  # Checkpoints every 300K steps
|-- logs/                    # Training logs
|-- tensorboard/             # TensorBoard logs
```

## Future Plans

- **v13:** Re-enable external push perturbations for balance robustness
- **3D Migration:** Port to PyBullet/MuJoCo with Blender visualization
- **Larger Networks:** 1024-neuron MLP with GPU acceleration (planned RTX 5070)

## License

MIT

---

*Built with Python, Gymnasium, Pymunk, and Stable-Baselines3.*
