# 2D AI Dodgeball

A 2-player 2D top-down dodgeball game where two dots learn to play against each
other (or against heuristic bots) via **PPO + self-play + curriculum learning**.
Includes a polished Godot 4 arcade game built on the same physics/observation
contract as the Python training env.

Built to the spec in [`AI_Dodgeball_2D_Project_Plan.md`](./AI_Dodgeball_2D_Project_Plan.md).

---

## Table of contents

1. [Quick start](#quick-start)
2. [Play the game](#play-the-game-godot)
3. [Train an agent](#train-an-agent-python)
4. [Evaluate and record](#evaluate-and-record)
5. [Project layout](#project-layout)
6. [Configuration](#configuration)
7. [Troubleshooting](#troubleshooting)
8. [Status of trained models](#status-of-trained-models)

---

## Quick start

### Prerequisites

- **Python 3.10+** (tested with Anaconda Python 3.11)
- **CUDA-capable GPU** recommended for training (CPU also works, just slower)
- **Git** (for cloning + LFS for large model files, optional)
- **Godot 4.2+** (only required to play the game, not for training)

### Clone and install

```bash
git clone <repo-url>
cd "dodge ball"

# (Recommended) Create a fresh conda env
conda create -n dodgeball python=3.11 -y
conda activate dodgeball

# Install dependencies (CUDA 12.1 build of torch — change if you need CPU-only)
pip install -r requirements.txt
```

> If you're on CPU only, replace the `torch` line in `requirements.txt` with
> `torch --index-url https://download.pytorch.org/whl/cpu`.

### Verify the install (1 minute)

```bash
# Unit tests (physics, rewards, env API, snapshot opponent)
python -m unittest discover -s tests -v

# Or run them individually:
python tests/test_physics.py
python tests/test_rewards.py
python tests/test_env_api.py
python tests/test_snapshot_opponent.py

# Random-agent sanity check (rolls 3 episodes, saves MP4)
python scripts/sanity_random.py --episodes 3 --max-steps 400 --save-video
# Output: videos/random_sanity.mp4
```

If those all pass, you're ready to train.

---

## Play the game (Godot)

The Godot 4 game is fully playable with built-in heuristic bots. You do NOT
need a trained model to play.

1. Download Godot 4.2+ from https://godotengine.org
2. Open Godot → **Import** → select `godot-dodgeball/project.godot`
3. Press **F5** (or click ▶ Play) to launch

**Modes:**
- **Single Player** — you vs AI bot (Easy / Normal / Hard)
- **Watch AI vs AI** — two bots duel
- **Local 2 Player** — two humans on one keyboard

**Controls (Single Player):**
- `WASD` — move
- `Q` or `SPACE` — throw (must be holding the ball)
- Mouse — aim direction
- `R` — reset round, `ESC` — quit to menu

See [`godot-dodgeball/README.md`](./godot-dodgeball/README.md) for full controls.

---

## Train an agent (Python)

The training pipeline lives in `scripts/train.py`. It supports three modes:

### 1. Sanity training (5 min) — vs a stationary bot

```bash
python scripts/train.py --mode vs_bot --opponent stationary --phase 1 \
    --total-timesteps 200000 --num-envs 4 \
    --save-path models/sanity_p1_stationary
```

This is a fast smoke test. The agent should learn to walk over the ball
and pick it up within 100K steps.

### 2. Phase 1 vs heuristic bots (30-60 min) — recommended bootstrap

```bash
# vs normal heuristic bot (chases ball, throws at you)
python scripts/train.py --mode vs_bot --opponent heuristic --phase 1 \
    --total-timesteps 1000000 --num-envs 4 \
    --save-path models/dodgeball_p1_heuristic
```

### 3. Full self-play curriculum (hours)

```bash
# Trains Phases 1-4 with frozen-snapshot self-play (Bansal-style).
# Resume a prior run with --resume-from if it gets interrupted.
python scripts/train.py --mode selfplay \
    --total-timesteps 5000000 --num-envs 4 \
    --save-path models/dodgeball_final
```

**Resume an interrupted run:**
```bash
python scripts/train.py --mode selfplay \
    --resume-from models/snapshots/step_03400000.zip \
    --total-timesteps 5000000 --num-envs 4 \
    --save-path models/dodgeball_final
```

**Performance tuning:**
- `--num-envs N` — number of parallel envs (default 1, recommended 4-8)
- `--device cpu|cuda|auto` — auto detects GPU
- `--seed N` — for reproducibility
- `--phase N` — restrict to one curriculum phase (1-4)

Snapshots are auto-saved every `snapshot_interval` steps to
`models/snapshots/`. The opponent pool samples 80% latest / 20% older.

---

## Evaluate and record

### Deterministic eval (no rendering)

```bash
# 50 matches vs a bot, deterministic policy
python scripts/evaluate.py --model models/dodgeball_final \
    --matches 50 --opponent heuristic --phase 4
```

Output:
```
Matches: 50, opponent=heuristic, phase=4
  player_0 wins: 12
  player_1 wins: 35
  draws:         3
  avg episode length: 412.3
```

### Record video

```bash
# Save an MP4 of one match
python scripts/record_video.py --model models/dodgeball_final \
    --out videos/my_match.mp4 --opponent stationary \
    --phase 4 --max-steps 1500 --fps 30
```

### Export trained policy to ONNX

```bash
python scripts/export_onnx.py --model models/dodgeball_final \
    --out models/dodgeball_policy.onnx
```

This produces:
- `models/dodgeball_policy.onnx` — the policy MLP
- `models/dodgeball_policy.onnx.vecnorm.json` — VecNormalize stats

Wire these into the Godot game to play against the trained model
(requires an ONNX runtime GDExtension — see `godot-dodgeball/README.md`).

---

## Project layout

```
dodge ball/
├── README.md                      # ← you are here
├── AI_Dodgeball_2D_Project_Plan.md  # original spec
├── requirements.txt               # python deps
├── .gitignore
│
├── configs/                       # YAML configuration
│   ├── env.yaml                   #   arena + physics constants
│   ├── rewards.yaml               #   reward shaping per curriculum phase
│   └── ppo.yaml                   #   PPO hyperparameters + curriculum schedule
│
├── dodgeball/                     # main Python package
│   ├── env/
│   │   ├── entities.py            #   Arena / Agent / Ball + physics
│   │   ├── rewards.py             #   Reward function + curriculum context
│   │   └── dodgeball_parallel.py  #   PettingZoo ParallelEnv (17-dim obs, 5-dim act)
│   ├── scripted/
│   │   └── bots.py                #   Stationary / Random / Heuristic / HeuristicDodger
│   ├── selfplay/
│   │   ├── elo.py                 #   ELO rating
│   │   └── opponent_pool.py       #   Snapshot pool (Bansal-style)
│   ├── render/
│   │   └── pygame_renderer.py     #   RGB renderer (headless via SDL_VIDEODRIVER=dummy)
│   └── wrappers.py                #   gymnasium.Env wrapper for SB3
│
├── scripts/                       # entrypoints
│   ├── sanity_random.py           #   random-agent sanity check
│   ├── train.py                   #   training (curriculum + self-play)
│   ├── evaluate.py                #   deterministic eval over N matches
│   ├── record_video.py            #   export MP4 of a match
│   └── export_onnx.py             #   export trained policy to ONNX
│
├── tests/                         # unit tests
│   ├── test_physics.py            #   arena, agent, ball physics
│   ├── test_rewards.py            #   reward function + curriculum
│   ├── test_env_api.py            #   PettingZoo parallel_api_test
│   └── test_snapshot_opponent.py  #   self-play snapshot opponent obs shape
│
├── godot-dodgeball/               # Godot 4 arcade game
│   ├── README.md                  #   game-specific docs
│   ├── project.godot              #   project config + input map
│   ├── scenes/                    #   TitleScreen / MainMenu / Match / ResultScreen
│   └── scripts/                   #   game scripts (Agent / Ball / Arena / Match / etc.)
│
├── models/                        # ← gitignored, created by training
│   ├── *.zip                      #   trained PPO models
│   └── snapshots/                 #   intermediate snapshots for self-play
│
├── runs/                          # ← gitignored, TensorBoard logs
├── videos/                        # ← gitignored, recorded MP4s
└── logs/                          # ← gitignored
```

---

## Configuration

All hyperparameters live in YAML under `configs/`. Edit and re-train.

### `configs/env.yaml`

Arena + physics. Touch this if you want different arena size, ball speed, etc.

### `configs/rewards.yaml`

Reward coefficients per curriculum phase (1-4). Phase 1 has heavy shaping;
phase 4 is mostly sparse (just terminal hits).

### `configs/ppo.yaml`

- `ppo.*` — PPO hyperparameters (n_steps, batch_size, gamma, gae_lambda, etc.)
- `training.*` — total_timesteps, n_envs, tensorboard_log
- `curriculum.*` — phases list, snapshot_interval, opponent_pool_max, latest_opponent_prob

---

## Troubleshooting

### "Out of memory" / Windows paging file error

Windows sometimes can't mmap torch's CUDA DLLs after a long idle period.
Kill any leftover python processes:
```powershell
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
```
Then re-launch training.

### "Can't pickle local object" with SubprocVecEnv

This is a Windows + spawn-method issue with the PettingZoo→Gym wrapper
closure. The wrapper now uses `DummyVecEnv` for all `num_envs > 1`,
which is fine — env stepping is fast enough that parallelism isn't the
bottleneck.

### "ModuleNotFoundError: No module named 'onnx'" (for export_onnx.py)

```bash
pip install onnx onnxruntime
```

### Training is slow

The bottleneck is PPO gradient updates, not env stepping. Increase
`--num-envs` to give PPO more rollout data per `learn()` call:
```bash
python scripts/train.py ... --num-envs 8
```
You should see 3-5x throughput improvement on a single GPU.

### Tests fail with "mutable default" from hydra

This is a `hydra-core` 1.3+ dataclass bug, unrelated to our code.
Pin to `hydra-core<1.3` or just run tests with `python -m unittest`.

---

## Status of trained models

| Model file | Description | Eval win rate |
|---|---|---|
| `models/sanity_p1_stationary.zip` | 200K vs stationary, sanity smoke | ~0% (smoke only) |
| `models/dodgeball_p1_vsbot_v3.zip` | 1M vs stationary, fixes applied | 0% (96% pickup, throw aim unsolved) |
| `models/dodgeball_final.zip` | 3.5M self-play, full curriculum | 0% (warm-start only) |

**Known issue:** the trained agent navigates and picks up the ball reliably
but cannot land throws — `throw_alignment` reward fires only on the throw
event, so the agent gets no per-step feedback while aiming. See the project
plan for the long-term fix.

**The Godot game is fully playable against the heuristic bots regardless.**
They work well and are fun to play.

---

## License

Internal project — adjust before public release.