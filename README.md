# 2D AI Dodgeball

A 2-player 2D top-down dodgeball game where two dots learn to play against each
other (or against heuristic bots) via **PPO + self-play + curriculum learning**.
Includes a polished Godot 4 arcade game built on the same physics/observation
contract as the Python training env.

Built to the spec in [`AI_Dodgeball_2D_Project_Plan.md`](./AI_Dodgeball_2D_Project_Plan.md).

---

## Table of contents

1. [Quick start](#quick-start)
2. [Reproducing trained models (v6 → v10)](#reproducing-trained-models-v6--v10)
3. [Play the game](#play-the-game-godot)
4. [Train an agent](#train-an-agent-python)
5. [Evaluate and record](#evaluate-and-record)
6. [Project layout](#project-layout)
7. [Configuration](#configuration)
8. [Troubleshooting](#troubleshooting)
9. [Status of trained models](#status-of-trained-models)

---

## Quick start

### Prerequisites

- **Python 3.10+** (tested with Anaconda Python 3.11)
- **CUDA-capable GPU** recommended for training (CPU also works, just slower)
- **Git**
- **Godot 4.2+** (only required to play the game, not for training)

### Clone and install

```bash
git clone https://github.com/Bao-NQ06/dodge-ball.git
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
# Run each test file directly (pytest collection hits an unrelated
# pre-existing hydra/import bug on some setups):
python tests/test_physics.py
python tests/test_rewards.py
python tests/test_env_api.py
python tests/test_snapshot_opponent.py
python tests/test_throw_escape.py        # regression: thrower self-catch fix

# Random-agent sanity check (rolls 3 episodes, saves MP4)
python scripts/sanity_random.py --episodes 3 --max-steps 400 --save-video
# Output: videos/random_sanity.mp4
```

If those all pass, you're ready to train. For the recommended trained
checkpoint, jump to [v10](#v10----holder-bot-dodging-learned) — it's
the best deployable model.

---

## Reproducing trained models (v6 → v10)

The trained checkpoints are the result of an ablation sweep on a single
bug fix (the thrower self-catch) plus entropy and reward shaping. Each
model is a 5M-step, 4-phase curriculum run that takes ~70–75 min on a
single GPU (~1290 fps).

All three commands below:

- Train **5M steps** across all 4 curriculum phases (1.25M each).
- Use **8 parallel envs** (default), the v8 entropy anneal
  (`ent_coef: {start: 0.01, end: 0.005}` in `configs/ppo.yaml`).
- Write snapshots to an **isolated** `models/snapshots_vN/` so prior
  runs aren't overwritten (each snapshot is named by step number and
  would otherwise collide).

The current `configs/rewards.yaml` matches **v9 and v10** out of the
box. **v8** needs `dodge_held` zeroed first (v8 was trained before that
reward existed).

### v8 — engagement + accurate throws (the thrower-tuned baseline)

Trained vs the `heuristic` (thrower) bot. Wins 10/0/0 deterministically
vs both `heuristic` and `heuristic_dodger`. Did **not** learn dodging
(it didn't face a held ball long enough).

**One-time config edit** — `configs/rewards.yaml`, set `dodge_held: 0.0`
in phases 2, 3, and 4 (phase 1 is already 0):

```yaml
  2:
    ...
    dodge_held: 0.0   # was 0.10
  3:
    ...
    dodge_held: 0.0   # was 0.30
  4:
    ...
    dodge_held: 0.0   # was 0.10
```

Then train:

```bash
python scripts/train.py --mode vs_bot --opponent heuristic \
    --total-timesteps 5000000 \
    --snapshot-root models/snapshots_v8 \
    --save-path models/dodgeball_p1_vsbot_v8
```

> After v8 finishes, restore `dodge_held` in `configs/rewards.yaml`
> before running v9 or v10 (or those runs won't reproduce).

### v9 — + dodge_held reward (no dodging learned, opp-hold window too brief)

Trained vs the `heuristic` bot **with** the `dodge_held` reward on.
Default configs match. The reward was correctly wired but the thrower
bot only holds the ball for ~1 step before throwing, so `dodge_held`
fires too briefly to outweigh the pickup+throw cycle. The agent
became **more** throwing-aggressive, not more dodging.

```bash
python scripts/train.py --mode vs_bot --opponent heuristic \
    --total-timesteps 5000000 \
    --snapshot-root models/snapshots_v9 \
    --save-path models/dodgeball_p1_vsbot_v9
```

### v10 — + holder bot (DODGING LEARNED)

Trained vs the `heuristic_holder` bot (picks up the ball, stands still
for 30 steps, then throws). That 30-step held window finally gives
`dodge_held` a long enough signal to outweigh grabbing, and the
agent learns to **back off while opp has the ball, dodge the throw,
counter-attack**. Wins 10/0/0 vs all three scripted bots
(`heuristic`, `heuristic_dodger`, `heuristic_holder`).

```bash
python scripts/train.py --mode vs_bot --opponent heuristic_holder \
    --total-timesteps 5000000 \
    --snapshot-root models/snapshots_v10 \
    --save-path models/dodgeball_p1_vsbot_v10
```

### Useful flags

- `--resume-from <path>` — continue a previous run (loads weights +
  VecNormalize stats). Only useful with the exact same config.
- `--snapshot-root <dir>` — where to save snapshot checkpoints
  (default `models/snapshots/`).
- `--num-envs N` — parallel envs (default 8).
- `--device cpu|cuda|auto` — default `auto`. Note: SB3 warns that PPO
  on GPU with an MLP policy is often *slower* than CPU due to transfer
  overhead; you can pass `--device cpu` to force CPU.

---

## Play the game (Godot)

The Godot 4 frontend lives in `godot-dodgeball/`. Open it in the
Godot editor and run; play against the scripted bots (no model
needed) or wire in the ONNX export — see that folder's README.

---

## Train an agent (Python)

The training pipeline lives in `scripts/train.py`. For full-curriculum
reproductions, see [Reproducing trained models](#reproducing-trained-models-v6--v10)
above. For other entrypoints:

### Sanity training (5 min) — vs a stationary bot

```bash
python scripts/train.py --mode vs_bot --opponent stationary --phase 1 \
    --total-timesteps 200000 --num-envs 4 \
    --save-path models/sanity_p1_stationary
```

Fast smoke test. The agent should learn to walk over the ball and
pick it up within ~100K steps.

### Phase 1 vs heuristic bot (30-60 min) — recommended bootstrap

```bash
python scripts/train.py --mode vs_bot --opponent heuristic --phase 1 \
    --total-timesteps 1000000 --num-envs 4 \
    --save-path models/dodgeball_p1_heuristic
```

### Self-play with snapshot opponent pool

```bash
python scripts/train.py --mode selfplay \
    --total-timesteps 5000000 --num-envs 4 \
    --save-path models/dodgeball_final
```

### Resume an interrupted run

```bash
python scripts/train.py --mode selfplay \
    --resume-from models/snapshots/step_03400000.zip \
    --total-timesteps 5000000 --num-envs 4 \
    --save-path models/dodgeball_final
```

Snapshots are auto-saved every `snapshot_interval` steps to the
snapshot root. The opponent pool samples 80% latest / 20% older.

---

## Evaluate and record

### ELO learning curve across snapshots — `eval_elo.py`

Compares each saved snapshot against a fixed reference opponent and
prints a table of win-rate and a one-shot ELO performance rating
(anchored at 1200). This is the tool to check whether your run
actually improved over training.

```bash
python scripts/eval_elo.py \
    --snapshots "models/snapshots_v10/step_*.zip" \
    --vecnorm models/dodgeball_p1_vsbot_v10_vecnorm.pkl \
    --ref heuristic
```

Reference can be any bot name (`heuristic`, `heuristic_dodger`,
`heuristic_holder`) or a model `.zip` path. Add `--stochastic` to
sample actions instead of using the deterministic argmax (useful to
check sampling robustness). Output goes to `models/eval_elo.json`.

### Behavior trace — `diag_throw.py`

For a single model, reports per-match: wins/draws/losses, pickups,
throws, ball-possession breakdown, agent move distance, throw-trigger
behaviour, and the **dodge metric**: mean agent↔opponent distance
both overall and specifically *while the opponent holds the ball*.

```bash
python scripts/diag_throw.py \
    --model models/dodgeball_p1_vsbot_v10.zip \
    --vecnorm models/dodgeball_p1_vsbot_v10_vecnorm.pkl \
    --opponent heuristic_holder
```

`p0↔opp dist (opp holds): …` is the number to watch when comparing
v8 vs v9 vs v10: it jumps dramatically only when the model has
actually learned to back off while opp has the ball.

### Match counts eval — `evaluate.py`

```bash
python scripts/evaluate.py --model models/dodgeball_final \
    --matches 50 --opponent heuristic --phase 4
```

### Record a match to MP4 — `record_video.py`

```bash
python scripts/record_video.py --model models/dodgeball_p1_vsbot_v10.zip \
    --opponent heuristic_holder --out videos/v10_vs_holder.mp4
```

Tip: the wrapper used by this script resets on done and overwrites
the terminal `infos`, so the printed "winner" is unreliable — but
the recorded video itself is the real match. For accurate behaviour
metrics use `diag_throw.py`.

### Export trained policy to ONNX

```bash
python scripts/export_onnx.py --model models/dodgeball_p1_vsbot_v10 \
    --out models/dodgeball_policy.onnx
```

This produces `models/dodgeball_policy.onnx` and a `*.onnx.vecnorm.json`
sidecar. Wire these into the Godot game (requires an ONNX runtime
GDExtension — see `godot-dodgeball/README.md`).

---

## Project layout

```
dodge ball/
├── README.md                         # ← you are here
├── AI_Dodgeball_2D_Project_Plan.md   # original spec
├── requirements.txt                  # python deps
├── .gitignore
│
├── configs/                          # YAML configuration
│   ├── env.yaml                      #   arena + physics constants
│   ├── rewards.yaml                  #   reward shaping per curriculum phase
│   └── ppo.yaml                      #   PPO hyperparameters + curriculum schedule
│
├── dodgeball/                        # main Python package
│   ├── env/
│   │   ├── entities.py               #   Arena / Agent / Ball + physics
│   │   │                             #   (incl. THROWER_PICKUP_COOLDOWN)
│   │   ├── rewards.py                #   Reward function + curriculum context
│   │   │                             #   (incl. near_miss_bonus + dodge_held)
│   │   └── dodgeball_parallel.py     #   PettingZoo ParallelEnv (17-dim obs, 5-dim act)
│   ├── scripted/
│   │   └── bots.py                   #   Stationary / Random / Heuristic
│   │                                 #   / HeuristicDodger / HeuristicHolder
│   ├── selfplay/
│   │   ├── elo.py                    #   ELO rating
│   │   └── opponent_pool.py          #   Snapshot pool (Bansal-style)
│   ├── render/
│   │   └── pygame_renderer.py        #   RGB renderer (headless via SDL_VIDEODRIVER=dummy)
│   └── wrappers.py                   #   gymnasium.Env wrapper for SB3
│
├── scripts/                          # entrypoints
│   ├── sanity_random.py              #   random-agent sanity check
│   ├── train.py                      #   training (curriculum + self-play + anneal)
│   ├── evaluate.py                   #   deterministic eval over N matches
│   ├── diag_throw.py                 #   per-match behaviour trace + dodge metric
│   ├── eval_elo.py                   #   snapshot-vs-ref ELO learning curve
│   ├── record_video.py               #   export MP4 of a match
│   └── export_onnx.py                #   export trained policy to ONNX
│
├── tests/                            # unit tests
│   ├── test_physics.py               #   arena, agent, ball physics
│   ├── test_rewards.py               #   reward function + curriculum
│   ├── test_env_api.py               #   PettingZoo parallel_api_test
│   ├── test_snapshot_opponent.py     #   self-play snapshot opponent obs shape
│   └── test_throw_escape.py          #   regression: thrower can't self-catch
│
├── models/                           # ← gitignored, created by training
│   ├── *.zip                         #   trained PPO models
│   ├── *_vecnorm.pkl                 #   VecNormalize stats per model
│   └── snapshots_vN/                 #   isolated snapshot dirs per run
│
├── runs/                             # ← gitignored, TensorBoard logs
├── videos/                           # ← gitignored, recorded MP4s
└── godot-dodgeball/                  # Godot 4 arcade game
```

---

## Configuration

All hyperparameters live in YAML under `configs/`. Edit and re-train.

### `configs/env.yaml`

Arena + physics. Touch this if you want different arena size, ball
speed, etc.

### `configs/rewards.yaml`

Reward coefficients per curriculum phase (1-4). Phase 1 has heavy
shaping; phase 4 is mostly sparse. The `dodge_held` coefficient
controls the "back off when opp has the ball" reward — set it to
`0.0` in all phases to reproduce v8.

### `configs/ppo.yaml`

- `ppo.*` — PPO hyperparameters (n_steps, batch_size, gamma,
  gae_lambda, etc.). `ent_coef` is the entropy coefficient; the
  current value `{start: 0.01, end: 0.005}` is a linear anneal
  across the whole run (see `scripts/train.py` for the per-chunk
  progress wiring). To reproduce the v6 baseline (constant
  entropy), set `ent_coef: 0.01`.
- `training.*` — total_timesteps, n_envs, tensorboard_log.
- `curriculum.*` — phases list, snapshot_interval,
  opponent_pool_max, latest_opponent_prob.

### Scripted bots (`dodgeball/scripted/bots.py`)

- `stationary` — does nothing (phase-1 dummy).
- `random` — uniform random actions.
- `heuristic` — chases ball, throws at opponent when holding.
- `heuristic_dodger` — dodges incoming throws perpendicular to
  the ball's trajectory.
- `heuristic_holder` — picks up the ball, **holds** for 30 steps
  (no throw trigger), then throws. Used to train v10 so that
  `dodge_held` has a sustained window to shape behaviour.

---

## Troubleshooting

### "Out of memory" / Windows paging file error

Windows sometimes can't mmap torch's CUDA DLLs after a long idle
period. Kill any leftover python processes:
```powershell
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
```
Then re-launch training.

### "Can't pickle local object" with SubprocVecEnv

Windows + spawn-method issue with the PettingZoo→Gym wrapper
closure. The wrapper uses `DummyVecEnv` for all `num_envs > 1`,
which is fine — env stepping is fast enough that parallelism isn't
the bottleneck.

### `pytest` collection fails with `hydra` "mutable default" error

Pre-existing environment issue with `hydra-core` 1.3+ on some
Python versions — unrelated to this codebase. Run the tests
directly with `python tests/test_<name>.py` instead, or pin
`hydra-core<1.3`.

### `ModuleNotFoundError: No module named 'onnx'` (for `export_onnx.py`)

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

SB3 also warns that PPO on GPU with an MLP policy is often *slower*
than CPU due to transfer overhead. Pass `--device cpu` to force CPU.

### Eval shows 0% win-rate but training metrics looked fine

Two common causes:

1. **Snapshots from different runs got mixed.** The default
   `models/snapshots/` collides by step number across runs. Always
   pass `--snapshot-root models/snapshots_vN` per run (or use git
   branches) — that's why `train.py` accepts the flag.
2. **VecNormalize stats mismatch.** Each model has its own
   `<model>_vecnorm.pkl`. Pass the matching one to `eval_elo.py` /
   `diag_throw.py` via `--vecnorm`.

---

## Status of trained models

The repo's `models/` is gitignored; these are the checkpoints the
training commands above produce.

| Model | Trained vs | Entropy | `dodge_held` | Eval (det, 10 matches) |
|---|---|---|---|---|
| `dodgeball_p1_vsbot_v6.zip` | `heuristic` | const 0.01 | — | vs heuristic: 10/0/0 |
| `dodgeball_p1_vsbot_v7.zip` | `heuristic` | anneal 0.01→0.001 | — | vs heuristic: 0/0/10 (overshot — always throws) |
| `dodgeball_p1_vsbot_v8.zip` | `heuristic` | anneal 0.01→0.005 | — | vs heuristic: 10/0/0, vs dodger: 10/0/0 |
| `dodgeball_p1_vsbot_v9.zip` | `heuristic` | anneal 0.01→0.005 | yes | vs heuristic: 10/0/0 (dodge didn't transfer — opp-hold window was 1 step) |
| **`dodgeball_p1_vsbot_v10.zip`** | **`heuristic_holder`** | anneal 0.01→0.005 | yes | vs heuristic: 10/0/0, vs dodger: 10/0/0, vs holder: 10/0/0 |

**Recommended deployable model: v10.** It generalizes to all three
scripted bots and exhibits the intended conditional behaviour:
when the opponent has the ball, it backs off; when the ball is
free, it approaches and grabs.

### Key engineering decisions behind these models

- **Thrower self-catch fix** ([entities.py](dodgeball/env/entities.py)):
  a thrown ball travels 17.3 units in one step but `pickup_radius`
  is 18, so the thrower used to instantly re-catch its own throw —
  the ball never left the hand, the scripted bot monopolised
  possession, and the policy learned to be passive. The
  `THROWER_PICKUP_COOLDOWN = 10` lets the ball clear the thrower.
- **`near_miss_bonus` wired** ([rewards.py](dodgeball/env/rewards.py)):
  the coefficient was in `rewards.yaml` and `RewardContext` but
  never computed — a dead-config pattern. Wiring it gives dense
  signal toward throwing at the opponent.
- **`dodge_held` reward** ([rewards.py](dodgeball/env/rewards.py)):
  rewards the agent for being far from the opponent *while the
  opponent holds the ball*. Only produces visible behaviour when
  the training opponent holds the ball long enough — hence
  `heuristic_holder`.
- **Curriculum phases actually advance** ([train.py](scripts/train.py)):
  the previous training loop had `while steps_remaining > 0` inside
  `for phase in phases`, which drained all steps in phase 1.
  Phases now get an even split of `total_steps`.
- **`--snapshot-root` flag** ([train.py](scripts/train.py)): snapshot
  filenames collide by step number across runs; isolated snapshot
  dirs prevent one run from silently overwriting another's
  checkpoints.

---

## License

Internal project — adjust before public release.