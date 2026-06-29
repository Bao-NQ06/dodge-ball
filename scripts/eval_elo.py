"""Score saved snapshots vs a FIXED reference opponent -> win-rate + ELO line.

train.py registers snapshots in ELORater but never plays a match, so every
rating sits at the 1200 default. This actually plays the games.

We drive the RAW env (not _PettingZooToGymWrapper): that wrapper auto-resets
on terminal and overwrites infos, so the `winner` key is lost -- which is why
the existing evaluate.py always reports draws. The raw env exposes winner.

ELO here is a one-shot *performance rating* vs a fixed opponent (FIDE
formula):  anchor + 400*log10(s/(1-s)). Iterative elo.update against a single
pinned opponent collapses to ~0.5 expected, so it carries no signal; the
formula is the honest one-shot rating for a fixed-reference series.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from dodgeball.env.dodgeball_parallel import DodgeBall2DParallelEnv


class _ObsOnly(gym.Env):
    """Throwaway env so VecNormalize.load has something to attach to.
    Only its observation_space matters; it is never stepped."""
    observation_space = spaces.Box(-np.inf, np.inf, (34,), np.float32)
    action_space = spaces.Box(-1.0, 1.0, (5,), np.float32)

    def reset(self, *, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, True, False, {}


def load_normalizer(path: str):
    """normalize_obs(obs)->obs, or identity if no pkl. Snapshots are saved
    without VecNormalize stats, so pass the run's final *_vecnorm.pkl."""
    if not path or not os.path.exists(path):
        return lambda x: np.asarray(x, dtype=np.float32)
    vn = VecNormalize.load(path, DummyVecEnv([lambda: _ObsOnly()]))
    vn.training = False
    return vn.normalize_obs


def perf_elo(score: float, anchor: float = 1200.0) -> float:
    """One-shot ELO for an observed score vs a fixed `anchor`-rated opponent.
    Clamped so 0%/100% don't diverge."""
    s = min(max(score, 0.01), 0.99)
    return anchor + 400.0 * math.log10(s / (1.0 - s))


def _step_from_path(p: str) -> int:
    digits = "".join(c for c in os.path.basename(p) if c.isdigit())
    return int(digits) if digits else 0


def play_matches(model, normalize, opponent_kind, ref_fn, phase, max_steps,
                 matches, base_seed, deterministic=True):
    """Snapshot = player_0, reference = player_1.
    Returns (wins, draws, losses, mean_ep_len, mean_p0_throws)."""
    wins = draws = losses = 0
    ep_lens, p0_throws = [], []
    for i in range(matches):
        env = DodgeBall2DParallelEnv(
            curriculum_phase=phase,
            opponent=opponent_kind,
            max_steps=max_steps,
        )
        if ref_fn is not None:
            env.set_opponent_policy(ref_fn)
        obs, _ = env.reset(seed=base_seed + i)
        winner = -1
        throws = 0
        for t in range(max_steps):
            # Model input is the 34-dim [p0, p1] concat (matches training
            # and the obs_rms shape); a single 17-dim obs won't normalize.
            flat = normalize(np.concatenate(
                [obs["player_0"], obs["player_1"]]).astype(np.float32))
            action, _ = model.predict(flat, deterministic=deterministic)
            action = np.asarray(action, dtype=np.float32)
            held_before = env.ball.held_by
            obs, _r, terms, truncs, infos = env.step({"player_0": action})
            # player_0 threw this step if it held the ball and pulled the trigger
            if held_before == 0 and action[4] > 0.0 and env.ball.thrown_by == 0:
                throws += 1
            if terms["player_0"] or truncs["player_0"]:
                winner = infos["player_0"].get("winner", -1)
                ep_lens.append(t + 1)
                break
        else:
            ep_lens.append(max_steps)
        p0_throws.append(throws)
        env.close()
        if winner == 0:
            wins += 1
        elif winner == -1:
            draws += 1
        else:
            losses += 1
    n = max(matches, 1)
    return wins, draws, losses, sum(ep_lens) / n, sum(p0_throws) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", nargs="+",
                    default=["models/snapshots/step_01215808.zip",
                             "models/snapshots/step_02000000.zip"],
                    help="Snapshot paths to score as player_0 (globs ok).")
    ap.add_argument("--ref", default="heuristic",
                    help="Fixed reference (player_1): a bot name "
                         "(stationary/random/heuristic/heuristic_dodger) or a "
                         "model .zip path.")
    ap.add_argument("--vecnorm", default="models/dodgeball_p1_vsbot_v5_vecnorm.pkl",
                    help="Obs-normalizer pkl for the snapshots.")
    ap.add_argument("--matches", type=int, default=20)
    ap.add_argument("--phase", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--stochastic", action="store_true",
                    help="Sample actions instead of deterministic argmax. Use "
                         "to check whether a 0-hit result is a deterministic-"
                         "policy artifact.")
    ap.add_argument("--anchor-elo", type=float, default=1200.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="models/eval_elo.json")
    args = ap.parse_args()

    paths = []
    for s in args.snapshots:
        paths.extend(sorted(glob.glob(s)) if any(c in s for c in "*?[") else [s])
    paths = sorted(set(paths), key=_step_from_path)
    if not paths:
        print("No snapshots found.")
        return

    normalize = load_normalizer(args.vecnorm)

    ref_fn = None
    opponent_kind = "heuristic"
    if args.ref.endswith(".zip"):
        opp_model = PPO.load(args.ref, device="cpu")
        # Prefer the ref model's own vecnorm; fall back to the snapshots'.
        ref_vn = args.ref[:-4] + "_vecnorm.pkl"
        opp_norm = load_normalizer(ref_vn if os.path.exists(ref_vn) else args.vecnorm)

        def ref_fn(obs_p0, obs_p1):
            swap = opp_norm(np.concatenate([obs_p1, obs_p0]).astype(np.float32))
            a, _ = opp_model.predict(swap, deterministic=True)
            a = np.asarray(a, dtype=np.float32).copy()
            a[0] = -a[0]  # mirror x-components back to player_1 frame
            a[2] = -a[2]
            return a

        opponent_kind = "self"  # env must not also run a scripted bot
        ref_name = os.path.basename(args.ref)
    else:
        opponent_kind = args.ref
        ref_name = args.ref

    rows = []
    det = not args.stochastic
    print(f"Reference (player_1): {ref_name}  |  {args.matches} matches/snapshot  "
          f"|  {'stochastic' if args.stochastic else 'deterministic'}\n")
    print(f"{'step':>10}  {'W':>3} {'D':>3} {'L':>3}  {'winrate':>7}  {'ELO':>7}"
          f"  {'ep_len':>6}  {'p0_thr':>6}")
    for p in paths:
        model = PPO.load(p, device="cpu")
        w, d, l, ep_len, throws = play_matches(
            model, normalize, opponent_kind, ref_fn, args.phase,
            args.max_steps, args.matches, args.seed, deterministic=det)
        score = (w + 0.5 * d) / args.matches
        elo = perf_elo(score, args.anchor_elo)
        print(f"{_step_from_path(p):>10}  {w:>3} {d:>3} {l:>3}  {score:>6.1%}  "
              f"{elo:>7.0f}  {ep_len:>6.0f}  {throws:>6.1f}")
        rows.append({"step": _step_from_path(p), "path": p, "wins": w,
                     "draws": d, "losses": l, "winrate": score, "elo": elo,
                     "mean_ep_len": ep_len, "mean_p0_throws": throws,
                     "ref": ref_name, "deterministic": det})

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"ref": ref_name, "matches": args.matches, "rows": rows},
                  f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
