"""Trace what the policy actually does per match: pickup / hold / throw / move.

Diagnostic for the 'rarely throws, can't land hits' bug. Run deterministic to
see the deployable behavior, --stochastic to see training-time behavior.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from dodgeball.env.dodgeball_parallel import DodgeBall2DParallelEnv
from dodgeball.env.entities import distance


class _ObsOnly(gym.Env):
    observation_space = spaces.Box(-np.inf, np.inf, (34,), np.float32)
    action_space = spaces.Box(-1.0, 1.0, (5,), np.float32)

    def reset(self, *, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, a):
        return self.observation_space.sample(), 0.0, True, False, {}


def load_norm(path):
    if not path or not os.path.exists(path):
        return lambda x: np.asarray(x, dtype=np.float32)
    vn = VecNormalize.load(path, DummyVecEnv([lambda: _ObsOnly()]))
    vn.training = False
    return vn.normalize_obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/snapshots/step_02000000.zip")
    ap.add_argument("--vecnorm", default="models/dodgeball_p1_vsbot_v5_vecnorm.pkl")
    ap.add_argument("--opponent", default="heuristic")
    ap.add_argument("--matches", type=int, default=10)
    ap.add_argument("--phase", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--stochastic", action="store_true")
    args = ap.parse_args()

    norm = load_norm(args.vecnorm)
    model = PPO.load(args.model, device="cpu")

    T = dict(pickups=0, throws=0, hold=0, free=0, opp=0,
             wins=0, draws=0, losses=0, dist=0.0, hold_a4=[], throw_a4=[],
             p0_opp_dist_sum=0.0, p0_opp_steps=0,
             opp_hold_dist_sum=0.0, opp_hold_steps=0)
    for i in range(args.matches):
        env = DodgeBall2DParallelEnv(
            curriculum_phase=args.phase, opponent=args.opponent,
            max_steps=args.max_steps)
        obs, _ = env.reset(seed=1000 + i)
        prev = np.array([env.agents_state[0].x, env.agents_state[0].y])
        winner = -1
        for _ in range(args.max_steps):
            flat = norm(np.concatenate(
                [obs["player_0"], obs["player_1"]]).astype(np.float32))
            a, _ = model.predict(flat, deterministic=not args.stochastic)
            a = np.asarray(a, dtype=np.float32)
            held_before = env.ball.held_by
            obs, _r, terms, truncs, infos = env.step({"player_0": a})
            hb = env.ball.held_by
            if hb == 0:
                T["hold"] += 1
                T["hold_a4"].append(float(a[4]))
            elif hb == 1:
                T["opp"] += 1
            else:
                T["free"] += 1
            if hb == 0 and held_before != 0:
                T["pickups"] += 1
            d_p0_p1 = distance(env.agents_state[0].pos(),
                               env.agents_state[1].pos())
            T["p0_opp_dist_sum"] += d_p0_p1
            T["p0_opp_steps"] += 1
            if hb == 1:
                T["opp_hold_dist_sum"] += d_p0_p1
                T["opp_hold_steps"] += 1
            if held_before == 0 and a[4] > 0.0 and env.ball.thrown_by == 0:
                T["throws"] += 1
                T["throw_a4"].append(float(a[4]))
            pos = np.array([env.agents_state[0].x, env.agents_state[0].y])
            T["dist"] += float(np.linalg.norm(pos - prev))
            prev = pos
            if terms["player_0"] or truncs["player_0"]:
                winner = infos["player_0"].get("winner", -1)
                break
        if winner == 0:
            T["wins"] += 1
        elif winner == -1:
            T["draws"] += 1
        else:
            T["losses"] += 1
        env.close()

    n = args.matches
    ha = np.asarray(T["hold_a4"])
    print(f"model={os.path.basename(args.model)}  opp={args.opponent}  "
          f"matches={n}  {'stochastic' if args.stochastic else 'deterministic'}")
    print(f"  result W/D/L       : {T['wins']}/{T['draws']}/{T['losses']}")
    print(f"  p0 pickups/match   : {T['pickups']/n:.2f}")
    print(f"  p0 throws/match    : {T['throws']/n:.2f}")
    print(f"  ball steps/match   : p0-hold {T['hold']/n:.0f}  "
          f"free {T['free']/n:.0f}  opp-hold {T['opp']/n:.0f}")
    print(f"  p0 move dist/match : {T['dist']/n:.0f}  (max ~{220*0.0333*1500:.0f})")
    if len(ha):
        print(f"  a[4] while holding : mean={ha.mean():+.3f}  frac>0="
              f"{(ha > 0).mean():.2f}  (n={len(ha)})")
    if T["throw_a4"]:
        print(f"  a[4] on throws     : mean={np.mean(T['throw_a4']):+.3f}  "
              f"(n={len(T['throw_a4'])})")
    if T["p0_opp_steps"]:
        print(f"  p0<->opp dist (all)      : "
              f"{T['p0_opp_dist_sum']/T['p0_opp_steps']:.1f}")
    if T["opp_hold_steps"]:
        print(f"  p0<->opp dist (opp holds): "
              f"{T['opp_hold_dist_sum']/T['opp_hold_steps']:.1f}  "
              f"(n={T['opp_hold_steps']})")


if __name__ == "__main__":
    main()
