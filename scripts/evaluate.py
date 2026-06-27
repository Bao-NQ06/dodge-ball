"""Evaluate a trained PPO model over N deterministic matches."""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from dodgeball.env.dodgeball_parallel import DodgeBall2DParallelEnv
from dodgeball.wrappers import _PettingZooToGymWrapper


class _IdentityNormalizer:
    def normalize_obs(self, x):
        return x
    training = False


def make_vecnorm(model_path: str, inner_factory):
    vn_path = model_path + "_vecnorm.pkl"
    if os.path.exists(vn_path):
        venv = DummyVecEnv([inner_factory])
        vn = VecNormalize.load(vn_path, venv)
        vn.training = False
        vn.norm_reward = False
        return vn
    return _IdentityNormalizer()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--matches", type=int, default=20)
    ap.add_argument("--opponent", default="self",
                    choices=["self", "stationary", "random", "heuristic"])
    ap.add_argument("--phase", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=1500)
    args = ap.parse_args()

    model = PPO.load(args.model, device="cpu")
    vecnorm = make_vecnorm(args.model, lambda: _PettingZooToGymWrapper(
        DodgeBall2DParallelEnv(curriculum_phase=args.phase,
                               opponent=args.opponent,
                               max_steps=args.max_steps),
        selfplay_mirror=(args.opponent == "self"),
    ))

    results = defaultdict(int)
    lengths = []
    for i in range(args.matches):
        env = DodgeBall2DParallelEnv(curriculum_phase=args.phase,
                                     opponent=args.opponent,
                                     max_steps=args.max_steps)
        wrapped = _PettingZooToGymWrapper(
            env,
            selfplay_mirror=(args.opponent == "self"),
        )
        obs, infos = wrapped.reset(seed=i)
        done = False
        ep_len = 0
        winner = -1
        while not done:
            flat = vecnorm.normalize_obs(obs)
            action, _ = model.predict(flat, deterministic=True)
            obs, _r, done, _trunc, infos = wrapped.step(action)
            ep_len += 1
        winner = infos.get("winner", -1)
        results[winner] += 1
        lengths.append(ep_len)
        wrapped.close()

    print(f"Matches: {args.matches}, opponent={args.opponent}, phase={args.phase}")
    print(f"  player_0 wins: {results[0]}")
    print(f"  player_1 wins: {results[1]}")
    print(f"  draws:         {results[-1]}")
    print(f"  avg episode length: {np.mean(lengths):.1f}")


if __name__ == "__main__":
    main()