"""Render a match and export to MP4/GIF."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import imageio.v2 as imageio

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from dodgeball.env.dodgeball_parallel import DodgeBall2DParallelEnv
from dodgeball.wrappers import _PettingZooToGymWrapper


class _Id:
    def normalize_obs(self, x): return x
    training = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="videos/match.mp4")
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--opponent", default="self")
    ap.add_argument("--phase", type=int, default=4)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    env = DodgeBall2DParallelEnv(curriculum_phase=args.phase,
                                 opponent=args.opponent,
                                 max_steps=args.max_steps,
                                 render_mode="rgb_array")
    wrapped = _PettingZooToGymWrapper(env,
                                     selfplay_mirror=(args.opponent == "self"))
    model = PPO.load(args.model, device="cpu")
    vn_path = args.model + "_vecnorm.pkl"
    if os.path.exists(vn_path):
        venv = DummyVecEnv([lambda: wrapped])
        vecnorm = VecNormalize.load(vn_path, venv)
        vecnorm.training = False
    else:
        vecnorm = _Id()

    frames = []
    obs, infos = wrapped.reset()
    done = False
    winner = "draw"
    while not done:
        flat = vecnorm.normalize_obs(obs)
        action, _ = model.predict(flat, deterministic=True)
        obs, _r, done, _trunc, infos = wrapped.step(action)
        frames.append(env.render())
    winner = infos.get("winner", "draw")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    imageio.mimsave(args.out, frames, fps=args.fps)
    print(f"Saved {len(frames)} frames to {args.out}")
    print(f"Episode length: {len(frames)} steps  (max {args.max_steps})")
    wrapped.close()


if __name__ == "__main__":
    main()