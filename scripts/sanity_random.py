"""Random-agent sanity check: roll out the env with random actions,
print per-step info, save frames as an MP4."""
from __future__ import annotations

import os
import sys
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dodgeball.env.dodgeball_parallel import DodgeBall2DParallelEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--out", default="videos/random_sanity.mp4")
    args = ap.parse_args()

    env = DodgeBall2DParallelEnv(curriculum_phase=1,
                                 max_steps=args.max_steps,
                                 opponent="random",
                                 render_mode="rgb_array")
    frames = []

    for ep in range(args.episodes):
        obs, infos = env.reset(seed=42 + ep)
        ep_r0 = ep_r1 = 0.0
        steps = 0
        winner = "draw"
        for t in range(args.max_steps):
            # build actions using possible_agents (env clears self.agents after termination)
            actions = {a: np.random.uniform(-1, 1, size=5).astype(np.float32)
                       for a in env.possible_agents}
            try:
                obs, r, terms, truncs, infos = env.step(actions)
            except KeyError:
                break
            ep_r0 += r["player_0"]
            ep_r1 += r["player_1"]
            steps += 1
            if args.save_video and env.agents:
                # can only render while agents are alive
                frames.append(env.render())
            if any(terms.values()) or any(truncs.values()):
                winner = infos.get("player_0", {}).get("winner", "draw")
                break
        print(f"episode {ep}: steps={steps}  R0={ep_r0:+.2f}  R1={ep_r1:+.2f}  "
              f"winner={'player_' + str(winner) if isinstance(winner, int) else winner}")

    if args.save_video and frames:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        import imageio.v2 as imageio
        imageio.mimsave(args.out, frames, fps=30)
        print(f"Saved {len(frames)} frames to {args.out}")

    env.close()


if __name__ == "__main__":
    main()