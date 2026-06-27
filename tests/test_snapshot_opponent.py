"""Targeted test: use a saved PPO model as the snapshot opponent for player_1."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from stable_baselines3 import PPO
from dodgeball.env.dodgeball_parallel import DodgeBall2DParallelEnv


def main():
    env = DodgeBall2DParallelEnv(curriculum_phase=1, opponent="self")
    opp = PPO.load("models/dodgeball_p1_selfplay.zip", device="cpu")

    def opp_act(obs_p0, obs_p1):
        swap = np.concatenate([obs_p1, obs_p0], axis=0)
        a, _ = opp.predict(swap, deterministic=False)
        a[0] = -a[0]
        a[2] = -a[2]
        return a

    env.set_opponent_policy(opp_act)
    obs, infos = env.reset(seed=0)
    done = False
    ep_len = 0
    while not done and ep_len < 1500:
        a0 = np.random.uniform(-1, 1, size=5).astype(np.float32)
        obs, r, terms, truncs, infos = env.step({"player_0": a0})
        ep_len += 1
        if any(terms.values()) or any(truncs.values()):
            break
    print(f"OK: snapshot opponent ran for {ep_len} steps")
    print(f"  final rewards: {r}")
    print(f"  winner: {infos['player_0'].get('winner', '-')}")


if __name__ == "__main__":
    main()