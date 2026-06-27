"""Vectorization wrappers around the PettingZoo ParallelEnv for SB3 PPO.

We use parameter-sharing self-play: both agents see the same policy
input, so we concat the two observations into one 34-dim vector (mirror the
second half so player 1's frame is egocentric).

For non-self-play (scripted opponents), only player_0 is the learner and
player_1 is driven by the scripted bot. In that case we still concat as 34-dim
but the second half is the opponent's view.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
from gymnasium import Env, spaces

from dodgeball.env.dodgeball_parallel import (
    DodgeBall2DParallelEnv, OBS_DIM, ACT_DIM,
)


CONCAT_OBS_DIM = OBS_DIM * 2  # 34


class _PettingZooToGymWrapper(Env):
    """Converts PettingZoo ParallelEnv to gymnasium.Env for SB3 Monitor.

    Concat both agents' observations. Player_0 controls the env with a
    5-dim action; player_1 is filled in by the inner env's own logic
    (scripted bot, snapshot opponent, or mirrored action).
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, inner_env: DodgeBall2DParallelEnv,
                 selfplay_mirror: bool = False):
        super().__init__()
        self.env = inner_env
        self.selfplay_mirror = selfplay_mirror
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(CONCAT_OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32)
        self.render_mode = "rgb_array"

    def reset(self, *, seed: Optional[int] = None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        return self._flat(obs), infos

    def step(self, action):
        a = np.asarray(action, dtype=np.float32)
        if self.selfplay_mirror:
            a0 = a
            a1 = a.copy()
            a1[0] = -a1[0]
            a1[2] = -a1[2]
            obs, r, terms, truncs, infos = self.env.step(
                {"player_0": a0, "player_1": a1})
        else:
            obs, r, terms, truncs, infos = self.env.step({"player_0": a})
        r_total = float(r["player_0"] + r["player_1"])
        done = bool(terms["player_0"] or truncs["player_0"])
        # If terminated/truncated, reset internally so SB3 sees a continuous stream.
        if done:
            obs, infos = self.env.reset()
        return self._flat(obs), r_total, done, False, infos

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    def set_curriculum_phase(self, phase: int) -> None:
        self.env.set_curriculum_phase(phase)

    def set_opponent_kind(self, kind: str) -> None:
        self.env.set_opponent_kind(kind)

    def set_opponent_policy(self, fn) -> None:
        self.env.set_opponent_policy(fn)
    def _flat(self, obs_dict):
        return np.concatenate([obs_dict["player_0"], obs_dict["player_1"]], axis=0)


def make_vec_env(num_envs: int = 1, mode: str = "selfplay",
                 curriculum_phase: int = 1,
                 opponent: str = "self") -> Any:
    """Build a vectorized env for SB3.

    mode='selfplay' -> mirror actions for player_1.
    mode='vs_bot'   -> player_1 driven by scripted bot (inner env handles it).
    """
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

    selfplay_mirror = (mode == "selfplay")

    def _thunk(rank: int) -> Callable[[], Env]:
        def _f() -> Env:
            inner = DodgeBall2DParallelEnv(
                curriculum_phase=curriculum_phase,
                opponent=opponent if mode == "vs_bot" else "self",
            )
            env = _PettingZooToGymWrapper(inner, selfplay_mirror=selfplay_mirror)
            env = Monitor(env)
            return env
        return _f

    if num_envs == 1:
        venv = DummyVecEnv([_thunk(0)])
    else:
        # ponytail: Windows + spawn can't pickle the wrapper closure carrying
        # the SB3 Monitor parent. DummyVecEnv at n_envs>1 still beats 1 env
        # because PPO gets n_envs×rollout per learn() call, keeping the GPU
        # busy. Switch to SubprocVecEnv only if env stepping shows up in profiles.
        venv = DummyVecEnv([_thunk(i) for i in range(num_envs)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    return venv