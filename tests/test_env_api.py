"""PettingZoo parallel_api_test for DodgeBall2DParallelEnv."""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pettingzoo.test import parallel_api_test
from dodgeball.env.dodgeball_parallel import DodgeBall2DParallelEnv


def test_env_api():
    env = DodgeBall2DParallelEnv(curriculum_phase=1, opponent="self", max_steps=200)
    parallel_api_test(env, num_cycles=20)


if __name__ == "__main__":
    test_env_api()
    print("env api test passed")