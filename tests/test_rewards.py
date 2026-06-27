"""Unit tests for the reward function."""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dodgeball.env.dodgeball_parallel import DodgeBall2DParallelEnv
from dodgeball.env.rewards import RewardState, compute_rewards, make_context, load_rewards


def test_terminal_hit_signs():
    env = DodgeBall2DParallelEnv(curriculum_phase=1, opponent="self", max_steps=100)
    env.ball.held_by = 0
    env.ball.thrown_by = 0
    rewards, comps = compute_rewards(
        env.ball, env.ball, env.agents_state, env.reward_ctx,
        env.rs, env.arena, done_terminal=1, timeout=False,
    )
    # 10.0 terminal minus 0.001 time_penalty per agent per step
    assert abs(rewards[0] - (env.reward_ctx.terminal_hit - env.reward_ctx.time_penalty)) < 1e-6, rewards
    assert abs(rewards[1] - (env.reward_ctx.terminal_got_hit - env.reward_ctx.time_penalty)) < 1e-6, rewards


def test_timeout_signs():
    env = DodgeBall2DParallelEnv(curriculum_phase=1, opponent="self", max_steps=100)
    rewards, _ = compute_rewards(
        env.ball, env.ball, env.agents_state, env.reward_ctx,
        env.rs, env.arena, done_terminal=None, timeout=True,
    )
    expected = env.reward_ctx.terminal_timeout - env.reward_ctx.time_penalty
    assert abs(rewards[0] - expected) < 1e-6
    assert abs(rewards[1] - expected) < 1e-6


def test_phase_change_alters_coefficients():
    cfg = load_rewards("configs/rewards.yaml")
    c1 = make_context(cfg, 1)
    c4 = make_context(cfg, 4)
    assert c1.shaping_scale > c4.shaping_scale
    assert c1.approach_ball >= c4.approach_ball


def test_pickup_bonus_pays_only_holder():
    env = DodgeBall2DParallelEnv(curriculum_phase=1, opponent="self", max_steps=100)
    env.ball.held_by = 0
    env._prev_ball_snapshot.held_by = None
    rewards, _ = compute_rewards(
        env._prev_ball_snapshot, env.ball, env.agents_state,
        env.reward_ctx, env.rs, env.arena,
        done_terminal=None, timeout=False,
    )
    # player_0 gets pickup_bonus, player_1 only gets the step penalty.
    # difference = pickup_ball (>= 1.0 in phase 1) + any approach shaping.
    assert rewards[0] - rewards[1] > 0.5, f"holder should get more reward: {rewards}"


if __name__ == "__main__":
    test_terminal_hit_signs()
    test_timeout_signs()
    test_phase_change_alters_coefficients()
    test_pickup_bonus_pays_only_holder()
    print("all reward tests passed")