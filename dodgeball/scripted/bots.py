"""Scripted opponents for curriculum phases 1-3."""
from __future__ import annotations

import numpy as np

from dodgeball.env.entities import unit_vec


def stationary_bot(env, idx, learner_action=None) -> np.ndarray:
    """Does nothing - lets the ball come to it. Phase 1 dummy."""
    return np.zeros(5, dtype=np.float32)


def random_bot(env, idx, learner_action=None) -> np.ndarray:
    """Uniform random actions - baseline."""
    return np.random.uniform(-1.0, 1.0, size=5).astype(np.float32)


def heuristic_thrower_bot(env, idx, learner_action=None) -> np.ndarray:
    """If holding ball, throw toward opponent. Otherwise move toward ball."""
    me = env.agents_state[idx]
    opp = env.agents_state[1 - idx]
    ball = env.ball
    act = np.zeros(5, dtype=np.float32)

    if ball.held_by == idx:
        dx, dy = unit_vec(opp.x - me.x, opp.y - me.y)
        act[2], act[3] = dx, dy
        act[4] = 1.0
    else:
        dx, dy = unit_vec(ball.x - me.x, ball.y - me.y)
        act[0], act[1] = dx, dy
    return act


def heuristic_dodger_bot(env, idx, learner_action=None) -> np.ndarray:
    """If a thrown ball is incoming, dodge perpendicular to its trajectory.
    Otherwise move toward ball."""
    me = env.agents_state[idx]
    opp = env.agents_state[1 - idx]
    ball = env.ball
    act = np.zeros(5, dtype=np.float32)

    incoming = (ball.thrown_by is not None
                and ball.thrown_by != idx
                and ball.held_by is None
                and ball.vx != 0.0)
    if incoming:
        bx, by = ball.x - me.x, ball.y - me.y
        n = max((bx * bx + by * by) ** 0.5, 1e-6)
        bx /= n
        by /= n
        # perp
        p1x, p1y = -by, bx
        p2x, p2y = by, -bx
        # pick the side that increases y-distance from ball's predicted y
        ball_pred_y = ball.y + ball.vy * 0.3  # 0.3s lookahead
        if abs(ball.y - ball_pred_y) < 5.0:
            move = (p1x, p1y) if p1y > 0 else (p2x, p2y)
        else:
            move = (p1x, p1y) if (ball_pred_y - me.y) > 0 else (p2x, p2y)
        act[0], act[1] = float(move[0]), float(move[1])
        return act

    if ball.held_by == idx:
        dx, dy = unit_vec(opp.x - me.x, opp.y - me.y)
        act[2], act[3] = dx, dy
        act[4] = 1.0
    else:
        dx, dy = unit_vec(ball.x - me.x, ball.y - me.y)
        act[0], act[1] = dx, dy
    return act


def heuristic_holder_bot(env, idx, learner_action=None) -> np.ndarray:
    """Picks up ball, then HOLDS for HOLD_TIME steps (no throw trigger),
    then throws at opponent. Creates a sustained 'opp has the ball' window
    so the learner can learn to back off and dodge -- versus the thrower
    bots which grab-and-throw instantly."""
    me = env.agents_state[idx]
    opp = env.agents_state[1 - idx]
    ball = env.ball
    act = np.zeros(5, dtype=np.float32)
    HOLD_TIME = 30

    if ball.held_by == idx:
        if getattr(env, "_holder_pickup_step", None) is None:
            env._holder_pickup_step = env.step_count
        held_for = env.step_count - env._holder_pickup_step
        if held_for >= HOLD_TIME:
            dx, dy = unit_vec(opp.x - me.x, opp.y - me.y)
            act[2], act[3] = dx, dy
            act[4] = 1.0
        # else: hold -- act stays zero (no move, no throw trigger)
    else:
        env._holder_pickup_step = None
        dx, dy = unit_vec(ball.x - me.x, ball.y - me.y)
        act[0], act[1] = dx, dy
    return act


SCRIPTED_FUNCS = {
    "stationary": stationary_bot,
    "random": random_bot,
    "heuristic": heuristic_thrower_bot,
    "heuristic_dodger": heuristic_dodger_bot,
    "heuristic_holder": heuristic_holder_bot,
}