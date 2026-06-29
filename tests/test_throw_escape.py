"""Regression: a thrower must not instantly re-catch their own throw.

Before the cooldown fix, a thrown ball travelled <pickup_radius in one step
(520 * 0.033 = 17.3 < 18) and was re-caught by the stationary thrower at the
exact release point, with thrown_by wiped to None -- so throws never escaped
and nobody could land a hit. This is the root cause of the 'rarely throws /
can't land hits' bug.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dodgeball.env.dodgeball_parallel import DodgeBall2DParallelEnv


def test_throw_escapes_stationary_holder():
    env = DodgeBall2DParallelEnv(opponent="stationary", max_steps=50)
    a0 = env.agents_state[0]
    env.ball.held_by = 0
    env.ball.x, env.ball.y = a0.x, a0.y
    env.ball.thrown_by = None
    # Throw full-speed toward the opponent (+x), don't move.
    env.step({"player_0": [0.0, 0.0, 1.0, 0.0, 1.0]})
    assert env.ball.held_by is None, "thrower re-caught its own throw"
    assert env.ball.thrown_by == 0, "throw was not registered"
    assert env.ball.x > a0.x, "ball did not leave the thrower"


def test_opponent_can_still_catch():
    """The cooldown restricts only the thrower, not the opponent."""
    from dodgeball.env.entities import Ball, Agent, try_pickup
    ball = Ball(x=100, y=100)
    ball.thrown_by = 0
    ball.last_throw_step = 0
    thrower = Agent(x=100, y=100, side=0)  # on top of the ball
    opp = Agent(x=110, y=100, side=1)       # also within pickup_radius (18)
    picked = try_pickup(ball, [thrower, opp], pickup_radius=18,
                        now=0, cooldown=10)
    assert picked and ball.held_by == 1, "opponent should catch; thrower skipped"


if __name__ == "__main__":
    test_throw_escapes_stationary_holder()
    test_opponent_can_still_catch()
    print("OK: throws escape the thrower; opponent catching unaffected")
