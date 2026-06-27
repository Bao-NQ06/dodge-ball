"""Unit tests for entities & physics."""
from __future__ import annotations

import math
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dodgeball.env.entities import (
    Arena, Agent, Ball, step_agent, step_ball, throw_ball,
    try_pickup, check_hit, reset_state, distance,
)


def test_step_agent_stays_in_half():
    a = Arena(width=800, height=400)
    ag = Agent(x=100, y=200, side=0)
    # push right toward center line
    for _ in range(60):
        step_agent(ag, 1.0, 0.0, 1/30, 220.0, 1200.0, 4.0, a)
    assert ag.x < a.center_x, f"player 0 crossed center: {ag.x}"


def test_step_ball_bounces_off_wall():
    a = Arena(width=800, height=400)
    b = Ball(x=10, y=200, vx=-300, vy=0)
    step_ball(b, 1/30, a, 0.9, 0.0)
    assert b.vx > 0, f"ball did not bounce: vx={b.vx}"


def test_throw_ball_and_hit():
    a = Arena(width=800, height=400)
    a0 = Agent(x=100, y=200, side=0)
    a1 = Agent(x=700, y=200, side=1)
    b = Ball(x=100, y=200, held_by=0)
    # throw straight right at opponent
    throw_ball(b, 0, 1.0, 0.0, 1.0, 520.0)
    assert b.held_by is None
    assert b.thrown_by == 0
    # step a long time, ball should reach player 1
    hit = None
    for _ in range(120):
        step_ball(b, 1/30, a, 0.9, 0.0)
        hit = check_hit(b, [a0, a1])
        if hit is not None:
            break
    assert hit == 1, f"player 1 should have been hit, got {hit}"


def test_pickup_when_overlapping():
    a = Arena(width=800, height=400)
    a0 = Agent(x=400, y=200, side=0)
    a1 = Agent(x=400, y=200, side=1)  # both on top of ball
    b = Ball(x=400, y=200)
    picked = try_pickup(b, [a0, a1], pickup_radius=20.0)
    assert picked
    assert b.held_by in (0, 1)


def test_center_line_constraint():
    a = Arena(width=800, height=400)
    ag = Agent(x=100, y=200, side=0)
    for _ in range(100):
        step_agent(ag, 1.0, 0.0, 1/30, 1000.0, 5000.0, 0.0, a)
    assert ag.x <= a.center_x - ag.radius + 1e-3


if __name__ == "__main__":
    test_step_agent_stays_in_half()
    test_step_ball_bounces_off_wall()
    test_throw_ball_and_hit()
    test_pickup_when_overlapping()
    test_center_line_constraint()
    print("all physics tests passed")